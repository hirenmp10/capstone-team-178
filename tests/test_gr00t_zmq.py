"""GR00T production transport (ZeroMQ + msgpack) and the closed loop, end to end.

What this file proves, and what it does not
-------------------------------------------
``configs/default.yaml`` sets ``use_mock_server: false``, so the transport a real
run uses is :class:`~mfw.gr00t_bridge.zmq_client.Gr00tZmqClient` talking to
NVIDIA's ``PolicyServer``. Until this file existed no test imported that client:
the suite exercised only the pickle transport the mock uses.

The server here is an in-thread ZeroMQ REP loop that reproduces NVIDIA's
``gr00t/policy/server_client.py`` (n1.7-release) dispatch rule for rule:

* the request is a msgpack map ``{"endpoint", "data", "api_token"?}`` decoded
  with msgpack-numpy;
* a bad token answers ``{"error": "Unauthorized: Invalid API token"}``;
* ``handler(**data)`` for endpoints that take input, ``handler()`` otherwise;
* ``get_action`` returns the tuple ``(action, info)``, which msgpack delivers as
  a list;
* any exception, including an unknown endpoint, answers ``{"error": str(e)}``.

The policy behind it is a scripted **fake**. It validates the observation
against the ``oxe_droid_relative_eef_relative_joint`` layout and emits absolute
``eef_9d`` poses. **No model inference happens in these tests** -- they validate
the bridge (framing, token, errors, timeouts, the executor's closed loop and its
working-memory contract), not GR00T's behaviour.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")
msgpack_numpy = pytest.importorskip("msgpack_numpy")

from mfw.config.schema import MemoryConfig, load_config  # noqa: E402
from mfw.core.errors import PolicyError  # noqa: E402
from mfw.core.types import (  # noqa: E402
    BoundingBox3D,
    Frame,
    GripperState,
    JointState,
    ObjectHypothesis,
    Pose,
    RobotState,
    SceneGraph,
    SkillStatus,
)
from mfw.gr00t_bridge.executor import Gr00tExecutor, support_height  # noqa: E402
from mfw.gr00t_bridge.safety import DEFAULT_SUPPORT_CLEARANCE_M  # noqa: E402
from mfw.gr00t_bridge.zmq_client import API_TOKEN_ENV, Gr00tZmqClient  # noqa: E402
from mfw.memory.working_memory import WorkingMemory  # noqa: E402

pytestmark = pytest.mark.phase7

REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
HORIZON = 40


# ---------------------------------------------------------------------------
# A REP server with NVIDIA's PolicyServer dispatch semantics
# ---------------------------------------------------------------------------


class NvidiaProtocolServer:
    """In-thread stand-in for ``gr00t.policy.server_client.PolicyServer``.

    Only the loop is different: it polls so the fixture can stop it. Everything
    a client can observe -- framing, dispatch, the error envelope, token
    handling -- follows the upstream ``run()`` method.
    """

    def __init__(self, policy: Any, api_token: str | None = None) -> None:
        self.policy = policy
        self.api_token = api_token
        self.requests: list[dict[str, Any]] = []
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self.port = self._socket.bind_to_random_port("tcp://127.0.0.1")
        self._endpoints: dict[str, tuple[Callable[..., Any], bool]] = {
            "ping": (lambda: {"status": "ok", "message": "Server is running"}, False),
            "kill": (self._kill, False),
            "get_action": (policy.get_action, True),
            "reset": (policy.reset, True),
            "get_modality_config": (policy.get_modality_config, False),
        }
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "NvidiaProtocolServer":
        self._thread.start()
        return self

    def _kill(self) -> None:
        self._running.clear()

    @staticmethod
    def _pack(obj: Any) -> bytes:
        return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)

    @staticmethod
    def _unpack(raw: bytes) -> Any:
        return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running.is_set():
            if not poller.poll(50):
                continue
            message = self._socket.recv()
            try:
                request = self._unpack(message)
                self.requests.append(request)
                if self.api_token is not None and request.get("api_token") != self.api_token:
                    self._socket.send(self._pack({"error": "Unauthorized: Invalid API token"}))
                    continue
                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                handler, requires_input = self._endpoints[endpoint]
                result = handler(**request.get("data", {})) if requires_input else handler()
                self._socket.send(self._pack(result))
            except Exception as exc:  # noqa: BLE001 - mirrors upstream's blanket catch
                self._socket.send(self._pack({"error": str(exc)}))

    def close(self) -> None:
        self._running.clear()
        self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()


class DroidReachPolicy:
    """Scripted fake policy: reach the object, close, lift; (for 'put down') open.

    ``get_action`` enforces the observation layout NVIDIA's validator requires
    for oxe_droid, so a regression in :class:`ObservationBuilder` or in the
    request wrapping fails here as a server error. It then emits **absolute**
    ``eef_9d`` poses (what N1.7 returns after its processor decodes relative
    actions), 5 mm per step toward ``goal``, with the gripper closing once the
    TCP is within 1 cm. ``gripper_position`` is DROID closure (0 open .. 1
    closed) in both the state it reads and the action it emits.
    """

    #: The fake reads DROID closure (0 open .. 1 closed) from the observed
    #: state: above this the fingers have closed on something.
    HOLDING_ABOVE = 0.1

    def __init__(self, goal: np.ndarray, delay_s: float = 0.0, lift_m: float = 0.10) -> None:
        self.goal = np.asarray(goal, dtype=np.float64)
        self.delay_s = delay_s
        self.lift_m = float(lift_m)
        self.calls = 0
        self.observations: list[dict[str, Any]] = []
        self.instructions: list[str] = []

    def get_action(self, observation: dict[str, Any], options: Any = None):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        self._check(observation)
        self.observations.append(observation)

        eef = np.asarray(observation["state"]["eef_9d"], dtype=np.float64).reshape(9)
        closure = float(np.asarray(observation["state"]["gripper_position"]).reshape(-1)[0])
        instruction = observation["language"]["annotation.language.language_instruction"][0][0]
        self.instructions.append(instruction)
        releasing = instruction.startswith("put down")
        # "pick UP": once the fingers have closed on something, carry it
        # upward, still closed.
        holding = not releasing and closure > self.HOLDING_ABOVE
        goal = self.goal + np.array([0.0, 0.0, self.lift_m]) if holding else self.goal

        position, rot6d = eef[:3], eef[3:]
        to_goal = goal - position
        distance = float(np.linalg.norm(to_goal))
        direction = to_goal / distance if distance > 1e-9 else np.zeros(3)
        travel = np.minimum(np.arange(1, HORIZON + 1) * 0.005, distance)
        poses = np.hstack([position + travel[:, None] * direction, np.tile(rot6d, (HORIZON, 1))])

        close = holding or (distance < 0.01 and not releasing)
        # DROID closure, as N1.7 emits it: 1 = close, 0 = open.
        gripper = np.full((1, HORIZON, 1), 1.0 if close else 0.0, dtype=np.float32)
        action = {
            "eef_9d": poses[None].astype(np.float32),
            "gripper_position": gripper,
            "joint_position": np.zeros((1, HORIZON, 7), dtype=np.float32),
        }
        return action, {}

    @staticmethod
    def _check(observation: dict[str, Any]) -> None:
        video = observation["video"]
        for key in ("exterior_image_1_left", "wrist_image_left"):
            frames = np.asarray(video[key])
            if frames.ndim != 5 or frames.shape[:2] != (1, 2) or frames.dtype != np.uint8:
                raise ValueError(f"Video key must be (B, T, H, W, C) uint8, got {frames.shape}")
        state = observation["state"]
        for key, dim in (("eef_9d", 9), ("gripper_position", 1), ("joint_position", 7)):
            if np.asarray(state[key]).shape != (1, 1, dim):
                raise ValueError(f"state {key} must be (1, 1, {dim})")
        language = observation["language"]["annotation.language.language_instruction"]
        if not (isinstance(language, list) and isinstance(language[0], list)):
            raise ValueError("language must be nested (B, T)")

    def reset(self, options: Any = None) -> dict[str, Any]:
        return {}

    def get_modality_config(self) -> dict[str, Any]:
        return {"video": ["exterior_image_1_left", "wrist_image_left"], "state": ["eef_9d"]}


@pytest.fixture
def serve():
    servers: list[NvidiaProtocolServer] = []

    def _serve(policy: Any, api_token: str | None = None) -> NvidiaProtocolServer:
        server = NvidiaProtocolServer(policy, api_token=api_token).start()
        servers.append(server)
        return server

    yield _serve
    for server in servers:
        server.close()


def _config(port: int, timeout_s: float = 3.0):
    return load_config(
        overrides={
            "gr00t": {
                "enabled": True,
                "use_mock_server": False,
                "port": port,
                "request_timeout_s": timeout_s,
            }
        }
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class TestZmqTransport:
    def test_connect_pings_and_predict_round_trips(self, serve):
        policy = DroidReachPolicy(goal=[0.5, 0.0, 0.45])
        server = serve(policy)
        client = Gr00tZmqClient(_config(server.port).gr00t)
        client.connect(retries=1)
        try:
            assert client.is_ready()
            from mfw.gr00t_bridge.observation import ObservationBuilder

            builder = ObservationBuilder(client.config)
            builder.prime(np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64, 3), np.uint8))
            observation = builder.build(_state([0.3, 0.0, 0.6]), "pick up the can")
            action = client.predict(observation)
        finally:
            client.close()

        assert action["eef_9d"].shape == (1, HORIZON, 9)
        assert action["gripper_position"].shape == (1, HORIZON, 1)
        # The request carried NVIDIA's envelope: get_action(**data) needs both keys.
        request = server.requests[-1]
        assert request["endpoint"] == "get_action"
        assert set(request["data"]) == {"observation", "options"}
        assert "api_token" not in request

    def test_get_modality_config_returns_the_servers_mapping(self, serve):
        server = serve(DroidReachPolicy(goal=[0.5, 0.0, 0.45]))
        client = Gr00tZmqClient(_config(server.port).gr00t)
        client.connect(retries=1)
        try:
            assert client.get_modality_config()["video"] == [
                "exterior_image_1_left",
                "wrist_image_left",
            ]
        finally:
            client.close()

    def test_a_malformed_observation_surfaces_the_server_error(self, serve):
        """The fake validator's message must reach the caller, not a transport error."""
        server = serve(DroidReachPolicy(goal=[0.5, 0.0, 0.45]))
        client = Gr00tZmqClient(_config(server.port).gr00t)
        client.connect(retries=1)
        try:
            bad = {
                "video": {"exterior_image_1_left": np.zeros((2, 8, 8, 3), np.uint8)},
                "state": {},
                "language": {},
            }
            with pytest.raises(PolicyError, match=r"server error: .*\(B, T, H, W, C\)"):
                client.predict(bad)
            # A server-side error is a complete REP exchange, so the socket is
            # still usable for the next request.
            assert client.is_ready()
            assert client.ping()
        finally:
            client.close()

    def test_token_protected_server_rejects_a_client_without_the_token(self, serve, monkeypatch):
        monkeypatch.delenv(API_TOKEN_ENV, raising=False)
        server = serve(DroidReachPolicy(goal=[0.5, 0.0, 0.45]), api_token="s3cret")
        client = Gr00tZmqClient(_config(server.port).gr00t)
        with pytest.raises(PolicyError, match="Unauthorized"):
            client.connect(retries=1, backoff_s=0.0)

    def test_token_is_sent_from_the_argument_or_the_environment(self, serve, monkeypatch):
        server = serve(DroidReachPolicy(goal=[0.5, 0.0, 0.45]), api_token="s3cret")
        explicit = Gr00tZmqClient(_config(server.port).gr00t, api_token="s3cret")
        explicit.connect(retries=1)
        explicit.close()

        monkeypatch.setenv(API_TOKEN_ENV, "s3cret")
        from_env = Gr00tZmqClient(_config(server.port).gr00t)
        from_env.connect(retries=1)
        from_env.close()
        assert all(r.get("api_token") == "s3cret" for r in server.requests)

    def test_a_timed_out_request_forces_a_fresh_socket(self, serve):
        """ZeroMQ REQ sockets wedge after a missed reply; the client must rebuild."""
        policy = DroidReachPolicy(goal=[0.5, 0.0, 0.45], delay_s=1.0)
        server = serve(policy)
        client = Gr00tZmqClient(_config(server.port, timeout_s=0.3).gr00t)
        client.connect(retries=1)  # ping does not go through the slow policy
        try:
            started = time.perf_counter()
            with pytest.raises(PolicyError, match="policy request failed"):
                client.predict({"state": {}})
            assert time.perf_counter() - started < 0.9, "timeout was not honoured"
            assert not client.is_ready(), "a wedged REQ socket must not be reused"
        finally:
            client.close()
        time.sleep(1.0)  # let the slow handler finish before reconnecting
        client.connect(retries=2, backoff_s=0.1)
        assert client.ping()
        client.close()

    def test_nothing_listening_is_a_policy_error_with_a_start_hint(self):
        with zmq.Context() as context:
            probe = context.socket(zmq.REP)
            port = probe.bind_to_random_port("tcp://127.0.0.1")
            probe.close(linger=0)
        client = Gr00tZmqClient(_config(port, timeout_s=0.3).gr00t)
        with pytest.raises(PolicyError, match="groot_server.py"):
            client.connect(retries=1, backoff_s=0.0)


# ---------------------------------------------------------------------------
# Gr00tExecutor closed loop against the ZMQ server
# ---------------------------------------------------------------------------


class FakeWorld:
    """Robot + controller + cameras + vision over one shared state.

    The object is attached to the hand when the gripper closes within 2 cm of
    it, so "the object followed the hand" is decided by geometry, not by a flag
    the test sets.
    """

    def __init__(self, config, object_position=(0.5, 0.0, 0.45)) -> None:
        self.config = config
        self.pose = Pose(np.array([0.3, 0.0, 0.6]), TOP_DOWN_QUAT, Frame.WORLD)
        self.width = config.robot.gripper_open_width
        self.object_position = np.asarray(object_position, dtype=np.float64)
        self.attached = False
        self.servo_targets: list[np.ndarray] = []

    # robot ------------------------------------------------------------
    def tcp_pose(self) -> Pose:
        return self.pose

    def get_gripper_state(self) -> GripperState:
        return GripperState(
            width=self.width, target_width=self.width, is_moving=False, is_grasping=self.attached
        )

    def get_state(self) -> RobotState:
        return RobotState(
            joint_state=JointState(positions=np.zeros(7), names=tuple(f"j{i}" for i in range(7))),
            tcp_pose=self.pose,
            gripper=self.get_gripper_state(),
            sim_time=0.0,
            step_index=0,
        )

    def open_gripper(self) -> None:
        self.width = self.config.robot.gripper_open_width
        self.attached = False

    def close_gripper(self) -> None:
        if np.linalg.norm(self.object_position - self.pose.position) < 0.02:
            self.attached = True
            self.width = 0.06
        else:
            self.width = self.config.robot.gripper_closed_width

    # controller -------------------------------------------------------
    def servo_to_pose(self, pose: Pose) -> bool:
        self.servo_targets.append(np.asarray(pose.position, dtype=np.float64).copy())
        self.pose = pose
        if self.attached:
            self.object_position = np.asarray(pose.position, dtype=np.float64).copy()
        return True

    # vision -----------------------------------------------------------
    def _scene(self) -> SceneGraph:
        pose = Pose(self.object_position.copy(), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        can = ObjectHypothesis(
            track_id="obj_7",
            label="can",
            pose=pose,
            bbox=BoundingBox3D(center=pose, extents=np.array([0.066, 0.066, 0.1])),
            confidence=0.9,
            num_points=500,
            last_seen_sim_time=0.0,
            last_seen_step=0,
        )
        return SceneGraph(objects={"obj_7": can}, sim_time=0.0, step_index=0)

    def require_fresh_scene(self) -> SceneGraph:
        return self._scene()

    def observe(self) -> SceneGraph:
        return self._scene()

    def cameras(self) -> dict[str, Any]:
        frame = types.SimpleNamespace(rgb=np.zeros((120, 160, 3), np.uint8))
        camera = types.SimpleNamespace(capture=lambda: frame)
        return {self.config.exterior_camera.name: camera, self.config.wrist_camera.name: camera}


class EventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


def _executor(server_port: int, world: FakeWorld, memory: WorkingMemory, events: EventSink):
    config = _config(server_port)
    client = Gr00tZmqClient(config.gr00t)
    client.connect(retries=1)
    return Gr00tExecutor(
        client=client,
        robot=world,
        vision=world,
        controller=world,
        cameras=world.cameras(),
        config=config,
        memory=memory,
        events=events,
    ), client


def _state(position) -> RobotState:
    return RobotState(
        joint_state=JointState(positions=np.zeros(7), names=tuple(f"j{i}" for i in range(7))),
        tcp_pose=Pose(np.asarray(position, dtype=float), TOP_DOWN_QUAT, Frame.WORLD),
        gripper=GripperState(width=0.08, target_width=0.08, is_moving=False, is_grasping=False),
        sim_time=0.0,
        step_index=0,
    )


class TestExecutorOverZmq:
    def test_pick_succeeds_and_records_the_held_object(self, serve):
        """The GR00T pick must leave working memory as the classical pick does.

        Before the fix the executor never called ``set_held_object``, so the
        classical "place it" that follows a GR00T pick was refused with "not
        holding anything to place".
        """
        config = _config(1)  # only for world geometry
        world = FakeWorld(config)
        server = serve(DroidReachPolicy(goal=world.object_position))
        memory = WorkingMemory(MemoryConfig())
        events = EventSink()
        executor, client = _executor(server.port, world, memory, events)
        try:
            result = executor.execute("pick", {"target": "can", "max_iterations": 10})
        finally:
            client.close()

        assert result.status is SkillStatus.SUCCESS, result.message
        assert memory.get_held_object() == "obj_7"
        assert result.data["track_id"] == "obj_7"
        assert result.data["clamped_actions"] == 0, "absolute 5 mm steps must pass unclamped"
        assert any(name == "gr00t.iteration" for name, _ in events.events)

        # The classical Place's own precondition now passes.
        from mfw.skills.base import SkillContext
        from mfw.skills.primitives import Place

        context = SkillContext(
            sim=None, robot=None, vision=None, planner=None, controller=None,
            grasp_scorer=None, memory=memory, config=config,
        )
        assert Place(context).validate({"target": "it"}) is None

    def test_policy_targets_never_enter_the_table(self, serve):
        """A policy that aims below the table top is clamped to the support floor.

        The workspace box's z-min is the floor (0.0) while the table top is at
        0.4 m; the old filter clamped into the box and let servo targets reach
        z = 0.005.
        """
        config = _config(1)
        world = FakeWorld(config)
        floor = support_height(config) + DEFAULT_SUPPORT_CLEARANCE_M
        server = serve(DroidReachPolicy(goal=[0.5, 0.0, 0.05]))  # 35 cm into the table
        executor, client = _executor(server.port, world, WorkingMemory(MemoryConfig()), EventSink())
        try:
            result = executor.execute("move_to", {"target": "table", "max_iterations": 20})
        finally:
            client.close()

        assert result.status is SkillStatus.TIMEOUT
        lowest = min(target[2] for target in world.servo_targets)
        assert lowest >= floor - 1e-9, f"servo target at z={lowest:.3f} is inside the table"
        assert lowest == pytest.approx(floor, abs=1e-6), "the arm should come to rest on the floor"

    def test_place_with_nothing_held_is_refused_without_querying_the_policy(self, serve):
        config = _config(1)
        world = FakeWorld(config)
        policy = DroidReachPolicy(goal=world.object_position)
        server = serve(policy)
        executor, client = _executor(server.port, world, WorkingMemory(MemoryConfig()), EventSink())
        try:
            result = executor.execute("place", {"target": "box"})
        finally:
            client.close()

        assert result.status is SkillStatus.INFEASIBLE
        assert "not holding" in result.message
        assert policy.calls == 0

    def test_policy_place_after_policy_pick_clears_memory(self, serve):
        config = _config(1)
        world = FakeWorld(config)
        server = serve(DroidReachPolicy(goal=world.object_position))
        memory = WorkingMemory(MemoryConfig())
        executor, client = _executor(server.port, world, memory, EventSink())
        try:
            picked = executor.execute("pick", {"target": "can", "max_iterations": 10})
            placed = executor.execute("place", {"max_iterations": 3})
        finally:
            client.close()

        assert picked.ok and placed.ok, (picked.message, placed.message)
        assert placed.data["track_id"] == "obj_7"
        assert memory.get_held_object() is None

    def test_server_error_mid_skill_is_a_failed_result_not_a_crash(self, serve):
        class Broken(DroidReachPolicy):
            def get_action(self, observation, options=None):
                raise RuntimeError("CUDA out of memory")

        config = _config(1)
        world = FakeWorld(config)
        server = serve(Broken(goal=world.object_position))
        executor, client = _executor(server.port, world, WorkingMemory(MemoryConfig()), EventSink())
        try:
            result = executor.execute("pick", {"target": "can"})
        finally:
            client.close()

        assert result.status is SkillStatus.FAILED
        assert "CUDA out of memory" in result.message
        assert world.servo_targets == []


# ---------------------------------------------------------------------------
# scripts/groot_server.py (argument handling only; nothing heavy is imported)
# ---------------------------------------------------------------------------


def _load_groot_server():
    spec = importlib.util.spec_from_file_location(
        "groot_server_under_test", REPO_ROOT / "scripts" / "groot_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGrootServerScript:
    def test_defaults_to_loopback(self, monkeypatch):
        monkeypatch.delenv(API_TOKEN_ENV, raising=False)
        args = _load_groot_server().build_parser().parse_args([])
        assert args.host == "127.0.0.1"
        assert args.api_token is None

    def test_refuses_a_routable_bind_without_a_token(self, monkeypatch, capsys):
        monkeypatch.delenv(API_TOKEN_ENV, raising=False)
        module = _load_groot_server()
        # Returns before importing torch, so this runs in any interpreter.
        assert module.main(["--host", "0.0.0.0"]) == 2
        assert "API token" in capsys.readouterr().err

    def test_token_or_explicit_opt_out_permits_a_routable_bind(self):
        module = _load_groot_server()
        assert module.check_bind("0.0.0.0", "s3cret", False) is None
        assert module.check_bind("0.0.0.0", None, True) is None
        assert module.check_bind("127.0.0.1", None, False) is None

    # -- model resolution and the in-process environment ----------------------

    @staticmethod
    def _fake_cache(root: Path, commit: str = "abc123", ref: str | None = "abc123") -> Path:
        repo = root / "models--nvidia--GR00T-N1.7-3B"
        snapshot = repo / "snapshots" / commit
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        if ref is not None:
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text(ref + "\n", encoding="utf-8")
        return snapshot

    def test_a_cached_repo_id_resolves_to_its_snapshot_without_the_hub(self, tmp_path, monkeypatch):
        """The Isaac-day requirement: offline, and the repo id must still load."""
        snapshot = self._fake_cache(tmp_path)
        # Importing huggingface_hub now raises: the cache path must not need it.
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        module = _load_groot_server()
        env = {"HF_HUB_CACHE": str(tmp_path), "HF_HUB_OFFLINE": "1"}
        assert module._resolve_model_path("nvidia/GR00T-N1.7-3B", env) == str(snapshot)

    def test_a_stale_ref_falls_back_to_a_complete_snapshot(self, tmp_path):
        snapshot = self._fake_cache(tmp_path, commit="real", ref="gone")
        (tmp_path / "models--nvidia--GR00T-N1.7-3B" / "snapshots" / "partial").mkdir()
        module = _load_groot_server()
        assert module.cached_snapshot("nvidia/GR00T-N1.7-3B", tmp_path) == snapshot

    def test_uncached_and_offline_is_a_clear_error_not_a_download(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        module = _load_groot_server()
        with pytest.raises(FileNotFoundError, match="HF_HUB_OFFLINE"):
            module._resolve_model_path(
                "nvidia/GR00T-N1.7-3B", {"HF_HUB_CACHE": str(tmp_path), "HF_HUB_OFFLINE": "1"}
            )

    def test_hf_cache_dir_follows_huggingface_precedence(self, tmp_path):
        module = _load_groot_server()
        assert module.hf_hub_cache_dir({"HF_HUB_CACHE": "A", "HF_HOME": "B"}) == Path("A")
        assert module.hf_hub_cache_dir({"HF_HOME": str(tmp_path)}) == tmp_path / "hub"

    def test_patch_mistral_is_on_by_default_and_offline_gaps_warn(self):
        module = _load_groot_server()
        env: dict[str, str] = {}
        warnings = module.configure_environment(env)
        assert env["GROOT_PATCH_MISTRAL"] == "1"
        assert len(warnings) == 1 and "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE" in warnings[0]

        quiet = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        assert module.configure_environment(quiet) == []

        opted_out = {"GROOT_PATCH_MISTRAL": "0"}
        module.configure_environment(opted_out, patch_mistral=False)
        assert "GROOT_PATCH_MISTRAL" not in opted_out, "'0' would still enable gr00t's patch"

    def test_main_sets_the_patch_before_anything_heavy_is_imported(
        self, tmp_path, monkeypatch, capsys
    ):
        """main() configures the environment, then fails fast on an unresolvable
        model -- before torch or gr00t are imported, so this runs anywhere."""
        monkeypatch.delenv("GROOT_PATCH_MISTRAL", raising=False)
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)
        module = _load_groot_server()
        assert module.main([]) == 2
        err = capsys.readouterr().err
        assert "TRANSFORMERS_OFFLINE not set" in err
        assert "not in the HuggingFace cache" in err
        assert os.environ.get("GROOT_PATCH_MISTRAL") == "1"


# ---------------------------------------------------------------------------
# scripts/start_groot_server.ps1 (Windows PowerShell; nothing listens on a port)
# ---------------------------------------------------------------------------

_LAUNCHER = REPO_ROOT / "scripts" / "start_groot_server.ps1"
_SCOPED = ("PYTHONPYCACHEPREFIX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "GROOT_PATCH_MISTRAL")
_POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(_POWERSHELL is None, reason="PowerShell is not available")
class TestLauncher:
    @staticmethod
    def _run(command: str) -> subprocess.CompletedProcess:
        # The caller's environment without the scoped variables, so "after"
        # checks see only what the launcher itself might have leaked.
        env = {k: v for k, v in os.environ.items() if k not in _SCOPED}
        return subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", command],
            capture_output=True, text=True, timeout=120, env=env, check=False,
        )

    def test_dry_run_resolves_the_snapshot_and_lists_the_child_env(self, tmp_path):
        snapshot = TestGrootServerScript._fake_cache(tmp_path / "hub")
        result = self._run(
            f"& '{_LAUNCHER}' -DryRun -GrootPython '{sys.executable}' "
            f"-HfCache '{tmp_path / 'hub'}' -PycachePrefix '{tmp_path / 'pyc'}'"
        )
        assert result.returncode == 0, result.stderr
        assert f"MODEL={snapshot}" in result.stdout
        assert "--host 127.0.0.1" in result.stdout
        for name in _SCOPED:
            assert f"CHILD_ENV {name}=" in result.stdout

    def test_only_the_server_process_receives_the_variables(self, tmp_path):
        """A stand-in server records its environment; the calling PowerShell
        session must see none of it afterwards (Isaac would inherit it)."""
        TestGrootServerScript._fake_cache(tmp_path / "hub")
        record = tmp_path / "child_env.json"
        stand_in = tmp_path / "fake_server.py"
        stand_in.write_text(
            "import json, os, sys\n"
            f"names = {list(_SCOPED)!r}\n"
            "payload = {'env': {n: os.environ.get(n) for n in names}, 'argv': sys.argv[1:]}\n"
            f"json.dump(payload, open({str(record)!r}, 'w'))\n"
            "sys.exit(7)\n",
            encoding="utf-8",
        )
        names = ",".join(f"'{n}'" for n in _SCOPED)
        result = self._run(
            f"& '{_LAUNCHER}' -GrootPython '{sys.executable}' -ServerScript '{stand_in}' "
            f"-HfCache '{tmp_path / 'hub'}' -PycachePrefix '{tmp_path / 'pyc'}'; "
            "Write-Output ('EXIT=' + $LASTEXITCODE); "
            f"foreach ($n in {names}) {{ Write-Output ('AFTER ' + $n + '=' + "
            "[Environment]::GetEnvironmentVariable($n)) }"
        )
        assert result.returncode == 0, result.stderr
        assert "EXIT=7" in result.stdout, "the server's exit code must propagate"

        child = json.loads(record.read_text())
        assert child["env"] == {
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pyc"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "GROOT_PATCH_MISTRAL": "1",
        }
        assert child["argv"][child["argv"].index("--host") + 1] == "127.0.0.1"

        after = dict(
            line[len("AFTER "):].split("=", 1)
            for line in result.stdout.splitlines()
            if line.startswith("AFTER ")
        )
        assert after == {name: "" for name in _SCOPED}, "the launcher leaked into its caller"

    def test_an_uncached_model_fails_before_starting_anything(self, tmp_path):
        result = self._run(
            f"& '{_LAUNCHER}' -DryRun -GrootPython '{sys.executable}' -HfCache '{tmp_path}'; "
            "exit $LASTEXITCODE"
        )
        assert result.returncode == 2
        assert "not in the HuggingFace cache" in result.stderr
