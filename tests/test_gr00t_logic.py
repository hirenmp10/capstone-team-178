"""Phase 7 GR00T bridge tests: transport, observation contract, safety filter.

No Isaac Sim, no torch, no checkpoint. The whole point of this suite is that the
GR00T *integration* is verifiable before a model exists -- the mock server runs the
real wire protocol, so transport, history buffering and safety clamping are all
exercised for real.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from mfw.config.schema import Gr00tConfig, load_config
from mfw.core.errors import PolicyError, SafetyViolation
from mfw.core.types import Frame, GripperState, JointState, Pose, RobotState
from mfw.gr00t_bridge.client import Gr00tTcpClient
from mfw.gr00t_bridge.observation import ObservationBuilder, resize_image
from mfw.gr00t_bridge.safety import ActionSafetyFilter
from mfw.gr00t_bridge.server import MockPolicy, PolicyServer
from mfw.utils import transforms as tf

pytestmark = pytest.mark.phase7

IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


@pytest.fixture
def gr00t_config():
    """Config for tests that never open a socket.

    Uses a real port number rather than 0: the validator rejects 0 because a
    client cannot connect to it, even though it is the correct value for an
    ephemeral *server* bind.
    """
    return Gr00tConfig(enabled=True, port=5555, use_mock_server=True)


@pytest.fixture
def mock_server():
    """A real PolicyServer serving MockPolicy on an ephemeral port."""
    server = PolicyServer("127.0.0.1", 0, MockPolicy())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()


def _robot_state(position=(0.45, 0.0, 0.5), width=0.08):
    return RobotState(
        joint_state=JointState(positions=np.zeros(7), names=tuple(f"j{i}" for i in range(7))),
        tcp_pose=Pose(np.asarray(position, dtype=float), np.array([1.0, 0, 0, 0]), Frame.WORLD),
        gripper=GripperState(width=width, target_width=width, is_moving=False, is_grasping=False),
        sim_time=0.0,
        step_index=0,
    )


class TestTransport:
    def test_handshake_and_predict_round_trip(self, mock_server):
        """The real wire protocol, exercised end to end without a model."""
        host, port = mock_server
        client = Gr00tTcpClient(Gr00tConfig(enabled=True, host=host, port=port))
        client.connect()
        try:
            assert client.is_ready()

            observation = {
                "state": {"eef_9d": np.concatenate([[0.5, 0.0, 0.5], IDENTITY_ROT6D])[None, :]}
            }
            action = client.predict(observation)

            assert "eef_9d" in action
            assert action["eef_9d"].shape == (40, 9), (
                "the oxe_droid contract declares a 40-step horizon"
            )
            assert "gripper_position" in action
        finally:
            client.close()

    def test_large_arrays_survive_the_wire(self, mock_server):
        """Observations carry image stacks; a naive protocol truncates them."""
        host, port = mock_server
        client = Gr00tTcpClient(Gr00tConfig(enabled=True, host=host, port=port))
        client.connect()
        try:
            images = np.random.default_rng(0).integers(0, 255, (2, 224, 224, 3), dtype=np.uint8)
            action = client.predict(
                {
                    "video": {"exterior_image_1_left": images, "wrist_image_left": images},
                    "state": {"eef_9d": np.zeros((1, 9))},
                }
            )
            assert action["eef_9d"].shape[0] == 40
        finally:
            client.close()

    def test_connect_failure_is_reported_clearly(self):
        """An unreachable server must not look like a modelling problem."""
        client = Gr00tTcpClient(Gr00tConfig(enabled=True, host="127.0.0.1", port=1))
        with pytest.raises(PolicyError, match="could not connect"):
            client.connect(retries=1, backoff_s=0.0)

    def test_predict_before_connect_raises(self, gr00t_config):
        client = Gr00tTcpClient(gr00t_config)
        with pytest.raises(PolicyError, match="not connected"):
            client.predict({})

    def test_embodiment_mismatch_is_rejected(self, mock_server):
        """A tag mismatch means a different action layout.

        Accepting it would make the framework misread every chunk while everything
        appeared to work.
        """
        host, port = mock_server
        client = Gr00tTcpClient(
            Gr00tConfig(enabled=True, host=host, port=port, embodiment_tag="unitree_g1_sonic")
        )
        with pytest.raises(PolicyError, match="could not connect|does not match"):
            client.connect(retries=1, backoff_s=0.0)


class TestObservationContract:
    """Everything here is dictated by the oxe_droid embodiment config."""

    def test_requires_a_full_history_before_building(self, gr00t_config):
        """delta_indices=[-15, 0] cannot be served from a short buffer."""
        builder = ObservationBuilder(gr00t_config)
        builder.push(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
        with pytest.raises(PolicyError, match="history"):
            builder.build(_robot_state(), "pick the can")

    def test_prime_fills_the_history(self, gr00t_config):
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
        assert builder.is_ready
        assert builder.history_length == gr00t_config.observation_history

    def test_observation_has_both_required_camera_keys(self, gr00t_config):
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((64, 64, 3), np.uint8), np.ones((64, 64, 3), np.uint8))
        observation = builder.build(_robot_state(), "pick the can")

        assert set(observation["video"]) == {"exterior_image_1_left", "wrist_image_left"}

    def test_video_carries_two_timesteps(self, gr00t_config):
        """Frame t-15 and frame t, per delta_indices -- on the TIME axis.

        Time is axis 1, not 0: the layout is (B, T, H, W, C). This test used to
        assert ``shape[0] == 2``, which encoded the missing batch axis and so
        passed while the live model rejected the very same observation.
        """
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
        observation = builder.build(_robot_state(), "x")
        for stack in observation["video"].values():
            assert stack.shape[1] == 2

    def test_newest_frame_is_last(self, gr00t_config):
        """Ordering matters: the policy expects (t-15, t), not (t, t-15)."""
        builder = ObservationBuilder(gr00t_config)
        old = np.zeros((32, 32, 3), np.uint8)
        builder.prime(old, old)

        new = np.full((32, 32, 3), 200, np.uint8)
        builder.push(new, new)
        observation = builder.build(_robot_state(), "x")

        # Index the batch axis first: (B, T, H, W, C).
        stack = observation["video"]["exterior_image_1_left"][0]
        assert stack[0].mean() < stack[-1].mean(), "newest frame is not last"

    def test_state_uses_the_declared_keys(self, gr00t_config):
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))
        state = builder.build(_robot_state(), "x")["state"]

        assert set(state) == {"eef_9d", "gripper_position", "joint_position"}
        # (B, T, D) with T=1, since state delta_indices=[0].
        assert state["eef_9d"].shape == (1, 1, 9)
        assert state["joint_position"].shape == (1, 1, 7)

    def test_eef_9d_round_trips_through_the_pose(self, gr00t_config):
        """The state must describe where the TCP actually is."""
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))

        state = _robot_state(position=(0.51, -0.12, 0.43))
        encoded = builder.build(state, "x")["state"]["eef_9d"][0]
        assert np.allclose(Pose.from_eef_9d(encoded).position, [0.51, -0.12, 0.43])

    def test_language_uses_the_declared_key(self, gr00t_config):
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))
        language = builder.build(_robot_state(), "pick up the can")["language"]
        # Nested for (B, T); a flat list is rejected by the model.
        assert language["annotation.language.language_instruction"] == [["pick up the can"]]

    def test_images_are_resized_to_the_policy_input(self, gr00t_config):
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640, 3), np.uint8))
        stack = builder.build(_robot_state(), "x")["video"]["wrist_image_left"]
        # (B, T, H, W, C): height and width are axes 2 and 3.
        assert stack.shape[2:4] == gr00t_config.image_size

    def test_video_carries_a_batch_axis(self, gr00t_config):
        """(B, T, H, W, C) -- five dimensions, not four.

        Verified against the live checkpoint. The embodiment config declares
        modality keys and delta_indices but says nothing about batching, so this
        was wrong until the real model rejected it:
            "Video key must be (B, T, H, W, C), got (2, 224, 224, 3)"
        """
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
        observation = builder.build(_robot_state(), "pick up the can")

        for key, stack in observation["video"].items():
            assert stack.ndim == 5, f"{key} has {stack.ndim} dims, expected 5 (B,T,H,W,C)"
            assert stack.shape[0] == 1, f"{key} batch axis is {stack.shape[0]}, expected 1"
            assert stack.shape[1] == 2, f"{key} horizon is {stack.shape[1]}, expected 2"

    def test_state_carries_a_batch_axis(self, gr00t_config):
        """(B, T, D). state delta_indices=[0], so T is 1."""
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))
        state = builder.build(_robot_state(), "x")["state"]

        assert state["eef_9d"].shape == (1, 1, 9)
        assert state["gripper_position"].shape == (1, 1, 1)
        assert state["joint_position"].shape == (1, 1, 7)

    def test_language_is_nested_for_batch_and_horizon(self, gr00t_config):
        """[[str]] -- (B, T).

        A flat [str] is rejected with the notably unhelpful
        "horizon must be 1. Got 1".
        """
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))
        language = builder.build(_robot_state(), "pick up the can")["language"]

        value = language["annotation.language.language_instruction"]
        assert value == [["pick up the can"]]

    def test_reset_clears_history_between_episodes(self, gr00t_config):
        """Carrying frames across a command shows the policy motion that never happened."""
        builder = ObservationBuilder(gr00t_config)
        builder.prime(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8))
        builder.reset()
        assert not builder.is_ready

    def test_resize_preserves_channels_and_dtype(self):
        image = np.random.default_rng(0).integers(0, 255, (100, 200, 3), dtype=np.uint8)
        out = resize_image(image, (50, 25))
        assert out.shape == (25, 50, 3) and out.dtype == np.uint8


class TestSafetyFilter:
    @pytest.fixture
    def safety(self):
        return ActionSafetyFilter(
            config=Gr00tConfig(
                enabled=True, max_relative_translation=0.05, max_relative_rotation=0.35
            ),
            workspace_min=np.array([0.2, -0.5, 0.0]),
            workspace_max=np.array([0.8, 0.5, 0.7]),
            gripper_open_width=0.08,
            gripper_closed_width=0.0,
        )

    def test_passes_a_small_absolute_action_unchanged(self, safety):
        """GR00T emits ABSOLUTE world poses, despite the "relative" embodiment tag.

        These numbers are from the live model: with the TCP at [0.45, 0, 0.45],
        action[0] was [0.4569, -0.0059, 0.4435] -- an 11 mm move. Read as a delta
        it becomes a 0.61 m step and is clamped every single time, which is
        exactly what produced a 100% clamp rate and looked like a failing policy.
        """
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.4569, -0.0059, 0.4435], IDENTITY_ROT6D])

        filtered = safety.apply(current, action, gripper_command=0.08, absolute=True)
        assert not filtered.was_clamped, "an 11 mm absolute move must pass untouched"
        assert np.allclose(filtered.target_pose.position, [0.4569, -0.0059, 0.4435], atol=1e-6)

    def test_same_action_read_as_a_delta_is_clamped(self, safety):
        """The regression guard for the semantics bug itself."""
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.4569, -0.0059, 0.4435], IDENTITY_ROT6D])

        filtered = safety.apply(current, action, gripper_command=0.08, absolute=False)
        assert filtered.was_clamped, "a 0.61 m delta must be clamped"

    def test_passes_a_small_delta_unchanged(self, safety):
        """Delta mode still works, for policies that genuinely emit deltas."""
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.01, 0.0, 0.0], IDENTITY_ROT6D])

        filtered = safety.apply(current, action, gripper_command=0.08, absolute=False)
        assert not filtered.was_clamped
        assert np.allclose(filtered.target_pose.position, [0.46, 0.0, 0.45], atol=1e-9)

    def test_clamps_an_oversized_translation_but_keeps_direction(self, safety):
        """The policy's direction is usually right even when the magnitude is not."""
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.30, 0.0, 0.0], IDENTITY_ROT6D])

        filtered = safety.apply(current, action, gripper_command=0.08)
        assert filtered["translation_clamped"]
        step = float(np.linalg.norm(filtered.target_pose.position - current.position))
        assert step == pytest.approx(0.05, abs=1e-6)

    def test_clamps_an_oversized_rotation(self, safety):
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        # 90 degrees about Z, well beyond the 0.35 rad limit.
        big = tf.matrix_to_rot6d(
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )
        filtered = safety.apply(current, np.concatenate([[0.0, 0, 0], big]), 0.08)

        assert filtered["rotation_clamped"]
        angle = current.angular_distance(filtered.target_pose)
        assert angle <= 0.35 + 1e-6

    def test_clamped_rotation_is_still_a_valid_rotation(self, safety):
        """Scaling matrix entries would not produce a rotation; the axis must be preserved."""
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        big = tf.matrix_to_rot6d(
            np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )
        filtered = safety.apply(current, np.concatenate([[0.0, 0, 0], big]), 0.08)

        rotation = filtered.target_pose.rotation_matrix()
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)

    def test_clamps_an_absolute_target_into_the_workspace(self, safety):
        """An absolute pose clamped onto the envelope is safe by construction.

        Aborting instead is disproportionate for a continuously-steering policy:
        measured here, a target 4 mm outside x-min killed an entire 30-iteration
        run on its very first step.
        """
        current = Pose(np.array([0.78, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.95, 0.0, 0.45], IDENTITY_ROT6D])  # beyond x-max 0.8

        filtered = safety.apply(current, action, 0.08, absolute=True)
        assert filtered.was_clamped
        assert filtered.target_pose.position[0] <= 0.8 + 1e-9
        assert np.all(filtered.target_pose.position >= np.array([0.2, -0.5, 0.0]) - 1e-9)

    def test_rejects_a_delta_that_leaves_the_workspace(self, safety):
        """Deltas still raise. Clamping a delta invents a direction the policy
        never chose, which could drag the arm along the boundary."""
        current = Pose(np.array([0.78, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.05, 0.0, 0.0], IDENTITY_ROT6D])

        with pytest.raises(SafetyViolation, match="outside the workspace"):
            safety.apply(current, action, 0.08, absolute=False)

    def test_rejects_non_finite_actions(self, safety):
        """A NaN carries no usable direction, so it cannot be clamped."""
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[np.nan, 0.0, 0.0], IDENTITY_ROT6D])

        with pytest.raises(SafetyViolation, match="non-finite"):
            safety.apply(current, action, 0.08)

    def test_rejects_a_degenerate_rotation(self, safety):
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.0, 0, 0], [1.0, 0, 0, 2.0, 0, 0]])  # collinear basis

        with pytest.raises(SafetyViolation, match="invalid rotation"):
            safety.apply(current, action, 0.08)

    def test_rejects_a_wrong_sized_action(self, safety):
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        with pytest.raises(SafetyViolation, match="9-element"):
            safety.apply(current, np.zeros(7), 0.08)

    def test_gripper_command_is_clamped_to_range(self, safety):
        current = Pose(np.array([0.45, 0.0, 0.45]), np.array([1.0, 0, 0, 0]), Frame.WORLD)
        action = np.concatenate([[0.0, 0, 0], IDENTITY_ROT6D])

        assert safety.apply(current, action, 5.0).gripper_width == pytest.approx(0.08)
        assert safety.apply(current, action, -1.0).gripper_width == pytest.approx(0.0)

    def test_action_is_relative_to_the_current_tcp_frame(self, safety):
        """A relative action composed in the world frame would ignore wrist orientation."""
        # Rotated 90 degrees about Z, so local +X points along world +Y.
        rotated = Pose(
            np.array([0.45, 0.0, 0.45]),
            tf.matrix_to_quat(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
            Frame.WORLD,
        )
        action = np.concatenate([[0.02, 0.0, 0.0], IDENTITY_ROT6D])
        filtered = safety.apply(rotated, action, 0.08, absolute=False)

        delta = filtered.target_pose.position - rotated.position
        assert delta[1] == pytest.approx(0.02, abs=1e-9), "action was not applied in the TCP frame"
        assert abs(delta[0]) < 1e-9


class TestActionUnwrapping:
    """The policy's reply shape, verified against the live checkpoint."""

    def test_unwraps_a_tuple_reply(self):
        """get_action returns a TUPLE containing the dict, not a bare dict."""
        from mfw.gr00t_bridge.executor import _unwrap_action

        payload = {"eef_9d": np.zeros((1, 40, 9))}
        assert _unwrap_action((payload, {"meta": 1})) is payload
        assert _unwrap_action(payload) is payload

    def test_rejects_a_reply_with_no_action_dict(self):
        from mfw.core.errors import PolicyError
        from mfw.gr00t_bridge.executor import _unwrap_action

        with pytest.raises(PolicyError, match="could not find an action dict"):
            _unwrap_action(("not", "a", "dict"))

    def test_drops_the_singleton_batch_axis(self):
        """(1, 40, 9) -> (40, 9), matching the safety filter's expectation."""
        from mfw.gr00t_bridge.executor import _drop_batch

        assert _drop_batch(np.zeros((1, 40, 9))).shape == (40, 9)
        assert _drop_batch(np.zeros((40, 9))).shape == (40, 9)

    def test_refuses_a_real_batch(self):
        """B>1 would be several robots' actions arriving together.

        Silently taking the first would drive this arm with another's commands,
        so it raises instead.
        """
        from mfw.core.errors import PolicyError
        from mfw.gr00t_bridge.executor import _drop_batch

        with pytest.raises(PolicyError, match="batch size 1"):
            _drop_batch(np.zeros((4, 40, 9)))


class TestExecutorRouting:
    def test_policy_declines_skills_it_cannot_express(self):
        """Returning False lets the planner fall back rather than faking capability."""
        from mfw.gr00t_bridge.executor import POLICY_SKILLS

        assert "pick" in POLICY_SKILLS
        assert "observe" not in POLICY_SKILLS
        assert "stop" not in POLICY_SKILLS
        assert "emergency_stop" not in POLICY_SKILLS

    def test_config_rejects_gr00t_selection_while_disabled(self):
        from mfw.config.schema import ConfigError

        with pytest.raises(ConfigError, match="gr00t.enabled"):
            load_config(overrides={"default_executor": "gr00t"})
