"""First-observe colours must not be black.

Regression for the runs where a scene of a red block, a blue can and a green
box answered the operator's first "what do you see" with "black block, black
can, black box" (logs/20260924_112235_93d40e/events.jsonl, seq 6 -> seq 8),
and a benchmark run called a soup can, a banana and a bowl black. Depth and
segmentation were correct throughout; only colour was wrong.

Four independent defences, each tested on synthetic frames with no Isaac:

* an unlit RGB frame contributes geometry but no colour (``VisionManager``);
* a track's colour is held across frames: a hue replaces at once, an
  achromatic reading only after repeated agreement (``ObjectTracker``);
* start-up warm-up renders until object colours stop changing, not merely
  until buffers exist (``Runtime._warm_up_cameras``);
* start-up priming observes again while colours are unsettled
  (``Assistant._prime_colors``).

A genuinely black object must still be reported "black".
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import numpy as np
import pytest

from mfw.config.schema import PerceptionConfig
from mfw.core.errors import PerceptionError, SimulationError
from mfw.core.types import (
    BoundingBox3D,
    CameraFrame,
    CameraIntrinsics,
    Frame,
    ObjectHypothesis,
    Pose,
)
from mfw.vision.geometry import (
    UNLIT_FRAME_VALUE,
    frame_brightness,
    is_chromatic_color,
    is_unlit_frame,
)
from mfw.vision.manager import VisionManager, object_color_signature
from mfw.vision.tracking import COLOR_CHANGE_CONFIRMATIONS, ObjectTracker

pytestmark = pytest.mark.phase2

RED = (215, 115, 113)  # measured rendered red block
BLUE = (113, 155, 212)
GREEN = (118, 203, 146)
TABLE = (170, 160, 150)
BLACK_MARKER = (18, 18, 20)
UNLIT = (1, 1, 1)  # measured value of an unrendered RGB buffer


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _hyp(position, color, label="block", step=0):
    pose = Pose(np.asarray(position, dtype=float), np.array([1.0, 0, 0, 0]), Frame.WORLD)
    return ObjectHypothesis(
        track_id="",
        label=label,
        pose=pose,
        bbox=BoundingBox3D(center=pose, extents=np.array([0.05, 0.05, 0.05])),
        confidence=0.9,
        num_points=500,
        last_seen_sim_time=step * 0.01,
        last_seen_step=step,
        attributes={"color": color},
    )


_SCENE = {
    "block": (0.46, -0.15, 0.43),
    "can": (0.52, 0.12, 0.47),
    "box": (0.66, 0.32, 0.44),
}


def _observe(tracker, colors, step):
    """One tracker update over the three-object default scene."""
    detections = [_hyp(_SCENE[label], colors[label], label=label, step=step) for label in _SCENE]
    objects = tracker.update(detections, step_index=step)
    return {obj.label: obj.attributes.get("color") for obj in objects.values()}


W, H = 64, 48
_INTRINSICS = CameraIntrinsics(fx=60.0, fy=60.0, cx=W / 2.0, cy=H / 2.0, width=W, height=H)
# Looking straight down from 1 m: optical +Z is world -Z (180 degrees about X).
_DOWN = Pose(np.array([0.5, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 0.0]), Frame.WORLD)
_OBJECT_PRIM = "/World/objects/thing"


def _frame(name, object_rgb, table_rgb=TABLE, step=0, with_object=True):
    """A synthetic top-down RGB-D frame: a table at z=0.4 and one 5 cm object on it."""
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    rgb[...] = table_rgb
    depth = np.full((H, W), 0.6, dtype=np.float32)
    seg = np.ones((H, W), dtype=np.int32)
    if with_object:
        rows, cols = slice(14, 34), slice(22, 42)
        rgb[rows, cols] = object_rgb
        depth[rows, cols] = 0.55
        seg[rows, cols] = 2
    return CameraFrame(
        camera_name=name,
        rgb=rgb,
        depth=depth,
        segmentation=seg,
        seg_id_to_label={1: "table", 2: "thing"},
        seg_id_to_prim={1: "/World/table", 2: _OBJECT_PRIM},
        intrinsics=_INTRINSICS,
        extrinsics=_DOWN,
        sim_time=step * 0.01,
        step_index=step,
    )


class _FakeSim:
    def __init__(self):
        self.step_index = 0
        self.sim_time = 0.0
        self.renders = 0

    def render_step(self, count=1):
        self.renders += count
        self.step_index += count
        self.sim_time = self.step_index / 120.0


class _ScriptedCamera:
    """A camera whose frame is a function of how many frames have been rendered."""

    def __init__(self, name, sim, script, enable_depth=True, enable_segmentation=True):
        self._name = name
        self._sim = sim
        self._script = script
        self.config = types.SimpleNamespace(
            enable_depth=enable_depth, enable_segmentation=enable_segmentation
        )

    @property
    def name(self):
        return self._name

    def capture(self):
        frame = self._script(self._sim.renders)
        if frame is None:
            raise PerceptionError(f"{self._name} produced no RGB data")
        return frame

    def get_extrinsics(self):
        return _DOWN


class _Events:
    def __init__(self):
        self.records: list[tuple[str, dict[str, Any]]] = []

    def emit(self, name, payload):
        self.records.append((name, payload))


def _vision(cameras, sim):
    config = PerceptionConfig(min_points_per_object=20, render_frames_before_capture=1)
    events = _Events()
    manager = VisionManager(
        sim=sim, cameras=cameras, config=config, event_logger=events, support_height=0.4
    )
    return manager, events


def _only_color(scene):
    assert len(scene.objects) == 1, scene.objects
    return next(iter(scene.objects.values())).attributes["color"]


# ----------------------------------------------------------------------
# frame brightness
# ----------------------------------------------------------------------


class TestUnlitFrame:
    def test_unrendered_buffer_is_unlit(self):
        assert is_unlit_frame(_frame("c", UNLIT, table_rgb=UNLIT).rgb)
        assert is_unlit_frame(np.zeros((H, W, 3), np.uint8))
        assert is_unlit_frame(None)

    def test_black_object_on_a_lit_table_is_not_an_unlit_frame(self):
        """The frame is judged, not the object: a black marker stays nameable."""
        frame = _frame("c", BLACK_MARKER)
        assert not is_unlit_frame(frame.rgb)
        assert frame_brightness(frame.rgb) > UNLIT_FRAME_VALUE

    def test_a_few_hot_pixels_do_not_make_an_unlit_frame_lit(self):
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[:2, :2] = 255
        assert is_unlit_frame(rgb)

    def test_hue_names_are_chromatic_and_lightness_names_are_not(self):
        for name in ("red", "blue", "green", "yellow", "brown", "orange", "pink"):
            assert is_chromatic_color(name)
        for name in ("black", "grey", "white", ""):
            assert not is_chromatic_color(name)


# ----------------------------------------------------------------------
# tracker colour hold
# ----------------------------------------------------------------------


class TestTrackerColorHold:
    def test_replays_the_run_f_startup_sequence(self):
        """seq 3 (unconfirmed), seq 6 red/blue/green, seq 8 black/black/black.

        Before the fix the third observation -- the operator's first "what do
        you see" -- reported all three objects black.
        """
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        first = {"block": "red", "can": "blue", "box": "green"}
        assert _observe(tracker, first, step=76) == {}
        assert _observe(tracker, first, step=86) == first
        dark = {"block": "black", "can": "black", "box": "black"}
        assert _observe(tracker, dark, step=96) == first
        assert _observe(tracker, first, step=106) == first

    def test_replays_the_benchmark_partial_blackout(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        good = {"block": "red", "can": "yellow", "box": "grey"}
        _observe(tracker, good, step=0)
        _observe(tracker, good, step=1)
        partial = {"block": "red", "can": "black", "box": "black"}
        assert _observe(tracker, partial, step=2) == good

    def test_raw_reading_is_kept_for_the_event_log(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        tracker.update([_hyp((0.5, 0, 0.45), "red")], 0)
        tracker.update([_hyp((0.5, 0, 0.45), "red", step=1)], 1)
        obj = next(iter(tracker.update([_hyp((0.5, 0, 0.45), "black", step=2)], 2).values()))
        assert obj.attributes["color"] == "red"
        assert obj.attributes["color_observed"] == "black"

    def test_early_wrong_black_is_corrected_by_the_first_hue(self):
        """First lit frames can be black; a hue cannot be an artefact of darkness."""
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        tracker.update([_hyp((0.5, 0, 0.45), "black")], 0)
        assert _only(tracker.update([_hyp((0.5, 0, 0.45), "black", step=1)], 1)) == "black"
        assert _only(tracker.update([_hyp((0.5, 0, 0.45), "red", step=2)], 2)) == "red"

    def test_a_genuinely_black_object_is_black(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        for step in range(4):
            objects = tracker.update([_hyp((0.5, 0, 0.45), "black", label="marker", step=step)], step)
        assert _only(objects) == "black"
        assert tracker.colors_settled()

    def test_persistent_achromatic_reading_wins_eventually(self):
        """The hold is hysteresis, not a lock."""
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        tracker.update([_hyp((0.5, 0, 0.45), "red")], 0)
        tracker.update([_hyp((0.5, 0, 0.45), "red", step=1)], 1)
        colors = [
            _only(tracker.update([_hyp((0.5, 0, 0.45), "grey", step=s)], s))
            for s in range(2, 2 + COLOR_CHANGE_CONFIRMATIONS)
        ]
        assert colors[:-1] == ["red"] * (COLOR_CHANGE_CONFIRMATIONS - 1)
        assert colors[-1] == "grey"

    def test_an_interrupted_run_of_dark_frames_does_not_accumulate(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        sequence = ["red", "red"] + ["black", "red"] * COLOR_CHANGE_CONFIRMATIONS + ["black"]
        for step, color in enumerate(sequence):
            objects = tracker.update([_hyp((0.5, 0, 0.45), color, step=step)], step)
        assert _only(objects) == "red"

    def test_empty_reading_never_erases_a_known_colour(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        tracker.update([_hyp((0.5, 0, 0.45), "blue")], 0)
        for step in range(1, 6):
            objects = tracker.update([_hyp((0.5, 0, 0.45), "", step=step)], step)
        assert _only(objects) == "blue"

    def test_hue_to_hue_changes_immediately(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        tracker.update([_hyp((0.5, 0, 0.45), "red")], 0)
        tracker.update([_hyp((0.5, 0, 0.45), "red", step=1)], 1)
        assert _only(tracker.update([_hyp((0.5, 0, 0.45), "brown", step=2)], 2)) == "brown"

    def test_colors_settled_tracks_the_latest_frame(self):
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        assert tracker.colors_settled()  # empty scene: nothing to wait for
        tracker.update([_hyp((0.5, 0, 0.45), "")], 0)
        tracker.update([_hyp((0.5, 0, 0.45), "", step=1)], 1)
        assert not tracker.colors_settled()  # unknown colour
        tracker.update([_hyp((0.5, 0, 0.45), "red", step=2)], 2)
        assert tracker.colors_settled()
        tracker.update([_hyp((0.5, 0, 0.45), "black", step=3)], 3)
        assert not tracker.colors_settled()  # held red, but the frame disagreed

    def test_detections_without_a_colour_attribute_are_untouched(self):
        """The hardware lane's detections: colour "" always, and no new keys."""
        tracker = ObjectTracker(match_distance=0.06, max_age_steps=600)
        for step in range(3):
            hyp = _hyp((0.5, 0, 0.45), "", step=step)
            objects = tracker.update([hyp], step)
        obj = next(iter(objects.values()))
        assert obj.attributes == {"color": ""}
        bare = _hyp((0.5, 0, 0.45), "", step=4)
        bare.attributes = {}
        obj = next(iter(tracker.update([bare], 4).values()))
        assert obj.attributes == {}


def _only(objects):
    assert len(objects) == 1
    return next(iter(objects.values())).attributes["color"]


# ----------------------------------------------------------------------
# VisionManager colour path
# ----------------------------------------------------------------------


class TestVisionManagerColour:
    def test_lit_frame_names_the_colour(self):
        sim = _FakeSim()
        cam = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", RED, step=n))
        vision, _ = _vision({"exterior": cam}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == "red"

    def test_unlit_frame_reports_unknown_not_black(self):
        sim = _FakeSim()
        cam = _ScriptedCamera(
            "exterior", sim, lambda n: _frame("exterior", UNLIT, table_rgb=UNLIT, step=n)
        )
        vision, events = _vision({"exterior": cam}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == ""
        payload = [p for name, p in events.records if name == "perception.observe"][-1]
        assert payload["unlit_cameras"] == ["exterior"]
        assert payload["rgb_brightness"]["exterior"] <= UNLIT_FRAME_VALUE

    def test_unlit_camera_does_not_outvote_a_lit_one(self):
        """The benchmark's partial blackout: one camera dark, the other fine.

        Colour pixels from both cameras are pooled before the median, so a dark
        camera that sees the object with more pixels used to decide its colour.
        """
        sim = _FakeSim()
        lit = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", GREEN, step=n))
        # Two dark views against one lit one, so the dark pixels are the
        # majority of the pooled samples -- as the close-range wrist view is.
        darks = {
            name: _ScriptedCamera(
                name, sim, lambda n, name=name: _frame(name, UNLIT, table_rgb=UNLIT, step=n)
            )
            for name in ("wrist", "wrist_b")
        }
        vision, _ = _vision({"exterior": lit, **darks}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == "green"

    def test_black_frame_after_correct_colours_keeps_them(self):
        """End to end through perception: the run_f sequence on one object."""
        sim = _FakeSim()
        script = {"dark": False}

        def frames(n):
            if script["dark"]:
                return _frame("exterior", UNLIT, table_rgb=UNLIT, step=n)
            return _frame("exterior", BLUE, step=n)

        cam = _ScriptedCamera("exterior", sim, frames)
        vision, _ = _vision({"exterior": cam}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == "blue"
        script["dark"] = True
        assert _only_color(vision.observe()) == "blue"
        assert vision.colors_settled() is False

    def test_objects_dark_on_a_lit_table_are_held_too(self):
        """Materials unresolved while the frame is lit: the tracker hold covers it."""
        sim = _FakeSim()
        script = {"object": RED}
        cam = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", script["object"], step=n))
        vision, _ = _vision({"exterior": cam}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == "red"
        script["object"] = UNLIT
        assert _only_color(vision.observe()) == "red"

    def test_genuinely_black_object_on_lit_table_is_black(self):
        sim = _FakeSim()
        cam = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", BLACK_MARKER, step=n))
        vision, _ = _vision({"exterior": cam}, sim)
        vision.observe()
        assert _only_color(vision.observe()) == "black"
        assert vision.colors_settled()


# ----------------------------------------------------------------------
# start-up warm-up
# ----------------------------------------------------------------------


def _runtime_with(cameras, sim):
    from mfw.simulation.runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    runtime.sim = sim
    runtime.wrist_camera = cameras.get("wrist")
    runtime.exterior_camera = cameras.get("exterior")
    return runtime


class TestCameraWarmUp:
    def test_warm_up_waits_for_object_colours_not_just_buffers(self):
        """Buffers appear at render 5; the object fades in from black until render 20.

        The old warm-up returned at render 5, so the first observation measured
        a black object.
        """
        sim = _FakeSim()

        def script(n):
            if n < 5:
                return None
            fade = min(1.0, (n - 5) / 15.0)
            color = tuple(round(1 + fade * (c - 1)) for c in RED)
            return _frame("exterior", color, step=n)

        cam = _ScriptedCamera("exterior", sim, script)
        _runtime_with({"exterior": cam}, sim)._warm_up_cameras()
        assert sim.renders >= 20
        signature = object_color_signature(cam.capture())
        assert signature is not None
        assert np.allclose(signature[_OBJECT_PRIM], RED)

    def test_warm_up_waits_for_an_unlit_frame_to_light(self):
        sim = _FakeSim()

        def script(n):
            if n < 3:
                return None
            if n < 12:
                return _frame("exterior", UNLIT, table_rgb=UNLIT, step=n)
            return _frame("exterior", GREEN, step=n)

        cam = _ScriptedCamera("exterior", sim, script)
        _runtime_with({"exterior": cam}, sim)._warm_up_cameras()
        assert sim.renders >= 12
        assert not is_unlit_frame(cam.capture().rgb)

    def test_warm_up_is_quick_once_colours_are_stable(self):
        sim = _FakeSim()
        cam = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", RED, step=n))
        runtime = _runtime_with({"exterior": cam}, sim)
        runtime._warm_up_cameras()
        assert sim.renders <= 1 + runtime._COLOR_SETTLE_STREAK

    def test_warm_up_is_bounded_when_colours_never_settle(self):
        sim = _FakeSim()

        def flicker(n):
            return _frame("exterior", RED if n % 2 else BLUE, step=n)

        cam = _ScriptedCamera("exterior", sim, flicker)
        runtime = _runtime_with({"exterior": cam}, sim)
        runtime._warm_up_cameras()  # warns, does not raise
        assert sim.renders <= 1 + runtime._MAX_COLOR_SETTLE_STEPS

    def test_warm_up_still_fails_loudly_when_no_buffer_ever_arrives(self):
        sim = _FakeSim()
        cam = _ScriptedCamera("exterior", sim, lambda n: None)
        runtime = _runtime_with({"exterior": cam}, sim)
        with pytest.raises(SimulationError):
            runtime._warm_up_cameras()

    def test_every_camera_must_settle(self):
        sim = _FakeSim()
        ext = _ScriptedCamera("exterior", sim, lambda n: _frame("exterior", RED, step=n))
        wrist = _ScriptedCamera(
            "wrist",
            sim,
            lambda n: _frame("wrist", UNLIT, table_rgb=UNLIT, step=n)
            if n < 15
            else _frame("wrist", BLUE, step=n),
        )
        _runtime_with({"exterior": ext, "wrist": wrist}, sim)._warm_up_cameras()
        assert sim.renders >= 15

    def test_signature_ignores_non_object_prims_and_falls_back_to_the_frame(self):
        frame = _frame("c", RED, with_object=False)
        signature = object_color_signature(frame)
        assert set(signature) == {""}
        assert np.allclose(signature[""], TABLE)
        assert object_color_signature(_frame("c", UNLIT, table_rgb=UNLIT)) is None


# ----------------------------------------------------------------------
# start-up priming
# ----------------------------------------------------------------------


class _PrimingRuntime:
    def __init__(self, settled_after, has_colors=True):
        self.observes = 0
        runtime = self

        class _Skills:
            names = ("observe",)

            def execute(self, name, **_):
                assert name == "observe"
                runtime.observes += 1

        self.skills = _Skills()
        if has_colors:
            self.vision = types.SimpleNamespace(
                colors_settled=lambda: runtime.observes >= settled_after
            )
        else:
            self.vision = types.SimpleNamespace()
        self.executor = object()
        self.memory = object()
        self.events = _Events()

    def build(self):
        pass


class TestStartupPriming:
    def _assistant(self, monkeypatch, runtime):
        import mfw.assistant as assistant_module

        monkeypatch.setattr(assistant_module, "_build_runtime", lambda config: runtime)
        monkeypatch.setattr(assistant_module, "TaskPlanner", lambda **kwargs: object())
        config = assistant_module.load_config(assistant_module.DEFAULT_CONFIG_PATH)
        config = dataclasses.replace(
            config, gr00t=dataclasses.replace(config.gr00t, enabled=False)
        )
        return assistant_module.Assistant(config=config)

    def test_priming_observes_until_colours_settle(self, monkeypatch):
        runtime = _PrimingRuntime(settled_after=4)
        self._assistant(monkeypatch, runtime)
        assert runtime.observes == 4

    def test_priming_is_unchanged_when_colours_settle_at_once(self, monkeypatch):
        runtime = _PrimingRuntime(settled_after=2)
        self._assistant(monkeypatch, runtime)
        assert runtime.observes == 2

    def test_priming_is_bounded(self, monkeypatch):
        import mfw.assistant as assistant_module

        runtime = _PrimingRuntime(settled_after=10**6)
        self._assistant(monkeypatch, runtime)
        assert runtime.observes == 2 + assistant_module.Assistant._MAX_COLOR_PRIMING_OBSERVES

    def test_backend_without_colour_is_skipped(self, monkeypatch):
        """The hardware lane measures no colour; its start-up must not change."""
        runtime = _PrimingRuntime(settled_after=10**6, has_colors=False)
        self._assistant(monkeypatch, runtime)
        assert runtime.observes == 2
