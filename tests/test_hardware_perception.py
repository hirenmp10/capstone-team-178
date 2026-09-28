"""Phase 9: depthless perception, from synthetic pixels to a tracked scene graph.

No Isaac Sim, no torch, no webcam, well under 15 s. The detector service runs
in a thread on the scripted backend; the camera is an analytic pinhole whose
pixel -> table homography is known exactly, which is what lets these tests
put numbers on things the real bench cannot: the homography round-trips to
micrometres, the pose estimator's bottom-centre correction recovers a 15 cm
bowl to millimetres, and ``PlanarPerception`` drops exactly the detections it
should (outside the workspace, off-vocabulary).

The review's geometry findings are pinned here too, against the same
analytic cameras: the pinhole refinement runs only with a *measured* camera
(GEO-2), the first-order step follows the image-up direction and never the
look-at offset (GEO-5 and the nadir-5-mm case), an object's 0/90 degree yaw
is read off its pixel box (GEO-3), and ``predict_lift`` agrees with what the
honest scripted camera renders for a carried object (F1/F2). The scripted
detector itself is exercised in its honest mode (lifted objects stay visible;
only the jaw footprint hides them) and its legacy hide-above-z mode.

Also covers the two laptop scripts this stream owns: ``serve_detector.py``
(scripted backend through its own entry point; Florence helpers without a
model) and ``calibrate_table.py`` (fit, RANSAC, report, YAML splice).

Deviation from the plan, recorded: the plan says a dead detector port raises
``ExecutionError``. The code on disk raises ``DetectorError``, a
``PerceptionError`` -- the runtime catches it as such -- so that is what is
asserted here.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from jetson.detector_service import (
    HIDE_ABOVE_Z,
    DetectorServer,
    ScriptedDetector,
    SyntheticPinhole,
    add_common_arguments,
    build_scripted,
    format_homography,
    warm_up,
)
from mfw.config.schema import load_config
from mfw.core.errors import ConfigurationError, PerceptionError
from mfw.core.types import CameraFrame
from mfw.hardware.detector import DetectorError, RemoteDetector
from mfw.hardware.grasp import chord_width
from mfw.hardware.perception import (
    HomographyPoseEstimator,
    PinholeModel,
    PlanarPerception,
    homography_from_config,
)
from mfw.hardware.remote_camera import RemoteCamera

pytestmark = pytest.mark.phase9

REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_FAKE_YAML = REPO_ROOT / "configs" / "hardware_fake.yaml"
SCRIPTS = REPO_ROOT / "scripts"

MARKER_XY = (0.18, 0.05)
BOWL_XY = (0.15, -0.12)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (scripts is not a package)."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _table_grid() -> np.ndarray:
    xs = np.linspace(0.06, 0.30, 7)
    ys = np.linspace(-0.24, 0.24, 9)
    return np.array([[x, y, 0.0] for x in xs for y in ys])


class FakeClock:
    """The four attributes perception reads off ``sim``; advanced by hand."""

    def __init__(self, dt: float = 1.0 / 120.0) -> None:
        self.dt = dt
        self.step_index = 0

    @property
    def sim_time(self) -> float:
        return self.step_index * self.dt

    def advance(self, seconds: float) -> None:
        self.step_index += int(round(seconds / self.dt))

    def step(self, count: int = 1) -> None:
        self.step_index += int(count)


class FakeFrameSource:
    """Duck-types ``JetsonClient.get_frame`` over a rendered synthetic frame."""

    def __init__(self, camera: SyntheticPinhole, scene: dict[str, tuple[float, ...]]) -> None:
        import cv2

        w, h = camera.image_size
        bgr = np.full((h, w, 3), 40, dtype=np.uint8)
        detector = ScriptedDetector(scene, camera)
        for label, xyz in detector.current_scene().items():
            box = detector.box_for(label, xyz)
            if box is not None:
                cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), (0, 200, 255), -1)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok
        self.jpeg = buf.tobytes()
        self.size = (w, h)
        self.calls = 0

    def get_frame(self, quality: int | None = None) -> dict[str, Any]:
        self.calls += 1
        return {"jpeg": self.jpeg, "width": self.size[0], "height": self.size[1],
                "t_capture": time.time(), "seq": self.calls}


class DeadFrameSource:
    def get_frame(self, quality: int | None = None) -> dict[str, Any]:
        from mfw.hardware.jetson_client import RpcError

        raise RpcError("get_frame timed out")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config(HARDWARE_FAKE_YAML)


@pytest.fixture
def scene() -> dict[str, tuple[float, ...]]:
    return {"marker": MARKER_XY, "bowl": BOWL_XY}


@pytest.fixture
def detector_server(scene):
    backend = ScriptedDetector(lambda: scene, SyntheticPinhole())
    server = DetectorServer("127.0.0.1", 0, backend)
    _, port = server.serve_in_thread(port=0)
    try:
        yield server, port
    finally:
        server.stop()


@pytest.fixture
def remote_detector(cfg, detector_server):
    _server, port = detector_server
    detector = RemoteDetector("127.0.0.1", port, labels=cfg.hardware.labels,
                              min_score=cfg.hardware.detection_min_score, timeout_s=2.0)
    detector.connect(retries=2, backoff_s=0.1)
    yield detector
    detector.close()


def _perception(cfg, clock, detector, **overrides) -> PlanarPerception:
    perception_cfg = cfg.perception
    if "perception" in overrides:
        perception_cfg = dataclasses.replace(perception_cfg, **overrides.pop("perception"))
    kwargs: dict[str, Any] = dict(
        clock=clock,
        detector=detector,
        config=perception_cfg,
        hardware=cfg.hardware,
        homography=cfg.exterior_camera.homography,
        support_height=cfg.perception.ground_plane_z,
        workspace_min=cfg.scene.workspace_min,
        workspace_max=cfg.scene.workspace_max,
        camera=cfg.exterior_camera,
    )
    kwargs.update(overrides)
    return PlanarPerception(**kwargs)


@pytest.fixture
def perception(cfg, remote_detector):
    clock = FakeClock()
    return _perception(cfg, clock, remote_detector), clock


def _labels(scene_graph) -> dict[str, Any]:
    return {obj.label: obj for obj in scene_graph.objects.values()}


# ----------------------------------------------------------------------
# synthetic camera + homography
# ----------------------------------------------------------------------


class TestSyntheticCamera:
    def test_grid_round_trips_through_exact_homography(self):
        cam = SyntheticPinhole()
        pts = _table_grid()
        uv = cam.project(pts)
        assert np.all(np.isfinite(uv))
        h = cam.homography_pixel_to_table()
        back = np.array([cam.pixel_to_table(u, v) for u, v in uv])
        err = np.linalg.norm(back - pts[:, :2], axis=1)
        assert err.max() < 1e-3, err.max()
        assert err.max() < 1e-9  # analytic: really exact
        assert h[2, 2] == pytest.approx(1.0)

    def test_printed_homography_maps_grid_back_within_a_millimetre(self):
        """What ``--print-homography`` prints, pasted into YAML, must agree."""
        cam = SyntheticPinhole()
        printed = format_homography(cam.homography_pixel_to_table())
        values = json.loads(printed)
        assert len(values) == 9
        h = homography_from_config(values)
        pts = _table_grid()
        uv = cam.project(pts)
        hom = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ h.T
        back = hom[:, :2] / hom[:, 2:3]
        assert np.linalg.norm(back - pts[:, :2], axis=1).max() < 1e-3

    def test_hardware_fake_yaml_carries_the_synthetic_homography(self, cfg):
        """The e2e config is only right while this matrix matches the camera."""
        stored = homography_from_config(cfg.exterior_camera.homography)
        exact = SyntheticPinhole().homography_pixel_to_table()
        np.testing.assert_allclose(stored, exact, atol=1e-9)
        assert tuple(cfg.exterior_camera.resolution) == SyntheticPinhole().image_size

    def test_points_behind_camera_are_nan_not_mirrored(self):
        cam = SyntheticPinhole()
        behind = np.array([-2.0, 0.0, 0.0])
        assert np.all(np.isnan(cam.project(behind)))

    def test_homography_from_config_rejects_garbage(self):
        with pytest.raises(ConfigurationError, match="calibrate_table"):
            homography_from_config([])
        with pytest.raises(ConfigurationError):
            homography_from_config([0.0] * 9)


# ----------------------------------------------------------------------
# pose estimator
# ----------------------------------------------------------------------


class TestHomographyPoseEstimator:
    @pytest.fixture
    def estimator(self, cfg):
        return HomographyPoseEstimator(
            homography=cfg.exterior_camera.homography,
            object_sizes=cfg.hardware.object_sizes,
            default_size=cfg.hardware.default_object_size,
            support_height=0.0,
            pixel_anchor=cfg.hardware.pixel_anchor,
            camera=cfg.exterior_camera,
        )

    @pytest.mark.parametrize("label,xy", [("marker", MARKER_XY), ("bowl", BOWL_XY), ("cube", (0.25, 0.15))])
    def test_known_object_xy_z_extents(self, cfg, estimator, label, xy):
        scripted = ScriptedDetector({label: xy}, SyntheticPinhole())
        box = scripted.box_for(label, (*xy, 0.0))
        assert box is not None
        est = estimator.estimate(None, {"label": label, "confidence": 0.9, "bbox_px": list(box)})
        assert est is not None
        pose, extents = est
        size = np.asarray(cfg.hardware.object_sizes[label])
        assert np.linalg.norm(pose.position[:2] - np.array(xy)) < 0.010, pose.position
        assert pose.position[2] == pytest.approx(size[2] / 2.0)
        np.testing.assert_allclose(extents, size)

    def test_unknown_label_uses_default_size(self, cfg, estimator):
        np.testing.assert_allclose(estimator.size_for("stapler"), cfg.hardware.default_object_size)

    def test_degenerate_box_is_none(self, estimator):
        assert estimator.estimate(None, {"label": "marker", "bbox_px": [10, 10, 10.5, 40]}) is None
        assert estimator.estimate(None, {"label": "marker"}) is None

    def test_center_anchor_skips_the_correction(self, cfg):
        est = HomographyPoseEstimator(
            homography=cfg.exterior_camera.homography, object_sizes=cfg.hardware.object_sizes,
            default_size=cfg.hardware.default_object_size, pixel_anchor="center", camera=None,
        )
        box = [100.0, 200.0, 140.0, 260.0]
        pose, _ = est.estimate(None, {"label": "cube", "bbox_px": box})
        np.testing.assert_allclose(pose.position[:2], est.pixel_to_table(120.0, 230.0))

    def test_bad_anchor_rejected(self, cfg):
        with pytest.raises(ConfigurationError):
            HomographyPoseEstimator(cfg.exterior_camera.homography, {}, (0.04,) * 3, pixel_anchor="top_left")


# ----------------------------------------------------------------------
# PlanarPerception end to end
# ----------------------------------------------------------------------


class TestPlanarPerception:
    def test_two_observes_corroborate_marker_and_bowl(self, perception):
        vision, clock = perception
        first = vision.observe()
        assert first.objects == {}  # one sighting is not evidence
        clock.step()
        second = vision.observe()
        seen = _labels(second)
        assert set(seen) == {"marker", "bowl"}
        assert np.linalg.norm(seen["marker"].pose.position[:2] - np.array(MARKER_XY)) < 0.015
        assert np.linalg.norm(seen["bowl"].pose.position[:2] - np.array(BOWL_XY)) < 0.015
        assert seen["marker"].attributes["primary_camera"] == "exterior_camera"
        assert vision.last_scene_graph() is second
        # A third observe keeps the same track ids (stable identity for "it").
        clock.step()
        third = vision.observe()
        assert set(third.objects) == set(second.objects)

    def test_object_outside_workspace_is_dropped(self, cfg, scene, perception, remote_detector):
        vision, clock = perception
        scene["cube"] = (0.20, 0.34)  # in the image, 7 cm beyond y max + margin
        raw = {d["label"] for d in remote_detector.detect(None)}
        assert "cube" in raw, "the drop must be perception's, not the camera's"
        for _ in range(2):
            vision.observe()
            clock.step()
        assert set(_labels(vision.last_scene_graph())) == {"marker", "bowl"}

    def test_unknown_label_is_filtered_by_vocabulary(self, cfg, scene, detector_server):
        _server, port = detector_server
        scene["stapler"] = (0.20, 0.10)
        # Empty vocabulary: the server reports everything, so the filter under test is perception's.
        detector = RemoteDetector("127.0.0.1", port, labels=(), timeout_s=2.0)
        detector.connect(retries=1)
        assert "stapler" in {d["label"] for d in detector.detect(None)}
        clock = FakeClock()
        vision = _perception(cfg, clock, detector)
        for _ in range(2):
            vision.observe()
            clock.step()
        assert set(_labels(vision.last_scene_graph())) == {"marker", "bowl"}

    def test_require_fresh_scene_reobserves_after_the_clock_advances(self, cfg, perception, detector_server):
        server, _port = detector_server
        vision, clock = perception
        scene = vision.require_fresh_scene()
        served = server.requests_served
        assert vision.require_fresh_scene() is scene
        assert server.requests_served == served
        clock.advance(cfg.perception.max_scene_graph_age_s + 0.05)
        fresh = vision.require_fresh_scene()
        assert fresh is not scene
        assert server.requests_served == served + 1

    def test_lifted_object_stays_visible_bigger_and_parallax_shifted(self, cfg, scene, perception, remote_detector):
        """What a real overhead camera does with a carried marker (review F1/F2):
        it is still detected, its box is ~10 % bigger, and the table-plane
        estimate moves only 10-30 mm -- under the old 30 mm "displaced" gate --
        so the same track survives and is seen *this* frame."""
        vision, clock = perception
        for _ in range(2):
            vision.observe()
            clock.step()
        before = _labels(vision.last_scene_graph())["marker"]
        scene["marker"] = (*MARKER_XY, 0.06)
        clock.step()
        after = vision.observe()
        marker = _labels(after)["marker"]
        assert marker.track_id == before.track_id and marker.last_seen_step == after.step_index

        def diag(box):
            return float(np.hypot(box[2] - box[0], box[3] - box[1]))

        growth = diag(marker.attributes["bbox_px"]) / diag(before.attributes["bbox_px"])
        assert 1.05 < growth < 1.2
        shift = float(np.linalg.norm(marker.pose.position[:2] - before.pose.position[:2]))
        assert 0.005 < shift < cfg.grasp.verify_min_displacement

    def test_legacy_hide_rule_still_available_for_the_verdict_tests(self, cfg):
        legacy = ScriptedDetector({"marker": (*MARKER_XY, HIDE_ABOVE_Z + 0.02)}, SyntheticPinhole(),
                                  hide_above_z=HIDE_ABOVE_Z)
        honest = ScriptedDetector({"marker": (*MARKER_XY, HIDE_ABOVE_Z + 0.02)}, SyntheticPinhole())
        assert legacy.detect(None, ["marker"]) == [] and not legacy.honest
        assert [d["label"] for d in honest.detect(None, ["marker"])] == ["marker"] and honest.honest

    def test_removed_object_ages_out_with_a_short_track_life(self, cfg, scene, remote_detector):
        clock = FakeClock()
        vision = _perception(cfg, clock, remote_detector, perception={"track_max_age_steps": 2})
        for _ in range(2):
            vision.observe()
            clock.step()
        assert "marker" in _labels(vision.last_scene_graph())
        del scene["marker"]
        clock.step(5)
        assert set(_labels(vision.observe())) == {"bowl"}

    def test_hypotheses_carry_the_pixel_box_and_yaw_evidence(self, cfg, scene, perception):
        vision, clock = perception
        for _ in range(2):
            vision.observe()
            clock.step()
        marker = _labels(vision.last_scene_graph())["marker"]
        assert len(marker.attributes["bbox_px"]) == 4
        assert marker.attributes["yaw_rad"] == 0.0 and marker.attributes["yaw_ambiguous"] is False
        assert vision.pose_measured is cfg.exterior_camera.pose_measured

    def test_support_height_at_reads_the_tallest_object(self, cfg, perception):
        vision, clock = perception
        for _ in range(2):
            vision.observe()
            clock.step()
        assert vision.support_height_at(np.array(BOWL_XY), 0.0) == pytest.approx(cfg.hardware.object_sizes["bowl"][2])
        assert vision.support_height_at(np.array([0.29, 0.23]), 0.0) == 0.0

    def test_dead_detector_surfaces_as_perception_error(self, cfg):
        detector = RemoteDetector("127.0.0.1", _free_port(), labels=cfg.hardware.labels, timeout_s=0.5)
        vision = _perception(cfg, FakeClock(), detector)
        with pytest.raises(PerceptionError):
            vision.observe()


# ----------------------------------------------------------------------
# camera model gating, view direction, yaw (review GEO-2/3/5)
# ----------------------------------------------------------------------

#: The hardware.yaml placeholder camera, as a real (synthetic) camera.
NADIR = SyntheticPinhole(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480,
                         position=(0.18, 0.0, 0.60), look_at=(0.18, 0.0, 0.0), up=(1.0, 0.0, 0.0))

POSES = [(0.10, -0.15), (0.14, 0.05), (0.18, -0.05), (0.22, 0.15), (0.26, 0.0)]


def _camera_config(cfg, camera: SyntheticPinhole, measured: bool, **pose):
    """An ``exterior_camera`` describing ``camera`` (or, with ``pose``, a wrong placeholder)."""
    values = dict(position=tuple(camera.position), look_at=tuple(camera.look_at), up=tuple(camera.up))
    values.update(pose)
    return dataclasses.replace(
        cfg.exterior_camera, fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy,
        resolution=(camera.width, camera.height), pose_measured=measured, **values,
    )


def _estimator(cfg, camera: SyntheticPinhole, cam_cfg, **kwargs) -> HomographyPoseEstimator:
    return HomographyPoseEstimator(
        homography=camera.homography_pixel_to_table().reshape(-1),
        object_sizes=cfg.hardware.object_sizes,
        default_size=cfg.hardware.default_object_size,
        camera=cam_cfg,
        **kwargs,
    )


def _error_mm(estimator, camera, label, xy, yaw: float = 0.0) -> float:
    box = ScriptedDetector({}, camera).box_for(label, (*xy, 0.0, yaw))
    pose, _ = estimator.estimate(None, {"label": label, "bbox_px": list(box)})
    return float(np.linalg.norm(pose.position[:2] - np.asarray(xy)) * 1000.0)


class TestCameraModelGating:
    def test_a_placeholder_pose_10_cm_off_cannot_move_estimates_when_unmeasured(self, cfg):
        """GEO-2 case (b): the true camera is nadir, the config claims one 10 cm
        forward. Unmeasured, only the first-order step runs and the pose is
        irrelevant; before, the refinement turned it into ~100 mm everywhere."""
        wrong = dict(position=(0.28, 0.0, 0.60), look_at=(0.28, 0.0, 0.0))
        est = _estimator(cfg, NADIR, _camera_config(cfg, NADIR, measured=False, **wrong))
        assert est.camera is None and not est.pose_measured
        limits = {"marker": 8.0, "cube": 15.0, "bowl": 25.0}
        for label, limit in limits.items():
            for xy in POSES:
                assert _error_mm(est, NADIR, label, xy) < limit, (label, xy)

    def test_a_measured_pose_that_contradicts_the_homography_is_refused(self, cfg, caplog):
        wrong = dict(position=(0.28, 0.0, 0.60), look_at=(0.28, 0.0, 0.0))
        est = _estimator(cfg, NADIR, _camera_config(cfg, NADIR, measured=True, **wrong))
        box = ScriptedDetector({}, NADIR).box_for("bowl", (0.18, 0.05, 0.0))
        with pytest.raises(PerceptionError, match="pose_measured"):
            est.estimate(None, {"label": "bowl", "bbox_px": list(box)})
        # PlanarPerception drops it loudly rather than placing it anywhere.
        vision = PlanarPerception(
            clock=FakeClock(), detector=None, config=cfg.perception, hardware=cfg.hardware,
            homography=NADIR.homography_pixel_to_table().reshape(-1),
            camera=_camera_config(cfg, NADIR, measured=True, **wrong),
        )
        with caplog.at_level("WARNING"):
            assert vision._hypotheses([{"label": "bowl", "confidence": 0.9, "bbox_px": list(box)}]) == []
        assert "pose_measured" in caplog.text

    @pytest.mark.parametrize("label", ["marker", "cube", "bowl", "bin"])
    def test_a_matched_measured_camera_recovers_centres_to_a_millimetre(self, cfg, label):
        cam = SyntheticPinhole()
        est = _estimator(cfg, cam, _camera_config(cfg, cam, measured=True))
        assert est.camera is not None
        for xy in POSES:
            assert _error_mm(est, cam, label, xy) < 1.0, (label, xy)

    def test_nadir_camera_5_mm_off_still_steps_up_the_image(self, cfg):
        """Missed item 4: the step used to follow look_at - position, which for a
        near-nadir camera 5 mm off toward +Y points along -Y; the bbox bottom is
        fixed by the roll (up = +X). Bowl was 103-126 mm off with an exact H."""
        off = SyntheticPinhole(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480,
                               position=(0.18, 0.005, 0.60), look_at=(0.18, 0.0, 0.0), up=(1.0, 0.0, 0.0))
        for measured, limit in ((False, 25.0), (True, 1.0)):
            est = _estimator(cfg, off, _camera_config(cfg, off, measured=measured))
            np.testing.assert_allclose(est.view_direction, [1.0, 0.0], atol=1e-6)
            for label in ("marker", "bowl"):
                for xy in POSES:
                    assert _error_mm(est, off, label, xy) < limit, (measured, label, xy)

    def test_oblique_camera_view_direction_is_away_from_the_camera(self, cfg):
        cam = SyntheticPinhole()
        est = _estimator(cfg, cam, _camera_config(cfg, cam, measured=False))
        np.testing.assert_allclose(est.view_direction, [1.0, 0.0], atol=1e-6)

    def test_bottom_center_without_any_camera_orientation_is_refused(self, cfg):
        """GEO-5: camera=None used to return the near face as the centre (70 mm
        off for the marker) while the warning claimed a first-order step."""
        with pytest.raises(ConfigurationError, match="bottom_center"):
            HomographyPoseEstimator(cfg.exterior_camera.homography, cfg.hardware.object_sizes,
                                    cfg.hardware.default_object_size, camera=None)

    def test_explicit_view_direction_applies_the_first_order_step(self, cfg):
        cam = SyntheticPinhole()
        est = HomographyPoseEstimator(cam.homography_pixel_to_table().reshape(-1), cfg.hardware.object_sizes,
                                      cfg.hardware.default_object_size, camera=None, view_direction=(1.0, 0.0))
        for xy in POSES:
            assert _error_mm(est, cam, "marker", xy) < 8.0

    def test_a_pinhole_model_argument_counts_as_measured(self, cfg):
        cam = SyntheticPinhole()
        model = PinholeModel(cam.fx, cam.fy, cam.cx, cam.cy, cam.position, cam.look_at, cam.up)
        est = _estimator(cfg, cam, model)
        assert est.pose_measured and _error_mm(est, cam, "bowl", BOWL_XY) < 1.0


class TestYawObservation:
    @pytest.mark.parametrize("measured", [True, False])
    @pytest.mark.parametrize("xy", [(0.16, 0.0), (0.12, -0.10), (0.22, 0.12)])
    def test_marker_across_the_ray_is_placed_within_15_mm_with_its_true_jaw_width(self, cfg, measured, xy):
        """GEO-3: the review's marker lying along Y was placed 60 mm off and
        reported 19 mm across the tangential jaw it would really meet end-on."""
        cam = SyntheticPinhole()
        est = _estimator(cfg, cam, _camera_config(cfg, cam, measured=measured))
        box = ScriptedDetector({}, cam).box_for("marker", (*xy, 0.0, np.pi / 2.0))
        result = est.estimate_footprint({"label": "marker", "bbox_px": list(box)})
        assert np.linalg.norm(result.pose.position[:2] - np.asarray(xy)) < 0.015
        assert result.yaw_rad == pytest.approx(np.pi / 2.0) and not result.yaw_ambiguous
        np.testing.assert_allclose(result.extents, [0.019, 0.140, 0.019])
        bearing = np.arctan2(xy[1], xy[0])
        tangential = np.array([-np.sin(bearing), np.cos(bearing), 0.0])
        radial = np.array([np.cos(bearing), np.sin(bearing), 0.0])
        rotation = result.pose.rotation_matrix()
        truth = np.array([0.019, 0.140, 0.019])
        for closing in (tangential, radial):
            assert chord_width(result.extents, rotation, closing) == pytest.approx(
                chord_width(truth, np.eye(3), closing), rel=1e-6
            )
        if xy[1] == 0.0:
            # Straight ahead the fixed tangential jaw meets its full length.
            assert chord_width(result.extents, rotation, tangential) > cfg.grasp.max_grasp_width

    @pytest.mark.parametrize("camera", [SyntheticPinhole(), NADIR], ids=["oblique", "nadir"])
    def test_marker_along_x_keeps_the_size_table_orientation(self, cfg, camera):
        est = _estimator(cfg, camera, _camera_config(cfg, camera, measured=True))
        box = ScriptedDetector({}, camera).box_for("marker", (*MARKER_XY, 0.0, 0.0))
        result = est.estimate_footprint({"label": "marker", "bbox_px": list(box)})
        assert result.yaw_rad == 0.0 and not result.yaw_ambiguous
        np.testing.assert_allclose(result.extents, cfg.hardware.object_sizes["marker"])

    @pytest.mark.parametrize("measured", [True, False])
    def test_a_diagonal_marker_is_flagged_not_guessed(self, cfg, measured):
        cam = SyntheticPinhole()
        est = _estimator(cfg, cam, _camera_config(cfg, cam, measured=measured))
        for xy in POSES:
            box = ScriptedDetector({}, cam).box_for("marker", (*xy, 0.0, np.pi / 4.0))
            assert est.estimate_footprint({"label": "marker", "bbox_px": list(box)}).yaw_ambiguous, xy

    def test_every_accepted_marker_estimate_is_within_12_mm(self, cfg):
        """Whatever the true heading, an estimate that is *not* flagged is good
        enough to close the jaw on the marker."""
        for camera in (SyntheticPinhole(), NADIR):
            for measured in (True, False):
                est = _estimator(cfg, camera, _camera_config(cfg, camera, measured=measured))
                for yaw_deg in range(0, 180, 5):
                    for xy in POSES:
                        box = ScriptedDetector({}, camera).box_for("marker", (*xy, 0.0, np.radians(yaw_deg)))
                        result = est.estimate_footprint({"label": "marker", "bbox_px": list(box)})
                        if not result.yaw_ambiguous:
                            error = np.linalg.norm(result.pose.position[:2] - np.asarray(xy))
                            assert error < 0.012, (camera.width, measured, yaw_deg, xy, error)

    def test_square_objects_skip_the_choice(self, cfg):
        cam = SyntheticPinhole()
        est = _estimator(cfg, cam, _camera_config(cfg, cam, measured=True))
        box = ScriptedDetector({}, cam).box_for("bowl", (*BOWL_XY, 0.0, np.pi / 2.0))
        result = est.estimate_footprint({"label": "bowl", "bbox_px": list(box)})
        assert result.yaw_scores is None and not result.yaw_ambiguous


# ----------------------------------------------------------------------
# lift prediction vs the honest scripted camera (review F1/F2)
# ----------------------------------------------------------------------


class TestLiftPrediction:
    @staticmethod
    def _vision(cfg, measured: bool) -> PlanarPerception:
        cam = SyntheticPinhole()
        return PlanarPerception(
            clock=FakeClock(), detector=None, config=cfg.perception, hardware=cfg.hardware,
            homography=cam.homography_pixel_to_table().reshape(-1),
            camera=_camera_config(cfg, cam, measured=measured),
        )

    @staticmethod
    def _rest(vision, label="marker", xy=MARKER_XY):
        box = ScriptedDetector({}, SyntheticPinhole()).box_for(label, (*xy, 0.0))
        hyp = vision._hypotheses([{"label": label, "confidence": 0.9, "bbox_px": list(box)}])
        assert len(hyp) == 1
        return hyp[0]

    def test_measured_prediction_matches_what_the_camera_renders_for_a_carried_marker(self, cfg):
        vision = self._vision(cfg, measured=True)
        rest = self._rest(vision)
        half = cfg.hardware.object_sizes["marker"][2] / 2.0
        tcp = np.array([*MARKER_XY, half + cfg.grasp.lift_height])
        prediction = vision.predict_lift(rest, tcp)
        assert prediction.model_available
        detector = ScriptedDetector({}, SyntheticPinhole())
        carried = detector.box_for("marker", (*MARKER_XY, cfg.grasp.lift_height))
        np.testing.assert_allclose(prediction.predicted_bbox_px, carried, atol=1.0)

        def diag(b):
            return float(np.hypot(b[2] - b[0], b[3] - b[1]))

        assert prediction.expected_pixel_scale == pytest.approx(
            diag(carried) / diag(rest.attributes["bbox_px"]), rel=0.01
        )
        assert 1.05 < prediction.expected_pixel_scale < 1.2
        u, v = prediction.tcp_px
        assert carried[0] <= u <= carried[2] and carried[1] <= v <= carried[3]
        # The laptop's occluder is the detector's occluder, to the pixel: the two
        # files mirror their jaw-footprint constants and must keep doing so.
        np.testing.assert_allclose(prediction.arm_bbox_px, detector.arm_box(tcp), atol=1e-6)
        # And the table-plane estimate of that carried box is the prediction.
        observed = vision.estimator.xy_for_box(carried, rest.bbox.extents)
        assert np.linalg.norm(observed - np.asarray(prediction.predicted_xy)) < 0.002

    def test_unmeasured_prediction_is_growth_only(self, cfg):
        vision = self._vision(cfg, measured=False)
        rest = self._rest(vision)
        prediction = vision.predict_lift(rest, np.array([*MARKER_XY, 0.0095 + 0.06]))
        assert not prediction.model_available
        assert prediction.predicted_bbox_px is None and prediction.tcp_px is None and prediction.arm_bbox_px is None
        height = cfg.exterior_camera.position[2]
        assert prediction.expected_pixel_scale == pytest.approx((height - 0.0095) / (height - 0.0095 - 0.06))


# ----------------------------------------------------------------------
# the honest scripted camera and the detector warm-up (review F2, P3)
# ----------------------------------------------------------------------


class TestHonestScriptedDetector:
    def test_only_the_jaw_footprint_hides_objects(self):
        tcp = {"at": None}
        scene = {"marker": (*MARKER_XY, 0.0), "cube": (0.25, 0.15, 0.0)}
        detector = ScriptedDetector(lambda: scene, SyntheticPinhole(), footprints={"cube": (0.03, 0.03, 0.03)},
                                    arm=lambda: tcp["at"])
        assert {d["label"] for d in detector.detect(None, [])} == {"marker", "cube"}
        # Carried: the 30 mm cube vanishes under the 45 mm jaw...
        tcp["at"] = (0.25, 0.15, 0.075)
        scene["cube"] = (0.25, 0.15, 0.06)
        assert {d["label"] for d in detector.detect(None, [])} == {"marker"}
        # ...while a 140 mm marker in the same jaw sticks out and stays in view.
        half = 0.0095
        tcp["at"] = (*MARKER_XY, half + 0.06)
        scene["marker"] = (*MARKER_XY, 0.06)
        scene["cube"] = (0.25, 0.15, 0.0)
        assert {d["label"] for d in detector.detect(None, [])} == {"marker", "cube"}

    def test_a_parked_arm_occludes_nothing(self):
        detector = ScriptedDetector({"marker": MARKER_XY}, SyntheticPinhole(), arm=lambda: (0.041, 0.0, 0.299))
        assert detector.occluder() is None
        assert [d["label"] for d in detector.detect(None, [])] == ["marker"]

    def test_yaw_rotates_the_rendered_box(self):
        detector = ScriptedDetector({}, SyntheticPinhole())
        along = detector.box_for("marker", (0.16, 0.0, 0.0, 0.0))
        across = detector.box_for("marker", (0.16, 0.0, 0.0, np.pi / 2.0))
        assert (across[2] - across[0]) > 3 * (along[2] - along[0])

    def test_real_detector_imperfections_can_be_produced(self):
        """Re-review: exact boxes never fool the verifier, so the fake must be
        able to render what did -- a loose box, a box that takes in the
        fingers above a left-behind marker, a marker split by the jaw."""
        cam = SyntheticPinhole()
        exact = ScriptedDetector({}, cam)
        rest = exact.box_for("marker", (*MARKER_XY, 0.0))
        loose = ScriptedDetector({}, cam, box_scale=1.06).box_for("marker", (*MARKER_XY, 0.0))
        assert (loose[2] - loose[0]) == pytest.approx(1.06 * (rest[2] - rest[0]), abs=1.0)
        assert (loose[0] + loose[2]) == pytest.approx(rest[0] + rest[2], abs=1.0)
        jaw = exact.arm_box((*MARKER_XY, 0.0095 + 0.06))
        merged = ScriptedDetector({}, cam, merge_jaw=True).box_for("marker", (*MARKER_XY, 0.0), jaw)
        assert merged[0] <= min(rest[0], jaw[0]) + 1 and merged[2] >= max(rest[2], jaw[2]) - 1
        assert exact.box_for("marker", (*MARKER_XY, 0.0), jaw) == rest, "the default stays exact"
        carried_jaw = exact.arm_box((*MARKER_XY, 0.0095 + 0.06))
        carried = exact.box_for("marker", (*MARKER_XY, 0.06))
        split = ScriptedDetector({}, cam, partial_occlusion=True).box_for("marker", (*MARKER_XY, 0.06), carried_jaw)
        assert split is not None
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        assert area(split) < 0.8 * area(carried)
        with pytest.raises(ValueError, match="box_scale"):
            ScriptedDetector({}, cam, box_scale=0.0)

    def test_cli_and_builder_default_to_honest(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([])
        assert args.hide_above is None
        assert build_scripted(args.scene, None, args.hide_above).honest
        assert not build_scripted(args.scene, None, 0.04).honest


class TestDetectorWarmUp:
    class _Backend:
        name = "recording"

        def __init__(self, fail: bool = False) -> None:
            self.calls: list[tuple[Any, list[str], float]] = []
            self.fail = fail

        def detect(self, rgb, labels, min_score=0.0):
            self.calls.append((rgb, list(labels), min_score))
            if self.fail:
                raise RuntimeError("engine file missing")
            return []

    def test_one_dummy_detect_on_a_black_frame(self, caplog):
        backend = self._Backend()
        with caplog.at_level("INFO", logger="jetson.detector"):
            elapsed = warm_up(backend, labels=["marker"], image_size=(64, 48))
        assert elapsed is not None and elapsed >= 0.0
        (rgb, labels, _score), = backend.calls
        assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8 and not rgb.any()
        assert labels == ["marker"]
        assert "warm in" in caplog.text

    def test_a_failing_backend_is_logged_not_raised(self, caplog):
        with caplog.at_level("ERROR", logger="jetson.detector"):
            assert warm_up(self._Backend(fail=True)) is None
        assert "engine file missing" in caplog.text


# ----------------------------------------------------------------------
# RemoteDetector / RemoteCamera
# ----------------------------------------------------------------------


class TestRemoteDetector:
    def test_dead_port_raises_detector_error(self):
        detector = RemoteDetector("127.0.0.1", _free_port(), timeout_s=0.5)
        started = time.perf_counter()
        with pytest.raises(DetectorError, match="unreachable") as info:
            detector.connect(retries=2, backoff_s=0.05)
        assert isinstance(info.value, PerceptionError)
        assert not detector.is_ready()
        assert time.perf_counter() - started < 5.0

    def test_ping_records_backend(self, remote_detector):
        assert remote_detector.backend == "scripted"
        assert remote_detector.is_ready()

    def test_detect_returns_wire_boxes_only(self, remote_detector):
        objects = remote_detector.detect(None)
        assert {o["label"] for o in objects} == {"marker", "bowl"}
        for o in objects:
            assert set(o) == {"label", "confidence", "bbox_px"} and len(o["bbox_px"]) == 4
        assert remote_detector.last_image_size == SyntheticPinhole().image_size

    def test_bad_port_and_score_rejected(self):
        with pytest.raises(ValueError):
            RemoteDetector("127.0.0.1", 0)
        with pytest.raises(ValueError):
            RemoteDetector("127.0.0.1", 5558, min_score=1.5)


class TestRemoteCamera:
    def test_capture_over_fake_frame_source(self, cfg, scene, remote_detector):
        cam = SyntheticPinhole()
        source = FakeFrameSource(cam, scene)
        clock = FakeClock()
        clock.step(7)
        camera = RemoteCamera(source, cfg.exterior_camera, clock)
        frame = camera.capture()
        assert isinstance(frame, CameraFrame)
        assert frame.rgb.shape == (cam.height, cam.width, 3) and frame.rgb.dtype == np.uint8
        assert frame.depth is None and frame.segmentation is None
        assert frame.intrinsics.fx == pytest.approx(cam.fx) and frame.intrinsics.width == cam.width
        assert frame.step_index == 7 and frame.sim_time == pytest.approx(7 / 120.0)
        assert camera.name == "exterior_camera"
        # Sending that frame through the detector: the server decodes the JPEG
        # and reports its size, the Florence-fallback path end to end.
        objects = remote_detector.detect(frame)
        assert {o["label"] for o in objects} == {"marker", "bowl"}
        assert remote_detector.last_reply["width"] == cam.width
        assert remote_detector.last_reply["height"] == cam.height

    def test_rpc_failure_becomes_perception_error(self, cfg):
        camera = RemoteCamera(DeadFrameSource(), cfg.exterior_camera, FakeClock())
        with pytest.raises(PerceptionError, match="exterior_camera"):
            camera.capture()


# ----------------------------------------------------------------------
# scripts/serve_detector.py
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def serve_detector():
    return _load_script("serve_detector")


class TestServeDetectorScript:
    def test_scripted_backend_through_build_server(self, serve_detector):
        args = serve_detector.build_parser().parse_args(
            ["--backend", "scripted", "--port", "0", "--scene", "marker:0.18,0.05 bowl:0.15,-0.12"]
        )
        server, owner = serve_detector.build_server(args)
        assert owner is None
        _, port = server.serve_in_thread(port=0)
        try:
            detector = RemoteDetector("127.0.0.1", port, labels=("marker", "bowl"))
            assert detector.connect(retries=1)["backend"] == "scripted"
            assert {o["label"] for o in detector.detect(None)} == {"marker", "bowl"}
        finally:
            server.stop()

    def test_fake_flag_is_scripted(self, serve_detector):
        args = serve_detector.build_parser().parse_args(["--fake"])
        assert isinstance(serve_detector.build_backend(args), ScriptedDetector)
        args = serve_detector.build_parser().parse_args([])
        assert args.backend == "florence-onnx"  # the Jetson default: the sim's model, ONNX FP16
        backend = serve_detector.build_backend(args)
        assert isinstance(backend, serve_detector.FlorenceOnnxDetector) and backend.name == "florence-onnx"
        args = serve_detector.build_parser().parse_args(["--backend", "florence"])
        backend = serve_detector.build_backend(args)
        assert isinstance(backend, serve_detector.FlorenceDetector) and backend.name == "florence"
        assert backend.model_name == "florence-community/Florence-2-base"
        assert backend.max_new_tokens == 128 and backend.num_beams == 1

    def test_print_homography_matches_fake_yaml(self, serve_detector, cfg, capsys):
        assert serve_detector.main(["--fake", "--print-homography"]) == 0
        printed = json.loads(capsys.readouterr().out.strip())
        np.testing.assert_allclose(printed, cfg.exterior_camera.homography, atol=1e-12)

    def test_entry_point_serves_scripted(self):
        """The real command line, as a subprocess, answers the 5558 protocol."""
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "serve_detector.py"), "--backend", "scripted",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "WARNING"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            detector = RemoteDetector("127.0.0.1", port, labels=("marker", "bowl"), timeout_s=1.0)
            deadline = time.time() + 10.0
            reply = None
            while time.time() < deadline and reply is None:
                if proc.poll() is not None:
                    pytest.fail(f"serve_detector exited early with {proc.returncode}")
                try:
                    reply = detector.connect(retries=1)
                except DetectorError:
                    time.sleep(0.2)
            assert reply is not None and reply["backend"] == "scripted"
            assert {o["label"] for o in detector.detect(None)} == {"marker", "bowl"}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_importing_the_script_never_loads_torch(self):
        """Fresh interpreter: import the script, build both backends, no torch/transformers."""
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('sd', {str(SCRIPTS / 'serve_detector.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "m.build_backend(m.build_parser().parse_args([]))\n"
            "m.build_backend(m.build_parser().parse_args(['--fake']))\n"
            "m.FlorenceDetector().detect(None, ['marker']) if False else None\n"
            "bad = [k for k in ('torch', 'transformers', 'isaacsim', 'omni', 'pxr', 'carb') if k in sys.modules]\n"
            "print(','.join(bad))\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", f"heavy modules imported: {result.stdout.strip()}"

    def test_florence_prompt_and_label_matching(self, serve_detector):
        prompt = serve_detector.build_prompt(["Marker", "bowl", "marker", " "])
        assert prompt.startswith(serve_detector.GROUNDING_TASK)
        assert prompt.endswith("a marker. a bowl.")
        assert serve_detector.match_label("a whiteboard marker", ["marker", "bowl"]) == "marker"
        assert serve_detector.match_label("cardboard box", ["box", "cardboard box"]) == "cardboard box"
        assert serve_detector.match_label("a coffee mug", ["marker", "bowl"]) is None

    def test_grounding_to_objects(self, serve_detector):
        parsed = {
            "bboxes": [[10.5, 20.2, 110.0, 80.7], [-5.0, 0.0, 30.0, 25.0], [50, 50, 50.2, 60], [0, 0, 20, 20]],
            "labels": ["a marker", "a bowl", "a bowl", "a coffee mug"],
        }
        objects = serve_detector.grounding_to_objects(parsed, ["marker", "bowl"], 0.6, 0.1, (640, 480))
        assert [o["label"] for o in objects] == ["marker", "bowl"]
        assert objects[0]["bbox_px"] == [10, 20, 110, 81] and objects[0]["confidence"] == 0.6
        assert objects[1]["bbox_px"][0] == 0  # clipped to the image
        assert serve_detector.grounding_to_objects(parsed, ["marker"], 0.6, 0.9, (640, 480)) == []

    def test_florence_detect_without_frame_fails_before_loading(self, serve_detector):
        det = serve_detector.FlorenceDetector()
        with pytest.raises(RuntimeError, match="needs a frame"):
            det.detect(None, ["marker"])
        assert det.detect(np.zeros((8, 8, 3), dtype=np.uint8), []) == []
        assert det.detect(np.zeros((8, 8, 3), dtype=np.uint8), ["marker"], min_score=0.95) == []
        assert det.loader is None
        with pytest.raises(ValueError):
            serve_detector.FlorenceDetector(confidence=1.5)


# ----------------------------------------------------------------------
# scripts/calibrate_table.py
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibrate_table():
    return _load_script("calibrate_table")


def _calibration_points(n: int = 12, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = SyntheticPinhole()
    rng = np.random.default_rng(seed)
    xy = np.column_stack([rng.uniform(0.07, 0.29, n), rng.uniform(-0.22, 0.22, n)])
    uv = cam.project(np.column_stack([xy, np.zeros(n)]))
    return uv, xy, cam.homography_pixel_to_table()


class TestCalibrateTableMaths:
    def test_exact_points_recover_the_camera_homography(self, calibrate_table):
        uv, xy, exact = _calibration_points()
        h, inliers = calibrate_table.fit_homography(uv, xy)
        assert inliers.all()
        np.testing.assert_allclose(h, exact, atol=1e-6)
        assert calibrate_table.reprojection_errors(h, uv, xy).max() < 1e-6
        assert calibrate_table.pixel_errors(h, uv, xy).max() < 1e-3

    def test_four_points_exact_fit(self, calibrate_table):
        uv, xy, exact = _calibration_points(4)
        h, inliers = calibrate_table.fit_homography(uv, xy)
        assert inliers.tolist() == [True] * 4
        np.testing.assert_allclose(h, exact, atol=1e-6)

    def test_ransac_rejects_a_mistyped_point(self, calibrate_table):
        uv, xy, exact = _calibration_points(12)
        xy_bad = xy.copy()
        xy_bad[5] += np.array([0.08, -0.05])  # typed 8 cm off
        h, inliers = calibrate_table.fit_homography(uv, xy_bad, ransac_threshold_px=3.0)
        assert not inliers[5] and inliers.sum() == 11
        err = calibrate_table.reprojection_errors(h, uv, xy)
        assert err.max() < 1e-3
        report = calibrate_table.format_report(uv, xy_bad, h, inliers)
        assert "OUTLIER" in report and "inliers 11/12" in report

    def test_pixel_noise_stays_within_two_millimetres(self, calibrate_table):
        uv, xy, _ = _calibration_points(12, seed=7)
        rng = np.random.default_rng(1)
        h, _ = calibrate_table.fit_homography(uv + rng.normal(0.0, 0.5, uv.shape), xy)
        assert calibrate_table.reprojection_errors(h, uv, xy).max() < 0.002

    def test_too_few_or_degenerate_points_rejected(self, calibrate_table):
        uv, xy, _ = _calibration_points(3)
        with pytest.raises(ValueError, match="at least 4"):
            calibrate_table.fit_homography(uv, xy)
        line_uv = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
        line_xy = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
        with pytest.raises(ValueError):
            calibrate_table.fit_homography(line_uv, line_xy)

    def test_parse_xy_and_csv_round_trip(self, calibrate_table, tmp_path):
        assert calibrate_table.parse_xy("0.18, -0.05") == (0.18, -0.05)
        with pytest.raises(ValueError):
            calibrate_table.parse_xy("0.18")
        uv, xy, _ = _calibration_points(5)
        path = calibrate_table.write_points_csv(tmp_path / "pts.csv", uv, xy)
        uv2, xy2 = calibrate_table.read_points_csv(path)
        np.testing.assert_allclose(uv2, uv, atol=1e-3)
        np.testing.assert_allclose(xy2, xy, atol=1e-5)


class TestCalibrateTableYaml:
    COMMENTED = (
        "# Hardware lane config\n"
        "extends: hardware.yaml\n"
        "\n"
        "hardware:\n"
        "  jetson_host: 127.0.0.1\n"
        "\n"
        "exterior_camera:\n"
        "  # MEASURE: keep this comment\n"
        "  fx: 600.0\n"
        "  # MEASURE (scripts/calibrate_table.py --touch)\n"
        "  homography: []\n"
        "  position: [0.18, 0.0, 0.60]\n"
        "\n"
        "perception:\n"
        "  ground_plane_z: 0.0\n"
    )

    def test_splice_keeps_extends_comments_and_order(self, calibrate_table, tmp_path):
        import yaml

        path = tmp_path / "cal.yaml"
        path.write_bytes(self.COMMENTED.encode("utf-8"))  # LF, whatever the platform
        _, _, exact = _calibration_points()
        out = calibrate_table.write_homography(path, exact * 2.0)  # un-normalised on purpose
        assert out == path
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        assert data["extends"] == "hardware.yaml"
        assert list(data) == ["extends", "hardware", "exterior_camera", "perception"]
        np.testing.assert_allclose(homography_from_config(data["exterior_camera"]["homography"]), exact, atol=1e-12)
        assert data["exterior_camera"]["fx"] == 600.0 and data["exterior_camera"]["position"] == [0.18, 0.0, 0.6]
        assert "# MEASURE: keep this comment" in text
        assert "# Hardware lane config" in text
        assert text.count("homography:") == 1
        assert b"\r" not in path.read_bytes()  # LF in, LF out (no whole-file diff)
        assert (tmp_path / "cal.yaml.bak").read_text(encoding="utf-8") == self.COMMENTED
        # Second write replaces the multi-line entry, not appends.
        calibrate_table.write_homography(path, exact)
        again = path.read_text(encoding="utf-8")
        assert again.count("homography:") == 1 and again.count("[") == 2  # matrix + position

    def test_multiline_flow_entry_is_replaced(self, calibrate_table, tmp_path):
        import yaml

        path = tmp_path / "fake.yaml"
        path.write_bytes(HARDWARE_FAKE_YAML.read_bytes())
        h = np.array([[1e-3, 0.0, -0.3], [0.0, -1e-3, 0.4], [0.0, 2e-4, 1.0]])
        calibrate_table.write_homography(path, h, backup=False)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        assert data["extends"] == "hardware.yaml"
        np.testing.assert_allclose(data["exterior_camera"]["homography"], h.reshape(-1))
        assert data["hardware"]["fake_clock"] is True
        assert text.count("homography:") == 1
        assert "Regenerate if SyntheticPinhole" in text  # comments survive
        assert not (tmp_path / "fake.yaml.bak").exists()

    def test_missing_block_is_added_and_loadable(self, calibrate_table, tmp_path):
        import yaml

        path = tmp_path / "minimal.yaml"
        path.write_bytes(b"extends: hardware.yaml\nbackend: hardware\n")
        _, _, exact = _calibration_points()
        calibrate_table.write_homography(path, exact)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["extends"] == "hardware.yaml" and data["backend"] == "hardware"
        assert len(data["exterior_camera"]["homography"]) == 9
        # And the framework loader accepts the result.
        cfg = load_config(REPO_ROOT / "configs" / "hardware.yaml",
                          overrides={"exterior_camera": {"homography": data["exterior_camera"]["homography"]}})
        np.testing.assert_allclose(homography_from_config(cfg.exterior_camera.homography), exact, atol=1e-12)

    def test_block_without_entry_gets_one(self, calibrate_table, tmp_path):
        import yaml

        path = tmp_path / "noentry.yaml"
        path.write_bytes(b"extends: hardware.yaml\nexterior_camera:\n  fx: 600.0\n  # tail comment\n")
        _, _, exact = _calibration_points()
        calibrate_table.write_homography(path, exact, backup=False)
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        assert data["exterior_camera"]["fx"] == 600.0 and len(data["exterior_camera"]["homography"]) == 9
        assert "# tail comment" in text

    def test_crlf_file_stays_crlf(self, calibrate_table, tmp_path):
        import yaml

        path = tmp_path / "crlf.yaml"
        path.write_bytes(self.COMMENTED.replace("\n", "\r\n").encode("utf-8"))
        _, _, exact = _calibration_points()
        calibrate_table.write_homography(path, exact, backup=False)
        raw = path.read_bytes()
        assert raw.count(b"\r\n") == raw.count(b"\n")
        assert len(yaml.safe_load(raw.decode("utf-8"))["exterior_camera"]["homography"]) == 9

    def test_non_finite_rejected(self, calibrate_table, tmp_path):
        with pytest.raises(ValueError):
            calibrate_table.write_homography(tmp_path / "x.yaml", np.full((3, 3), np.nan))
