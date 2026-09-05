"""Transform math tests.

These run without Isaac Sim. The rot6d round-trip tests matter most: that
encoding is an external contract with GR00T, and getting the column/row
convention backwards produces plausible-looking but consistently wrong
orientations that are painful to debug from robot behaviour alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mfw.utils import transforms as tf


def _random_quats(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q


class TestQuaternionMatrix:
    def test_identity_roundtrip(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(tf.quat_to_matrix(q), np.eye(3))
        assert np.allclose(tf.matrix_to_quat(np.eye(3)), q)

    @pytest.mark.parametrize("seed", range(5))
    def test_quat_matrix_roundtrip(self, seed):
        for q in _random_quats(20, seed):
            r = tf.quat_to_matrix(q)
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-10), "not orthonormal"
            assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10), "not a proper rotation"
            assert np.allclose(tf.matrix_to_quat(r), q, atol=1e-9)

    def test_180_degree_rotations_are_stable(self):
        """The naive w-only formula loses precision here; Shepperd's method must not.

        Wrist-flip grasps live at exactly these orientations.
        """
        for axis in np.eye(3):
            q = np.concatenate([[0.0], axis])  # 180 deg about axis
            r = tf.quat_to_matrix(q)
            recovered = tf.matrix_to_quat(r)
            assert np.allclose(tf.quat_to_matrix(recovered), r, atol=1e-9)

    def test_zero_norm_quat_rejected(self):
        with pytest.raises(ValueError, match="zero-norm"):
            tf.quat_to_matrix(np.zeros(4))

    def test_double_cover_canonicalised(self):
        """q and -q are the same rotation; matrix_to_quat must pick one consistently."""
        for q in _random_quats(10, seed=7):
            assert tf.matrix_to_quat(tf.quat_to_matrix(q))[0] >= 0.0


class TestQuaternionAlgebra:
    def test_multiply_matches_matrix_composition(self):
        a, b = _random_quats(2, seed=3)
        assert np.allclose(
            tf.quat_to_matrix(tf.quat_multiply(a, b)),
            tf.quat_to_matrix(a) @ tf.quat_to_matrix(b),
            atol=1e-10,
        )

    def test_conjugate_is_inverse(self):
        for q in _random_quats(10, seed=4):
            prod = tf.quat_multiply(q, tf.quat_conjugate(q))
            assert np.allclose(np.abs(prod), [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_rotate_preserves_length(self):
        v = np.array([0.3, -0.7, 1.1])
        for q in _random_quats(10, seed=5):
            assert np.isclose(np.linalg.norm(tf.quat_rotate(q, v)), np.linalg.norm(v))

    def test_scalar_order_conversions_roundtrip(self):
        q = _random_quats(1, seed=6)[0]
        assert np.allclose(tf.quat_scalar_last_to_first(tf.quat_scalar_first_to_last(q)), q)

    def test_angular_distance_handles_double_cover(self):
        q = _random_quats(1, seed=8)[0]
        assert tf.quat_angular_distance(q, q) == pytest.approx(0.0, abs=1e-7)
        assert tf.quat_angular_distance(q, -q) == pytest.approx(0.0, abs=1e-7)

    def test_angular_distance_known_value(self):
        q90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # 90 deg about z
        assert tf.quat_angular_distance([1.0, 0, 0, 0], q90) == pytest.approx(np.pi / 2, abs=1e-9)


class TestRot6D:
    """GR00T's eef_9d contract: [xyz(3), rot6d(6)] with rot6d as the first two columns."""

    @pytest.mark.parametrize("seed", range(5))
    def test_roundtrip(self, seed):
        for q in _random_quats(20, seed):
            r = tf.quat_to_matrix(q)
            assert np.allclose(tf.rot6d_to_matrix(tf.matrix_to_rot6d(r)), r, atol=1e-10)

    def test_encodes_columns_not_rows(self):
        """Guards the exact convention. A transpose here is silent and costly."""
        r = tf.quat_to_matrix(_random_quats(1, seed=11)[0])
        six = tf.matrix_to_rot6d(r)
        assert np.allclose(six[:3], r[:, 0]), "first 3 must be column 0"
        assert np.allclose(six[3:], r[:, 1]), "next 3 must be column 1"

    def test_decodes_unnormalised_network_output(self):
        """Raw policy output is neither unit-length nor orthogonal; decoding must cope."""
        raw = np.array([2.0, 0.0, 0.0, 0.5, 3.0, 0.0])
        r = tf.rot6d_to_matrix(raw)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-10)

    def test_collinear_basis_rejected(self):
        with pytest.raises(ValueError, match="collinear"):
            tf.rot6d_to_matrix([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])

    def test_degenerate_first_vector_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            tf.rot6d_to_matrix([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


class TestHomogeneousTransforms:
    def test_invert_is_true_inverse(self):
        for q in _random_quats(10, seed=12):
            t = tf.make_transform([0.4, -0.2, 0.9], q)
            assert np.allclose(tf.invert_transform(t) @ t, np.eye(4), atol=1e-10)

    def test_invert_matches_numpy_inv(self):
        t = tf.make_transform([1.0, 2.0, 3.0], _random_quats(1, seed=13)[0])
        assert np.allclose(tf.invert_transform(t), np.linalg.inv(t), atol=1e-10)

    def test_transform_points_matches_manual(self):
        q = _random_quats(1, seed=14)[0]
        t = tf.make_transform([0.1, 0.2, 0.3], q)
        pts = np.random.default_rng(0).normal(size=(50, 3))
        expected = (t[:3, :3] @ pts.T).T + t[:3, 3]
        assert np.allclose(tf.transform_points(t, pts), expected, atol=1e-12)

    def test_transform_points_rejects_bad_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            tf.transform_points(np.eye(4), np.zeros((5, 2)))

    def test_matrix_to_pose_roundtrip(self):
        q = _random_quats(1, seed=15)[0]
        pos = np.array([0.5, -0.3, 0.7])
        p, r = tf.matrix_to_pose(tf.make_transform(pos, q))
        assert np.allclose(p, pos)
        assert np.allclose(r, q, atol=1e-9)


class TestOrthonormalize:
    def test_repairs_drifted_rotation(self):
        r = tf.quat_to_matrix(_random_quats(1, seed=16)[0])
        drifted = r + np.random.default_rng(1).normal(scale=1e-3, size=(3, 3))
        fixed = tf.orthonormalize(drifted)
        assert np.allclose(fixed @ fixed.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(fixed), 1.0, atol=1e-10)
        assert np.allclose(fixed, r, atol=5e-3)

    def test_never_returns_a_reflection(self):
        """det must be +1 even when the input is a mirror; a reflection is not a pose."""
        reflection = np.diag([1.0, 1.0, -1.0])
        assert np.isclose(np.linalg.det(tf.orthonormalize(reflection)), 1.0, atol=1e-10)
