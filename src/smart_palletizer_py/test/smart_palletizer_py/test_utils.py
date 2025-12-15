import math
import numpy as np
import pytest
from unittest.mock import MagicMock

import smart_palletizer_py.utils as utils


def test_time_filter_with_no_prev():
    """If prev_filtered_img is not provided, it should return the original image."""
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    result = utils.time_filter(img, prev_filtered_img=None)
    np.testing.assert_array_equal(result, img)


def test_time_filter_basic_ema():
    """Test that EMA is applied correctly when differences are below delta."""
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    prev = np.array([[12, 18], [28, 42]], dtype=np.uint8)
    alpha = 0.5
    delta = 20
    result = utils.time_filter(img, prev, alpha=alpha, delta=delta)

    expected = alpha * img + (1 - alpha) * prev
    np.testing.assert_array_equal(result, expected.astype(np.uint8))


def test_time_filter_large_difference():
    """Test that when pixel difference exceeds delta, the current value is kept."""
    img = np.array([[100, 200], [150, 250]], dtype=np.uint8)
    prev = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    delta = 50
    result = utils.time_filter(img, prev, alpha=0.5, delta=delta)

    # All differences are larger than delta, so result should equal img
    np.testing.assert_array_equal(result, img)


@pytest.mark.parametrize("angles", [
    (0, 0, 0),
    (math.pi / 2, 0, 0),
    (0, math.pi / 2, 0),
    (0, 0, math.pi / 2),
])
def test_quaternion_from_euler_basic(angles):
    """Test that quaternions have unit norm."""
    q = utils.quaternion_from_euler(*angles)
    norm = np.linalg.norm(q)
    # The quaternion should be normalized
    assert math.isclose(norm, 1.0, rel_tol=1e-9)


def test_quaternion_from_euler_known_value():
    """Test known conversion for zero angles."""
    q = utils.quaternion_from_euler(0, 0, 0)
    # Zero rotation corresponds to quaternion [0,0,0,1]
    np.testing.assert_array_almost_equal(q, [0, 0, 0, 1])


def test_get_XYZ_from_Pixels_scaling():
    """Test that the scaling produces correct depth."""
    # Create a mock camera
    camera = MagicMock()
    camera.project_pixel_to_3d_ray.return_value = (1.0, 2.0, 2.0)
    
    u, v, d = 10, 20, 1000  # depth in mm
    X, Y, Z = utils.get_XYZ_from_Pixels(camera, u, v, d)
    
    # Z should be approximately equal to d / 1000
    expected_factor = d / (1000 * 2.0)
    np.testing.assert_allclose([X, Y, Z], [1.0 * expected_factor, 2.0 * expected_factor, 2.0 * expected_factor])


def test_get_XYZ_from_Pixels_vector_direction():
    """Test that the returned vector is scaled version of unit vector."""
    camera = MagicMock()
    camera.project_pixel_to_3d_ray.return_value = (0.5, 0.5, 1.0)
    d = 500
    X, Y, Z = utils.get_XYZ_from_Pixels(camera, 0, 0, d)
    factor = d / (1000 * 1.0)
    np.testing.assert_allclose([X, Y, Z], [0.5 * factor, 0.5 * factor, 1.0 * factor])
