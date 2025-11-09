"""
Test suite for XFT Array class.
Tests both C++ bindings and Python wrapper.
"""

import pytest
import numpy as np
import xft

class TestArrayCreation:
    """Test array creation and initialization."""
    
    def test_create_1d_array(self):
        """Test 1D array creation."""
        arr = xft.Array([10], dtype=xft.DType.Float32)
        assert arr.shape == (10,)
        assert arr.ndim == 1
        assert arr.size == 10
        assert arr.dtype == xft.DType.Float32
    
    def test_create_2d_array(self):
        """Test 2D array creation."""
        arr = xft.Array([3, 4], dtype=xft.DType.Float64)
        assert arr.shape == (3, 4)
        assert arr.ndim == 2
        assert arr.size == 12
        assert arr.dtype == xft.DType.Float64
    
    def test_create_3d_array(self):
        """Test 3D array creation."""
        arr = xft.Array([2, 3, 4], dtype=xft.DType.Float32)
        assert arr.shape == (2, 3, 4)
        assert arr.ndim == 3
        assert arr.size == 24
    
    def test_itemsize(self):
        """Test itemsize property."""
        arr32 = xft.Array([10], dtype=xft.DType.Float32)
        arr64 = xft.Array([10], dtype=xft.DType.Float64)
        assert arr32.itemsize == 4
        assert arr64.itemsize == 8
    
    def test_nbytes(self):
        """Test total bytes calculation."""
        arr = xft.Array([10, 20], dtype=xft.DType.Float32)
        assert arr.nbytes == 10 * 20 * 4  # 800 bytes

class TestArrayFromNumPy:
    """Test wrapping NumPy arrays."""
    
    def test_wrap_numpy_float32(self):
        """Test wrapping float32 NumPy array."""
        np_arr = np.random.randn(5, 6).astype(np.float32)
        xft_arr = xft.Array(np_arr)
        
        assert xft_arr.shape == (5, 6)
        assert xft_arr.dtype == xft.DType.Float32
    
    def test_wrap_numpy_float64(self):
        """Test wrapping float64 NumPy array."""
        np_arr = np.random.randn(3, 4).astype(np.float64)
        xft_arr = xft.Array(np_arr)
        
        assert xft_arr.shape == (3, 4)
        assert xft_arr.dtype == xft.DType.Float64
    
    def test_zero_copy_wrap(self):
        """Test that wrapping is zero-copy (shares memory)."""
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        xft_arr = xft.Array(np_arr)
        
        # Modify via XFT
        xft_arr[0, 0] = 99.0
        
        # Should reflect in NumPy array
        assert np_arr[0, 0] == 99.0
    
    def test_from_numpy_helper(self):
        """Test from_numpy convenience function."""
        np_arr = np.ones((4, 5), dtype=np.float32)
        xft_arr = xft.from_numpy(np_arr)
        
        assert xft_arr.shape == (4, 5)
        assert xft_arr.size == 20

class TestArrayToNumPy:
    """Test converting XFT arrays to NumPy."""
    
    def test_to_numpy_float32(self):
        """Test converting float32 array to NumPy."""
        xft_arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        np_arr = xft_arr.numpy()
        
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (3, 4)
        assert np_arr.dtype == np.float32
    
    def test_to_numpy_float64(self):
        """Test converting float64 array to NumPy."""
        xft_arr = xft.Array([2, 5], dtype=xft.DType.Float64)
        np_arr = xft_arr.numpy()
        
        assert np_arr.dtype == np.float64
    
    def test_zero_copy_to_numpy(self):
        """Test that conversion to NumPy is zero-copy."""
        xft_arr = xft.Array([2, 3], dtype=xft.DType.Float32)
        np_arr = xft_arr.numpy()
        
        # Modify via NumPy
        np_arr[0, 0] = 42.0
        
        # Should reflect in XFT array
        assert xft_arr[0, 0] == 42.0
    
    def test_round_trip(self):
        """Test NumPy -> XFT -> NumPy round trip."""
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        xft_arr = xft.from_numpy(original)
        back = xft_arr.numpy()
        
        # Should share memory (same pointer)
        assert np.shares_memory(original, back)
        np.testing.assert_array_equal(original, back)

class TestStridesAndLayout:
    """Test stride calculation and memory layout."""
    
    def test_c_contiguous_strides(self):
        """Test C-order (row-major) strides."""
        arr = xft.Array([3, 4, 5], dtype=xft.DType.Float32, order=xft.Order.C)
        
        # C-order: strides = [4*5*4, 5*4, 4] = [80, 20, 4]
        assert arr.strides == (80, 20, 4)
        assert arr.is_contiguous() is True
    
    def test_f_contiguous_strides(self):
        """Test F-order (column-major) strides."""
        arr = xft.Array([3, 4, 5], dtype=xft.DType.Float32, order=xft.Order.F)
        
        # F-order: strides = [4, 3*4, 3*4*4] = [4, 12, 48]
        assert arr.strides == (4, 12, 48)
        assert arr.is_f_contiguous() is True
    
    def test_contiguity_checks(self):
        """Test contiguity detection."""
        c_arr = xft.Array([10, 20], dtype=xft.DType.Float64, order=xft.Order.C)
        f_arr = xft.Array([10, 20], dtype=xft.DType.Float64, order=xft.Order.F)
        
        assert c_arr.is_contiguous() is True
        assert c_arr.is_f_contiguous() is False
        
        assert f_arr.is_contiguous() is False
        assert f_arr.is_f_contiguous() is True

class TestElementAccess:
    """Test element access and modification."""
    
    def test_1d_get_set_float32(self):
        """Test 1D element access for float32."""
        arr = xft.Array([5], dtype=xft.DType.Float32)
        
        arr[0] = 1.5
        arr[1] = 2.5
        arr[4] = 9.5
        
        assert arr[0] == 1.5
        assert arr[1] == 2.5
        assert arr[4] == 9.5
    
    def test_1d_get_set_float64(self):
        """Test 1D element access for float64."""
        arr = xft.Array([3], dtype=xft.DType.Float64)
        
        arr[0] = 1.23456789
        arr[2] = 9.87654321
        
        assert abs(arr[0] - 1.23456789) < 1e-10
        assert abs(arr[2] - 9.87654321) < 1e-10
    
    def test_2d_get_set_float32(self):
        """Test 2D element access for float32."""
        arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        
        arr[0, 0] = 1.0
        arr[1, 2] = 5.5
        arr[2, 3] = 9.9
        
        assert arr[0, 0] == 1.0
        assert arr[1, 2] == 5.5
        assert abs(arr[2, 3] - 9.9) < 1e-6  # Float32 precision
    
    def test_element_access_via_numpy(self):
        """Test that element access matches NumPy."""
        xft_arr = xft.Array([2, 3], dtype=xft.DType.Float32)
        np_arr = xft_arr.numpy()
        
        # Set via XFT
        xft_arr[0, 1] = 42.0
        assert np_arr[0, 1] == 42.0
        
        # Set via NumPy
        np_arr[1, 2] = 99.0
        assert xft_arr[1, 2] == 99.0

class TestArrayFunction:
    """Test array() convenience function."""
    
    def test_array_from_shape(self):
        """Test creating array from shape."""
        arr = xft.array([5, 6])
        assert arr.shape == (5, 6)
    
    def test_array_from_numpy(self):
        """Test creating array from NumPy."""
        np_arr = np.ones((3, 4), dtype=np.float32)
        arr = xft.array(np_arr)
        assert arr.shape == (3, 4)
    
    def test_array_from_nested_list(self):
        """Test creating array from nested lists."""
        data = [[1, 2, 3], [4, 5, 6]]
        arr = xft.array(data)
        assert arr.shape == (2, 3)
        assert arr[0, 0] == 1.0
        assert arr[1, 2] == 6.0

class TestEdgeCases:
    """Test error handling and edge cases."""
    
    def test_empty_shape_raises(self):
        """Test that empty shape raises error."""
        with pytest.raises(ValueError):
            xft.Array([])
    
    def test_zero_dimension_raises(self):
        """Test that zero in shape raises error."""
        with pytest.raises(ValueError):
            xft.Array([3, 0, 5])
    
    def test_out_of_bounds_1d(self):
        """Test bounds checking for 1D access."""
        arr = xft.Array([5], dtype=xft.DType.Float32)
        with pytest.raises(Exception):  # C++ throws out_of_range
            arr[10]
    
    def test_out_of_bounds_2d(self):
        """Test bounds checking for 2D access."""
        arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        with pytest.raises(Exception):
            arr[5, 2]
    
    def test_type_mismatch(self):
        """Test that type mismatches are caught."""
        arr = xft.Array([5], dtype=xft.DType.Float32)
        # Accessing float32 array as float64 should fail in C++
        with pytest.raises(Exception):
            arr.core.get_float64(0)

class TestRepr:
    """Test string representations."""

    def test_repr_shows_metadata(self):
        """Test that __repr__ shows shape, dtype, and ndim."""
        arr = xft.Array([3, 4], dtype=xft.DType.Float32)
        repr_str = repr(arr)

        # Expected format: xft.Array(shape=(3, 4), dtype=float32, ndim=2)
        assert repr_str.startswith("xft.Array(")
        assert "shape=(3, 4)" in repr_str
        assert "dtype=float32" in repr_str
        assert "ndim=2" in repr_str


if __name__ == "__main__":
    import time
    import sys
    import pytest

    print("\nPython Array Tests")
    print("------------------")
    print("Testing C++ bindings + Python wrapper for XFT Array...")
    print("Includes: creation, NumPy interop, strides/layout, element access, and repr.\n")

    start_time = time.time()
    result = pytest.main([__file__, "-v", "--disable-warnings"])
    duration = time.time() - start_time

    # Summary footer
    print("\n------------------")
    print("Test Summary")
    print("------------------")
    print(f"Total tests run: {result.testscollected if hasattr(result, 'testscollected') else 'N/A'}")
    print(f"Exit code: {result}")
    print(f"Time taken: {duration:.2f} seconds")
    print("\n✅ All array tests completed.\n" if result == 0 else "\n❌ Some tests failed.\n")

    sys.exit(result)

