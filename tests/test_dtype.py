import numpy as np
import ml_dtypes
import jax
import jax.numpy as jnp
import itertools

# Import the code we are testing
from xft._internal import dtype as xft_dtypes

# Import our test infrastructure
from tests.test_base import XftTestCase, parameterized

# This class will contain all the tests for the xft dtype system.
# Each method starting with 'test_' will be automatically discovered
# and run by pytest.
#
# Use the 'parameterized' library to run this single test method
# multiple times with different inputs. This is incredibly powerful for
# ensuring our code works across all dtypes.

# This list defines all the types we want to use for our exhaustive test.
# It includes Python weak types and all the native types corresponding to our DTypes.
_JAX_COMPAT_TYPES = [
    # Weak Types
    bool, int, float, complex,
    # Strong Types
    np.bool_, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    ml_dtypes.bfloat16,
    np.float16, np.float32, np.float64,
    np.complex64, np.complex128,
    # Your custom small int types
    ml_dtypes.int2, ml_dtypes.int4,
    ml_dtypes.uint2, ml_dtypes.uint4,
]

def get_name(t):
    """Helper function to get a clean name for a type."""
    try: return np.dtype(t).name
    except TypeError: return t.__name__
    
class DTypeTest(XftTestCase):

    # 1. Test dtype properties of DType objects
    @parameterized.expand([
        # Format: (test_name_suffix, xft_dtype_obj, expected_native_type)
        ("float32", xft_dtypes.dtypes.float32, np.float32),
        ("int64", xft_dtypes.dtypes.int64, np.int64),
        ("bool", xft_dtypes.dtypes.bool, np.bool_),
        ("bfloat16", xft_dtypes.dtypes.bfloat16, ml_dtypes.bfloat16),
        ("complex128", xft_dtypes.dtypes.complex128, np.complex128),
    ])
    def test_dtype_properties(self, name, dtype_obj, native_type):
        """
        Tests that the basic properties of DType objects are correct.
        This test will be run for each set of parameters defined in the decorator.
        """
        # We check that the `name` attribute of our DType object (e.g., 'float32')
        # matches the name we expect.
        self.assertEqual(dtype_obj.name, name)

        # This is a crucial check. It ensures that our DType wrapper is correctly
        # holding onto the underlying NumPy or ml_dtypes object.
        self.assertEqual(dtype_obj.native_type, np.dtype(native_type))

        # We check that the priority is an integer, which is essential for
        # the type promotion logic.
        self.assertIsInstance(dtype_obj.priority, int)

        # A good __repr__ is important for debugging. We ensure it looks correct.
        expected_repr = f'xft.dtypes.{name}'
        self.assertEqual(repr(dtype_obj), expected_repr)

    # We create a comprehensive list of test cases for dtype inference.
    # Each entry is a tuple: (test_name, input_value, expected_dtype_obj)
    # This covers Python scalars, NumPy scalars, NumPy arrays, and our own DType objects.
    DTYPE_INFERENCE_CASES = [
        # --- Python Native Types ---
        ("python_bool", True, xft_dtypes.dtypes.bool),
        ("python_int", 10, xft_dtypes.dtypes.int64),
        ("python_float", 3.14, xft_dtypes.dtypes.float64),
        ("python_complex", 1 + 2j, xft_dtypes.dtypes.complex128),

        # --- NumPy Scalar Values ---
        ("numpy_bool", np.bool_(True), xft_dtypes.dtypes.bool),
        ("numpy_int8", np.int8(1), xft_dtypes.dtypes.int8),
        ("numpy_int32", np.int32(1), xft_dtypes.dtypes.int32),
        ("numpy_int64", np.int64(1), xft_dtypes.dtypes.int64),
        ("numpy_uint32", np.uint32(1), xft_dtypes.dtypes.uint32),
        ("numpy_float16", np.float16(1.0), xft_dtypes.dtypes.float16),
        ("numpy_float32", np.float32(1.0), xft_dtypes.dtypes.float32),
        ("numpy_float64", np.float64(1.0), xft_dtypes.dtypes.float64),
        ("numpy_complex64", np.complex64(1+1j), xft_dtypes.dtypes.complex64),

        # --- ml_dtypes Scalar Values ---
        ("numpy_bfloat16", ml_dtypes.bfloat16(1.0), xft_dtypes.dtypes.bfloat16),

        # --- NumPy Arrays ---
        ("array_int32", np.array([1, 2], dtype=np.int32), xft_dtypes.dtypes.int32),
        ("array_float64", np.array([1., 2.], dtype=np.float64), xft_dtypes.dtypes.float64),

        # --- xft DType Objects ---
        ("xft_dtype_obj", xft_dtypes.dtypes.uint16, xft_dtypes.dtypes.uint16),

        # --- String Aliases ---
        ("string_alias_float", "float32", xft_dtypes.dtypes.float32),
        ("string_alias_int", "int64", xft_dtypes.dtypes.int64),
    ]

    @parameterized.expand(DTYPE_INFERENCE_CASES)
    def test_dtype_inference(self, name, value_to_infer, expected_dtype):
        """
        Tests that the main `dtype()` function correctly infers the DType
        from a variety of input values and types.
        """
        # We call our main public function `xft_dtypes.dtype()` with the input value.
        # Note: We use `enable_x64=True` for this test to ensure that Python ints
        # and floats are promoted to their 64-bit versions, which is the default
        # behavior we've defined in our test cases.
        xft_dtypes.enable_x64 = True
        
        inferred_dtype = xft_dtypes.dtype(value_to_infer)

        # We assert that the returned DType object is the exact one we expect.
        # `assertIs` is stronger than `assertEqual`; it checks that they are the
        # same object in memory, which is what we want for our singleton DType instances.
        self.assertIs(inferred_dtype, expected_dtype)

    def test_canonicalize_dtype_with_x64_disabled(self):
        """
        Tests that canonicalize_dtype correctly downcasts 64-bit types when
        the x64 configuration flag is disabled.
        """
        # ... (the downcast_map dictionary is correct and does not need to change) ...
        downcast_map = {
            xft_dtypes.dtypes.int64: xft_dtypes.dtypes.int32,
            xft_dtypes.dtypes.uint64: xft_dtypes.dtypes.uint32,
            xft_dtypes.dtypes.float64: xft_dtypes.dtypes.float32,
            xft_dtypes.dtypes.complex128: xft_dtypes.dtypes.complex64,
        }

        # --- Setup: Disable x64 mode and save the original state ---
        original_x64_state = xft_dtypes.x64_enabled  # <-- FIX #1
        try:
            xft_dtypes.x64_enabled = False  # <-- FIX #2

            # --- Test Execution and Assertions ---
            # 1. Test that 64-bit types are correctly downcast.
            for high_prec_dtype, expected_low_prec_dtype in downcast_map.items():
                with self.subTest(f"downcasting_{high_prec_dtype.name}"):
                    result = xft_dtypes.canonicalize_dtype(high_prec_dtype)
                    self.assertIs(result, expected_low_prec_dtype)

            # 2. Test that 32-bit (and smaller) types are unaffected.
            unaffected_dtypes = [
                xft_dtypes.dtypes.int32,
                xft_dtypes.dtypes.float32,
                xft_dtypes.dtypes.bool,
                xft_dtypes.dtypes.int16,
            ]
            for dtype_obj in unaffected_dtypes:
                with self.subTest(f"unaffected_{dtype_obj.name}"):
                    result = xft_dtypes.canonicalize_dtype(dtype_obj)
                    self.assertIs(result, dtype_obj)

        finally:
            # --- Teardown: Restore the original configuration state ---
            xft_dtypes.x64_enabled = original_x64_state

    # We will create a list of test cases for various utility functions.
    # Format: (test_name, function_to_test, input_dtype, expected_output)
    UTILITY_FUNCTION_CASES = [
        # --- Type Category Checks ---
        ("is_int_for_int32", xft_dtypes.dtypes.is_int, xft_dtypes.dtypes.int32, True),
        ("is_int_for_float32", xft_dtypes.dtypes.is_int, xft_dtypes.dtypes.float32, False),
        ("is_float_for_float64", xft_dtypes.dtypes.is_float, xft_dtypes.dtypes.float64, True),
        ("is_float_for_int64", xft_dtypes.dtypes.is_float, xft_dtypes.dtypes.int64, False),
        ("is_unsigned_for_uint16", xft_dtypes.dtypes.is_unsigned, xft_dtypes.dtypes.uint16, True),
        ("is_unsigned_for_int16", xft_dtypes.dtypes.is_unsigned, xft_dtypes.dtypes.int16, False),

        # --- issubdtype Checks (This is a top-level function, so it's correct) ---
        ("issubdtype_int_int", xft_dtypes.issubdtype, (xft_dtypes.dtypes.int32, xft_dtypes.dtypes.int32), True),
        ("issubdtype_int_float", xft_dtypes.issubdtype, (xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32), False),
        ("issubdtype_int_np_integer", xft_dtypes.issubdtype, (xft_dtypes.dtypes.int32, np.integer), True),
        ("issubdtype_float_np_integer", xft_dtypes.issubdtype, (xft_dtypes.dtypes.float32, np.integer), False),

        # --- Conversion Function Checks (These are top-level, so they are correct) ---
        ("to_inexact_from_int32", xft_dtypes.to_inexact, xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32),
        ("to_inexact_from_float64", xft_dtypes.to_inexact, xft_dtypes.dtypes.float64, xft_dtypes.dtypes.float64),
        ("to_complex_from_int32", xft_dtypes.to_complex, xft_dtypes.dtypes.int32, xft_dtypes.dtypes.complex64),
        ("to_complex_from_float64", xft_dtypes.to_complex, xft_dtypes.dtypes.float64, xft_dtypes.dtypes.complex128),
    ]

    @parameterized.expand(UTILITY_FUNCTION_CASES)
    def test_dtype_utility_functions(self, name, func, inputs, expected_output):
        """
        Tests various helper functions from the dtype module.
        """
        # The `issubdtype` function takes two arguments, so we handle that case.
        if isinstance(inputs, tuple):
            result = func(*inputs)
        else:
            result = func(inputs)

        # For functions that return a boolean, we use assertEqual.
        if isinstance(expected_output, bool):
            self.assertEqual(result, expected_output)
        # For functions that return a DType, we use assertIs to check for the exact object.
        else:
            self.assertIs(result, expected_output)

    # Test cases for various DType properties.
    # Format: (dtype_obj, expected_bit_width, has_inf, expected_min, expected_max)
    DTYPE_PROPERTY_CASES = [
        (xft_dtypes.dtypes.bool, 8, False, False, True),
        (xft_dtypes.dtypes.int8, 8, False, -128, 127),
        (xft_dtypes.dtypes.uint16, 16, False, 0, 65535),
        (xft_dtypes.dtypes.int32, 32, False, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        (xft_dtypes.dtypes.float32, 32, True, -np.inf, np.inf),
        (xft_dtypes.dtypes.float64, 64, True, -np.inf, np.inf),
        (xft_dtypes.dtypes.complex64, 64, True, None, None), # min/max not defined for complex
    ]

    @parameterized.expand(DTYPE_PROPERTY_CASES)
    def test_dtype_extra_properties(self, dtype_obj, bit_width, has_inf, min_val, max_val):
        """
        Tests metadata properties on the DType class like .bit_width, .min, .max, etc.
        """
        self.assertEqual(dtype_obj.bit_width, bit_width)
        self.assertEqual(dtype_obj.supports_inf, has_inf)

        # min and max are not defined for complex numbers, so we only test others.
        if min_val is not None:
            self.assertEqual(dtype_obj.min, min_val)
        if max_val is not None:
            self.assertEqual(dtype_obj.max, max_val)

        # Also test the top-level utility functions that call these properties.
        self.assertEqual(xft_dtypes.bit_width(dtype_obj), bit_width)
        self.assertEqual(xft_dtypes.supports_inf(dtype_obj), has_inf)
    
    def test_check_and_canonicalize_user_dtype(self):
        """
        Tests the primary user-facing dtype validation and canonicalization function.
        """
        # --- Case 1: Test with a valid input ---
        # It should correctly identify and return the DType object.
        result = xft_dtypes.check_and_canonicalize_user_dtype("float32")
        self.assertIs(result, xft_dtypes.dtypes.float32)

        # --- Case 2: Test that it raises an error for None ---
        # The `with self.assertRaises(...)` block is a context manager that
        # asserts that the code inside it raises the specified exception.
        # The test will fail if no exception is raised.
        with self.assertRaisesRegex(ValueError, "dtype cannot be None"):
            xft_dtypes.check_and_canonicalize_user_dtype(None)

        # --- Case 3: Test that it raises an error for an invalid string ---
        with self.assertRaisesRegex(TypeError, "'invalid_string' is not a valid xft dtype"):
            xft_dtypes.check_and_canonicalize_user_dtype("invalid_string")

        # --- Case 4: Test that it issues a warning when downcasting ---
        original_x64_state = xft_dtypes.get_x64_enabled()
        try:
            xft_dtypes.set_x64_enabled(False)

            # The `assertWarns` context manager checks that the code inside it
            # triggers a warning of the specified type.
            with self.assertWarns(UserWarning):
                # We expect this to issue a warning because float64 will be truncated.
                result = xft_dtypes.check_and_canonicalize_user_dtype("float64")

            # We also check that the result is the correctly downcasted dtype.
            self.assertIs(result, xft_dtypes.dtypes.float32)
        finally:
            xft_dtypes.set_x64_enabled(original_x64_state)

    # We select a representative set of dtypes for this test.
    NUMPY_INTEROP_CASES = [
        ("float32", xft_dtypes.dtypes.float32, np.dtype('float32')),
        ("int16", xft_dtypes.dtypes.int16, np.dtype('int16')),
        ("uint64", xft_dtypes.dtypes.uint64, np.dtype('uint64')),
        ("bfloat16", xft_dtypes.dtypes.bfloat16, np.dtype(ml_dtypes.bfloat16)),
    ]

    @parameterized.expand(NUMPY_INTEROP_CASES)
    def test_numpy_interop_roundtrip(self, name, xft_dtype, np_dtype):
        """
        Tests the conversion functions between xft.DType and np.dtype.
        """
        # 1. Test conversion from xft.DType to NumPy dtype.
        converted_np_dtype = xft_dtypes.dtype_to_numpy(xft_dtype)
        self.assertEqual(converted_np_dtype, np_dtype)

        # 2. Test conversion from NumPy dtype back to xft.DType.
        converted_xft_dtype = xft_dtypes.numpy_to_dtype(np_dtype)
        self.assertIs(converted_xft_dtype, xft_dtype)

    def test_as_numpy_dtype_function(self):
        """
        Tests the versatile `as_numpy_dtype` helper with various input types.
        """
        # Test with an xft.DType object
        self.assertEqual(xft_dtypes.as_numpy_dtype(xft_dtypes.dtypes.float16), np.dtype('float16'))

        # Test with a standard np.dtype object
        self.assertEqual(xft_dtypes.as_numpy_dtype(np.dtype('int32')), np.dtype('int32'))

        # Test with a string alias
        self.assertEqual(xft_dtypes.as_numpy_dtype('uint8'), np.dtype('uint8'))

        # Test with default Python types (sensitive to x64 mode)
        original_x64_state = xft_dtypes.get_x64_enabled()
        try:
            # Check behavior with x64 enabled
            xft_dtypes.set_x64_enabled(True)
            self.assertEqual(xft_dtypes.as_numpy_dtype(int), np.dtype('int64'))
            self.assertEqual(xft_dtypes.as_numpy_dtype(float), np.dtype('float64'))
            # Also test the default dtype functions
            self.assertIs(xft_dtypes.default_int_dtype(), xft_dtypes.dtypes.int64)

            # Check behavior with x64 disabled
            xft_dtypes.set_x64_enabled(False)
            self.assertEqual(xft_dtypes.as_numpy_dtype(int), np.dtype('int32'))
            self.assertEqual(xft_dtypes.as_numpy_dtype(float), np.dtype('float32'))
            # Also test the default dtype functions
            self.assertIs(xft_dtypes.default_int_dtype(), xft_dtypes.dtypes.int32)

        finally:
            xft_dtypes.set_x64_enabled(original_x64_state)

    # Final suite of tests for the remaining utility functions.
    # Format: (test_name, function_to_test, args_tuple, expected_output)
    FINAL_UTILITY_CASES = [
        # --- isdtype ---
        ("isdtype_int32_integral", xft_dtypes.isdtype, (xft_dtypes.dtypes.int32, 'integral'), True),
        ("isdtype_float32_integral", xft_dtypes.isdtype, (xft_dtypes.dtypes.float32, 'integral'), False),
        ("isdtype_complex64_numeric", xft_dtypes.isdtype, (xft_dtypes.dtypes.complex64, 'numeric'), True),
        ("isdtype_int32_specific", xft_dtypes.isdtype, (xft_dtypes.dtypes.int32, xft_dtypes.dtypes.int32), True),
        ("isdtype_int32_different", xft_dtypes.isdtype, (xft_dtypes.dtypes.int32, xft_dtypes.dtypes.int64), False),

        # --- is_string_dtype, is_scalar_type, is_weakly_typed ---
        ("is_scalar_type_for_int", xft_dtypes.is_scalar_type, (5,), True),
        ("is_scalar_type_for_np_array", xft_dtypes.is_scalar_type, (np.array(5),), False),
        ("is_weakly_typed_for_float", xft_dtypes.is_weakly_typed, (3.14,), True),
        ("is_weakly_typed_for_np_float", xft_dtypes.is_weakly_typed, (np.float32(3.14),), False),
        ("is_string_dtype_false", xft_dtypes.is_string_dtype, (xft_dtypes.dtypes.int32,), False),

        # --- to_numeric ---
        ("to_numeric_from_int", xft_dtypes.to_numeric, (xft_dtypes.dtypes.int32,), xft_dtypes.dtypes.int32),
        ("to_numeric_from_bool", xft_dtypes.to_numeric, (xft_dtypes.dtypes.bool,), xft_dtypes.default_int_dtype()),
        ("to_numeric_from_float", xft_dtypes.to_numeric, (xft_dtypes.dtypes.float32,), xft_dtypes.dtypes.float32),

        # --- scalar_type ---
        ("scalar_type_from_int32", xft_dtypes.scalar_type, (xft_dtypes.dtypes.int32,), int),
        ("scalar_type_from_float64", xft_dtypes.scalar_type, (xft_dtypes.dtypes.float64,), float),
        ("scalar_type_from_complex128", xft_dtypes.scalar_type, (xft_dtypes.dtypes.complex128,), complex),
        ("scalar_type_from_bool", xft_dtypes.scalar_type, (xft_dtypes.dtypes.bool,), bool),

        # --- ensure_array (CORRECTED TUPLES) ---
        ("ensure_array_from_list", xft_dtypes.ensure_array, ([1, 2],), np.array([1, 2])),
        ("ensure_array_with_dtype", xft_dtypes.ensure_array, ([1., 2.], 'int32'), np.array([1, 2], dtype=np.int32)),
        ("ensure_array_from_scalar", xft_dtypes.ensure_array, (5,), np.array(5)),

        # --- compatible_dtypes ---
        ("compatible_dtypes_true", xft_dtypes.compatible_dtypes, (xft_dtypes.dtypes.int8, xft_dtypes.dtypes.float32), True),

        # --- register_weak_type ---
        ("register_weak_type_and_check", "test_register_weak_type", (None,), True),
    ]

    @parameterized.expand(FINAL_UTILITY_CASES)
    def test_final_utility_suite(self, name, func_or_name, args_tuple, expected_output):
        """A comprehensive test for the remaining dtype utility functions."""
        # Special case for testing register_weak_type, which modifies global state.
        if func_or_name == "test_register_weak_type":
            class MyWeakScalar: pass
            self.assertFalse(xft_dtypes.is_weakly_typed(MyWeakScalar()))
            xft_dtypes.register_weak_type(MyWeakScalar)
            self.assertTrue(xft_dtypes.is_weakly_typed(MyWeakScalar()))
            return

        # SIMPLIFIED and corrected runner logic
        result = func_or_name(*args_tuple)

        # Special case for testing ensure_array, which returns a numpy array.
        if func_or_name == xft_dtypes.ensure_array:
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_output)
            return

        # General case for most other functions.
        self.assertEqual(result, expected_output)
    
    def test_compatible_dtypes_failure(self):
        """Tests that compatible_dtypes returns False for incompatible types."""
        # Custom floats don't promote with each other in the current lattice.
        self.assertFalse(xft_dtypes.compatible_dtypes(
            xft_dtypes.dtypes.float8_e4m3fn, xft_dtypes.dtypes.bfloat16
        ))

    # testing type promotion
    @parameterized.expand([
        # (test_name_suffix, dtype1, dtype2, expected)
        ("float32_float64", xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float64, xft_dtypes.dtypes.float64),
        ("int32_float32", xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float32),
        ("bool_int8", xft_dtypes.dtypes.bool, xft_dtypes.dtypes.int8, xft_dtypes.dtypes.int8),
        ("complex64_float32", xft_dtypes.dtypes.complex64, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.complex64),
        ("int16_int64", xft_dtypes.dtypes.int16, xft_dtypes.dtypes.int64, xft_dtypes.dtypes.int64),
    ])
    def test_promote_types(self, name, dtype1, dtype2, expected):
        """
        Test dtype promotion logic.
        Ensures promote_types() returns the correct resulting dtype.
        """
        result = xft_dtypes.promote_types(dtype1, dtype2)
        self.assertEqual(result, expected, f"Promotion failed for {dtype1} and {dtype2}")

    @parameterized.expand([
        # (test_name_suffix, inputs, expected_result_type)
        ("int32_float32", [xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32], xft_dtypes.dtypes.float32),
        ("float32_float64", [xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float64], xft_dtypes.dtypes.float64),
        ("complex64_float64", [xft_dtypes.dtypes.complex64, xft_dtypes.dtypes.float64], xft_dtypes.dtypes.complex128),
        ("bool_int8", [xft_dtypes.dtypes.bool, xft_dtypes.dtypes.int8], xft_dtypes.dtypes.int8),
        ("int16_int64_bool", [xft_dtypes.dtypes.int16, xft_dtypes.dtypes.int64, xft_dtypes.dtypes.bool], xft_dtypes.dtypes.int64),
    ])
    def test_result_type(self, name, dtypes, expected):
        """
        Test result_type() logic.
        Ensures the resulting dtype matches the highest precision among inputs.
        """
        result = xft_dtypes.result_type(*dtypes)
        self.assertEqual(result, expected, f"Result type mismatch for {dtypes}")

    # Promotion Mode Behaviour
    @parameterized.expand([
        ("strict_blocks_unsafe_promotion", "strict", xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32, xft_dtypes.TypePromotionError),
        ("standard_allows_promotion", "standard", xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float32),
    ])
    def test_promotion_mode_behavior(self, name, mode, d1, d2, expected):
        """
        Checks how strict vs standard promotion modes behave.
        """
        xft_dtypes.set_promotion_mode(mode)
        if isinstance(expected, type) and issubclass(expected, Exception):
            with self.assertRaises(expected):
                _ = xft_dtypes.promote_types(d1, d2)
        else:
            result = xft_dtypes.promote_types(d1, d2)
            self.assertEqual(result, expected)

    # Edge and Error Cases
    @parameterized.expand([
        ("string_and_float", xft_dtypes.dtypes.string, xft_dtypes.dtypes.float32, ValueError),
        ("unsupported_object", xft_dtypes.dtypes.int32, "not_a_dtype", TypeError),
        ("same_type_float32", xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float32),
        ("bool_and_bool", xft_dtypes.dtypes.bool, xft_dtypes.dtypes.bool, xft_dtypes.dtypes.bool),
        ("bfloat16_and_float32", xft_dtypes.dtypes.bfloat16, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float32),
        ("complex128_and_bool", xft_dtypes.dtypes.complex128, xft_dtypes.dtypes.bool, xft_dtypes.dtypes.complex128),
    ])
    def test_promote_types_edge_and_error(self, name, d1, d2, expected):
        """
        Covers unsupported types, same-type idempotency, and mixed categories.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with self.assertRaises(expected):
                _ = xft_dtypes.promote_types(d1, d2)
        else:
            result = xft_dtypes.promote_types(d1, d2)
            self.assertEqual(result, expected)

    # 4. Multi-input Result Type Chains
    @parameterized.expand([
        ("int_chain", [xft_dtypes.dtypes.int8, xft_dtypes.dtypes.int16, xft_dtypes.dtypes.int32], xft_dtypes.dtypes.int32),
        ("int_to_float_chain", [xft_dtypes.dtypes.int16, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float64], xft_dtypes.dtypes.float64),
        ("float_to_complex_chain", [xft_dtypes.dtypes.float16, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.complex64], xft_dtypes.dtypes.complex64),
        ("mixed_chain_upgrades_to_complex128", [xft_dtypes.dtypes.int32, xft_dtypes.dtypes.float64, xft_dtypes.dtypes.complex64], xft_dtypes.dtypes.complex128),
        ("bool_mixed_chain", [xft_dtypes.dtypes.bool, xft_dtypes.dtypes.int8, xft_dtypes.dtypes.float32], xft_dtypes.dtypes.float32),
    ])
    def test_result_type_multi_chain(self, name, dtypes, expected):
        """
        Ensures chained promotions yield consistent results.
        """
        result = xft_dtypes.result_type(*dtypes)
        self.assertEqual(result, expected)

    # Consistency with NumPy (result_type)
    @parameterized.expand([
        ("float32_float64", np.float32, np.float64),
        ("int32_float64", np.int32, np.float64),
        ("complex64_float32", np.complex64, np.float32),
        ("complex64_float64", np.complex64, np.float64),
        ("bool_int16", np.bool_, np.int16),
        ("bfloat16_float32", ml_dtypes.bfloat16, np.float32),
    ])
    def test_result_type_consistency_with_numpy(self, name, np_dt1, np_dt2):
        """
        Validates that our result_type aligns with NumPy's for numeric cases.
        """
        ours = xft_dtypes.result_type(
            xft_dtypes.numpy_to_dtype(np_dt1),
            xft_dtypes.numpy_to_dtype(np_dt2),
        )
        expected = xft_dtypes.numpy_to_dtype(np.result_type(np_dt1, np_dt2))
        self.assertEqual(ours, expected)

    # Consistency with NumPy (promote_types)
    @parameterized.expand([
        ("float32_float64", np.float32, np.float64),
        ("int32_float64", np.int32, np.float64),
        ("complex64_float64", np.complex64, np.float64),
        ("bool_float32", np.bool_, np.float32),
        ("complex128_int32", np.complex128, np.int32),
    ])
    def test_promote_types_consistency_with_numpy(self, name, np_dt1, np_dt2):
        """
        Validates promote_types() agrees with NumPy’s promotion behavior.
        """
        ours = xft_dtypes.promote_types(
            xft_dtypes.numpy_to_dtype(np_dt1),
            xft_dtypes.numpy_to_dtype(np_dt2),
        )
        expected = xft_dtypes.numpy_to_dtype(np.promote_types(np_dt1, np_dt2))
        self.assertEqual(ours, expected)

    # Algebraic Properties (Regression Tests)
    @parameterized.expand([
        ("commutativity_float32_float64", xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float64),
        ("commutativity_int16_float32", xft_dtypes.dtypes.int16, xft_dtypes.dtypes.float32),
        ("associativity_chain", [xft_dtypes.dtypes.int8, xft_dtypes.dtypes.float32, xft_dtypes.dtypes.float64]),
    ])
    def test_promotion_algebraic_properties(self, name, *args):
        """
        Checks that type promotion is commutative and associative.
        """
        if len(args) == 2:
            d1, d2 = args
            a = xft_dtypes.promote_types(d1, d2)
            b = xft_dtypes.promote_types(d2, d1)
            self.assertEqual(a, b, f"Promotion not commutative for {d1}, {d2}")
        else:
            d1, d2, d3 = args[0]
            left = xft_dtypes.promote_types(xft_dtypes.promote_types(d1, d2), d3)
            right = xft_dtypes.promote_types(d1, xft_dtypes.promote_types(d2, d3))
            self.assertEqual(left, right, "Promotion not associative")

    # =========================================================================
    # FINAL EXHAUSTIVE PROMOTION TEST
    # =========================================================================

    @parameterized.expand([
        (f"{get_name(t1)}_{get_name(t2)}", t1, t2)
        for t1, t2 in itertools.product(_JAX_COMPAT_TYPES, repeat=2)
    ])
    def test_exhaustive_promotion_matches_jax(self, name, type1, type2):
        """
        Exhaustively verifies that xft.promote_types() matches JAX's behavior
        for every possible pair of numeric and weak types.
        """
        # --- Setup: Configure both JAX and xft to a known state (x64 enabled) ---
        original_xft_x64 = xft_dtypes.get_x64_enabled()
        original_jax_x64 = jax.config.jax_enable_x64
        try:
            xft_dtypes.set_x64_enabled(True)
            jax.config.update("jax_enable_x64", True)
            
            # --- Get the "Ground Truth" result from JAX ---
            jax_result_name = "ERROR"
            try:
                jax_result = jnp.promote_types(type1, type2)
                jax_result_name = jax_result.name
            except (TypeError, ValueError):
                # If JAX raises an error, we expect our library to also raise an error.
                with self.assertRaises(xft_dtypes.TypePromotionError):
                    xft_dtypes.promote_types(type1, type2)
                # If both errored, the test for this pair is successful.
                return

            # --- Get the result from our xft implementation ---
            xft_result = xft_dtypes.promote_types(type1, type2)

            # --- Assert that our result's name matches JAX's result's name ---
            self.assertEqual(
                xft_result.name,
                jax_result_name,
                f"Promotion mismatch for {name}: expected {jax_result_name} (from JAX), got {xft_result.name}",
            )

        finally:
            # --- Teardown: Restore original configuration for other tests ---
            xft_dtypes.set_x64_enabled(original_xft_x64)
            jax.config.update("jax_enable_x64", original_jax_x64)