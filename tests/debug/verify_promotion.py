import numpy as np
import jax
import jax.numpy as jnp
import itertools

# Import your xft dtype system
from xft._internal import dtype as xft_dtypes

# This list contains every type we want to test.
ALL_TEST_TYPES = [
    # Weak Types
    bool, int, float, complex,
    # Strong Types
    xft_dtypes.dtypes.bool, xft_dtypes.dtypes.int8, xft_dtypes.dtypes.int16,
    xft_dtypes.dtypes.int32, xft_dtypes.dtypes.int64, xft_dtypes.dtypes.uint8,
    xft_dtypes.dtypes.uint16, xft_dtypes.dtypes.uint32, xft_dtypes.dtypes.uint64,
    xft_dtypes.dtypes.bfloat16, xft_dtypes.dtypes.float16, xft_dtypes.dtypes.float32,
    xft_dtypes.dtypes.float64, xft_dtypes.dtypes.complex64, xft_dtypes.dtypes.complex128,
    xft_dtypes.dtypes.int2, xft_dtypes.dtypes.int4, xft_dtypes.dtypes.uint2,
    xft_dtypes.dtypes.uint4,
]

def get_name(t):
    """Helper to get a clean name for any type."""
    if isinstance(t, xft_dtypes.DType): return t.name
    try: return np.dtype(t).name
    except TypeError: return t.__name__

def run_exhaustive_comparison():
    """
    Exhaustively compares and displays the type promotion rules of NumPy, JAX,
    and xft for all defined numeric and weak types in a formatted table.
    """
    # --- Configure Libraries ---
    try:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_numpy_dtype_promotion", "standard")
        xft_dtypes.set_x64_enabled(True)
        xft_dtypes.set_promotion_mode("standard")
    except Exception as e:
        print(f"[ERROR] Could not configure libraries: {e}")
        return

    # --- Table Formatting ---
    max_name_len = max(len(get_name(t)) for t in ALL_TEST_TYPES)
    col1_width = max_name_len
    col2_width = max_name_len
    col3_width = max(len("NumPy"), max_name_len)
    col4_width = max(len("JAX"), max_name_len)
    col5_width = max(len("xft"), max_name_len)
    col6_width = len("JAX == xft")

    header = (
        f"{'Type A':<{col1_width}} | {'Type B':<{col2_width}} | "
        f"{'NumPy':<{col3_width}} | {'JAX':<{col4_width}} | "
        f"{'xft':<{col5_width}} | {'JAX == xft':<{col6_width}}"
    )
    separator = "-" * len(header)

    print("=" * len(header))
    print("      Exhaustive Type Promotion Verification Report")
    print("=" * len(header))
    print(header)
    print(separator)

    mismatches = 0
    total_comparisons = 0
    
    # --- THIS IS THE NEW LOGIC ---
    last_type1 = None
    # --- END NEW LOGIC ---

    # --- Iterate and Build Table ---
    for type1, type2 in itertools.product(ALL_TEST_TYPES, repeat=2):
        
        # --- NEW LOGIC TO PRINT SEPARATOR ---
        if last_type1 is not None and type1 is not last_type1:
            print(separator)
        last_type1 = type1
        # --- END NEW LOGIC ---

        total_comparisons += 1
        name1, name2 = get_name(type1), get_name(type2)

        # --- NumPy Promotion ---
        np_result_name = "ERROR"
        try:
            np_t1 = type1.native_type if isinstance(type1, xft_dtypes.DType) else type1
            np_t2 = type2.native_type if isinstance(type2, xft_dtypes.DType) else type2
            np_result = np.promote_types(np_t1, np_t2)
            np_result_name = np_result.name
        except TypeError:
            np_result_name = "N/A"

        # --- JAX Promotion (Ground Truth) ---
        jax_result_name = "ERROR"
        try:
            jax_t1 = type1.native_type if isinstance(type1, xft_dtypes.DType) else type1
            jax_t2 = type2.native_type if isinstance(type2, xft_dtypes.DType) else type2
            jax_result = jnp.promote_types(jax_t1, jax_t2)
            jax_result_name = jax_result.name
        except (TypeError, ValueError):
            jax_result_name = "ERROR"

        # --- Your xft Promotion ---
        xft_result_name = "ERROR"
        try:
            xft_result = xft_dtypes.promote_types(type1, type2)
            xft_result_name = xft_result.name
        except Exception:
            xft_result_name = "ERROR"
            
        # --- Comparison and Row Formatting ---
        match = (xft_result_name == jax_result_name)
        status = "✅" if match else "❌"
        if not match:
            mismatches += 1

        print(
            f"{name1:<{col1_width}} | {name2:<{col2_width}} | "
            f"{np_result_name:<{col3_width}} | {jax_result_name:<{col4_width}} | "
            f"{xft_result_name:<{col5_width}} | {status:^{col6_width}}"
        )

    # --- Final Summary ---
    print(separator)
    print("\n" + "=" * len(header))
    print("Verification Summary")
    print("-" * len(header))
    print(f"Total pairs tested: {total_comparisons}")
    print(f"Mismatches with JAX: {mismatches}")
    if mismatches == 0:
        print("\n🎉 SUCCESS! xft promotion logic is fully JAX-compliant.")
    else:
        print(f"\n⚠️ Found {mismatches} mismatches that need to be fixed.")
    print("=" * len(header))

if __name__ == "__main__":
    run_exhaustive_comparison()