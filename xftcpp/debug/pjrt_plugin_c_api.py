import ctypes

#  Step 1: Define the path to your creation 
#  Step 2: Load the library
#  Step 3: Find the sacred function 
#  Step 4: Define the return type 
#  Step 5: Call the function 
#  Step 6: Inspect the artifact (Advanced) 

plugin_path = "/Users/aakritisuresh/Desktop/xla/bazel-out/darwin_arm64-opt/bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so"

try:
    pjrt_lib = ctypes.CDLL(plugin_path)
    print(f"Success: Successfully loaded the library from {plugin_path}")
except OSError as e: print(f"Failure: Could not load the library. Error: {e}")

try:
    get_pjrt_api_func = pjrt_lib.GetPjrtApi
    print("Success: Found the 'GetPjrtApi' function.")
except AttributeError: print("Failure: Could not find the 'GetPjrtApi' function in the library.")

# We must tell ctypes what kind of thing the function will return.
# It returns a pointer, so we use ctypes.c_void_p.
get_pjrt_api_func.restype = ctypes.c_void_p

# We execute the function and get back a raw memory address.
api_ptr = get_pjrt_api_func()
print(f"Success: 'GetPjrtApi' returned a pointer: {hex(api_ptr)}")


# This is the final proof. We define the C structure of the PJRT_Api_Version
# in Python so we can read the memory at the pointer's location.
class PjrtApiVersion(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]

# The PJRT_Api struct starts with the version struct.
class PjrtApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("pjrt_api_version", PjrtApiVersion),
        # ... we don't need to define the rest of the function pointers for this test
    ]

# Cast the raw pointer to our defined structure type
api_struct = ctypes.cast(api_ptr, ctypes.POINTER(PjrtApi)).contents

# Read the version numbers
major = api_struct.pjrt_api_version.major_version
minor = api_struct.pjrt_api_version.minor_version

print(f"SUCCESS: Read API version from the struct: {major}.{minor}")