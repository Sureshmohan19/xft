#ifndef XFT_MEMORY_H
#define XFT_MEMORY_H

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

namespace xft {
namespace memory {

// AVX2 requires 32-byte alignment for optimal SIMD performance.
// Larger than dtype natural alignment to prevent false sharing (cache line = 64 bytes).
constexpr size_t DEFAULT_ALIGNMENT = 32;

// Rounds size up to next multiple of alignment.
// Required because aligned_alloc on some platforms mandates size % alignment == 0.
// Example: align_size(50, 32) -> 64
inline size_t align_size(size_t size, size_t alignment) noexcept {
    // Equivalent to: ((size + alignment - 1) / alignment) * alignment
    // Using bitwise ops is faster when alignment is power of 2.
    return (size + alignment - 1) & ~(alignment - 1);
}

// Calculates total bytes needed for n elements of given size, aligned.
// Centralizes size calculation to ensure consistency across Array/Storage.
inline size_t calculate_aligned_size(size_t num_elements, size_t element_size, 
                                     size_t alignment = DEFAULT_ALIGNMENT) noexcept {
    size_t total = num_elements * element_size;
    return align_size(total, alignment);
}

// Allocates aligned memory block.
// alignment must be power of 2 and multiple of sizeof(void*).
// Returns nullptr on allocation failure (caller must check).
// Use free_aligned() to deallocate, not standard free().
inline void* alloc_aligned(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
    if (size == 0) {
        return nullptr;
    }
    
    // Enforce alignment constraints for posix_memalign.
    // Must be power of 2 and >= sizeof(void*).
    if (alignment < sizeof(void*) || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be power of 2 and >= sizeof(void*)");
    }
    
    // Round size up to alignment multiple for platform compatibility.
    size = align_size(size, alignment);
    
    void* ptr = nullptr;
    
    // posix_memalign preferred over aligned_alloc on Linux:
    // - More consistent behavior across glibc versions
    // - Returns error code instead of nullptr, clearer diagnostics
    int result = posix_memalign(&ptr, alignment, size);
    
    if (result != 0) {
        // EINVAL: alignment not power of 2 or not multiple of sizeof(void*)
        // ENOMEM: insufficient memory
        return nullptr;
    }
    
    return ptr;
}

// Frees memory allocated by alloc_aligned().
// Safe to call with nullptr (no-op).
inline void free_aligned(void* ptr) noexcept {
    // posix_memalign uses standard malloc heap, so standard free works.
    // Unlike _mm_malloc which requires _mm_free.
    std::free(ptr);
}

// Checks if pointer is aligned to specified boundary.
// Used in debug builds to validate alignment assumptions.
// Returns true if ptr % alignment == 0.
inline bool is_aligned(const void* ptr, size_t alignment = DEFAULT_ALIGNMENT) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

} // namespace memory
} // namespace xft

#endif // XFT_MEMORY_H

