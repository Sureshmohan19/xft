#ifndef XFT_STORAGE_H
#define XFT_STORAGE_H

#include "memory.h"
#include "scalar_types.h"
#include <memory>
#include <cstddef>
#include <functional>

namespace xft {

// Manages lifetime of array data buffer with automatic reference counting.
// Multiple Arrays can share same Storage (enables views, slicing without copy).
// Thread-safe refcounting via std::shared_ptr.
class Storage {
public:
    // Creates new storage by allocating aligned memory.
    // size_bytes: total bytes to allocate (already aligned by caller)
    // dtype: element type (for debugging/validation)
    // alignment: memory boundary alignment (default 32 for AVX2)
    static std::shared_ptr<Storage> create(size_t size_bytes, ScalarType dtype, 
                                           size_t alignment = memory::DEFAULT_ALIGNMENT) {
        if (size_bytes == 0) {
            throw std::invalid_argument("Cannot allocate zero-sized storage");
        }

        void* data = memory::alloc_aligned(size_bytes, alignment);
        if (data == nullptr) {
            throw std::bad_alloc();
        }

        // Custom deleter ensures aligned memory freed correctly.
        // Captures size for potential debugging/tracking.
        auto deleter = [size_bytes](void* ptr) {
            memory::free_aligned(ptr);
        };

        return std::shared_ptr<Storage>(
            new Storage(data, size_bytes, dtype, true, deleter)
        );
    }

    // Wraps existing external memory without taking ownership.
    // Caller responsible for keeping memory alive during Storage lifetime.
    // Use case: wrapping NumPy arrays, mmap'd files, device memory.
    static std::shared_ptr<Storage> wrap(void* data, size_t size_bytes, ScalarType dtype) {
        if (data == nullptr) {
            throw std::invalid_argument("Cannot wrap null pointer");
        }

        // No-op deleter: we don't own this memory.
        auto deleter = [](void*) {};

        return std::shared_ptr<Storage>(
            new Storage(data, size_bytes, dtype, false, deleter)
        );
    }

    // Returns raw pointer to data buffer.
    // Use with caution: pointer validity tied to Storage lifetime.
    void* data() noexcept { return data_ptr_; }
    const void* data() const noexcept { return data_ptr_; }

    // Total size in bytes of allocated buffer.
    size_t size_bytes() const noexcept { return size_bytes_; }

    // Element data type stored in this buffer.
    ScalarType dtype() const noexcept { return dtype_; }

    // Whether this Storage allocated and owns the memory.
    // false = wrapping external memory, caller manages lifetime.
    bool owns_data() const noexcept { return owns_data_; }

    // Number of Storage instances sharing this data buffer.
    // Useful for debugging/optimization decisions (copy-on-write).
    long use_count() const noexcept { 
        return data_shared_.use_count(); 
    }

    // Prevent copying - use shared_ptr to share Storage.
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

private:
    // Private constructor - use create() or wrap() factory methods.
    // This enforces that Storage is always managed by shared_ptr.
    Storage(void* data, size_t size_bytes, ScalarType dtype, bool owns_data,
            std::function<void(void*)> deleter)
        : data_ptr_(data)
        , size_bytes_(size_bytes)
        , dtype_(dtype)
        , owns_data_(owns_data)
        , data_shared_(data, deleter) // shared_ptr handles refcounting + cleanup
    {}

    void* data_ptr_;                      // Raw pointer for fast access
    size_t size_bytes_;                   // Total allocated bytes
    ScalarType dtype_;                    // Element type
    bool owns_data_;                      // Ownership flag
    std::shared_ptr<void> data_shared_;   // Refcounted pointer with custom deleter
};

} // namespace xft

#endif // XFT_STORAGE_H

