#ifndef XFT_ARRAY_H
#define XFT_ARRAY_H

#include "memory.h"
#include "storage.h"
#include "scalar_types.h"

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace xft {

// Multi-dimensional array with NumPy-like semantics.
// Supports shared views via reference-counted Storage.
// Layout defined by shape (dimensions) and strides (memory jumps).
class Array {
public:
    // Memory layout order for stride calculation.
    enum class Order {
        C,  // Row-major (C-style): rightmost index varies fastest
        F   // Column-major (Fortran-style): leftmost index varies fastest
    };

    // Creates array by allocating new memory.
    // shape: dimensions [d0, d1, ..., dn]
    // dtype: element type
    // order: memory layout (C=row-major default, F=column-major)
    Array(const std::vector<size_t>& shape, ScalarType dtype, Order order = Order::C)
        : shape_(shape)
        , dtype_(dtype)
        , offset_(0)
    {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty");
        }
        
        // Check for overflow in size calculation.
        size_t total_elements = 1;
        for (size_t dim : shape) {
            if (dim == 0) {
                throw std::invalid_argument("Shape dimensions must be > 0");
            }
            // Overflow check: if total * dim > SIZE_MAX, we'd overflow.
            if (total_elements > SIZE_MAX / dim) {
                throw std::overflow_error("Array size exceeds addressable memory");
            }
            total_elements *= dim;
        }
        
        size_t element_size = scalarTypeSize(dtype);
        size_t size_bytes = memory::calculate_aligned_size(total_elements, element_size);
        
        // Allocate storage.
        storage_ = Storage::create(size_bytes, dtype);
        
        // Calculate strides based on order.
        strides_ = calculate_strides(shape, element_size, order);
    }
    
    // Wraps existing memory without taking ownership.
    // Caller must ensure memory outlives this Array.
    // strides: if empty, computed as C-contiguous.
    Array(void* data, const std::vector<size_t>& shape, ScalarType dtype,
          const std::vector<size_t>& strides = {})
        : shape_(shape)
        , dtype_(dtype)
        , offset_(0)
    {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty");
        }
        if (data == nullptr) {
            throw std::invalid_argument("Cannot wrap null data pointer");
        }
        
        size_t total_elements = std::accumulate(shape.begin(), shape.end(), 
                                                 size_t(1), std::multiplies<size_t>());
        size_t element_size = scalarTypeSize(dtype);
        size_t size_bytes = total_elements * element_size;
        
        storage_ = Storage::wrap(data, size_bytes, dtype);
        
        if (strides.empty()) {
            // Default to C-contiguous strides.
            strides_ = calculate_strides(shape, element_size, Order::C);
        } else {
            if (strides.size() != shape.size()) {
                throw std::invalid_argument("Strides length must match shape length");
            }
            strides_ = strides;
        }
    }
    
    // Creates view sharing storage with another Array but different shape/strides.
    // Used internally for slicing (future implementation).
    Array(std::shared_ptr<Storage> storage, const std::vector<size_t>& shape,
          const std::vector<size_t>& strides, ScalarType dtype, size_t offset = 0)
        : storage_(storage)
        , shape_(shape)
        , strides_(strides)
        , dtype_(dtype)
        , offset_(offset)
    {
        if (shape.size() != strides.size()) {
            throw std::invalid_argument("Shape and strides must have same length");
        }
    }

    // Number of dimensions (rank).
    size_t ndim() const noexcept { return shape_.size(); }
    
    // Shape: array of dimension sizes.
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    
    // Size of specific dimension.
    size_t shape(size_t axis) const {
        if (axis >= shape_.size()) {
            throw std::out_of_range("Axis out of range");
        }
        return shape_[axis];
    }
    
    // Strides: bytes to jump for each dimension.
    const std::vector<size_t>& strides() const noexcept { return strides_; }
    
    // Total number of elements.
    size_t size() const noexcept {
        return std::accumulate(shape_.begin(), shape_.end(), 
                               size_t(1), std::multiplies<size_t>());
    }
    
    // Total size in bytes.
    size_t nbytes() const noexcept {
        return size() * scalarTypeSize(dtype_);
    }
    
    // Element data type.
    ScalarType dtype() const noexcept { return dtype_; }
    
    // Size in bytes of each element.
    size_t itemsize() const noexcept { return scalarTypeSize(dtype_); }
    
    // Raw data pointer (base address + offset).
    void* data() noexcept {
        return static_cast<char*>(storage_->data()) + offset_;
    }
    
    const void* data() const noexcept {
        return static_cast<const char*>(storage_->data()) + offset_;
    }
    
    // Access underlying storage (for creating views).
    std::shared_ptr<Storage> storage() const noexcept { return storage_; }
    
    // Byte offset into storage buffer.
    size_t offset() const noexcept { return offset_; }
    
    // Check if array layout is C-contiguous (row-major).
    // Required for many optimized operations (BLAS, etc).
    bool is_contiguous() const noexcept {
        if (shape_.empty()) return true;
        
        size_t expected_stride = scalarTypeSize(dtype_);
        // Check from innermost (last) to outermost dimension.
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }
    
    // Check if array layout is Fortran-contiguous (column-major).
    bool is_f_contiguous() const noexcept {
        if (shape_.empty()) return true;
        
        size_t expected_stride = scalarTypeSize(dtype_);
        // Check from outermost (first) to innermost dimension.
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (strides_[i] != expected_stride) {
                return false;
            }
            expected_stride *= shape_[i];
        }
        return true;
    }
    
    // Calculate byte offset for multi-dimensional index.
    // indices: [i0, i1, ..., in] where n = ndim()
    // Returns: byte offset from base data pointer.
    size_t calculate_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Index dimensions don't match array dimensions");
        }
        
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            offset += indices[i] * strides_[i];
        }
        return offset;
    }
    
    // Element access for 1D array.
    template<typename T>
    T& at(size_t i0) {
        check_dtype<T>();
        if (ndim() != 1) {
            throw std::invalid_argument("1D indexing requires 1D array");
        }
        if (i0 >= shape_[0]) {
            throw std::out_of_range("Index out of bounds");
        }
        return *reinterpret_cast<T*>(static_cast<char*>(data()) + i0 * strides_[0]);
    }
    
    // Element access for 2D array.
    template<typename T>
    T& at(size_t i0, size_t i1) {
        check_dtype<T>();
        if (ndim() != 2) {
            throw std::invalid_argument("2D indexing requires 2D array");
        }
        if (i0 >= shape_[0] || i1 >= shape_[1]) {
            throw std::out_of_range("Index out of bounds");
        }
        size_t offset = i0 * strides_[0] + i1 * strides_[1];
        return *reinterpret_cast<T*>(static_cast<char*>(data()) + offset);
    }
    
    // Element access for 3D array.
    template<typename T>
    T& at(size_t i0, size_t i1, size_t i2) {
        check_dtype<T>();
        if (ndim() != 3) {
            throw std::invalid_argument("3D indexing requires 3D array");
        }
        if (i0 >= shape_[0] || i1 >= shape_[1] || i2 >= shape_[2]) {
            throw std::out_of_range("Index out of bounds");
        }
        size_t offset = i0 * strides_[0] + i1 * strides_[1] + i2 * strides_[2];
        return *reinterpret_cast<T*>(static_cast<char*>(data()) + offset);
    }
    
    // General N-D element access (slower due to vector allocation).
    template<typename T>
    T& at(const std::vector<size_t>& indices) {
        check_dtype<T>();
        size_t offset = calculate_offset(indices);
        return *reinterpret_cast<T*>(static_cast<char*>(data()) + offset);
    }
    
    // String representation for debugging.
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Array(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape_[i];
        }
        oss << "], dtype=" << scalarTypeName(dtype_);
        oss << ", strides=[";
        for (size_t i = 0; i < strides_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << strides_[i];
        }
        oss << "], contiguous=" << (is_contiguous() ? "true" : "false");
        oss << ")";
        return oss.str();
    }

private:
    std::shared_ptr<Storage> storage_;  // Reference-counted data buffer
    std::vector<size_t> shape_;         // Dimension sizes
    std::vector<size_t> strides_;       // Byte jumps per dimension
    ScalarType dtype_;                  // Element type
    size_t offset_;                     // Byte offset into storage
    
    // Calculate strides from shape and element size.
    // C-order: rightmost index varies fastest (default for DL).
    // F-order: leftmost index varies fastest (MATLAB/Fortran style).
    static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape,
                                                  size_t element_size, Order order) {
        std::vector<size_t> strides(shape.size());
        
        if (order == Order::C) {
            // C-contiguous: stride[i] = product of all dims to the right * element_size
            size_t stride = element_size;
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        } else { // Order::F
            // Fortran-contiguous: stride[i] = product of all dims to the left * element_size
            size_t stride = element_size;
            for (size_t i = 0; i < shape.size(); ++i) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
        
        return strides;
    }
    
    // Type-checking helper for element access.
    // Ensures T matches the array's dtype to prevent UB.
    template<typename T>
    void check_dtype() const {
        ScalarType expected = get_dtype<T>();
        if (dtype_ != expected) {
            throw std::invalid_argument(
                "Type mismatch: array is " + scalarTypeName(dtype_) +
                " but accessed as " + scalarTypeName(expected)
            );
        }
    }
    
    // Map C++ type to ScalarType enum.
    template<typename T> static ScalarType get_dtype();
};

// Template specializations for dtype mapping.
template<> inline ScalarType Array::get_dtype<float>() { return ScalarType::Float32; }
template<> inline ScalarType Array::get_dtype<double>() { return ScalarType::Float64; }

} // namespace xft

#endif // XFT_ARRAY_H