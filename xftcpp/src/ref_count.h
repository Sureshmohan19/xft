// Reference Counting Utilities
// 
// Provides lightweight reference counting for manual memory management.
// Core components:
//   - ReferenceCounted<T>: CRTP base class for ref-counted objects
//   - RCReference<T>: Smart pointer for automatic ref count management
//   - Helper functions: MakeRef, FormRef, TakeRef
//
// This is a simplified implementation derived from XLA's reference counting
// system, keeping only essential features. Thread-safe with atomic operations.
//
// Usage:
//   class MyClass : public ReferenceCounted<MyClass> { ... };
//   auto ptr = MakeRef<MyClass>(args...);
//   // ptr automatically manages reference count

#ifndef XFT_REF_COUNT_H_
#define XFT_REF_COUNT_H_

#include <atomic>      
#include <cassert>     
#include <cstdint>     
#include <utility>     
#include <chrono>

namespace xft {

#ifndef NDEBUG

// Global atomic counter tracking the total number of reference-counted objects
// currently alive in the process. This is incremented when objects are created
// and decremented when they are destroyed. Used for leak detection and debugging.
inline std::atomic<size_t> total_reference_counted_objects{0};

// Each new object gets a unique, monotonically increasing ID.
// Starts at 1 (0 can be reserved for "invalid" or null objects if needed).
inline std::atomic<uint64_t> next_object_id{1};

// Debug metadata stored within each reference-counted object.
struct RefCountDebugInfo {
    // Unique identifier for this object, assigned at construction.
    uint64_t object_id;

    // Creation time of this object in nanoseconds since an arbitrary epoch.
    int64_t creation_timestamp_ns;

    // Constructor: automatically assigns a unique ID and captures creation time
    RefCountDebugInfo() 
    : object_id(next_object_id.fetch_add(1, std::memory_order_relaxed)),
      creation_timestamp_ns(std::chrono::steady_clock::now().time_since_epoch().count()) {
    }
};

// Returns the current number of live reference-counted objects in the process.
inline size_t GetNumReferenceCountedObjects() {
  return total_reference_counted_objects.load(std::memory_order_relaxed);
}

// Increments the global count of live reference-counted objects.
// Called automatically when a ReferenceCounted object is constructed.
inline void AddNumReferenceCountedObjects() {
  total_reference_counted_objects.fetch_add(1, std::memory_order_relaxed);
}

// Decrements the global count of live reference-counted objects.
// Called automatically when a ReferenceCounted object is destroyed.
inline void DropNumReferenceCountedObjects() {
  total_reference_counted_objects.fetch_sub(1, std::memory_order_relaxed);
}

#else  // NDEBUG defined (release mode)

// Release mode: Debug info becomes an empty struct with zero overhead.
struct RefCountDebugInfo {
  RefCountDebugInfo() = default;
};

// Release mode: No-op implementations of debug tracking functions.
// These exist so code can call them unconditionally without #ifdef checks.
inline void AddNumReferenceCountedObjects() {}
inline void DropNumReferenceCountedObjects() {}

#endif  // NDEBUG

// Base class for objects that use atomic reference counting for ownership management.
//
// This uses the CRTP (Curiously Recurring Template Pattern) where SubClass inherits
// from ReferenceCounted<SubClass>. This allows calling the subclass's Destroy() method
// without needing virtual functions or a vtable.
//
// Thread-safe: All reference counting operations use atomic operations.
// The object is destroyed when the reference count reaches zero.
template <typename SubClass>
class ReferenceCounted {
 public:
  // Default constructor: initializes with reference count of 1.
  ReferenceCounted() : ReferenceCounted(1) {}
  
  // Explicit constructor: allows starting with a custom reference count.
  // This is useful in rare cases where you need non-standard initialization.
  // In most cases, use the default constructor which starts at 1.
  explicit ReferenceCounted(unsigned ref_count) 
    : ref_count_(ref_count), 
      debug_info_() {  // Initialize debug info (captures ID and timestamp)
    AddNumReferenceCountedObjects();
  }
  
  // Destructor: verifies that reference count is zero before destruction.
  ~ReferenceCounted() {
    assert(ref_count_.load() == 0 &&
           "Shouldn't destroy a reference counted object with references!");
    DropNumReferenceCountedObjects();
  }
  
  // Reference-counted objects should never be copied or moved.
  // Use RCReference<T> smart pointers to share ownership instead.
  ReferenceCounted(const ReferenceCounted&) = delete;
  ReferenceCounted& operator=(const ReferenceCounted&) = delete;
  
  // Increment the reference count by one.
  void AddRef() {
    assert(ref_count_.load(std::memory_order_relaxed) >= 1);
    // Relaxed ordering is safe here: we're just incrementing a counter.
    // No other memory operations depend on this increment.
    ref_count_.fetch_add(1, std::memory_order_relaxed);
  }
  
  // Decrement the reference count by one, potentially destroying the object.
  void DropRef() {
    assert(ref_count_.load(std::memory_order_relaxed) > 0);
    
    // Optimization: if we're the sole owner (ref_count==1), we can skip
    // the atomic decrement and go straight to destruction.
    if (ref_count_.load(std::memory_order_acquire) == 1 ||
        ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // Last reference dropped - time to destroy the object.
      // Set count to 0 to make the destructor assertion happy.
      assert((ref_count_.store(0, std::memory_order_relaxed), true));
      
      // Call the subclass's Destroy() method via CRTP.
      // If SubClass doesn't define Destroy(), the base class version is used.
      static_cast<SubClass*>(this)->Destroy();
    }
  }
  
  // Return the current reference count.
  uint32_t NumRef() const { 
    return ref_count_.load(std::memory_order_relaxed); 
  }
  
  // Check if this is the only reference to the object.
  bool IsUnique() const {
    return ref_count_.load(std::memory_order_acquire) == 1;
  }
  
  // Access debug information for this object (ID, timestamp, etc.).
  const RefCountDebugInfo& GetDebugInfo() const {
    return debug_info_;
  }
  
 protected:
  // Default implementation of Destroy() - just deletes the object.
  // Subclasses can override this to provide custom destruction logic.
  void Destroy() { 
    delete static_cast<SubClass*>(this); 
  }
  
 private:
  // Atomic reference counter. Tracks how many references exist to this object.
  // When this reaches zero, the object is destroyed.
  std::atomic<unsigned> ref_count_;
  
  // Debug metadata: unique ID, creation timestamp, etc.
  // In release builds (NDEBUG), this is an empty struct with zero overhead.
  RefCountDebugInfo debug_info_;
};

// Smart pointer for automatic reference count management.
//
// RCReference<T> is similar to std::shared_ptr but specifically designed for
// objects that inherit from ReferenceCounted<T>. It automatically calls AddRef()
// when creating new references and DropRef() when references are destroyed.
template <typename T>
class RCReference {
 public:
  // Default constructor: creates an empty (null) reference.
  RCReference() : pointer_(nullptr) {}
  
  // Move constructor: transfers ownership without changing reference count.
  // The source reference becomes null after the move.
  RCReference(RCReference&& other) noexcept : pointer_(other.pointer_) {
    other.pointer_ = nullptr;
  }
  
  // Copy constructor: creates a new reference to the same object.
  // Both the source and destination will point to the same object.
  RCReference(const RCReference& other) : pointer_(other.pointer_) {
    if (pointer_) {
      pointer_->AddRef();
    }
  }
  
  // Move assignment: replaces current reference with another, transferring ownership.
  // Drops the old reference (if any) and takes ownership of the new one.
  // The source reference becomes null after the move.
  RCReference& operator=(RCReference&& other) noexcept {
    reset(other.pointer_);  // Drop current reference
    other.pointer_ = nullptr;
    return *this;
  }
  
  // Copy assignment: replaces current reference with a copy of another.
  // Drops the old reference (if any) and increments the new reference count.
  // Both the source and destination will point to the same object after this.
  RCReference& operator=(const RCReference& other) {
    reset(other.pointer_);  // Drop current reference
    if (pointer_) {
      pointer_->AddRef();   // Add reference to new object
    }
    return *this;
  }
  
  // Destructor: automatically drops the reference when going out of scope.
  ~RCReference() {
    if (pointer_ != nullptr) {
      pointer_->DropRef();
    }
  }
  
  // Replace the current pointer with a new one.
  // Drops the reference to the old object (if any) and stores the new pointer.
  // Does NOT increment the reference count of the new pointer - caller must
  // ensure the pointer has proper ownership (use FormRef or TakeRef).
  void reset(T* pointer = nullptr) {
    if (pointer_ != nullptr) {
      pointer_->DropRef();  // Drop old reference
    }
    pointer_ = pointer;     // Store new pointer
  }
  
  // Release ownership of the pointer without dropping the reference.
  // Returns the raw pointer and sets this RCReference to null.
  // The caller is now responsible for eventually calling DropRef().
  T* release() {
    T* tmp = pointer_;
    pointer_ = nullptr;
    return tmp;
  }
  
  // Dereference operator: access the pointed-to object.
  T& operator*() const {
    assert(pointer_ && "Dereferencing null RCReference");
    return *pointer_;
  }
  
  // Arrow operator: access members of the pointed-to object.
  T* operator->() const {
    assert(pointer_ && "Dereferencing null RCReference");
    return pointer_;
  }
  
  // Get the raw pointer without affecting reference count.
  // Returns nullptr if this RCReference is empty.
  T* get() const { 
    return pointer_; 
  }
  
  // Check if this RCReference points to an object (is non-null).
  // Allows usage in boolean contexts:
  explicit operator bool() const { 
    return pointer_ != nullptr; 
  }
  
  // Efficiently exchanges the pointers without any reference count changes.
  // Both references remain valid and point to each other's original objects.
  void swap(RCReference& other) noexcept {
    using std::swap;
    swap(pointer_, other.pointer_);
  }
  
  // Equality comparison: check if two RCReferences point to the same object.
  // Note: Compares pointer identity, not object contents.
  bool operator==(const RCReference& ref) const {
    return pointer_ == ref.pointer_;
  }
  
  // Inequality comparison: check if two RCReferences point to different objects.
  // Note: Compares pointer identity, not object contents.
  bool operator!=(const RCReference& ref) const {
    return pointer_ != ref.pointer_;
  }

  // Comparison with nullptr: allows explicit null checks.
  // Enables natural syntax like: if (ptr == nullptr) or if (ptr != nullptr)
  friend bool operator==(const RCReference& ref, std::nullptr_t) {
    return ref.pointer_ == nullptr;
  }
  friend bool operator==(std::nullptr_t, const RCReference& ref) {
    return ref.pointer_ == nullptr;
  }
  friend bool operator!=(const RCReference& ref, std::nullptr_t) {
    return ref.pointer_ != nullptr;
  }
  friend bool operator!=(std::nullptr_t, const RCReference& ref) {
    return ref.pointer_ != nullptr;
  }
  
  // Friend declarations to allow helper functions to access private pointer_.
  template <typename R>
  friend RCReference<R> FormRef(R*);
  
  template <typename R>
  friend RCReference<R> TakeRef(R*);
  
 private:
  // The raw pointer to the reference-counted object.
  // Can be nullptr if this RCReference is empty.
  T* pointer_;
  
  // Allow other RCReference instantiations to access our privates.
  // This is needed for potential future derived-to-base conversions
  // or other template interactions.
  template <typename R>
  friend class RCReference;
};

// Create a new RCReference from an existing raw pointer and increment its reference count.
// The pointer must be non-null and already have at least one reference.
//
// Example:
//   MyClass* raw = ...;  // Existing object with ref count
//   auto ref1 = FormRef(raw);  // Creates new reference, increments count
//   auto ref2 = FormRef(raw);  // Another reference, increments again
template <typename T>
RCReference<T> FormRef(T* pointer) {
  RCReference<T> ref;
  ref.pointer_ = pointer;
  pointer->AddRef();  // Increment reference count
  return ref;
}

// Create an RCReference that takes ownership of an existing +1 reference.
// Use this when you already have ownership of a reference and want to wrap it
// in an RCReference without incrementing the count.
//
// This is the counterpart to release() - it takes ownership without AddRef().
//
// Example:
//   MyClass* raw = new MyClass();  // ref count = 1
//   auto ref = TakeRef(raw);       // Takes ownership, still ref count = 1
//   // When ref is destroyed, it will call DropRef()
template <typename T>
RCReference<T> TakeRef(T* pointer) {
  RCReference<T> ref;
  ref.pointer_ = pointer;
  // Note: Does NOT call AddRef() - assumes pointer already has +1 reference
  return ref;
}

// Create a new reference-counted object, similar to std::make_shared.
// This is the preferred way to create reference-counted objects.
// The object is created with ref count = 1 and wrapped in an RCReference.
//
// Example:
//   auto obj = MakeRef<MyClass>(arg1, arg2);  // Creates MyClass(arg1, arg2)
template <typename T, typename... Args>
RCReference<T> MakeRef(Args&&... args) {
  auto t = new T(std::forward<Args>(args)...);
  return TakeRef(t);  // Take ownership of the initial reference
}

// ADL-findable swap function for RCReference.
// Allows std::swap and generic code to work efficiently with RCReference.
// Simply delegates to the member swap() function.
template <typename T>
void swap(RCReference<T>& a, RCReference<T>& b) noexcept {
  a.swap(b);
}

}  // namespace xft

#endif  // XFT_REF_COUNT_H_