/**
 * @file column.hpp
 * @brief Columnar storage for GPU-accelerated SQL processing
 * 
 * This implements Structure-of-Arrays (SoA) storage which is essential
 * for efficient GPU processing:
 * 
 * Row-oriented (bad for GPU):        Columnar (good for GPU):
 * [id1, val1, cat1]                  [id1, id2, id3, ...]     <- contiguous
 * [id2, val2, cat2]                  [val1, val2, val3, ...]  <- contiguous
 * [id3, val3, cat3]                  [cat1, cat2, cat3, ...]  <- contiguous
 * 
 * Why columnar is better for GPUs:
 * 1. Memory Coalescing: Adjacent threads access adjacent memory locations
 * 2. Cache Efficiency: Only load columns needed for the query
 * 3. SIMD-friendly: Same operation on many values
 * 4. Compression: Same-type data compresses better
 * 
 * Memory Layout:
 * - Host memory: std::vector with typed storage
 * - Device memory: cudaMalloc'd buffer, managed separately
 */

#pragma once

#include "types.hpp"
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <cstring>

namespace gpusql {

/**
 * @class Column
 * @brief Type-erased columnar storage for a single column
 * 
 * Stores data in a contiguous array suitable for GPU transfer.
 * The Column class manages host-side data; GPU memory is managed
 * by the kernel execution layer.
 * 
 * Example usage:
 *   Column col("price", DataType::FLOAT32);
 *   col.reserve(1000);
 *   col.push_back(19.99f);
 *   col.push_back(29.99f);
 *   
 *   // Get typed access for kernels
 *   float* data = col.data<float>();
 */
class Column {
public:
    // ========================================================================
    // Construction
    // ========================================================================
    
    /**
     * Create an empty column with the given name and type.
     * 
     * @param name Column name (e.g., "price", "user_id")
     * @param type Data type for this column
     */
    Column(std::string name, DataType type);
    
    /**
     * Create a column with pre-allocated capacity.
     * 
     * @param name Column name
     * @param type Data type
     * @param capacity Initial capacity (number of rows)
     */
    Column(std::string name, DataType type, size_t capacity);
    
    // Move semantics (columns can be large)
    Column(Column&& other) noexcept;
    Column& operator=(Column&& other) noexcept;
    
    // No copying (use explicit clone if needed)
    Column(const Column&) = delete;
    Column& operator=(const Column&) = delete;
    
    ~Column() = default;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /** Get column name */
    const std::string& name() const { return name_; }
    
    /** Get data type */
    DataType type() const { return type_; }
    
    /** Get number of rows */
    size_t size() const { return size_; }
    
    /** Check if column is empty */
    bool empty() const { return size_ == 0; }
    
    /** Get size in bytes of stored data */
    size_t bytes() const { return size_ * type_size(type_); }
    
    // ========================================================================
    // Typed Data Access
    // ========================================================================
    
    /**
     * Get typed pointer to underlying data.
     * 
     * @tparam T Expected type (must match column's DataType)
     * @return Pointer to first element
     * 
     * Usage:
     *   if (col.type() == DataType::INT32) {
     *       int32_t* ptr = col.data<int32_t>();
     *       // Use ptr...
     *   }
     */
    template<typename T>
    T* data() {
        assert(validate_type<T>());
        return reinterpret_cast<T*>(data_.data());
    }
    
    template<typename T>
    const T* data() const {
        assert(validate_type<T>());
        return reinterpret_cast<const T*>(data_.data());
    }
    
    /**
     * Get raw pointer to data (type-erased).
     * Use when type is not known at compile time.
     */
    void* raw_data() { return data_.data(); }
    const void* raw_data() const { return data_.data(); }
    
    /**
     * Access element at index with type checking.
     * 
     * @tparam T Element type
     * @param idx Row index
     * @return Reference to element
     */
    template<typename T>
    T& at(size_t idx) {
        assert(idx < size_ && validate_type<T>());
        return data<T>()[idx];
    }
    
    template<typename T>
    const T& at(size_t idx) const {
        assert(idx < size_ && validate_type<T>());
        return data<T>()[idx];
    }
    
    // ========================================================================
    // Modification
    // ========================================================================
    
    /**
     * Reserve memory for at least n elements.
     * Does not change size, only capacity.
     */
    void reserve(size_t n);
    
    /**
     * Resize to exactly n elements.
     * New elements are zero-initialized.
     */
    void resize(size_t n);
    
    /**
     * Clear all data (size becomes 0, capacity unchanged).
     */
    void clear();
    
    /**
     * Add a value to the end of the column.
     * 
     * @tparam T Value type (must match column's DataType)
     * @param value Value to add
     */
    template<typename T>
    void push_back(const T& value) {
        assert(validate_type<T>());
        size_t elem_size = type_size(type_);
        size_t new_size_bytes = (size_ + 1) * elem_size;
        
        if (new_size_bytes > data_.size()) {
            // Grow by 1.5x (amortized O(1) insertion)
            data_.resize(std::max(new_size_bytes, data_.size() * 3 / 2 + elem_size));
        }
        
        data<T>()[size_] = value;
        ++size_;
    }
    
    /**
     * Set value at specific index.
     */
    template<typename T>
    void set(size_t idx, const T& value) {
        assert(idx < size_ && validate_type<T>());
        data<T>()[idx] = value;
    }
    
    // ========================================================================
    // Bulk Operations (for efficient data loading)
    // ========================================================================
    
    /**
     * Copy data from external array.
     * Replaces existing data.
     * 
     * @tparam T Element type
     * @param src Source array
     * @param count Number of elements
     */
    template<typename T>
    void copy_from(const T* src, size_t count) {
        assert(validate_type<T>());
        size_t bytes_needed = count * sizeof(T);
        data_.resize(bytes_needed);
        std::memcpy(data_.data(), src, bytes_needed);
        size_ = count;
    }
    
    /**
     * Copy data from std::vector.
     */
    template<typename T>
    void copy_from(const std::vector<T>& src) {
        copy_from(src.data(), src.size());
    }
    
    // ========================================================================
    // Utility
    // ========================================================================
    
    /**
     * Create a deep copy of this column.
     */
    Column clone() const;
    
    /**
     * Print column contents for debugging (first n rows).
     */
    void debug_print(size_t max_rows = 10) const;

private:
    std::string name_;          // Column identifier
    DataType type_;             // Data type
    size_t size_;               // Number of elements
    std::vector<std::byte> data_;  // Raw storage (type-erased)
    
    /**
     * Validate that T matches the column's DataType.
     */
    template<typename T>
    bool validate_type() const {
        if constexpr (std::is_same_v<T, int32_t>) {
            return type_ == DataType::INT32;
        } else if constexpr (std::is_same_v<T, float>) {
            return type_ == DataType::FLOAT32;
        } else if constexpr (std::is_same_v<T, FixedString>) {
            return type_ == DataType::STRING;
        }
        return false;
    }
};

} // namespace gpusql

