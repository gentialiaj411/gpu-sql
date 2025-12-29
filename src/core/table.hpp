/**
 * @file table.hpp
 * @brief Columnar table representation for GPU SQL processing
 * 
 * A Table is a collection of Columns with the same number of rows.
 * This follows the columnar (SoA) layout for GPU efficiency.
 * 
 * Schema:
 *   Table represents a schema with named, typed columns.
 *   Example: CREATE TABLE users (id INT, score FLOAT, name STRING)
 * 
 * Data Generation:
 *   The Table class includes utilities for generating synthetic test data,
 *   which is essential for benchmarking.
 * 
 * Usage Example:
 *   Table users;
 *   users.add_column("id", DataType::INT32);
 *   users.add_column("score", DataType::FLOAT32);
 *   users.add_column("category", DataType::INT32);
 *   
 *   // Generate 1 million random rows
 *   users.generate_random_data(1'000'000);
 */

#pragma once

#include "column.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <random>
#include <optional>

namespace gpusql {

/**
 * @class Table
 * @brief A collection of columns forming a relational table
 * 
 * Invariants:
 * - All columns have the same number of rows
 * - Column names are unique within a table
 * - Columns cannot be removed (only added)
 */
class Table {
public:
    // ========================================================================
    // Construction
    // ========================================================================
    
    /**
     * Create an empty table with an optional name.
     */
    explicit Table(std::string name = "");
    
    // Tables can be moved
    Table(Table&& other) noexcept = default;
    Table& operator=(Table&& other) noexcept = default;
    
    // No copying (tables can be large)
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    
    ~Table() = default;
    
    // ========================================================================
    // Schema Definition
    // ========================================================================
    
    /**
     * Add a new column to the table.
     * If the table already has rows, the new column is zero-initialized.
     * 
     * @param name Column name (must be unique)
     * @param type Column data type
     * @return Reference to this table (for chaining)
     */
    Table& add_column(const std::string& name, DataType type);
    
    /**
     * Check if a column exists.
     */
    bool has_column(const std::string& name) const;
    
    /**
     * Get column index by name.
     * @return Column index, or nullopt if not found
     */
    std::optional<size_t> column_index(const std::string& name) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    /** Table name */
    const std::string& name() const { return name_; }
    
    /** Number of rows in the table */
    size_t num_rows() const { return num_rows_; }
    
    /** Number of columns */
    size_t num_columns() const { return columns_.size(); }
    
    /** Check if table is empty */
    bool empty() const { return num_rows_ == 0; }
    
    /**
     * Get column by index.
     */
    Column& column(size_t idx) { return columns_[idx]; }
    const Column& column(size_t idx) const { return columns_[idx]; }
    
    /**
     * Get column by name.
     * @throws std::runtime_error if column not found
     */
    Column& column(const std::string& name);
    const Column& column(const std::string& name) const;
    
    /**
     * Get all columns.
     */
    std::vector<Column>& columns() { return columns_; }
    const std::vector<Column>& columns() const { return columns_; }
    
    /**
     * Get column names.
     */
    std::vector<std::string> column_names() const;
    
    // ========================================================================
    // Data Manipulation
    // ========================================================================
    
    /**
     * Reserve memory for n rows in all columns.
     */
    void reserve(size_t n);
    
    /**
     * Resize all columns to n rows.
     * New rows are zero-initialized.
     */
    void resize(size_t n);
    
    /**
     * Clear all data (columns remain, rows are deleted).
     */
    void clear();
    
    // ========================================================================
    // Test Data Generation
    // ========================================================================
    
    /**
     * Generate random data for benchmarking.
     * 
     * This is essential for testing GPU kernels with realistic data volumes.
     * 
     * @param n Number of rows to generate
     * @param seed Random seed for reproducibility (default: random)
     * 
     * Generated values:
     * - INT32: uniform [0, 1000000)
     * - FLOAT32: uniform [0.0, 1000.0)
     * - STRING: "str_XXXXXX" where X is random digit
     */
    void generate_random_data(size_t n, uint32_t seed = 0);
    
    /**
     * Generate data with specific distribution for filter benchmarks.
     * 
     * @param n Number of rows
     * @param selectivity What fraction of rows should pass filter (0.0 to 1.0)
     * @param int_col_name Name of INT column to set up for filtering
     * @param threshold Threshold value (rows with col > threshold pass)
     * 
     * Example: selectivity=0.1 means 10% of rows have int_col > threshold
     */
    void generate_filter_benchmark_data(
        size_t n, 
        double selectivity,
        const std::string& int_col_name,
        int32_t threshold
    );
    
    // ========================================================================
    // Debug / Utility
    // ========================================================================
    
    /**
     * Print table schema and sample data.
     */
    void debug_print(size_t max_rows = 5) const;
    
    /**
     * Print schema only (no data).
     */
    void print_schema() const;
    
    /**
     * Get total memory usage in bytes.
     */
    size_t memory_usage() const;

private:
    std::string name_;
    size_t num_rows_ = 0;
    std::vector<Column> columns_;
    std::unordered_map<std::string, size_t> name_to_index_;
};

// ============================================================================
// Factory Functions for Common Test Tables
// ============================================================================

/**
 * Create a standard benchmark table with (id, value, category) schema.
 * This is the primary schema for filter and join benchmarks.
 * 
 * @param n Number of rows
 * @param seed Random seed
 * @return Populated table
 */
Table create_benchmark_table(size_t n, uint32_t seed = 42);

/**
 * Create a smaller "dimension" table for join benchmarks.
 * Schema: (id, label)
 * 
 * @param n Number of rows
 * @param seed Random seed
 */
Table create_dimension_table(size_t n, uint32_t seed = 123);

} // namespace gpusql

