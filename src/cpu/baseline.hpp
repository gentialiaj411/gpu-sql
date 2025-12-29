/**
 * @file baseline.hpp
 * @brief CPU reference implementations for benchmarking
 * 
 * These are intentionally simple, single-threaded implementations.
 * They serve as:
 * 1. Correctness reference for GPU implementations
 * 2. Performance baseline to measure GPU speedup
 * 
 * The implementations are straightforward loops without SIMD or
 * multi-threading - this makes the GPU speedup numbers more impressive
 * while still being a fair comparison (single CPU core vs GPU).
 */

#pragma once

#include "core/types.hpp"
#include "core/table.hpp"
#include <vector>
#include <chrono>
#include <functional>

namespace gpusql {
namespace cpu {

// ============================================================================
// Timing Utilities
// ============================================================================

/**
 * Simple high-resolution timer for benchmarking.
 */
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }
    
    /** Get elapsed time in milliseconds */
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
        return duration.count() / 1000.0;
    }
    
    /** Get elapsed time in seconds */
    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

/**
 * Run a function multiple times and return average time.
 * 
 * @param func Function to benchmark
 * @param warmup_runs Number of warmup runs (not timed)
 * @param timed_runs Number of timed runs
 * @return Average time in milliseconds
 */
double benchmark(std::function<void()> func, int warmup_runs = 2, int timed_runs = 5);

// ============================================================================
// Filter Operations
// ============================================================================

/**
 * Result of a filter operation.
 * Contains indices of rows that passed the filter.
 */
struct FilterResult {
    std::vector<int32_t> indices;  // Indices of matching rows
    size_t count;                   // Number of matching rows
    double elapsed_ms;              // Execution time
};

/**
 * Filter rows where int column > threshold.
 * 
 * This is the primary filter benchmark: "SELECT * WHERE col > value"
 * 
 * Implementation:
 * - Simple loop over all rows
 * - Check condition and collect matching indices
 * 
 * @param column Input int32 column
 * @param threshold Comparison threshold
 * @return FilterResult with matching row indices
 */
FilterResult filter_greater_than(const Column& column, int32_t threshold);

/**
 * Filter rows where int column == value.
 * "SELECT * WHERE col = value"
 */
FilterResult filter_equals(const Column& column, int32_t value);

/**
 * Filter rows where float column > threshold.
 */
FilterResult filter_float_greater(const Column& column, float threshold);

/**
 * Apply filter mask to extract matching rows from multiple columns.
 * This simulates the "compaction" phase of a SELECT query.
 * 
 * @param table Input table
 * @param indices Indices of rows to extract
 * @return New table with only matching rows
 */
Table apply_filter_indices(const Table& table, const std::vector<int32_t>& indices);

// ============================================================================
// Aggregation Operations
// ============================================================================

/**
 * Result of an aggregation operation.
 */
struct AggregateResult {
    int64_t count;
    double sum;
    double min;
    double max;
    double avg;
    double elapsed_ms;
};

/**
 * Compute COUNT(*) - just return number of rows.
 */
int64_t count_all(const Table& table);

/**
 * Compute COUNT(*) WHERE int_col > threshold.
 */
int64_t count_where(const Column& column, int32_t threshold);

/**
 * Compute SUM(column).
 */
double sum_column(const Column& column);

/**
 * Compute SUM(float_col) WHERE int_col > threshold.
 * 
 * This is a more realistic query that combines filter + aggregation.
 */
double sum_where(
    const Column& sum_column,
    const Column& filter_column,
    int32_t threshold
);

/**
 * Full aggregation: COUNT, SUM, MIN, MAX, AVG.
 */
AggregateResult aggregate_column(const Column& column);

// ============================================================================
// Join Operations
// ============================================================================

/**
 * Result of a join operation.
 */
struct JoinResult {
    std::vector<int32_t> left_indices;   // Matching row indices from left table
    std::vector<int32_t> right_indices;  // Matching row indices from right table
    size_t count;                         // Number of matches
    double elapsed_ms;
};

/**
 * Hash join: left_table JOIN right_table ON left_col = right_col
 * 
 * Algorithm:
 * 1. Build phase: Create hash map from right table (id -> row index)
 * 2. Probe phase: For each row in left table, look up in hash map
 * 
 * This is a simple implementation using std::unordered_map.
 * The GPU version will use a custom hash table in device memory.
 * 
 * @param left_column Join key column from left table
 * @param right_column Join key column from right table
 * @return JoinResult with matching index pairs
 */
JoinResult hash_join(const Column& left_column, const Column& right_column);

/**
 * Nested loop join (O(n*m) - for small tables or correctness testing).
 */
JoinResult nested_loop_join(const Column& left_column, const Column& right_column);

// ============================================================================
// Benchmark Suite
// ============================================================================

/**
 * Run all CPU baseline benchmarks and print results.
 * 
 * @param rows Number of rows to test with
 */
void run_all_benchmarks(size_t rows = 1'000'000);

} // namespace cpu
} // namespace gpusql

