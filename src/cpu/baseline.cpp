#include "baseline.hpp"
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>
namespace gpusql {
namespace cpu {

double benchmark(std::function<void()> func, int warmup_runs, int timed_runs) {
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }
    
    Timer timer;
    double total_ms = 0.0;
    
    for (int i = 0; i < timed_runs; ++i) {
        timer.start();
        func();
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    
    return total_ms / timed_runs;
}


FilterResult filter_greater_than(const Column& column, int32_t threshold) {
    Timer timer;
    timer.start();
    
    FilterResult result;
    result.indices.reserve(column.size() / 10);  
    const int32_t* data = column.data<int32_t>();
    size_t n = column.size();
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] > threshold) {
            result.indices.push_back(static_cast<int32_t>(i));
        }
    }
    
    timer.stop();
    result.count = result.indices.size();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}

FilterResult filter_equals(const Column& column, int32_t value) {
    Timer timer;
    timer.start();
    
    FilterResult result;
    result.indices.reserve(column.size() / 100); 
    
    const int32_t* data = column.data<int32_t>();
    size_t n = column.size();
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] == value) {
            result.indices.push_back(static_cast<int32_t>(i));
        }
    }
    
    timer.stop();
    result.count = result.indices.size();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}

FilterResult filter_float_greater(const Column& column, float threshold) {
    Timer timer;
    timer.start();
    
    FilterResult result;
    result.indices.reserve(column.size() / 10);
    
    const float* data = column.data<float>();
    size_t n = column.size();
    
    for (size_t i = 0; i < n; ++i) {
        if (data[i] > threshold) {
            result.indices.push_back(static_cast<int32_t>(i));
        }
    }
    
    timer.stop();
    result.count = result.indices.size();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}

Table apply_filter_indices(const Table& table, const std::vector<int32_t>& indices) {
    Table result(table.name() + "_filtered");
    
    for (const auto& col : table.columns()) {
        result.add_column(col.name(), col.type());
    }
    
    size_t output_size = indices.size();
    result.resize(output_size);
    
    for (size_t col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
        const Column& src = table.column(col_idx);
        Column& dst = result.column(col_idx);
        
        switch (src.type()) {
            case DataType::INT32: {
                const int32_t* src_data = src.data<int32_t>();
                int32_t* dst_data = dst.data<int32_t>();
                for (size_t i = 0; i < output_size; ++i) {
                    dst_data[i] = src_data[indices[i]];
                }
                break;
            }
            case DataType::FLOAT32: {
                const float* src_data = src.data<float>();
                float* dst_data = dst.data<float>();
                for (size_t i = 0; i < output_size; ++i) {
                    dst_data[i] = src_data[indices[i]];
                }
                break;
            }
            case DataType::STRING: {
                const FixedString* src_data = src.data<FixedString>();
                FixedString* dst_data = dst.data<FixedString>();
                for (size_t i = 0; i < output_size; ++i) {
                    dst_data[i] = src_data[indices[i]];
                }
                break;
            }
        }
    }
    
    return result;
}

int64_t count_all(const Table& table) {
    return static_cast<int64_t>(table.num_rows());
}

int64_t count_where(const Column& column, int32_t threshold) {
    const int32_t* data = column.data<int32_t>();
    size_t n = column.size();
    
    int64_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i] > threshold) {
            ++count;
        }
    }
    
    return count;
}

double sum_column(const Column& column) {
    size_t n = column.size();
    double sum = 0.0;
    
    switch (column.type()) {
        case DataType::INT32: {
            const int32_t* data = column.data<int32_t>();
            for (size_t i = 0; i < n; ++i) {
                sum += data[i];
            }
            break;
        }
        case DataType::FLOAT32: {
            const float* data = column.data<float>();
            for (size_t i = 0; i < n; ++i) {
                sum += data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Cannot sum STRING column");
    }
    
    return sum;
}

double sum_where(
    const Column& sum_col,
    const Column& filter_col,
    int32_t threshold
) {
    if (sum_col.size() != filter_col.size()) {
        throw std::runtime_error("Column sizes must match");
    }
    
    size_t n = sum_col.size();
    double sum = 0.0;
    
    const int32_t* filter_data = filter_col.data<int32_t>();
    
    switch (sum_col.type()) {
        case DataType::INT32: {
            const int32_t* sum_data = sum_col.data<int32_t>();
            for (size_t i = 0; i < n; ++i) {
                if (filter_data[i] > threshold) {
                    sum += sum_data[i];
                }
            }
            break;
        }
        case DataType::FLOAT32: {
            const float* sum_data = sum_col.data<float>();
            for (size_t i = 0; i < n; ++i) {
                if (filter_data[i] > threshold) {
                    sum += sum_data[i];
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Cannot sum STRING column");
    }
    
    return sum;
}

AggregateResult aggregate_column(const Column& column) {
    Timer timer;
    timer.start();
    
    AggregateResult result;
    result.count = column.size();
    result.sum = 0.0;
    result.min = std::numeric_limits<double>::max();
    result.max = std::numeric_limits<double>::lowest();
    
    size_t n = column.size();
    
    switch (column.type()) {
        case DataType::INT32: {
            const int32_t* data = column.data<int32_t>();
            for (size_t i = 0; i < n; ++i) {
                double val = data[i];
                result.sum += val;
                result.min = std::min(result.min, val);
                result.max = std::max(result.max, val);
            }
            break;
        }
        case DataType::FLOAT32: {
            const float* data = column.data<float>();
            for (size_t i = 0; i < n; ++i) {
                double val = data[i];
                result.sum += val;
                result.min = std::min(result.min, val);
                result.max = std::max(result.max, val);
            }
            break;
        }
        default:
            throw std::runtime_error("Cannot aggregate STRING column");
    }
    
    result.avg = (result.count > 0) ? (result.sum / result.count) : 0.0;
    
    timer.stop();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}


JoinResult hash_join(const Column& left_column, const Column& right_column) {
    Timer timer;
    timer.start();
    
    JoinResult result;
    
    std::unordered_map<int32_t, std::vector<int32_t>> hash_table;
    
    const int32_t* right_data = right_column.data<int32_t>();
    size_t right_size = right_column.size();
    
    hash_table.reserve(right_size);  
    for (size_t i = 0; i < right_size; ++i) {
        hash_table[right_data[i]].push_back(static_cast<int32_t>(i));
    }
    
    const int32_t* left_data = left_column.data<int32_t>();
    size_t left_size = left_column.size();
    
    result.left_indices.reserve(std::min(left_size, right_size));
    result.right_indices.reserve(std::min(left_size, right_size));
    
    for (size_t i = 0; i < left_size; ++i) {
        auto it = hash_table.find(left_data[i]);
        if (it != hash_table.end()) {
            for (int32_t right_idx : it->second) {
                result.left_indices.push_back(static_cast<int32_t>(i));
                result.right_indices.push_back(right_idx);
            }
        }
    }
    
    timer.stop();
    result.count = result.left_indices.size();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}

JoinResult nested_loop_join(const Column& left_column, const Column& right_column) {
    Timer timer;
    timer.start();
    
    JoinResult result;
    
    const int32_t* left_data = left_column.data<int32_t>();
    const int32_t* right_data = right_column.data<int32_t>();
    size_t left_size = left_column.size();
    size_t right_size = right_column.size();
    
    for (size_t i = 0; i < left_size; ++i) {
        for (size_t j = 0; j < right_size; ++j) {
            if (left_data[i] == right_data[j]) {
                result.left_indices.push_back(static_cast<int32_t>(i));
                result.right_indices.push_back(static_cast<int32_t>(j));
            }
        }
    }
    
    timer.stop();
    result.count = result.left_indices.size();
    result.elapsed_ms = timer.elapsed_ms();
    
    return result;
}

void run_all_benchmarks(size_t rows) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            CPU BASELINE BENCHMARKS                           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Rows: " << std::setw(12) << rows << "                                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Generating test data... " << std::flush;
    Timer gen_timer;
    gen_timer.start();
    Table table = create_benchmark_table(rows, 42);
    gen_timer.stop();
    std::cout << "done (" << std::fixed << std::setprecision(2) 
              << gen_timer.elapsed_ms() << " ms)\n\n";
    
    std::cout << "── FILTER BENCHMARKS ──────────────────────────────────────────\n";
    
    std::vector<int32_t> thresholds = {
        static_cast<int32_t>(rows * 0.1),  
        static_cast<int32_t>(rows * 0.5),  
        static_cast<int32_t>(rows * 0.9), 
        static_cast<int32_t>(rows * 0.99), 
    };
    
    for (int32_t threshold : thresholds) {
        auto result = filter_greater_than(table.column("id"), threshold);
        double selectivity = 100.0 * result.count / rows;
        double throughput = rows / (result.elapsed_ms / 1000.0) / 1e6;
        
        std::cout << "  Filter (id > " << std::setw(8) << threshold << "): "
                  << std::setw(8) << result.count << " rows ("
                  << std::fixed << std::setprecision(1) << std::setw(5) << selectivity << "%) in "
                  << std::setw(7) << std::setprecision(2) << result.elapsed_ms << " ms"
                  << " [" << std::setprecision(1) << throughput << " M rows/sec]\n";
    }
    
    auto float_result = filter_float_greater(table.column("value"), 500.0f);
    double float_throughput = rows / (float_result.elapsed_ms / 1000.0) / 1e6;
    std::cout << "  Filter (value > 500.0): "
              << std::setw(8) << float_result.count << " rows in "
              << std::setw(7) << std::fixed << std::setprecision(2) << float_result.elapsed_ms << " ms"
              << " [" << std::setprecision(1) << float_throughput << " M rows/sec]\n";
    
    std::cout << "\n── AGGREGATION BENCHMARKS ─────────────────────────────────────\n";
    
    auto agg_result = aggregate_column(table.column("value"));
    std::cout << "  Full aggregation (value): "
              << std::fixed << std::setprecision(2) << agg_result.elapsed_ms << " ms\n";
    std::cout << "    COUNT: " << agg_result.count << "\n";
    std::cout << "    SUM:   " << std::setprecision(2) << agg_result.sum << "\n";
    std::cout << "    MIN:   " << agg_result.min << "\n";
    std::cout << "    MAX:   " << agg_result.max << "\n";
    std::cout << "    AVG:   " << agg_result.avg << "\n";
    
    Timer count_timer;
    count_timer.start();
    int64_t count_result = count_where(table.column("category"), 50);
    count_timer.stop();
    std::cout << "  COUNT WHERE category > 50: " << count_result 
              << " in " << std::fixed << std::setprecision(2) 
              << count_timer.elapsed_ms() << " ms\n";
    
    std::cout << "\n── JOIN BENCHMARKS ────────────────────────────────────────────\n";
    
    size_t dim_size = std::min(rows / 100, size_t(10000));
    Table dim_table = create_dimension_table(dim_size, 123);
    
    Table join_left("join_left");
    join_left.add_column("id", DataType::INT32);
    join_left.resize(std::min(rows, size_t(100000)));
    
    int32_t* join_ids = join_left.column("id").data<int32_t>();
    const int32_t* orig_ids = table.column("category").data<int32_t>();  // Use category for more matches
    for (size_t i = 0; i < join_left.num_rows(); ++i) {
        join_ids[i] = orig_ids[i] % static_cast<int32_t>(dim_size);
    }
    
    auto join_result = hash_join(join_left.column("id"), dim_table.column("id"));
    double join_throughput = (join_left.num_rows() + dim_size) / (join_result.elapsed_ms / 1000.0) / 1e6;
    
    std::cout << "  Hash join (" << join_left.num_rows() << " x " << dim_size << "): "
              << join_result.count << " matches in "
              << std::fixed << std::setprecision(2) << join_result.elapsed_ms << " ms"
              << " [" << std::setprecision(1) << join_throughput << " M rows/sec]\n";
    
    std::cout << "\n── MEMORY USAGE ───────────────────────────────────────────────\n";
    double mb = table.memory_usage() / (1024.0 * 1024.0);
    std::cout << "  Main table: " << std::fixed << std::setprecision(2) << mb << " MB\n";
    
    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "CPU baseline benchmarks complete.\n";
    std::cout << "These timings will be compared against GPU implementations.\n\n";
}

}
}

