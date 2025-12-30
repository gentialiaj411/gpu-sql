#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include "core/types.hpp"
#include "core/table.hpp"
#include "cpu/baseline.hpp"
#include "kernels/filter.cuh"
#include "kernels/utils.cuh"

using namespace gpusql;

struct BenchmarkConfig {
    size_t num_rows = 1'000'000;
    bool run_cpu = true;
    bool run_gpu = true;
    bool verbose = false;
    int warmup_runs = 2;
    int timed_runs = 5;
};

void print_usage() {
    std::cout << "Usage: gpu_sql_benchmark [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --rows N       Number of rows (default: 1000000)\n";
    std::cout << "  --cpu-only     Run only CPU benchmarks\n";
    std::cout << "  --gpu-only     Run only GPU benchmarks\n";
    std::cout << "  --verbose      Print detailed output\n";
    std::cout << "  --help         Show this message\n";
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage();
            exit(0);
        } else if (arg == "--rows" && i + 1 < argc) {
            config.num_rows = std::stoull(argv[++i]);
        } else if (arg == "--cpu-only") {
            config.run_gpu = false;
        } else if (arg == "--gpu-only") {
            config.run_cpu = false;
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    return config;
}

void print_header(const std::string& title) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(65) << title << " ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
}

void print_section(const std::string& title) {
    std::cout << "\n── " << title << " ";
    std::cout << std::string(60 - title.length(), '─') << "\n";
}

struct BenchmarkResult {
    std::string name;
    size_t rows;
    size_t matches;
    double cpu_time_ms;
    double gpu_time_ms;
    double gpu_kernel_ms;
    double speedup;
    double throughput_mrps;  
};

void print_result(const BenchmarkResult& r) {
    double selectivity = 100.0 * r.matches / r.rows;
    
    std::cout << "  " << std::left << std::setw(30) << r.name << " │ ";
    std::cout << std::right << std::setw(8) << r.matches << " matches ";
    std::cout << "(" << std::fixed << std::setprecision(1) << std::setw(5) << selectivity << "%)";
    
    if (r.cpu_time_ms > 0) {
        std::cout << " │ CPU: " << std::setw(7) << std::setprecision(2) << r.cpu_time_ms << " ms";
    }
    
    if (r.gpu_time_ms > 0) {
        std::cout << " │ GPU: " << std::setw(7) << r.gpu_time_ms << " ms";
        std::cout << " (kernel: " << std::setw(5) << r.gpu_kernel_ms << " ms)";
    }
    
    if (r.speedup > 0) {
        std::cout << " │ " << std::setw(5) << std::setprecision(1) << r.speedup << "x speedup";
    }
    
    std::cout << "\n";
}

bool verify_filter_results(
    const std::vector<int32_t>& cpu_indices,
    const std::vector<int32_t>& gpu_indices
) {
    if (cpu_indices.size() != gpu_indices.size()) {
        std::cerr << "ERROR: Count mismatch! CPU: " << cpu_indices.size()
                  << ", GPU: " << gpu_indices.size() << "\n";
        return false;
    }
    
    std::vector<int32_t> cpu_sorted = cpu_indices;
    std::vector<int32_t> gpu_sorted = gpu_indices;
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    std::sort(gpu_sorted.begin(), gpu_sorted.end());
    
    if (cpu_sorted != gpu_sorted) {
        std::cerr << "ERROR: Index mismatch!\n";
        return false;
    }
    
    return true;
}

void run_filter_benchmarks(const BenchmarkConfig& config, const Table& table) {
    print_section("FILTER BENCHMARKS (WHERE col > threshold)");
    
    const Column& id_col = table.column("id");
    const Column& value_col = table.column("value");
    size_t n = table.num_rows();
    
    std::vector<std::pair<std::string, int32_t>> tests = {
        {"10% selectivity", static_cast<int32_t>(n * 0.9)},
        {"50% selectivity", static_cast<int32_t>(n * 0.5)},
        {"90% selectivity", static_cast<int32_t>(n * 0.1)},
        {"1% selectivity",  static_cast<int32_t>(n * 0.99)},
    };
    
    for (const auto& [name, threshold] : tests) {
        BenchmarkResult result;
        result.name = name;
        result.rows = n;
        result.cpu_time_ms = 0;
        result.gpu_time_ms = 0;
        result.gpu_kernel_ms = 0;
        result.speedup = 0;
        
        cpu::FilterResult cpu_result;
        if (config.run_cpu) {
            for (int i = 0; i < config.warmup_runs; ++i) {
                cpu::filter_greater_than(id_col, threshold);
            }
            
            double total_cpu = 0;
            for (int i = 0; i < config.timed_runs; ++i) {
                cpu_result = cpu::filter_greater_than(id_col, threshold);
                total_cpu += cpu_result.elapsed_ms;
            }
            result.cpu_time_ms = total_cpu / config.timed_runs;
            result.matches = cpu_result.count;
        }
        
        if (config.run_gpu) {
            for (int i = 0; i < config.warmup_runs; ++i) {
                auto gpu_res = cuda::filter_greater_than_gpu(
                    id_col.data<int32_t>(), n, threshold, true
                );
                cuda::free_filter_result(gpu_res);
            }
            
            double total_gpu = 0;
            double total_kernel = 0;
            cuda::GpuFilterResult gpu_result;
            
            for (int i = 0; i < config.timed_runs; ++i) {
                gpu_result = cuda::filter_greater_than_gpu(
                    id_col.data<int32_t>(), n, threshold, true
                );
                total_gpu += gpu_result.total_time_ms;
                total_kernel += gpu_result.kernel_time_ms;
                
                if (i < config.timed_runs - 1) {
                    cuda::free_filter_result(gpu_result);
                }
            }
            
            result.gpu_time_ms = total_gpu / config.timed_runs;
            result.gpu_kernel_ms = total_kernel / config.timed_runs;
            result.matches = gpu_result.count;
            
            if (config.run_cpu && config.verbose) {
                auto gpu_indices = cuda::copy_filter_results_to_host(gpu_result);
                if (verify_filter_results(cpu_result.indices, gpu_indices)) {
                    std::cout << "  ✓ Verification passed\n";
                }
            }
            
            cuda::free_filter_result(gpu_result);
        }
        
        if (config.run_cpu && config.run_gpu && result.cpu_time_ms > 0) {
            result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        }
        
        result.throughput_mrps = n / (result.gpu_time_ms > 0 ? result.gpu_time_ms : result.cpu_time_ms) / 1000.0;
        
        print_result(result);
    }
}

void run_kernel_comparison(const BenchmarkConfig& config, const Table& table) {
    if (!config.run_gpu) return;
    
    print_section("KERNEL OPTIMIZATION COMPARISON");
    
    const Column& id_col = table.column("id");
    size_t n = table.num_rows();
    int32_t threshold = static_cast<int32_t>(n * 0.5);
    
    std::cout << "  Comparing atomic vs warp-optimized kernels (50% selectivity):\n\n";
    
    {
        cuda::GpuTimer timer;
        timer.start();
        
        for (int i = 0; i < 10; ++i) {
            auto result = cuda::filter_greater_than_gpu(
                id_col.data<int32_t>(), n, threshold, false  // Use atomic version
            );
            cuda::free_filter_result(result);
        }
        
        timer.stop();
        double avg_ms = timer.elapsed_ms() / 10.0;
        double throughput = n / avg_ms / 1000.0;
        
        std::cout << "  Atomic kernel:         " << std::fixed << std::setprecision(2)
                  << avg_ms << " ms (" << std::setprecision(1) << throughput << " M rows/sec)\n";
    }
    
    {
        cuda::GpuTimer timer;
        timer.start();
        
        for (int i = 0; i < 10; ++i) {
            auto result = cuda::filter_greater_than_gpu(
                id_col.data<int32_t>(), n, threshold, true  
            );
            cuda::free_filter_result(result);
        }
        
        timer.stop();
        double avg_ms = timer.elapsed_ms() / 10.0;
        double throughput = n / avg_ms / 1000.0;
        
        std::cout << "  Warp-optimized kernel: " << std::fixed << std::setprecision(2)
                  << avg_ms << " ms (" << std::setprecision(1) << throughput << " M rows/sec)\n";
    }
}

void run_bandwidth_analysis(const BenchmarkConfig& config, const Table& table) {
    if (!config.run_gpu) return;
    
    print_section("MEMORY BANDWIDTH ANALYSIS");
    
    const Column& id_col = table.column("id");
    size_t n = table.num_rows();
    int32_t threshold = static_cast<int32_t>(n * 0.5);
    
    auto result = cuda::filter_greater_than_gpu(
        id_col.data<int32_t>(), n, threshold, true
    );
    
    size_t bytes_read = n * sizeof(int32_t);     
    size_t bytes_written = result.count * sizeof(int32_t);  
    double total_gb = (bytes_read + bytes_written) / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_gbps = total_gb / (result.kernel_time_ms / 1000.0);
    
    std::cout << "  Input size:         " << n << " elements (" 
              << bytes_read / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "  Output size:        " << result.count << " elements (" 
              << bytes_written / (1024.0 * 1024.0) << " MB)\n";
    std::cout << "  Kernel time:        " << std::fixed << std::setprecision(3) 
              << result.kernel_time_ms << " ms\n";
    std::cout << "  Effective bandwidth: " << std::setprecision(1) 
              << bandwidth_gbps << " GB/s\n";
    
    cuda::print_memory_usage();
    
    cuda::free_filter_result(result);
}

void run_scaling_benchmark(const BenchmarkConfig& config) {
    if (!config.run_gpu || !config.run_cpu) return;
    
    print_section("SCALING ANALYSIS");
    
    std::cout << "  Testing how speedup scales with data size:\n\n";
    std::cout << std::left << std::setw(15) << "  Rows" 
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << "Throughput\n";
    std::cout << "  " << std::string(60, '-') << "\n";
    
    std::vector<size_t> sizes = {
        10'000,
        100'000,
        1'000'000,
        5'000'000,
        10'000'000
    };
    
    for (size_t n : sizes) {
        auto [free_mem, total_mem] = cuda::get_memory_info();
        size_t required = n * sizeof(int32_t) * 3;  
        
        if (required > free_mem * 0.8) {
            std::cout << "  " << std::setw(13) << n << " (skipped - insufficient GPU memory)\n";
            continue;
        }
        
        Table table = create_benchmark_table(n, 42);
        int32_t threshold = static_cast<int32_t>(n * 0.5);
        
        auto cpu_result = cpu::filter_greater_than(table.column("id"), threshold);
        
        double gpu_total = 0;
        for (int i = 0; i < 3; ++i) {
            auto gpu_result = cuda::filter_greater_than_gpu(
                table.column("id").data<int32_t>(), n, threshold, true
            );
            gpu_total += gpu_result.total_time_ms;
            cuda::free_filter_result(gpu_result);
        }
        double gpu_time = gpu_total / 3.0;
        
        double speedup = cpu_result.elapsed_ms / gpu_time;
        double throughput = n / gpu_time / 1000.0;
        
        std::cout << "  " << std::setw(13) << n 
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << cpu_result.elapsed_ms
                  << std::setw(12) << gpu_time
                  << std::setw(12) << std::setprecision(1) << speedup << "x"
                  << std::setprecision(0) << throughput << " M rows/s\n";
    }
}

int main(int argc, char** argv) {
    BenchmarkConfig config = parse_args(argc, argv);
    
    print_header("GPU-SQL BENCHMARK SUITE");
    
    std::cout << "Configuration:\n";
    std::cout << "  Rows:        " << config.num_rows << "\n";
    std::cout << "  CPU tests:   " << (config.run_cpu ? "enabled" : "disabled") << "\n";
    std::cout << "  GPU tests:   " << (config.run_gpu ? "enabled" : "disabled") << "\n";
    std::cout << "  Warmup runs: " << config.warmup_runs << "\n";
    std::cout << "  Timed runs:  " << config.timed_runs << "\n";
    
    if (config.run_gpu) {
        cuda::print_device_info();
    }
    
    print_section("DATA GENERATION");
    
    cpu::Timer gen_timer;
    gen_timer.start();
    Table table = create_benchmark_table(config.num_rows, 42);
    gen_timer.stop();
    
    std::cout << "  Generated " << table.num_rows() << " rows in " 
              << std::fixed << std::setprecision(2) << gen_timer.elapsed_ms() << " ms\n";
    std::cout << "  Memory usage: " << table.memory_usage() / (1024.0 * 1024.0) << " MB\n";
    
    if (config.verbose) {
        table.debug_print(5);
    }
    
    run_filter_benchmarks(config, table);
    run_kernel_comparison(config, table);
    run_bandwidth_analysis(config, table);
    run_scaling_benchmark(config);
    
    // Summary
    print_section("SUMMARY");
    
    std::cout << R"(
  This benchmark demonstrates GPU acceleration of SQL filter operations.
  
  Key observations:
  1. GPU achieves significant speedup for large datasets
  2. Warp-optimized kernel reduces atomic contention
  3. Data transfer time is significant for small datasets
  4. Kernel-only time shows the true GPU compute advantage
  
  Next steps:
  - Implement hash join kernel
  - Add aggregation kernels
  - Optimize with CUDA streams for pipelined execution
  - Add multi-column filter support
)";
    
    std::cout << "\n═════════════════════════════════════════════════════════════════════\n";
    
    return 0;
}


