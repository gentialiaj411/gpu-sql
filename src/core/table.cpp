#include "table.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace gpusql {

Table::Table(std::string name)
    : name_(std::move(name))
    , num_rows_(0)
{
}

Table& Table::add_column(const std::string& name, DataType type) {
    if (has_column(name)) {
        throw std::runtime_error("Column '" + name + "' already exists");
    }
    
    // Create column and resize to match existing rows
    Column col(name, type);
    if (num_rows_ > 0) {
        col.resize(num_rows_);
    }
    
    // Store column
    size_t idx = columns_.size();
    columns_.push_back(std::move(col));
    name_to_index_[name] = idx;
    
    return *this;
}

bool Table::has_column(const std::string& name) const {
    return name_to_index_.find(name) != name_to_index_.end();
}

std::optional<size_t> Table::column_index(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it != name_to_index_.end()) {
        return it->second;
    }
    return std::nullopt;
}

Column& Table::column(const std::string& name) {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        throw std::runtime_error("Column '" + name + "' not found");
    }
    return columns_[it->second];
}

const Column& Table::column(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) {
        throw std::runtime_error("Column '" + name + "' not found");
    }
    return columns_[it->second];
}

std::vector<std::string> Table::column_names() const {
    std::vector<std::string> names;
    names.reserve(columns_.size());
    for (const auto& col : columns_) {
        names.push_back(col.name());
    }
    return names;
}

void Table::reserve(size_t n) {
    for (auto& col : columns_) {
        col.reserve(n);
    }
}

void Table::resize(size_t n) {
    for (auto& col : columns_) {
        col.resize(n);
    }
    num_rows_ = n;
}

void Table::clear() {
    for (auto& col : columns_) {
        col.clear();
    }
    num_rows_ = 0;
}


void Table::generate_random_data(size_t n, uint32_t seed) {
    if (columns_.empty()) {
        throw std::runtime_error("Cannot generate data for table with no columns");
    }

    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 gen(seed);
    
    
    resize(n);
    
    for (auto& col : columns_) {
        switch (col.type()) {
            case DataType::INT32: {
                std::uniform_int_distribution<int32_t> dist(0, 999999);
                int32_t* data = col.data<int32_t>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
                break;
            }
            
            case DataType::FLOAT32: {
                std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
                float* data = col.data<float>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
                break;
            }
            
            case DataType::STRING: {
                std::uniform_int_distribution<int> digit_dist(0, 9);
                FixedString* data = col.data<FixedString>();
                for (size_t i = 0; i < n; ++i) {
                    char buf[STRING_MAX_LENGTH];
                    snprintf(buf, sizeof(buf), "str_%d%d%d%d%d%d",
                             digit_dist(gen), digit_dist(gen), digit_dist(gen),
                             digit_dist(gen), digit_dist(gen), digit_dist(gen));
                    data[i] = FixedString(buf);
                }
                break;
            }
        }
    }
}

void Table::generate_filter_benchmark_data(
    size_t n, 
    double selectivity,
    const std::string& int_col_name,
    int32_t threshold
) {
    if (columns_.empty()) {
        throw std::runtime_error("Cannot generate data for table with no columns");
    }
    
    auto idx = column_index(int_col_name);
    if (!idx.has_value()) {
        throw std::runtime_error("Column '" + int_col_name + "' not found");
    }
    
    if (columns_[*idx].type() != DataType::INT32) {
        throw std::runtime_error("Column '" + int_col_name + "' must be INT32");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    resize(n);
    
    {
        int32_t* data = columns_[*idx].data<int32_t>();
        std::uniform_real_distribution<double> select_dist(0.0, 1.0);
        
        // pass values from threshold+1 to threshold+1000
        std::uniform_int_distribution<int32_t> pass_dist(threshold + 1, threshold + 1000);
        std::uniform_int_distribution<int32_t> fail_dist(0, threshold);
        
        for (size_t i = 0; i < n; ++i) {
            if (select_dist(gen) < selectivity) {
                data[i] = pass_dist(gen);  // Will pass filter
            } else {
                data[i] = fail_dist(gen);  // Will fail filter
            }
        }
    }
    
    for (size_t col_idx = 0; col_idx < columns_.size(); ++col_idx) {
        if (col_idx == *idx) continue;  
        
        auto& col = columns_[col_idx];
        switch (col.type()) {
            case DataType::INT32: {
                std::uniform_int_distribution<int32_t> dist(0, 999999);
                int32_t* data = col.data<int32_t>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
                break;
            }
            
            case DataType::FLOAT32: {
                std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
                float* data = col.data<float>();
                for (size_t i = 0; i < n; ++i) {
                    data[i] = dist(gen);
                }
                break;
            }
            
            case DataType::STRING: {
                std::uniform_int_distribution<int> digit_dist(0, 9);
                FixedString* data = col.data<FixedString>();
                for (size_t i = 0; i < n; ++i) {
                    char buf[STRING_MAX_LENGTH];
                    snprintf(buf, sizeof(buf), "str_%d%d%d%d%d%d",
                             digit_dist(gen), digit_dist(gen), digit_dist(gen),
                             digit_dist(gen), digit_dist(gen), digit_dist(gen));
                    data[i] = FixedString(buf);
                }
                break;
            }
        }
    }
}

void Table::debug_print(size_t max_rows) const {
    print_schema();
    
    if (columns_.empty() || num_rows_ == 0) {
        std::cout << "(no data)\n";
        return;
    }
    
    size_t rows_to_print = std::min(max_rows, num_rows_);
    
    std::cout << std::string(60, '-') << '\n';
    for (const auto& col : columns_) {
        std::cout << std::setw(15) << col.name() << " | ";
    }
    std::cout << '\n' << std::string(60, '-') << '\n';
    
    for (size_t row = 0; row < rows_to_print; ++row) {
        for (const auto& col : columns_) {
            switch (col.type()) {
                case DataType::INT32:
                    std::cout << std::setw(15) << col.data<int32_t>()[row];
                    break;
                case DataType::FLOAT32:
                    std::cout << std::setw(15) << std::fixed << std::setprecision(2) 
                             << col.data<float>()[row];
                    break;
                case DataType::STRING:
                    std::cout << std::setw(15) << col.data<FixedString>()[row].data;
                    break;
            }
            std::cout << " | ";
        }
        std::cout << '\n';
    }
    
    if (num_rows_ > max_rows) {
        std::cout << "... (" << (num_rows_ - max_rows) << " more rows)\n";
    }
}

void Table::print_schema() const {
    std::cout << "Table: " << (name_.empty() ? "(unnamed)" : name_) 
              << " (" << num_rows_ << " rows, " << columns_.size() << " columns)\n";
    
    for (const auto& col : columns_) {
        std::cout << "  - " << col.name() << ": " << type_name(col.type()) << '\n';
    }
}

size_t Table::memory_usage() const {
    size_t total = 0;
    for (const auto& col : columns_) {
        total += col.bytes();
    }
    return total;
}

Table create_benchmark_table(size_t n, uint32_t seed) {
    Table table("benchmark");
    table.add_column("id", DataType::INT32);
    table.add_column("value", DataType::FLOAT32);
    table.add_column("category", DataType::INT32);
    
    table.resize(n);
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> value_dist(0.0f, 1000.0f);
    std::uniform_int_distribution<int32_t> cat_dist(0, 99);  
    
    int32_t* ids = table.column("id").data<int32_t>();
    float* values = table.column("value").data<float>();
    int32_t* categories = table.column("category").data<int32_t>();
    
    for (size_t i = 0; i < n; ++i) {
        ids[i] = static_cast<int32_t>(i);
        values[i] = value_dist(gen);
        categories[i] = cat_dist(gen);
    }
    
    return table;
}

Table create_dimension_table(size_t n, uint32_t seed) {
    Table table("dimension");
    table.add_column("id", DataType::INT32);
    table.add_column("label", DataType::STRING);
    
    table.resize(n);
    
    std::mt19937 gen(seed);
    
    int32_t* ids = table.column("id").data<int32_t>();
    FixedString* labels = table.column("label").data<FixedString>();
    
    for (size_t i = 0; i < n; ++i) {
        ids[i] = static_cast<int32_t>(i);
        char buf[STRING_MAX_LENGTH];
        snprintf(buf, sizeof(buf), "label_%zu", i);
        labels[i] = FixedString(buf);
    }
    
    return table;
}

} 

