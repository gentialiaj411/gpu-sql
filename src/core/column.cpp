#include "column.hpp"
#include <iostream>
#include <iomanip>

namespace gpusql {

Column::Column(std::string name, DataType type)
    : name_(std::move(name))
    , type_(type)
    , size_(0)
    , data_()
{
}

Column::Column(std::string name, DataType type, size_t capacity)
    : name_(std::move(name))
    , type_(type)
    , size_(0)
    , data_()
{
    reserve(capacity);
}

Column::Column(Column&& other) noexcept
    : name_(std::move(other.name_))
    , type_(other.type_)
    , size_(other.size_)
    , data_(std::move(other.data_))
{
    other.size_ = 0;
}

Column& Column::operator=(Column&& other) noexcept {
    if (this != &other) {
        name_ = std::move(other.name_);
        type_ = other.type_;
        size_ = other.size_;
        data_ = std::move(other.data_);
        other.size_ = 0;
    }
    return *this;
}

void Column::reserve(size_t n) {
    size_t bytes_needed = n * type_size(type_);
    if (bytes_needed > data_.size()) {
        data_.resize(bytes_needed);
    }
}

void Column::resize(size_t n) {
    size_t bytes_needed = n * type_size(type_);
    data_.resize(bytes_needed);
    
    if (n > size_) {
        size_t old_bytes = size_ * type_size(type_);
        std::memset(data_.data() + old_bytes, 0, bytes_needed - old_bytes);
    }
    
    size_ = n;
}

void Column::clear() {
    size_ = 0;
}

Column Column::clone() const {
    Column copy(name_, type_);
    copy.data_ = data_;  
    copy.size_ = size_;
    return copy;
}

void Column::debug_print(size_t max_rows) const {
    std::cout << "Column '" << name_ << "' (" << type_name(type_) 
              << ", " << size_ << " rows):\n";
    
    size_t rows_to_print = std::min(max_rows, size_);
    
    for (size_t i = 0; i < rows_to_print; ++i) {
        std::cout << "  [" << std::setw(4) << i << "] ";
        
        switch (type_) {
            case DataType::INT32:
                std::cout << data<int32_t>()[i];
                break;
            case DataType::FLOAT32:
                std::cout << std::fixed << std::setprecision(2) 
                         << data<float>()[i];
                break;
            case DataType::STRING:
                std::cout << '"' << data<FixedString>()[i].data << '"';
                break;
        }
        std::cout << '\n';
    }
    
    if (size_ > max_rows) {
        std::cout << "  ... (" << (size_ - max_rows) << " more rows)\n";
    }
}

} 


