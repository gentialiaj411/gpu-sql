#pragma once
#include <cstdint>
#include <string>
#include <variant>
#include <stdexcept>

namespace gpusql{


enum class DataType {
    INT32,      //32-bit signed int
    FLOAT32,    //32-bit float
    STRING      //32 char string
};

constexpr size_t STRING_MAX_LENGTH = 32;

struct FixedString {
    char data[STRING_MAX_LENGTH];
    FixedString() {data[0] = '\0';}

    explicit FixedString(const std::string& s) {
        size_t len = std::min(s.length(), STRING_MAX_LENGTH - 1);
        data[len] = '\0';
    }

    explicit FixedString(const char* s) {
        size_t i = 0;
        while (s[i] && i < STRING_MAX_LENGTH - 1) {
            data[i] = s[i];
            ++i;
        }
        data[i] = '\0';
    }

    std::string to_string() const { return std::string(data); }

    bool operator == (const FixedString& other) const {
        for (size_t i = 0; i < STRING_MAX_LENGTH; ++i) {
            if (data[i] != other.data[i]) return false;
            if (data[i] == '\0') break;
        }
        return true;
    }
    
    bool operator != (const FixedString& other) const{ 
        return !(*this == other);
    }
};


inline size_t type_size(DataType type) {
    switch (type) {
        case DataType::INT32: return sizeof(int32_t);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::STRING: return sizeof(FixedString);
    }
    throw std::runtime_error("Unkown data type");
}


inline std::string type_name(DataType type) {
    switch (type) {
        case DataType::INT32: return "INT32";
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::STRING: return "STRING";
    }
    return "UNKOWN";
}

//comparison symbols (==, !=, <, <=, >, >=)
enum class CompareOp {
    EQ, 
    NE,
    LT,
    LE,
    GT,
    GE,
};

inline std::string op_to_string(CompareOp op) {
    switch(op) {
        case CompareOp::EQ: return "=";
        case CompareOp::NE: return "!=";
        case CompareOp::LT: return "<";
        case CompareOp::LE: return "<=";
        case CompareOp::GT: return ">";
        case CompareOp::GE: return ">=";
    }
    return "?";
}

enum class AggregateOp {
    COUNT, 
    SUM,
    MIN,
    MAX,
    AVG
};

inline std::string agg_to_string(AggregateOp op) {
    switch (op) {
        case AggregateOp::COUNT: return "COUNT";
        case AggregateOp::SUM: return "SUM";
        case AggregateOp::MIN: return "MIN";
        case AggregateOp::MAX: return "MAX";
        case AggregateOp::AVG: return "AVG";
    }
    return "?";
}
}
