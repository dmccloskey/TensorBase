/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTYPE_H
#define TENSORBASE_TENSORTYPE_H

#include <cstdint>
#include <string>
#include <type_traits>  // `is_signed`
#include <ostream>

namespace TensorBase
{
  /* Classes and structures for all data types used

  References:
  Primitive type wrappers using the following minimal wrapper structure with additional operator overloads
  ```
    template<typename T>
    class primitive
    {
      T value_t_;
    public:
      using value_type = T;
      constexpr primitive() noexcept : value_t_() {}
      template<typename U>
      constexpr primitive(U const& value) noexcept : value_t_(value) {}
      operator T&() { return value_t_; }
    };
  ```
  was inspired by https://github.com/jehugaleahsa/primitive and https://stackoverflow.com/questions/24213743/c-using-primitive-types-as-a-base-class?rq=1

  Type inheritance, list/map/dict types, and date/time class structures were inspired by the Apache Arrow types  
  https://github.com/apache/arrow/blob/master/cpp/src/arrow/type.h  

  STATUS and TODOS:
  - implement all operator overloads in `PrimitiveCType`
  - implement both CPU and GPU versions of `PrimitiveCType`s
  - implement `sizeof()` in TensorType and all derived types
  - clean up date, time, and interval classes
  */

  struct Type {
    /// @brief Main data type enumeration
    ///
    /// This enumeration provides a quick way to interrogate the category
    /// of a TensorType instance.
    enum type {
      /// A NULL type having no physical storage
      NA,
      /// Boolean as 1 bit, LSB bit-packed ordering
      BOOL,
      /// Character as 1 bit, ASCII
      CHAR,
      /// Unsigned 8-bit little-endian integer
      UINT8,
      /// Signed 8-bit little-endian integer
      INT8,
      /// Unsigned 16-bit little-endian integer
      UINT16,
      /// Signed 16-bit little-endian integer
      INT16,
      /// Unsigned 32-bit little-endian integer
      UINT32,
      /// Signed 32-bit little-endian integer
      INT32,
      /// Unsigned 64-bit little-endian integer
      UINT64,
      /// Signed 64-bit little-endian integer
      INT64,
      /// 2-byte floating point value
      HALF_FLOAT,
      /// 4-byte floating point value
      FLOAT,
      /// 8-byte floating point value
      DOUBLE,
      /// 16-byte floating point value
      LONG_DOUBLE,
      /// UTF8 variable-length string as List<Char>
      STRING,
      /// Variable-length bytes (no guarantee of UTF8-ness)
      BINARY,
      /// Fixed-size binary. Each value occupies the same number of bytes
      FIXED_SIZE_BINARY,
      /// int32_t days since the UNIX epoch
      DATE32,
      /// int64_t milliseconds since the UNIX epoch
      DATE64,
      /// Exact timestamp encoded with int64 since UNIX epoch
      /// Default unit millisecond
      TIMESTAMP,
      /// Time as signed 32-bit integer, representing either seconds or
      /// milliseconds since midnight
      TIME32,
      /// Time as signed 64-bit integer, representing either microseconds or
      /// nanoseconds since midnight
      TIME64,
      /// YEAR_MONTH or DAY_TIME interval in SQL style
      INTERVAL,
      /// Precision- and scale-based decimal type. Storage type depends on the
      /// parameters. (Not implemented)
      DECIMAL,
      /// A list of some logical data type (Not implemented)
      LIST,
      /// Struct of logical types (Not implemented)
      STRUCT,
      /// Unions of logical types (Not implemented)
      UNION,
      /// Dictionary aka Category type (Not implemented)
      DICTIONARY,
      /// Map, a repeated struct logical type (Not implemented)
      MAP,
      /// Custom data type, implemented by user
      EXTENSION
    };
  };

  /**
    @brief Abstract Base class for all Tensor Types
  */
  class TensorType
  {
  public:
    TensorType() = default;
    virtual ~TensorType() = default; 
    virtual Type::type getId() const = 0; ///< id getter
    virtual std::string getName() const = 0; ///< name getter
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_);
    //	}
  };

  /// @brief Base class for all data types representing primitive values
  template<typename C_TYPE>
  class PrimitiveCType: public TensorType
  {
    C_TYPE value_t_;
  public:
    using c_type = C_TYPE;
    constexpr PrimitiveCType() noexcept : value_t_() {}
    template<typename U>
    constexpr PrimitiveCType(U const& value) noexcept : value_t_(value) {}
    operator C_TYPE&() { return value_t_; }  ///< minimal operator overload needed
    constexpr C_TYPE const& get() const noexcept { return value_t_; }

    // TODO: implement all overloads
  };

  // TODO: implement all overloads

  /// Concrete type class for always-null data
  class NullType : public TensorType {
  public:
    Type::type getId() const { return Type::NA; }
    std::string getName() const override { return "null"; }
  };

  /// Concrete type class for boolean data
  class BooleanType : public PrimitiveCType<bool> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::BOOL; }
    std::string getName() const override { return "bool"; }
  };

  /// Concrete type class for character data
  class CharType : public PrimitiveCType<char> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::CHAR; }
    std::string getName() const override { return "char"; }
  };

  /// Concrete type class for unsigned 8-bit integer data
  class UInt8Type : public PrimitiveCType<uint8_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::UINT8; }
    std::string getName() const override { return "uint8"; }
  };

  /// Concrete type class for signed 8-bit integer data
  class Int8Type : public PrimitiveCType<int8_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; }
    std::string getName() const override { return "int8"; }
  };

  /// Concrete type class for unsigned 16-bit integer data
  class UInt16Type : public PrimitiveCType<uint16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::UINT16; }
    std::string getName() const override { return "uint16"; }
  };

  /// Concrete type class for signed 16-bit integer data
  class Int16Type : public PrimitiveCType<int16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT16; }
    std::string getName() const override { return "int16"; }
  };

  /// Concrete type class for unsigned 32-bit integer data
  class UInt32Type : public PrimitiveCType<uint32_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::UINT32; }
    std::string getName() const override { return "uint32"; }
  };

  /// Concrete type class for signed 32-bit integer data
  class Int32Type : public PrimitiveCType<int32_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT32; }
    std::string getName() const override { return "int32"; }
  };

  /// Concrete type class for unsigned 64-bit integer data
  class UInt64Type : public PrimitiveCType<uint64_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::UINT64; }
    std::string getName() const override { return "uint64"; }
  };

  /// Concrete type class for signed 64-bit integer data
  class Int64Type : public PrimitiveCType<int64_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT64; } 
    std::string getName() const override { return "int64"; }
  };

  /// Concrete type class for 16-bit floating-point data
  class HalfFloatType : public PrimitiveCType<uint16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::HALF_FLOAT; } 
    std::string getName() const override { return "halffloat"; }
  };

  /// Concrete type class for 32-bit floating-point data (C "float")
  class FloatType : public PrimitiveCType<float> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::FLOAT; } 
    std::string getName() const override { return "float"; }
  };

  /// Concrete type class for 64-bit floating-point data (C "double")
  class DoubleType : public PrimitiveCType<double> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::DOUBLE; } 
    std::string getName() const override { return "double"; }
  };

  /// Concrete type class for 64-bit floating-point data with extra precision (C "long double")
  class LongDoubleType : public PrimitiveCType<long double> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::LONG_DOUBLE; }
    std::string getName() const override { return "long_double"; }
  };

  /// @brief Concrete type class for variable-size binary data
  class BinaryType : public TensorType {
  public:
    Type::type getId() const { return Type::BINARY; }
    std::string getName() const override { return "binary"; }
  };
  
  /// @brief Concrete type class for fixed-size binary data
  class FixedSizeBinaryType : public TensorType {
  public:
    explicit FixedSizeBinaryType(int32_t byte_width): byte_width_(byte_width) {}
    std::string getName() const override { return "fixed_size_binary"; }
    Type::type getId() const { return Type::FIXED_SIZE_BINARY; }
    int32_t byte_width() const { return byte_width_; }
  protected:
    int32_t byte_width_;
  };

  /// @brief Concrete type class for variable-size string data, utf8-encoded
  class StringType : public BinaryType {
  public:
    using BinaryType::BinaryType;
    Type::type getId() const { return Type::STRING; }
    std::string getName() const override { return "utf8"; }
  };

  enum class DateUnit : char { DAY = 0, MILLI = 1 };

  /// @brief Base type class for date data
  class DateType : public TensorType {
  public:
    virtual DateUnit unit() const = 0;
  };

  /// Concrete type class for 32-bit date data (as number of days since UNIX epoch)
  class Date32Type : public DateType {
  public:
    static constexpr DateUnit UNIT = DateUnit::DAY;
    using c_type = int32_t;
    std::string getName() const override { return "date32"; }
    Type::type getId() const { return Type::DATE32; }
    DateUnit unit() const override { return UNIT; }
  };

  /// Concrete type class for 64-bit date data (as number of milliseconds since UNIX epoch)
  class Date64Type : public DateType {
  public:
    static constexpr DateUnit UNIT = DateUnit::MILLI;
    using c_type = int64_t;
    std::string getName() const override { return "date64"; }
    Type::type getId() const { return Type::DATE64; }
    DateUnit unit() const override { return UNIT; }
  };

  struct TimeUnit {
    /// The unit for a time or timestamp TensorType
    enum type { SECOND = 0, MILLI = 1, MICRO = 2, NANO = 3 };
  };

  static inline std::ostream& operator<<(std::ostream& os, TimeUnit::type unit) {
    switch (unit) {
    case TimeUnit::SECOND:
      os << "s";
      break;
    case TimeUnit::MILLI:
      os << "ms";
      break;
    case TimeUnit::MICRO:
      os << "us";
      break;
    case TimeUnit::NANO:
      os << "ns";
      break;
    }
    return os;
  }

  /// Base type class for time data
  class TimeType : public TensorType {
  public:
    TimeType(TimeUnit::type unit) : unit_(unit) {};
    TimeType() : unit_(TimeUnit::MILLI) {};
    TimeUnit::type unit() const { return unit_; }
  private:
    TimeUnit::type unit_;
  };

  class Time32Type : public TimeType {
  public:
    using c_type = int32_t;
    Type::type getId() const { return Type::TIME32; }
    std::string getName() const override { return "time32"; }
  };

  class Time64Type : public TimeType {
  public:
    using c_type = int64_t;
    Type::type getId() const { return Type::TIME64; }
    std::string getName() const override { return "time64"; }
  };

  class TimestampType : public TensorType {
  public:
    using Unit = TimeUnit;
    typedef int64_t c_type;
    explicit TimestampType(TimeUnit::type unit = TimeUnit::MILLI): unit_(unit) {}
    explicit TimestampType(TimeUnit::type unit, const std::string& timezone): unit_(unit), timezone_(timezone) {}
    Type::type getId() const { return Type::TIMESTAMP; }
    std::string getName() const override { return "timestamp"; }
    TimeUnit::type unit() const { return unit_; }
    const std::string& timezone() const { return timezone_; }
  private:
    TimeUnit::type unit_;
    std::string timezone_;
  };

  class IntervalType : public TensorType {
  public:
    enum class Unit : char { YEAR_MONTH = 0, DAY_TIME = 1 };
    using c_type = int64_t;
    explicit IntervalType(Unit unit = Unit::YEAR_MONTH): unit_(unit) {}
    Type::type getId() const { return Type::INTERVAL; }
    std::string getName() const override { return "interval"; }
    Unit unit() const { return unit_; }
  private:
    Unit unit_;
  };
};
#endif //TENSORBASE_TENSORTYPE_H