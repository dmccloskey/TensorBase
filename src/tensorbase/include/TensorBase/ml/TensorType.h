/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTYPE_H
#define TENSORBASE_TENSORTYPE_H

#include <climits> // `CHAR_BIT`
#include <cstdint>
#include <string>
#include <type_traits>  // `is_signed`

#include <unsupported/Eigen/CXX11/Tensor>

namespace TensorBase
{
  /* Classes and structures for all data types used

  References:
  Primitive type wrappers using the following minimal wrapper structure
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
  */

  struct Type {
    /// \brief Main data type enumeration
    ///
    /// This enumeration provides a quick way to interrogate the category
    /// of a TensorType instance.
    enum type {
      /// A NULL type having no physical storage
      NA,
      /// Boolean as 1 bit, LSB bit-packed ordering
      BOOL,
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
  template<typename BASE, Type::type TYPE_ID, typename C_TYPE>
  class PrimitiveCType: public BASE
  {
    C_TYPE value_t_;
  public:
    using c_type = C_TYPE;
    constexpr PrimitiveCType() noexcept : value_t_() {}
    template<typename U>
    constexpr PrimitiveCType(U const& value) noexcept : value_t_(value) {}
    operator C_TYPE&() { return value_t_; }  ///< minimal operator overload needed
  };

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

  /// Concrete type class for always-null data
  class NullType : public TensorType {
  public:
    Type::type getId() const { return Type::NA; }
    std::string getName() const override { return "null"; }
  };

  /// Concrete type class for boolean data
  class BooleanType : public PrimitiveCType<TensorType, Type::BOOL, bool> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::BOOL; }
    std::string getName() const override { return "bool"; }
  };

  /// Concrete type class for unsigned 8-bit integer data
  class UInt8Type : public PrimitiveCType<TensorType, Type::UINT8, uint8_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::UINT8; }
    std::string getName() const override { return "uint8"; }
  };

  /// Concrete type class for signed 8-bit integer data
  class Int8Type : public PrimitiveCType<TensorType, Type::INT8, int8_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; }
    std::string getName() const override { return "int8"; }
  };

  /// Concrete type class for unsigned 16-bit integer data
  class UInt16Type : public PrimitiveCType<TensorType, Type::UINT16, uint16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "uint16"; }
  };

  /// Concrete type class for signed 16-bit integer data
  class Int16Type : public PrimitiveCType<TensorType, Type::INT16, int16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "int16"; }
  };

  /// Concrete type class for unsigned 32-bit integer data
  class UInt32Type : public PrimitiveCType<TensorType, Type::UINT32, uint32_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "uint32"; }
  };

  /// Concrete type class for signed 32-bit integer data
  class Int32Type : public PrimitiveCType<TensorType, Type::INT32, int32_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "int32"; }
  };

  /// Concrete type class for unsigned 64-bit integer data
  class UInt64Type : public PrimitiveCType<TensorType, Type::UINT64, uint64_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "uint64"; }
  };

  /// Concrete type class for signed 64-bit integer data
  class Int64Type : public PrimitiveCType<TensorType, Type::INT64, int64_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "int64"; }
  };

  /// Concrete type class for 16-bit floating-point data
  class HalfFloatType : public PrimitiveCType<TensorType, Type::HALF_FLOAT, uint16_t> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "halffloat"; }
  };

  /// Concrete type class for 32-bit floating-point data (C "float")
  class FloatType : public PrimitiveCType<TensorType, Type::FLOAT, float> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "float"; }
  };

  /// Concrete type class for 64-bit floating-point data (C "double")
  class DoubleType : public PrimitiveCType<TensorType, Type::DOUBLE, double> {
  public:
    using PrimitiveCType::PrimitiveCType;
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "double"; }
  };

  /// @brief Concrete type class for variable-size binary data
  class BinaryType : public TensorType {
  public:
    static constexpr Type::type type_id = Type::BINARY;
    BinaryType() : BinaryType(Type::BINARY) {}
    std::string getName() const override { return "binary"; }
  protected:
    // Allow subclasses to change the logical type.
    explicit BinaryType(Type::type logical_type) : TensorType(logical_type) {}
  };
  
  /// @brief Concrete type class for fixed-size binary data
  class FixedSizeBinaryType : public TensorType {
  public:
    static constexpr Type::type type_id = Type::FIXED_SIZE_BINARY;
    explicit FixedSizeBinaryType(int32_t byte_width): TensorType(Type::FIXED_SIZE_BINARY), byte_width_(byte_width) {}
    explicit FixedSizeBinaryType(int32_t byte_width, Type::type override_type_id): TensorType(override_type_id), byte_width_(byte_width) {}
    std::string getName() const override { return "fixed_size_binary"; }
    int32_t byte_width() const { return byte_width_; }
  protected:
    int32_t byte_width_;
  };

  /// @brief Concrete type class for variable-size string data, utf8-encoded
  class StringType : public BinaryType {
  public:
    static constexpr Type::type type_id = Type::STRING;
    StringType() : BinaryType(Type::STRING) {}
    std::string getName() const override { return "utf8"; }
  };

  enum class DateUnit : char { DAY = 0, MILLI = 1 };

  /// @brief Base type class for date data
  class DateType : public TensorType {
  public:
    virtual DateUnit unit() const = 0;
  protected:
    explicit DateType(Type::type type_id);
  };

  /// Concrete type class for 32-bit date data (as number of days since UNIX epoch)
  class Date32Type : public DateType {
  public:
    static constexpr Type::type type_id = Type::DATE32;
    static constexpr DateUnit UNIT = DateUnit::DAY;
    using c_type = int32_t;
    Date32Type();
    std::string getName() const override { return "date32"; }
    Type::type getId() const { return Type::INT8; } // FIXME
    DateUnit unit() const override { return UNIT; }
  };

  /// Concrete type class for 64-bit date data (as number of milliseconds since UNIX epoch)
  class Date64Type : public DateType {
  public:
    static constexpr Type::type type_id = Type::DATE64;
    static constexpr DateUnit UNIT = DateUnit::MILLI;
    using c_type = int64_t;
    Date64Type();
    std::string getName() const override { return "date64"; }
    Type::type getId() const { return Type::INT8; } // FIXME
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
    TimeUnit::type unit() const { return unit_; }
    Type::type getId() const { return Type::INT8; } // FIXME
  protected:
    TimeType(Type::type type_id, TimeUnit::type unit);
    TimeUnit::type unit_;
  };

  class Time32Type : public TimeType {
  public:
    static constexpr Type::type type_id = Type::TIME32;
    using c_type = int32_t;
    explicit Time32Type(TimeUnit::type unit = TimeUnit::MILLI);
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "time32"; }
  };

  class Time64Type : public TimeType {
  public:
    static constexpr Type::type type_id = Type::TIME64;
    using c_type = int64_t;
    explicit Time64Type(TimeUnit::type unit = TimeUnit::MILLI);
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "time64"; }
  };

  class TimestampType : public TensorType {
  public:
    using Unit = TimeUnit;
    typedef int64_t c_type;
    static constexpr Type::type type_id = Type::TIMESTAMP;
    explicit TimestampType(TimeUnit::type unit = TimeUnit::MILLI): TensorType(Type::TIMESTAMP), unit_(unit) {}
    explicit TimestampType(TimeUnit::type unit, const std::string& timezone): TensorType(Type::TIMESTAMP), unit_(unit), timezone_(timezone) {}
    Type::type getId() const { return Type::INT8; } // FIXME
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
    static constexpr Type::type type_id = Type::INTERVAL;
    explicit IntervalType(Unit unit = Unit::YEAR_MONTH): TensorType(Type::INTERVAL), unit_(unit) {}
    Type::type getId() const { return Type::INT8; } // FIXME
    std::string getName() const override { return "interval"; }
    Unit unit() const { return unit_; }
  private:
    Unit unit_;
  };
};
#endif //TENSORBASE_TENSORTYPE_H