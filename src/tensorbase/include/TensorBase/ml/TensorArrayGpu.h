/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORARRAYGPU_H
#define TENSORBASE_TENSORARRAYGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorArray.h>
#include <iostream> // << operator overload

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    NOTEs to future developers:

    1. Fixed length vector class has to be implemented without the use of base classes
      and a common set of comparison functors
    2. This means that each fixed length vector class needs to re-implement the following each time:
      - constructor/destructors
      - getters/setters
      - at operator
      - comparison operators
    3. In addition, an associated set of comparison functors must be created for each fixed length vector class

    For example: the following does not work:

  ```
  template<typename TensorT>
  class TensorArrayGpu
  {
  public:
    TensorArrayGpu() = default;
    virtual ~TensorArrayGpu() = default;

    /// operators are defined on a DeviceT-basis and executed on the specific DeviceT
    virtual __host__ __device__ bool operator==(const TensorArrayGpu& other) const = 0;
    virtual __host__ __device__ bool operator!=(const TensorArrayGpu& other) const = 0;
    virtual __host__ __device__ bool operator<(const TensorArrayGpu& other) const = 0;
    virtual __host__ __device__ bool operator<=(const TensorArrayGpu& other) const = 0;
    virtual __host__ __device__ bool operator>(const TensorArrayGpu& other) const = 0;
    virtual __host__ __device__ bool operator>=(const TensorArrayGpu& other) const = 0;

    __host__ __device__ size_t getArraySize() const { return array_size_; } ///< array_size getter

    virtual void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) = 0; ///< tensor_array setter
    virtual Eigen::Tensor<TensorT, 1> getTensorArray() = 0; ///< tensor_array getter
    virtual __host__ __device__ TensorT at(const int& i) const = 0; /// tensor_array accessor

  protected:
    size_t array_size_ = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_);
    }
  };

  namespace TensorArrayComparisonGpu {
    template<typename TensorT>
    __host__ __device__ int compare(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return s1.at(i) - s2.at(i);
    }

    template<>
    __host__ __device__ int compare<char>(const TensorArrayGpu<char>& s1, const TensorArrayGpu<char>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1.at(i) - (const unsigned char)s2.at(i);
    }
  };

  struct TensorArrayFunctorsGpu {
    __host__ __device__ TensorArrayFunctorsGpu(const int& size) : size_(size) {};
    int size_ = 0;
  };
  struct isEqualToGpu : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) == 0) return true;
      else return false;
    }
  };

  struct isNotEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) != 0) return true;
      else return false;
    }
  };

  struct isLessThanGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) < 0) return true;
      else return false;
    }
  };

  struct isGreaterThanGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) > 0) return true;
      else return false;
    }
  };

  struct isLessThanOrEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) <= 0) return true;
      else return false;
    }
  };

  struct isGreaterThanOrEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__  bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) >= 0) return true;
      else return false;
    }
  };

  template<typename TensorT>
  class TensorArrayGpu8: TensorArrayGpu<TensorT>
  {
  public:
    TensorArrayGpu8() = default;
    ~TensorArrayGpu8() = default;
    TensorArrayGpu8(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    __host__ __device__ bool operator==(const TensorArrayGpu8& other) const override;
    __host__ __device__ bool operator!=(const TensorArrayGpu8& other) const override;
    __host__ __device__ bool operator<(const TensorArrayGpu8& other) const override;
    __host__ __device__ bool operator<=(const TensorArrayGpu8& other) const override;
    __host__ __device__ bool operator>(const TensorArrayGpu8& other) const override;
    __host__ __device__ bool operator>=(const TensorArrayGpu8& other) const override;
    __host__ __device__ size_t getArraySize() const { return array_size_; } ///< array_size getter
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    Eigen::Tensor<TensorT, 1> getTensorArray() override;
    __host__ __device__ TensorT at(const int& i) const override;
  protected:
    size_t array_size_;
    TensorT item_0_ = TensorT(0);
    TensorT item_1_ = TensorT(0);
    TensorT item_2_ = TensorT(0);
    TensorT item_3_ = TensorT(0);
    TensorT item_4_ = TensorT(0);
    TensorT item_5_ = TensorT(0);
    TensorT item_6_ = TensorT(0);
    TensorT item_7_ = TensorT(0);
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT>>(this));
    }
  };
  ```

  Instead, each class needs to be implemented in the following manner:
    1. Fixed length class declaration
    2. Comparison functors specialized for the class
    3. Member functions for the fixed length class (that use the comparison functors)
  */

  /**
    @brief Fixed length 8 vector class
  */
  template<typename TensorT>
  class TensorArrayGpu8
  {
  public:
    TensorArrayGpu8() = default;
    ~TensorArrayGpu8() = default;
    TensorArrayGpu8(const std::string& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArrayGpu8(const std::initializer_list<TensorT>& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArrayGpu8(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    __host__ __device__ bool operator==(const TensorArrayGpu8& other) const;
    __host__ __device__ bool operator!=(const TensorArrayGpu8& other) const;
    __host__ __device__ bool operator<(const TensorArrayGpu8& other) const;
    __host__ __device__ bool operator<=(const TensorArrayGpu8& other) const;
    __host__ __device__ bool operator>(const TensorArrayGpu8& other) const;
    __host__ __device__ bool operator>=(const TensorArrayGpu8& other) const;
    __host__ __device__ size_t getArraySize() const { return array_size_; } ///< array_size getter
    void setTensorArray(const std::string& tensor_array);
    void setTensorArray(const std::initializer_list<TensorT>& tensor_array);
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array);
    Eigen::Tensor<TensorT, 1> getTensorArray();
    std::string getTensorArrayAsString() const; ///< tensor_array getter as a string
    __host__ __device__ TensorT at(const int& i) const;

    /// Inline << operator overload
    friend std::ostream& operator<<(std::ostream& os, const TensorArrayGpu8& data) {
      os << data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_;
      return os;
    }
  protected:
    size_t array_size_;
    TensorT item_0_ = TensorT(0);
    TensorT item_1_ = TensorT(0);
    TensorT item_2_ = TensorT(0);
    TensorT item_3_ = TensorT(0);
    TensorT item_4_ = TensorT(0);
    TensorT item_5_ = TensorT(0);
    TensorT item_6_ = TensorT(0);
    TensorT item_7_ = TensorT(0);
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_, 
        item_0_, item_1_, item_2_, item_3_, item_4_, item_5_, item_6_, item_7_);
    }
  };

  /* Operators for Gpu classes
  */
  namespace TensorArrayComparisonGpu {
    template<typename TensorT>
    __host__ __device__ int compare(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return s1.at(i) - s2.at(i);
    }

    template<>
    __host__ __device__ int compare<char>(const TensorArrayGpu8<char>& s1, const TensorArrayGpu8<char>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i) || s1.at(i) == '\0' || s2.at(i) == '\0') break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1.at(i) - (const unsigned char)s2.at(i);
    }
  };

  struct TensorArrayFunctorsGpu {
    __host__ __device__ TensorArrayFunctorsGpu(const int& size) : size_(size) {};
    int size_ = 0;
  };
  struct isEqualToGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) == 0) return true;
      else return false;
    }
  };

  struct isNotEqualToGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) != 0) return true;
      else return false;
    }
  };

  struct isLessThanGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) < 0) return true;
      else return false;
    }
  };

  struct isGreaterThanGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) > 0) return true;
      else return false;
    }
  };

  struct isLessThanOrEqualToGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) <= 0) return true;
      else return false;
    }
  };

  struct isGreaterThanOrEqualToGpu8 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__  bool operator()(const TensorArrayGpu8<TensorT>& s1, const TensorArrayGpu8<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) >= 0) return true;
      else return false;
    }
  };

  template<typename TensorT>
  inline void TensorArrayGpu8<TensorT>::setTensorArray(const std::string & tensor_array)
  {
    // check the array size
    assert(8 >= tensor_array.size());
    this->array_size_ = 8;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) {
      this->item_0_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_0_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_1_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_1_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_2_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_2_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_3_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_3_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_4_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_4_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_5_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_5_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_6_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_6_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_7_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_7_ = TensorT(0);
  }

  template<typename TensorT>
  inline void TensorArrayGpu8<TensorT>::setTensorArray(const std::initializer_list<TensorT>& tensor_array)
  {
    // check the array size
    assert(8 >= tensor_array.size());
    this->array_size_ = 8;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) {
      this->item_0_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_0_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_1_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_1_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_2_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_2_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_3_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_3_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_4_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_4_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_5_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_5_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_6_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_6_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_7_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_7_ = TensorT(0);
  }

  template<typename TensorT>
  inline void TensorArrayGpu8<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // check the array size
    assert(8 == tensor_array.dimension(0));
    this->array_size_ = 8;

    // copy the data
    this->item_0_ = tensor_array(0);
    this->item_1_ = tensor_array(1);
    this->item_2_ = tensor_array(2);
    this->item_3_ = tensor_array(3);
    this->item_4_ = tensor_array(4);
    this->item_5_ = tensor_array(5);
    this->item_6_ = tensor_array(6);
    this->item_7_ = tensor_array(7);
  }

  template<typename TensorT>
  inline Eigen::Tensor<TensorT, 1> TensorArrayGpu8<TensorT>::getTensorArray()
  {
    Eigen::Tensor<TensorT, 1> tensor_array(this->array_size_);
    tensor_array(0) = this->item_0_;
    tensor_array(1) = this->item_1_;
    tensor_array(2) = this->item_2_;
    tensor_array(3) = this->item_3_;
    tensor_array(4) = this->item_4_;
    tensor_array(5) = this->item_5_;
    tensor_array(6) = this->item_6_;
    tensor_array(7) = this->item_7_;
    return tensor_array;
  }

  template<typename TensorT>
  inline std::string TensorArrayGpu8<TensorT>::getTensorArrayAsString() const
  {
    std::ostringstream os;
    os << *this;
    return std::string(os.str());
  }

  template<typename TensorT>
  inline __host__ __device__ TensorT TensorArrayGpu8<TensorT>::at(const int & i) const
  {
    if (i == 0) return item_0_;
    else if (i == 1) return item_1_;
    else if (i == 2) return item_2_;
    else if (i == 3) return item_3_;
    else if (i == 4) return item_4_;
    else if (i == 5) return item_5_;
    else if (i == 6) return item_6_;
    else if (i == 7) return item_7_;
    else return TensorT(0);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator==(const TensorArrayGpu8<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isEqualToGpu8 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator!=(const TensorArrayGpu8<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isNotEqualToGpu8 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator<(const TensorArrayGpu8<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanGpu8 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator<=(const TensorArrayGpu8<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanOrEqualToGpu8 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator>(const TensorArrayGpu8<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanGpu8 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu8<TensorT>::operator>=(const TensorArrayGpu8<TensorT>& other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanOrEqualToGpu8 comp(this->array_size_);
    return comp(*this, other);
  }


  /**
    @brief Fixed length 32 vector class
  */
  template<typename TensorT>
  class TensorArrayGpu32
  {
  public:
    TensorArrayGpu32() = default;
    ~TensorArrayGpu32() = default;
    TensorArrayGpu32(const std::string& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArrayGpu32(const std::initializer_list<TensorT>& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArrayGpu32(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    __host__ __device__ bool operator==(const TensorArrayGpu32& other) const;
    __host__ __device__ bool operator!=(const TensorArrayGpu32& other) const;
    __host__ __device__ bool operator<(const TensorArrayGpu32& other) const;
    __host__ __device__ bool operator<=(const TensorArrayGpu32& other) const;
    __host__ __device__ bool operator>(const TensorArrayGpu32& other) const;
    __host__ __device__ bool operator>=(const TensorArrayGpu32& other) const;
    __host__ __device__ size_t getArraySize() const { return array_size_; } ///< array_size getter
    void setTensorArray(const std::string& tensor_array);
    void setTensorArray(const std::initializer_list<TensorT>& tensor_array);
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array);
    Eigen::Tensor<TensorT, 1> getTensorArray();
    std::string getTensorArrayAsString() const; ///< tensor_array getter as a string
    __host__ __device__ TensorT at(const int& i) const;

    /// Inline << operator overload
    friend std::ostream& operator<<(std::ostream& os, const TensorArrayGpu32& data) {
      os << data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_ << data.item_8_ << data.item_9_ << data.item_10_ << data.item_11_ << data.item_12_ << data.item_13_ << data.item_14_ << data.item_15_ << data.item_16_ << data.item_17_ << data.item_18_ << data.item_19_ << data.item_20_ << data.item_21_ << data.item_22_ << data.item_23_ << data.item_24_ << data.item_25_ << data.item_26_ << data.item_27_ << data.item_28_ << data.item_29_ << data.item_30_ << data.item_31_;
      return os;
    }
  protected:
    size_t array_size_;
    TensorT item_0_ = TensorT(0); TensorT item_1_ = TensorT(1); TensorT item_2_ = TensorT(2); TensorT item_3_ = TensorT(3); TensorT item_4_ = TensorT(4); TensorT item_5_ = TensorT(5); TensorT item_6_ = TensorT(6); TensorT item_7_ = TensorT(7); TensorT item_8_ = TensorT(8); TensorT item_9_ = TensorT(9); TensorT item_10_ = TensorT(10); TensorT item_11_ = TensorT(11); TensorT item_12_ = TensorT(12); TensorT item_13_ = TensorT(13); TensorT item_14_ = TensorT(14); TensorT item_15_ = TensorT(15); TensorT item_16_ = TensorT(16); TensorT item_17_ = TensorT(17); TensorT item_18_ = TensorT(18); TensorT item_19_ = TensorT(19); TensorT item_20_ = TensorT(20); TensorT item_21_ = TensorT(21); TensorT item_22_ = TensorT(22); TensorT item_23_ = TensorT(23); TensorT item_24_ = TensorT(24); TensorT item_25_ = TensorT(25); TensorT item_26_ = TensorT(26); TensorT item_27_ = TensorT(27); TensorT item_28_ = TensorT(28); TensorT item_29_ = TensorT(29); TensorT item_30_ = TensorT(30); TensorT item_31_ = TensorT(31);
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_,
        item_0_, item_1_, item_2_, item_3_, item_4_, item_5_, item_6_, item_7_,
        item_8_, item_9_, item_10_, item_11_, item_12_, item_13_, item_14_, item_15_, item_16_, item_17_, item_18_, item_19_, item_20_, item_21_, item_22_, item_23_, item_24_, item_25_, item_26_, item_27_, item_28_, item_29_, item_30_, item_31_
      );
    }
  };

  /* Operators for Gpu classes
  */
  namespace TensorArrayComparisonGpu {
    template<typename TensorT>
    __host__ __device__ int compare(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return s1.at(i) - s2.at(i);
    }

    template<>
    __host__ __device__ int compare<char>(const TensorArrayGpu32<char>& s1, const TensorArrayGpu32<char>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i) || s1.at(i) == '\0' || s2.at(i) == '\0') break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1.at(i) - (const unsigned char)s2.at(i);
    }
  };

  struct isEqualToGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) == 0) return true;
      else return false;
    }
  };

  struct isNotEqualToGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) != 0) return true;
      else return false;
    }
  };

  struct isLessThanGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) < 0) return true;
      else return false;
    }
  };

  struct isGreaterThanGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) > 0) return true;
      else return false;
    }
  };

  struct isLessThanOrEqualToGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) <= 0) return true;
      else return false;
    }
  };

  struct isGreaterThanOrEqualToGpu32 : TensorArrayFunctorsGpu {
    using TensorArrayFunctorsGpu::TensorArrayFunctorsGpu;
    template<typename TensorT>
    __host__ __device__  bool operator()(const TensorArrayGpu32<TensorT>& s1, const TensorArrayGpu32<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) >= 0) return true;
      else return false;
    }
  };

  template<typename TensorT>
  inline void TensorArrayGpu32<TensorT>::setTensorArray(const std::string & tensor_array)
  {
    // check the array size
    assert(32 >= tensor_array.size());
    this->array_size_ = 32;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) { this->item_0_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_0_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_1_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_1_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_2_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_2_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_3_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_3_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_4_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_4_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_5_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_5_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_6_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_6_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_7_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_7_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_8_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_8_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_9_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_9_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_10_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_10_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_11_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_11_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_12_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_12_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_13_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_13_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_14_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_14_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_15_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_15_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_16_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_16_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_17_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_17_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_18_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_18_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_19_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_19_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_20_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_20_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_21_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_21_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_22_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_22_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_23_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_23_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_24_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_24_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_25_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_25_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_26_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_26_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_27_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_27_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_28_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_28_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_29_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_29_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_30_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_30_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_31_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_31_ = TensorT(0); }
  }

  template<typename TensorT>
  inline void TensorArrayGpu32<TensorT>::setTensorArray(const std::initializer_list<TensorT>& tensor_array)
  {
    // check the array size
    assert(32 >= tensor_array.size());
    this->array_size_ = 32;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) { this->item_0_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_0_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_1_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_1_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_2_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_2_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_3_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_3_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_4_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_4_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_5_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_5_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_6_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_6_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_7_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_7_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_8_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_8_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_9_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_9_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_10_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_10_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_11_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_11_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_12_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_12_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_13_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_13_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_14_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_14_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_15_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_15_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_16_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_16_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_17_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_17_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_18_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_18_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_19_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_19_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_20_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_20_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_21_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_21_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_22_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_22_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_23_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_23_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_24_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_24_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_25_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_25_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_26_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_26_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_27_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_27_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_28_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_28_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_29_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_29_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_30_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_30_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_31_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_31_ = TensorT(0); }
  }

  template<typename TensorT>
  inline void TensorArrayGpu32<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // check the array size
    assert(32 == tensor_array.dimension(0));
    this->array_size_ = 32;

    // copy the data
    this->item_0_ = tensor_array(0);
    this->item_1_ = tensor_array(1);
    this->item_2_ = tensor_array(2);
    this->item_3_ = tensor_array(3);
    this->item_4_ = tensor_array(4);
    this->item_5_ = tensor_array(5);
    this->item_6_ = tensor_array(6);
    this->item_7_ = tensor_array(7);
    this->item_8_ = tensor_array(8);
    this->item_9_ = tensor_array(9);
    this->item_10_ = tensor_array(10);
    this->item_11_ = tensor_array(11);
    this->item_12_ = tensor_array(12);
    this->item_13_ = tensor_array(13);
    this->item_14_ = tensor_array(14);
    this->item_15_ = tensor_array(15);
    this->item_16_ = tensor_array(16);
    this->item_17_ = tensor_array(17);
    this->item_18_ = tensor_array(18);
    this->item_19_ = tensor_array(19);
    this->item_20_ = tensor_array(20);
    this->item_21_ = tensor_array(21);
    this->item_22_ = tensor_array(22);
    this->item_23_ = tensor_array(23);
    this->item_24_ = tensor_array(24);
    this->item_25_ = tensor_array(25);
    this->item_26_ = tensor_array(26);
    this->item_27_ = tensor_array(27);
    this->item_28_ = tensor_array(28);
    this->item_29_ = tensor_array(29);
    this->item_30_ = tensor_array(30);
    this->item_31_ = tensor_array(31);
  }

  template<typename TensorT>
  inline Eigen::Tensor<TensorT, 1> TensorArrayGpu32<TensorT>::getTensorArray()
  {
    Eigen::Tensor<TensorT, 1> tensor_array(this->array_size_);
    tensor_array(0) = this->item_0_;
    tensor_array(1) = this->item_1_;
    tensor_array(2) = this->item_2_;
    tensor_array(3) = this->item_3_;
    tensor_array(4) = this->item_4_;
    tensor_array(5) = this->item_5_;
    tensor_array(6) = this->item_6_;
    tensor_array(7) = this->item_7_;
    tensor_array(8) = this->item_8_;
    tensor_array(9) = this->item_9_;
    tensor_array(10) = this->item_10_;
    tensor_array(11) = this->item_11_;
    tensor_array(12) = this->item_12_;
    tensor_array(13) = this->item_13_;
    tensor_array(14) = this->item_14_;
    tensor_array(15) = this->item_15_;
    tensor_array(16) = this->item_16_;
    tensor_array(17) = this->item_17_;
    tensor_array(18) = this->item_18_;
    tensor_array(19) = this->item_19_;
    tensor_array(20) = this->item_20_;
    tensor_array(21) = this->item_21_;
    tensor_array(22) = this->item_22_;
    tensor_array(23) = this->item_23_;
    tensor_array(24) = this->item_24_;
    tensor_array(25) = this->item_25_;
    tensor_array(26) = this->item_26_;
    tensor_array(27) = this->item_27_;
    tensor_array(28) = this->item_28_;
    tensor_array(29) = this->item_29_;
    tensor_array(30) = this->item_30_;
    tensor_array(31) = this->item_31_;
    return tensor_array;
  }

  template<typename TensorT>
  inline std::string TensorArrayGpu32<TensorT>::getTensorArrayAsString() const
  {
    std::ostringstream os;
    os << *this;
    return std::string(os.str());
  }

  template<typename TensorT>
  inline __host__ __device__ TensorT TensorArrayGpu32<TensorT>::at(const int & i) const
  {
    if (i == 0) return item_0_;
    else if (i == 1) return item_1_;
    else if (i == 2) return item_2_;
    else if (i == 3) return item_3_;
    else if (i == 4) return item_4_;
    else if (i == 5) return item_5_;
    else if (i == 6) return item_6_;
    else if (i == 7) return item_7_;
    else if (i == 8) return item_8_;
    else if (i == 9) return item_9_;
    else if (i == 10) return item_10_;
    else if (i == 11) return item_11_;
    else if (i == 12) return item_12_;
    else if (i == 13) return item_13_;
    else if (i == 14) return item_14_;
    else if (i == 15) return item_15_;
    else if (i == 16) return item_16_;
    else if (i == 17) return item_17_;
    else if (i == 18) return item_18_;
    else if (i == 19) return item_19_;
    else if (i == 20) return item_20_;
    else if (i == 21) return item_21_;
    else if (i == 22) return item_22_;
    else if (i == 23) return item_23_;
    else if (i == 24) return item_24_;
    else if (i == 25) return item_25_;
    else if (i == 26) return item_26_;
    else if (i == 27) return item_27_;
    else if (i == 28) return item_28_;
    else if (i == 29) return item_29_;
    else if (i == 30) return item_30_;
    else if (i == 31) return item_31_;
    else return TensorT(0);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator==(const TensorArrayGpu32<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isEqualToGpu32 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator!=(const TensorArrayGpu32<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isNotEqualToGpu32 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator<(const TensorArrayGpu32<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanGpu32 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator<=(const TensorArrayGpu32<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanOrEqualToGpu32 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator>(const TensorArrayGpu32<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanGpu32 comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline __host__ __device__ bool TensorArrayGpu32<TensorT>::operator>=(const TensorArrayGpu32<TensorT>& other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanOrEqualToGpu32 comp(this->array_size_);
    return comp(*this, other);
  }
};
#endif
#endif //TENSORBASE_TENSORARRAYGPU_H