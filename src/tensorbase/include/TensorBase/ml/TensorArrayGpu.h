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
#include <TensorBase/ml/TensorDataGpu.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{

  /**
    @brief Base class for fixed length vector types on the Gpu
  */
  template<typename TensorT>
  class TensorArrayGpu
  {
  public:
    TensorArrayGpu() = default;
    virtual ~TensorArrayGpu() = default;

    /// operators are defined on a DeviceT-basis and executed on the specific DeviceT
    virtual bool operator==(const TensorArrayGpu& other) const = 0;
    virtual bool operator!=(const TensorArrayGpu& other) const = 0;
    virtual bool operator<(const TensorArrayGpu& other) const = 0;
    virtual bool operator<=(const TensorArrayGpu& other) const = 0;
    virtual bool operator>(const TensorArrayGpu& other) const = 0;
    virtual bool operator>=(const TensorArrayGpu& other) const = 0;

    size_t getArraySize() const { return array_size_; } ///< array_size getter

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

  /* Operators for Gpu classes
  */
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

  // Operators for Gpu classes
  struct isEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
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

  /**
    @brief Fixed length 8 vector class
  */
  template<typename TensorT>
  class TensorArrayGpu8 : public TensorArrayGpu<TensorT>
  {
  public:
    TensorArrayGpu8() = default;
    ~TensorArrayGpu8() = default;
    TensorArrayGpu8(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    bool operator==(const TensorArrayGpu& other) const override;
    bool operator!=(const TensorArrayGpu& other) const override;
    bool operator<(const TensorArrayGpu& other) const override;
    bool operator<=(const TensorArrayGpu& other) const override;
    bool operator>(const TensorArrayGpu& other) const override;
    bool operator>=(const TensorArrayGpu& other) const override;
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    Eigen::Tensor<TensorT, 1> getTensorArray() override;
    __host__ __device__ TensorT at(const int& i) const;
  protected:
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
  inline bool TensorArrayGpu8<TensorT>::operator==(const TensorArrayGpu<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu8<TensorT>::operator!=(const TensorArrayGpu<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isNotEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu8<TensorT>::operator<(const TensorArrayGpu<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu8<TensorT>::operator<=(const TensorArrayGpu<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanOrEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu8<TensorT>::operator>(const TensorArrayGpu<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu8<TensorT>::operator>=(const TensorArrayGpu<TensorT>& other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanOrEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  class Point {
  public:
    Point() = default;
    Point(float x, float y) : x(x), y(y) {};
    float x;
    float y;
    __host__ __device__ bool operator==(const Point& other) const {
      if (this->x == other.x && this->y == other.y) return true;
      else return false;
    }
  };

  struct compOp {
    template<typename T>
    __host__ __device__ bool operator()(const T& lhs, const T& rhs) {
      if (lhs.x < rhs.x) return true;
      else return false;
    }
  };
};
#endif
#endif //TENSORBASE_TENSORARRAYGPU_H