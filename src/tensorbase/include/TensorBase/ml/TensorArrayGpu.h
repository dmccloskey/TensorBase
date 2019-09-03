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

  /* Operators for Gpu classes
  */
  namespace TensorArrayComparisonGpu {
    template<typename TensorT>
    __host__ __device__ int compare(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return s1.at(i) - s2.at(i);
    }

    template<>
    __host__ __device__ int compare<char>(const TensorArray<char>& s1, const TensorArray<char>& s2, const int& size)
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
    __host__ __device__ bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) == 0) return true;
      else return false;
    }
  };

  struct isNotEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) != 0) return true;
      else return false;
    }
  };

  struct isLessThanGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) < 0) return true;
      else return false;
    }
  };

  struct isGreaterThanGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) > 0) return true;
      else return false;
    }
  };

  struct isLessThanOrEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) <= 0) return true;
      else return false;
    }
  };

  struct isGreaterThanOrEqualToGpu : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    __host__ __device__  bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonGpu::compare(s1, s2, this->size_) >= 0) return true;
      else return false;
    }
  };

  /**
    @brief Fixed length 8 vector class
  */
  template<typename TensorT>
  class TensorArray8Gpu : public TensorArray8<TensorT>
  {
  public:
    using TensorArray8<TensorT>::TensorArray8;
    bool operator==(const TensorArray& other) override;
    bool operator!=(const TensorArray& other) override;
    bool operator<(const TensorArray& other) override;
    bool operator<=(const TensorArray& other) override;
    bool operator>(const TensorArray& other) override;
    bool operator>=(const TensorArray& other) override;
    TensorT at(const int& i) const override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT>>(this));
    }
  };


  template<typename TensorT>
  inline __host__ __device__ TensorT TensorArray8Gpu<TensorT>::at(const int & i) const
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
  inline bool TensorArray8Gpu<TensorT>::operator==(const TensorArray<TensorT> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    isEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray8Gpu<TensorT>::operator!=(const TensorArray<TensorT> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    isNotEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray8Gpu<TensorT>::operator<(const TensorArray<TensorT> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray8Gpu<TensorT>::operator<=(const TensorArray<TensorT> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanOrEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray8Gpu<TensorT>::operator>(const TensorArray<TensorT> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanGpu comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray8Gpu<TensorT>::operator>=(const TensorArray<TensorT>& other)
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanOrEqualToGpu comp(this->array_size_);
    return comp(*this, other);
  }


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