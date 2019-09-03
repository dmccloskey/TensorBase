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
  /* Operators for the Gpu
  */
  namespace TensorArrayOperatorsGpu {
    template<typename TensorT>
    __host__ __device__ int compare(const TensorT * s1, const TensorT * s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1[i] != s2[i]) break;
        if (i == size - 1) return 0;
      }
      return s1[i] - s2[i];
    }

    template<typename TensorT>
    __host__ __device__ bool isEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) == 0) return true;
      else return false;
    }

    template<typename TensorT>
    __host__ __device__ bool isNotEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) != 0) return true;
      else return false;
    }

    template<typename TensorT>
    __host__ __device__ bool isLessThan(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) < 0) return true;
      else return false;
    }

    template<typename TensorT>
    __host__ __device__ bool isGreaterThan(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) > 0) return true;
      else return false;
    }

    template<typename TensorT>
    __host__ __device__ bool isLessThanOrEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) <= 0) return true;
      else return false;
    }

    template<typename TensorT>
    __host__ __device__ bool isGreaterThanOrEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) >= 0) return true;
      else return false;
    }

    template<>
    __host__ __device__ int compare<char>(const char * s1, const char * s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1[i] != s2[i]) break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1[i] - (const unsigned char)s2[i];
    }
  };

  template<typename TensorT>
  class TensorArrayGpu : public TensorArray<TensorT, Eigen::GpuDevice>
  {
  public:
    TensorArrayGpu() = default;  ///< Default constructor
    TensorArrayGpu(const Eigen::Tensor<TensorT, 1>& tensor_array);
    ~TensorArrayGpu() = default; ///< Default destructor
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    bool operator==(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
    bool operator!=(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
    bool operator<(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
    bool operator<=(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
    bool operator>(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
    bool operator>=(const TensorArray<TensorT, Eigen::GpuDevice>& other) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT, Eigen::GpuDevice>>(this));
    }
  };

  template<typename TensorT>
  inline TensorArrayGpu<TensorT>::TensorArrayGpu(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    this->setTensorArray(tensor_array);
  }

  template<typename TensorT>
  inline void TensorArrayGpu<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // set the array size
    this->array_size_ = tensor_array.dimension(0);
    // copy the data

    this->tensor_array_ = new TensorT[this->array_size_];
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_copy(this->tensor_array_, this->array_size_);
    data_copy = tensor_array;

    //this->tensor_array_.reset(new TensorDataGpu<TensorT, 1>(tensor_array.dimensions()));
    //this->tensor_array_->setData(tensor_array);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator==(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator!=(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isNotEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator<(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isLessThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator<=(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isLessThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator>(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isGreaterThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayGpu<TensorT>::operator>=(const TensorArray<TensorT, Eigen::GpuDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsGpu::isGreaterThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  // Sort functors for Thrust
  struct SortThrust {
    SortThrust(const int& size) : size_(size) {};
    int size_ = 0;
  };
  struct SortLessThanThrust: SortThrust {
    using SortThrust::SortThrust;
    template<typename TensorT>
    __host__ __device__ bool operator()(const TensorArrayGpu<TensorT>& s1, const TensorArrayGpu<TensorT>& s2) {
      return TensorArrayOperatorsGpu::isLessThan(s1.tensor_array_, s2.tensor_array_, size_);
    }
  };
};
#endif
#endif //TENSORBASE_TENSORARRAYGPU_H