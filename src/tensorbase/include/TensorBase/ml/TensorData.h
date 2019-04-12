/**TODO:  Add copyright*/

#ifndef TENSORDATA_TENSORDATA_H
#define TENSORDATA_TENSORDATA_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Tensor data
  */
  template<typename TensorT, typename DeviceT, typename... Indices>
  class TensorData
  {
  public:
    TensorData() = default;
    //TensorData(const Indices&... indices) { setIndices(indices...); };  // uncommenting will throw an error...
    TensorData(const TensorData& other)
    {
      h_data_ = other.h_data_;
      d_data_ = other.d_data_;
      h_data_updated_ = other.h_data_updated_;
      d_data_updated_ = other.d_data_updated_;
      indices_ = other.indices_;
    };
    ~TensorData() = default; ///< Default destructor

    inline bool operator==(const TensorData& other) const
    {
      return
        std::tie(

        ) == std::tie(

        )
        ;
    }

    inline bool operator!=(const TensorData& other) const
    {
      return !(*this == other);
    }

    inline TensorData& operator=(const TensorData& other)
    {
      h_data_ = other.h_data_;
      d_data_ = other.d_data_;
      h_data_updated_ = other.h_data_updated_;
      d_data_updated_ = other.d_data_updated_;
      indices_ = other.indices_;
      return *this;
    }

    void setIndices(const Indices&... indices) { indices_ = std::tuple<Indices...>(indices...); }
    std::tuple<Indices...> getIndicesAsTuple() const { return indices_; }

    virtual void setData(const Eigen::Tensor<TensorT, sizeof...(Indices)>& data) = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, sizeof...(Indices)>> getData() { return getData_(indices_, std::index_sequence_for<Indices...>()); } ///< data copy getter
    std::shared_ptr<TensorT> getHDataPointer() { return h_data_; }; ///< data pointer getter
    std::shared_ptr<TensorT> getDDataPointer() { return d_data_; }; ///< data pointer getter

    size_t getTensorSize() { return prod_(1, indices_, std::index_sequence_for<Indices...>()) * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

    virtual bool syncHAndDData(DeviceT& device) = 0;  ///< Sync the host and device data
    std::pair<bool, bool> getDataStatus() { return std::make_pair(h_data_updated_, d_data_updated_); };   ///< Get the status of the host and device data

  protected:
    std::shared_ptr<TensorT> h_data_ = nullptr;
    std::shared_ptr<TensorT> d_data_ = nullptr;

    bool h_data_updated_ = false;
    bool d_data_updated_ = false;

    std::tuple<Indices...> indices_;
    size_t tensor_size_;

    template<typename T>
    int prod_(T t, Indices... indices) {
      return t * prod_(indices...);
    }

  private:
    template<std::size_t... Is>
    Eigen::TensorMap<Eigen::Tensor<TensorT, sizeof...(Indices)>> getData_(const std::tuple<Indices...>& tuple, std::index_sequence<Is...>) {
      std::shared_ptr<TensorT> h_data = h_data_;
      Eigen::TensorMap<Eigen::Tensor<TensorT, sizeof...(Indices)>> data(h_data.get(), std::get<Is>(tuple)...);
      return data;
    }

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(batch_size_, memory_size_, layer_size_, 
    //		h_data_, h_output_, h_error_, h_derivative_, h_dt_,
    //		d_data_, d_output_, d_error_, d_derivative_, d_dt_,
    //		h_data_updated_, h_output_updated_, h_error_updated_, h_derivative_updated_, h_dt_updated_,
    //		d_data_updated_, d_output_updated_, d_error_updated_, d_derivative_updated_, d_dt_updated_);
    //	}
  };

  template<typename TensorT, typename... Indices>
  class TensorDataCpu : public TensorData<TensorT, Eigen::DefaultDevice, Indices...> {
  public:
    //using TensorData<TensorT, Eigen::DefaultDevice, Dim, Indices...>::TensorData;
    TensorDataCpu(Indices... indices) { setIndices(indices...); };
    void setData(const Eigen::Tensor<TensorT, sizeof...(Indices)>& data) { setData_(data, this->indices_, std::index_sequence_for<Indices...>()); }; ///< data setter
    bool syncHAndDData(Eigen::DefaultDevice& device) { return true; }
  private:
    template<std::size_t... Is>
    void setData_(const Eigen::Tensor<TensorT, sizeof...(Indices)>& data, const std::tuple<Indices...>& tuple, std::index_sequence<Is...>) {
      // allocate cuda and pinned host memory
      TensorT* d_data;
      TensorT* h_data;
      assert(cudaMalloc((void**)(&d_data), getTensorSize()) == cudaSuccess);
      assert(cudaHostAlloc((void**)(&h_data), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, sizeof...(Indices)>> data_copy(h_data, std::get<Is>(tuple)...);
      data_copy = data;
      // define the deleters
      auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
      auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
      this->h_data_.reset(h_data, h_deleter);
      this->d_data_.reset(d_data, d_deleter);
      this->h_data_updated_ = true;
      this->d_data_updated_ = false;

    };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
    //	}
  };

#if COMPILE_WITH_CUDA

  template<typename TensorT, typename... Indices>
  class TensorDataGpu : public TensorData<TensorT, Eigen::GpuDevice, Indices...> {
  public:
    //using TensorData<TensorT, Eigen::DefaultDevice, Dim, Indices...>::TensorData;
    TensorDataGpu(Indices... indices) { setIndices(indices...); }
    void setData(const Eigen::Tensor<TensorT, sizeof...(Indices)>& data) { setData_(data, this->indices_, std::index_sequence_for<Indices...>()); }; ///< data setter
    bool syncHAndDData(Eigen::GpuDevice& device) {
      if (this->h_data_updated_ && !this->d_data_updated_) {
        device.memcpyHostToDevice(this->d_data_.get(), this->h_data_.get(), getTensorSize());
        this->d_data_updated_ = true;
        this->h_data_updated_ = false;
        return true;
      }
      else if (!this->h_data_updated_ && this->d_data_updated_) {
        device.memcpyDeviceToHost(this->h_data_.get(), this->d_data_.get(), getTensorSize());
        this->h_data_updated_ = true;
        this->d_data_updated_ = false;
        return true;
      }
      else {
        std::cout << "Both host and device are syncHAndDronized." << std::endl;
        return false;
      }
    }
  private:
    template<std::size_t... Is>
    void setData_(const Eigen::Tensor<TensorT, sizeof...(Indices)>& data, const std::tuple<Indices...>& tuple, std::index_sequence<Is...>) {
      // allocate cuda and pinned host memory
      TensorT* d_data;
      TensorT* h_data;
      assert(cudaMalloc((void**)(&d_data), getTensorSize()) == cudaSuccess);
      assert(cudaHostAlloc((void**)(&h_data), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, sizeof...(Indices)>> data_copy(h_data, std::get<Is>(tuple)...);
      data_copy = data;
      // define the deleters
      auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
      auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
      this->h_data_.reset(h_data, h_deleter);
      this->d_data_.reset(d_data, d_deleter);
      this->h_data_updated_ = true;
      this->d_data_updated_ = false;
    };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice>>(this));
    //	}
  };
#endif
}

//CEREAL_REGISTER_TYPE(TensorData::TensorDataCpu<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(TensorData::TensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //TENSORDATA_TENSORDATA_H