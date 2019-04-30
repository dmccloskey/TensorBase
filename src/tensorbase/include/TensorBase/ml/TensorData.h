/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATA_H
#define TENSORBASE_TENSORDATA_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/Meta.h>
#include <memory>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Tensor data base class to handle the underlying memory and resource
      allocation of tensor data

    LIMITATIONS: currently, the memory management assumes a single GPU environment.
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorData
  {
  public:
    TensorData() { device_name_ = typeid(DeviceT).name(); };
    TensorData(const Eigen::array<Eigen::Index, TDim>& dimensions) { 
      setDimensions(dimensions); 
      device_name_ = typeid(DeviceT).name(); };
    TensorData(const TensorData& other)
    {
      h_data_ = other.h_data_;
      d_data_ = other.d_data_;
      h_data_updated_ = other.h_data_updated_;
      d_data_updated_ = other.d_data_updated_;
      dimensions_ = other.dimensions_;
    };
    virtual ~TensorData() = default; ///< Default destructor

    inline bool operator==(const TensorData& other) const
    {
      return
        std::tie(
          dimensions_
        ) == std::tie(
          other.dimensions_
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
      dimensions_ = other.dimensions_;
      return *this;
    }

    /**
      @brief Set the tensor dimensions and calculate the tensor size
    */
    void setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions) { 
      dimensions_ = dimensions; 
      size_t tensor_size = 1;
      for (const auto& index : dimensions)
        tensor_size *= index;
      tensor_size_ = tensor_size;
    }
    Eigen::array<Eigen::Index, TDim> getDimensions() const { return dimensions_; }  ///< dimensions getter
    size_t getTensorSize() { return tensor_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    int getDims() { return dimensions_.size(); };  ///< TDims getter
    std::string getDeviceName() { return device_name_; }; ///< Device name getter

    virtual void setData(const Eigen::Tensor<TensorT, TDim>& data) = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { std::shared_ptr<TensorT> h_data = h_data_;  Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data(h_data.get(), this->getDimensions()); return data; } ///< data copy getter
    std::shared_ptr<TensorT> getHDataPointer() { return h_data_; }; ///< data pointer getter
    std::shared_ptr<TensorT> getDDataPointer() { return d_data_; }; ///< data pointer getter
    
    virtual bool syncHAndDData(DeviceT& device) = 0;  ///< Sync the host and device data
    std::pair<bool, bool> getDataStatus() { return std::make_pair(h_data_updated_, d_data_updated_); };   ///< Get the status of the host and device data

  protected:
    std::shared_ptr<TensorT> h_data_ = nullptr;  ///< Shared pointer implementation of the host tensor data
    std::shared_ptr<TensorT> d_data_ = nullptr;  ///< Shared pointer implementation of the device (GPU) tensor data

    bool h_data_updated_ = false;  ///< boolean indicator if the host data is up to date
    bool d_data_updated_ = false;  ///< boolean indicator if the device data is up to date
    // MULTI-GPU: more advanced syncronization will need to be implemented when transfering data between different GPUs    

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>(); ///< Tensor dimensions (initialized to all zeros)
    size_t tensor_size_ = 0;  ///< Tensor size
    std::string device_name_ = "";

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(dimensions_, tensor_size_,
    //		h_data_, d_data_, h_data_updated_, d_data_updated_);
    //	}
  };

  /**
    @brief Tensor data class specialization for Eigen::DefaultDevice (single thread CPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataDefaultDevice : public TensorData<TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::DefaultDevice, TDim>::TensorData;
    ~TensorDataDefaultDevice() = default;
    void setData(const Eigen::Tensor<TensorT, TDim>& data) {
      TensorT* h_data = new TensorT[this->tensor_size_];
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, this->getDimensions());
      data_copy = data;
      this->h_data_.reset(h_data);
      this->h_data_updated_ = true;
      this->d_data_updated_ = true;
    }; ///< data setter
    bool syncHAndDData(Eigen::DefaultDevice& device) { return true; }
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
    //	}
  };

  /**
    @brief Tensor data class specialization for Eigen::ThreadPoolDevice (Multi thread CPU)

    NOTE: Methods are exactly the same as DefaultDevice
  */
  template<typename TensorT, int TDim>
  class TensorDataCpu : public TensorData<TensorT, Eigen::ThreadPoolDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>::TensorData;
    ~TensorDataCpu() = default;
    void setData(const Eigen::Tensor<TensorT, TDim>& data) {
      TensorT* h_data = new TensorT[this->tensor_size_];
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, this->getDimensions());
      data_copy = data;
      //auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
      //this->h_data_.reset(h_data, h_deleter);
      this->h_data_.reset(h_data);
      this->h_data_updated_ = true;
      this->d_data_updated_ = true;
    }; ///< data setter
    bool syncHAndDData(Eigen::ThreadPoolDevice& device) { return true; }
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
    //	}
  };

#if COMPILE_WITH_CUDA
  /**
    @brief Tensor data class specialization for Eigen::GpuDevice (single GPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataGpu : public TensorData<TensorT, Eigen::GpuDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::GpuDevice, TDim>::TensorData;
    ~TensorDataGpu() = default;
    void setData(const Eigen::Tensor<TensorT, TDim>& data) {
      // allocate cuda and pinned host memory
      TensorT* d_data;
      TensorT* h_data;
      assert(cudaMalloc((void**)(&d_data), getTensorSize()) == cudaSuccess);
      assert(cudaHostAlloc((void**)(&h_data), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, getDimensions());
      data_copy = data;
      // define the deleters
      auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
      auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
      this->h_data_.reset(h_data, h_deleter);
      this->d_data_.reset(d_data, d_deleter);
      this->h_data_updated_ = true;
      this->d_data_updated_ = false;
    }; ///< data setter
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
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice>>(this));
    //	}
  };
#endif
}

//CEREAL_REGISTER_TYPE(TensorData::TensorDataDefaultDevice<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(TensorData::TensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //TENSORBASE_TENSORDATA_H