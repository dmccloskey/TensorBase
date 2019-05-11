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
    using tensorT = TensorT;
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
      tensor_size_ = other.tensor_size_;
      device_name_ = other.device_name_;
    };
    virtual ~TensorData() = default; ///< Default destructor

    template<typename TensorTOther, typename DeviceTOther, int TDimOther>
    inline bool operator==(const TensorData<TensorTOther, DeviceTOther, TDimOther>& other) const
    {
      if (!std::is_same<tensorT, TensorData<TensorTOther, DeviceTOther, TDimOther>::tensorT>::value)
        return false;
      return
        std::tie(
          dimensions_,
          device_name_          
        ) == std::tie(
          other.dimensions_,
          other.device_name_          
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
      tensor_size_ = other.tensor_size_;
      device_name_ = other.device_name_;
      return *this;
    }

    virtual std::shared_ptr<TensorData> copy(DeviceT& device) = 0; ///< returns a copy of the TensorData

    virtual void select(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< return a selection of the TensorData
    virtual void sortIndices(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, const std::string& sort_order, DeviceT& device) = 0; ///< sort the indices based on the TensorData
    virtual void sort(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< sort the TensorData in place

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
    size_t getTensorBytes() { return tensor_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    size_t getTensorSize() { return tensor_size_; }; ///< Get the size of the tensor
    int getDims() { return dimensions_.size(); };  ///< TDims getter
    std::string getDeviceName() { return device_name_; }; ///< Device name getter

    virtual void setData(const Eigen::Tensor<TensorT, TDim>& data) = 0; ///< data setter
    virtual void setData() = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { std::shared_ptr<TensorT> h_data = h_data_;  Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data(h_data.get(), this->getDimensions()); return data; } ///< data copy getter
    virtual std::shared_ptr<TensorT> getDataPointer() = 0; ///< data pointer getter
    
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

  // NOTE: could be useful in another context
//  template<typename TensorT, typename DeviceT, int TDim>
//  inline std::shared_ptr<TensorT> TensorData<TensorT, DeviceT, TDim>::getDataPointer(DeviceT& device)
//  {
//    // Sync the data
//    syncHAndDData(device);
//
//    // Get the appropriate data pointer depending upon the device
//    if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
//      return h_data_;
//    }
//    else if (typeid(device).name() == typeid(Eigen::ThreadPoolDevice).name()) {
//      return h_data_;
//    }
//#if COMPILE_WITH_CUDA
//    else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
//      return d_data_;
//    }
//#endif
//    else {
//      throw("Device not recognized!");
//    }
//  }

  /**
    @brief Tensor data class specialization for Eigen::DefaultDevice (single thread CPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataDefaultDevice : public TensorData<TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::DefaultDevice, TDim>::TensorData;
    ~TensorDataDefaultDevice() = default;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>> copy(Eigen::DefaultDevice& device);
    void select(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device);
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, const std::string& sort_order, Eigen::DefaultDevice& device);
    void sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device);
    std::shared_ptr<TensorT> getDataPointer() { return h_data_; }
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::DefaultDevice& device) { return true; }
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
    //	}
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>> TensorDataDefaultDevice<TensorT, TDim>::copy(Eigen::DefaultDevice& device) {
    // initialize the new data
    TensorDataDefaultDevice<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData();
    // copy over the values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    data_new_values.device(device) = data_values;
    return std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::select(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice & device)
  {
    // Copy over the selected values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_select_values(tensor_select->getDataPointer().get(), tensor_select->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_values(this->getDataPointer().get(), this->getDimensions());
    int iter_select = 0;
    int iter_tensor = 0;
    std::for_each(indices->getDataPointer().get(), indices->getDataPointer().get() + indices->getData().size(),
      [&](const int& index) {
      if (index > 0) {
        tensor_select_values.data()[iter_select] = tensor_values.data()[iter_tensor]; // works because all data is on the host
        ++iter_select;
      }
      ++iter_tensor;
    });
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, const std::string& sort_order, Eigen::DefaultDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());
    if (sort_order == "ASC") {
      std::sort(indices_values.data(), indices_values.data() + indices_values.size(),
        [&tensor_values](const int& lhs, const int& rhs) {
        return tensor_values(lhs - 1) < tensor_values(rhs - 1);
      });
    }
    else if (sort_order == "DESC") {
      std::sort(indices_values.data(), indices_values.data() + indices_values.size(),
        [&tensor_values](const int& lhs, const int& rhs) {
        return tensor_values(lhs - 1) > tensor_values(rhs - 1);
      });
    }
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    TensorDataDefaultDevice data_copy(this->getDimensions());
    data_copy.setData();
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy.getDataPointer().get(), (int)data_copy.getTensorSize());
    copy_values.device(device) = tensor_values;

    // Sort the data in place
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &device](const int& index) {
        tensor_values(index - 1) = copy_values(index - 1);
    });
  }
  template<typename TensorT, int TDim>
  void TensorDataDefaultDevice<TensorT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data) {
    TensorT* h_data = new TensorT[this->tensor_size_];
    // copy the tensor
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, this->getDimensions());
    data_copy = data;
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
  };
  template<typename TensorT, int TDim>
  void TensorDataDefaultDevice<TensorT, TDim>::setData() {
    TensorT* h_data = new TensorT[this->tensor_size_];
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
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
    std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>> copy(Eigen::ThreadPoolDevice& device);
    void select(std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice & device);
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, const std::string& sort_order, Eigen::ThreadPoolDevice & device);
    void sort(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice& device);
    std::shared_ptr<TensorT> getDataPointer() { return h_data_; }
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::ThreadPoolDevice& device) { return true; }
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
    //	}
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>> TensorDataCpu<TensorT, TDim>::copy(Eigen::ThreadPoolDevice& device) {
    // initialize the new data
    TensorDataCpu<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData();
    // copy over the values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    data_new_values.device(device) = data_values;
    return std::make_shared<TensorDataCpu<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataCpu<TensorT, TDim>::select(std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice & device)
  {
    // Copy over the selected values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_select_values(tensor_select->getDataPointer().get(), tensor_select->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_values(this->getDataPointer().get(), this->getDimensions());
    int iter_select = 0;
    int iter_tensor = 0;
    // TODO: add parallel execution policy C++17
    std::for_each(indices->getDataPointer().get(), indices->getDataPointer().get() + indices->getData().size(),
      [&](const int& index) {
      if (index > 0) {
        tensor_select_values.data()[iter_select] = tensor_values.data()[iter_tensor]; // works because all data is on the host
        ++iter_select;
      }
      ++iter_tensor;
    });
  }
  template<typename TensorT, int TDim>
  inline void TensorDataCpu<TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, const std::string& sort_order, Eigen::ThreadPoolDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());
    if (sort_order == "ASC") {
      std::sort(indices_values.data(), indices_values.data() + indices_values.size(),
        [&tensor_values](const int& lhs, const int& rhs) {
        return tensor_values(lhs - 1) < tensor_values(rhs - 1);
      });
    }
    else if (sort_order == "DESC") {
      std::sort(indices_values.data(), indices_values.data() + indices_values.size(),
        [&tensor_values](const int& lhs, const int& rhs) {
        return tensor_values(lhs - 1) > tensor_values(rhs - 1);
      });
    }
  }
  template<typename TensorT, int TDim>
  inline void TensorDataCpu<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice& device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    TensorDataCpu data_copy(this->getDimensions());
    data_copy.setData();
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy.getDataPointer().get(), (int)data_copy.getTensorSize());
    copy_values.device(device) = tensor_values;

    // Sort the data in place
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &device](const int& index) {
      tensor_values(index - 1) = copy_values(index - 1);
    });
  }
  template<typename TensorT, int TDim>
  void TensorDataCpu<TensorT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data) {
    TensorT* h_data = new TensorT[this->tensor_size_];
    // copy the tensor
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, this->getDimensions());
    data_copy = data;
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
  };
  template<typename TensorT, int TDim>
  void TensorDataCpu<TensorT, TDim>::setData() {
    TensorT* h_data = new TensorT[this->tensor_size_];
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
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
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> copy(Eigen::GpuDevice& device);
    std::shared_ptr<TensorT> getDataPointer();
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::GpuDevice& device);
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice>>(this));
    //	}
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> TensorDataGpu<TensorT, TDim>::copy(Eigen::GpuDevice& device) {
    // initialize the new data
    TensorDataGpu<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData();
    // copy over the values
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    data_new_values.device(device) = data_values;
    return std::make_shared<TensorDataGpu<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  std::shared_ptr<TensorT> TensorDataGpu<TensorT, TDim>::getDataPointer() {
    if (!this->d_data_updated_) {
      this->syncHAndDData(device);
    }
    return d_data_;
  }
  template<typename TensorT, int TDim>
  void TensorDataGpu<TensorT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data) {
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
  };
  template<typename TensorT, int TDim>
  void TensorDataGpu<TensorT, TDim>::setData() {
    // allocate cuda and pinned host memory
    TensorT* d_data;
    TensorT* h_data;
    assert(cudaMalloc((void**)(&d_data), getTensorSize()) == cudaSuccess);
    assert(cudaHostAlloc((void**)(&h_data), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
    // define the deleters
    auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
    auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
    this->h_data_.reset(h_data, h_deleter);
    this->d_data_.reset(d_data, d_deleter);
    this->h_data_updated_ = true;
    this->d_data_updated_ = false;
  };
  template<typename TensorT, int TDim>
  bool TensorDataGpu<TensorT, TDim>::syncHAndDData(Eigen::GpuDevice& device) {
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
      std::cout << "Both host and device are synchronized." << std::endl;
      return false;
    }
  }
#endif
}

//CEREAL_REGISTER_TYPE(TensorData::TensorDataDefaultDevice<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(TensorData::TensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //TENSORBASE_TENSORDATA_H