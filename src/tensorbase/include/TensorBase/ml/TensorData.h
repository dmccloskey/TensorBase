/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATA_H
#define TENSORBASE_TENSORDATA_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/Meta.h>
#include <memory>
#include <array>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/array.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{

  /**
    @brief Tensor data base class to handle the underlying memory and resource
      allocation of tensor data

    LIMITATIONS:
    - the memory management assumes a single GPU environment and does not allow for specifying which GPU to use
    - the GPU memory always uses pinned memory, and does not provide an option to use a different type of GPU memory
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
      // TODO: CUDA comiler error: use the "typename" keyword to treat nontype "TensorBase::TensorData<TensorT, DeviceT, TDim>::tensorT [with TensorT=TensorTOther, DeviceT=DeviceTOther, TDim=TDimOther]" as a type in a dependent context
      //if (!std::is_same<tensorT, TensorData<TensorTOther, DeviceTOther, TDimOther>::tensorT>::value)
      //  return false;
      return std::tie(
          dimensions_,
          device_name_          
        ) == std::tie(
          other.dimensions_,
          other.device_name_          
        );
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

    // NOT NEEDED
    //template<typename T, typename D>
    //TensorData(const TensorData<T,D,TDim>& other) {
    //  h_data_ = std::reinterpret_pointer_cast<TensorT>(other.h_data_);
    //  d_data_ = std::reinterpret_pointer_cast<TensorT>(other.d_data_);
    //  h_data_updated_ = other.h_data_updated_;
    //  d_data_updated_ = other.d_data_updated_;
    //  dimensions_ = other.dimensions_;
    //  tensor_size_ = other.tensor_size_;
    //  device_name_ = typeid(DeviceT).name();
    //};

    // NOT NEEDED
    //template<typename T>
    //TensorData(const TensorData<T, DeviceT, TDim>& other) {
    //  h_data_ = std::reinterpret_pointer_cast<TensorT>(other.h_data_);
    //  d_data_ = std::reinterpret_pointer_cast<TensorT>(other.d_data_);
    //  h_data_updated_ = other.h_data_updated_;
    //  d_data_updated_ = other.d_data_updated_;
    //  dimensions_ = other.dimensions_;
    //  tensor_size_ = other.tensor_size_;
    //  device_name_ = typeid(DeviceT).name();
    //};

    virtual std::shared_ptr<TensorData> copy(DeviceT& device) = 0; ///< returns a copy of the TensorData

    virtual void select(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< return a selection of the TensorData
    virtual void sortIndices(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, const std::string& sort_order, DeviceT& device) = 0; ///< sort the indices based on the TensorData
    virtual void sort(const std::string& sort_order, DeviceT& device) = 0; ///< sort the TensorData in place
    virtual void sort(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< sort the TensorData in place

    /*
    @brief Partition the data based on a flag.
      The flagged indices are moved to the front (in order)
      and the non-flagged indices are moved to the back (in reverse order)
    */
    virtual void partition(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0;

    void setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions); ///< Set the tensor dimensions and calculate the tensor size
    Eigen::array<Eigen::Index, TDim> getDimensions() const; ///< Get the tensor dimensions
    size_t getTensorBytes() { return tensor_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    size_t getTensorSize() { return tensor_size_; }; ///< Get the size of the tensor
    int getDims() { return dimensions_.size(); };  ///< TDims getter
    std::string getDeviceName() { return device_name_; }; ///< Device name getter

    virtual void setData(const Eigen::Tensor<TensorT, TDim>& data) = 0; ///< data setter
    virtual void setData() = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { std::shared_ptr<TensorT> h_data = h_data_;  Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data(h_data.get(), this->getDimensions()); return data; } ///< data copy getter
    virtual std::shared_ptr<TensorT> getDataPointer() = 0; ///< data pointer getter
    
    virtual bool syncHAndDData(DeviceT& device) = 0;  ///< Sync the host and device data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { h_data_updated_ = h_data_updated; d_data_updated_ = d_data_updated; } ///< Set the status of the host and device data
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

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(dimensions_, tensor_size_, device_name_, h_data_updated_, d_data_updated_);
    }
  };

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorData<TensorT, DeviceT, TDim>::setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions) {
    //dimensions_ = std::array<Eigen::Index, TDim>(); // works on gpu
    dimensions_ = dimensions; // works on cpu but not gpu
    size_t tensor_size = 1;
    for (int i = 0; i < TDim; ++i) {
      //dimensions_.at(i) = dimensions.at(i); // works on gpu
      tensor_size *= dimensions.at(i);
    }
    tensor_size_ = tensor_size;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline Eigen::array<Eigen::Index, TDim> TensorData<TensorT, DeviceT, TDim>::getDimensions() const
  { 
    return dimensions_; // works on cpu but not gpu
    //Eigen::array<Eigen::Index, TDim> dimensions;
    //for (int i = 0; i < TDim; ++i) {
    //  dimensions.at(i) = dimensions_.at(i);
    //}
    //return dimensions;
  }

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
    void sort(const std::string& sort_order, Eigen::DefaultDevice& device);
    void sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device);
    void partition(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device);
    std::shared_ptr<TensorT> getDataPointer() { return h_data_; }
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::DefaultDevice& device) { this->d_data_updated_ = true; this->h_data_updated_ = true; return true; }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice, TDim>>(this));
    }
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>> TensorDataDefaultDevice<TensorT, TDim>::copy(Eigen::DefaultDevice& device) {
    // initialize the new data
    TensorDataDefaultDevice<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //data_new_values.device(device) = data_values;
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
  inline void TensorDataDefaultDevice<TensorT, TDim>::sort(const std::string & sort_order, Eigen::DefaultDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    if (sort_order == "ASC") {
      std::sort(tensor_values.data(), tensor_values.data() + tensor_values.size(),
        [](const TensorT& lhs, const TensorT& rhs) {
        return lhs < rhs;
      });
    }
    else if (sort_order == "DESC") {
      std::sort(tensor_values.data(), tensor_values.data() + tensor_values.size(),
        [](const TensorT& lhs, const TensorT& rhs) {
        return lhs > rhs;
      });
    }
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    auto data_copy = this->copy(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());

    // Sort the data in place
    int iter = 0;
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &iter, &device](const int& index) {
        tensor_values(iter) = copy_values(index - 1);
        ++iter;
    });
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::partition(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    auto data_copy = this->copy(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());

    // Partition the data in place
    int iter = 0;
    int partition_iter = 0;
    int back_iter = data_copy->getTensorSize() - 1;
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &iter, &partition_iter, &back_iter, &device](const int& index) {
      if (index != 0) {
        tensor_values(partition_iter) = copy_values(iter);
        ++partition_iter;
      }
      else {
        tensor_values(back_iter) = copy_values(iter);
        --back_iter;
      }
      ++iter;
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
    void sort(const std::string& sort_order, Eigen::ThreadPoolDevice& device);
    void sort(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice& device);
    void partition(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice& device);
    std::shared_ptr<TensorT> getDataPointer() { return h_data_; }
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::ThreadPoolDevice& device) { this->d_data_updated_ = true; this->h_data_updated_ = true; return true; }
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(cereal::base_class<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>>(this));
    }
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, TDim>> TensorDataCpu<TensorT, TDim>::copy(Eigen::ThreadPoolDevice& device) {
    // initialize the new data
    TensorDataCpu<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //data_new_values.device(device) = data_values;
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
  inline void TensorDataCpu<TensorT, TDim>::sort(const std::string & sort_order, Eigen::ThreadPoolDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    if (sort_order == "ASC") {
      std::sort(tensor_values.data(), tensor_values.data() + tensor_values.size(),
        [](const TensorT& lhs, const TensorT& rhs) {
        return lhs < rhs;
      });
    }
    else if (sort_order == "DESC") {
      std::sort(tensor_values.data(), tensor_values.data() + tensor_values.size(),
        [](const TensorT& lhs, const TensorT& rhs) {
        return lhs > rhs;
      });
    }
  }
  template<typename TensorT, int TDim>
  inline void TensorDataCpu<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice& device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    auto data_copy = this->copy(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());

    // Sort the data in place
    int iter = 0;
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &iter, &device](const int& index) {
      tensor_values(iter) = copy_values(index - 1);
      ++iter;
    });
  }
  template<typename TensorT, int TDim>
  inline void TensorDataCpu<TensorT, TDim>::partition(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, TDim>>& indices, Eigen::ThreadPoolDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), (int)indices->getTensorSize());

    // Create a copy
    auto data_copy = this->copy(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> copy_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());

    // Partition the data in place
    int iter = 0;
    int partition_iter = 0;
    int back_iter = data_copy->getTensorSize() - 1;
    std::for_each(indices_values.data(), indices_values.data() + indices_values.size(),
      [&tensor_values, &copy_values, &iter, &partition_iter, &back_iter, &device](const int& index) {
      if (index != 0) {
        tensor_values(partition_iter) = copy_values(iter);
        ++partition_iter;
      }
      else {
        tensor_values(back_iter) = copy_values(iter);
        --back_iter;
      }
      ++iter;
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
}

// Cereal registration of TensorTs: float, int, char, double and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 4>);

CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataCpu<char, 4>);

#endif //TENSORBASE_TENSORDATA_H