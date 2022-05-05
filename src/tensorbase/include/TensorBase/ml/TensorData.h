/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATA_H
#define TENSORBASE_TENSORDATA_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/Meta.h>
#include <memory>
#include <array>
#include <TensorBase/ml/TensorArray.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/string.hpp> // appear not to be needed for enum type
#include <cereal/types/common.hpp> // appear not to be needed for enum type
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /// Flags for pinned memory
  enum TensorDataGpuPinnedFlags {
    HostAllocDefault = 0,
    HostAllocPortable,
    HostAllocMapped,
    HostAllocWriteCombined
  };

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
    /*
    @brief Constructor with arguments for using pinned memory with various CUDA flags

    The flags parameter enables different options to be specified that affect the allocation, as follows.
      cudaHostAllocDefault: This flag's value is defined to be 0 and causes cudaHostAlloc() to emulate cudaMallocHost().
      cudaHostAllocPortable: The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
      cudaHostAllocMapped: Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
      cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC). WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.

    @param[in] pinned_memory Boolean value indicating whether to use pinned or pageable memory
    @param[in] pinned_flags String of the CUDA flag to use when creating pinned memory
    */
    TensorData(const bool& pinned_memory, const TensorDataGpuPinnedFlags& pinned_flag) :
      pinned_memory_(pinned_memory), pinned_flag_(pinned_flag) {device_name_ = typeid(DeviceT).name();};
    TensorData(const Eigen::array<Eigen::Index, TDim>& dimensions, const bool& pinned_memory, const TensorDataGpuPinnedFlags& pinned_flag) :
      pinned_memory_(pinned_memory), pinned_flag_(pinned_flag) {
      setDimensions(dimensions);
      device_name_ = typeid(DeviceT).name();
    };
    TensorData(const TensorData& other) {
      h_data_ = other.h_data_;
      d_data_ = other.d_data_;
      h_data_updated_ = other.h_data_updated_;
      d_data_updated_ = other.d_data_updated_;
      dimensions_ = other.dimensions_;
      tensor_size_ = other.tensor_size_;
      device_name_ = other.device_name_;
      pinned_memory_ = other.pinned_memory_;
      pinned_flag_ = other.pinned_flag_;
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
      pinned_memory_ = other.pinned_memory_;
      pinned_flag_ = other.pinned_flag_;
      return *this;
    }

    virtual std::shared_ptr<TensorData> copyToHost(DeviceT& device) = 0; ///< returns a copy of the TensorData synchronized to the host
    virtual std::shared_ptr<TensorData> copyToDevice(DeviceT& device) = 0; ///< returns a copy of the TensorData synchronized to the device

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

    /*
    @brief Run Length Encode the data

    @param[out] unique The unique value in the run (same size as the data)
    @param[out] count The counts of the value in the run (same size as the data)
    @param[out] n_runs The number of runs in the data (size of 1);
    */
    virtual void runLengthEncode(std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& unique, std::shared_ptr<TensorData<int, DeviceT, 1>>& count, std::shared_ptr<TensorData<int, DeviceT, 1>>& n_runs, DeviceT& device) = 0;

		/*
		@brief Bin the data for display as a histogram

		NOTE: Only available for primitive types
		TODO: optionally, return the calculated bins

		@param[out] n_levels The number of bin levels where n_bins = n_levels -1
		@param[out] lower_level The lower sample value boundary of lowest bin
		@param[out] upper_level The upper sample value boundary of upper bin
		@param[out] histogram The bin counts;
		*/
		virtual void histogram(const int& n_levels, const TensorT& lower_level, const TensorT& upper_level, std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& histogram, DeviceT& device) = 0;

    void setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions); ///< Set the tensor dimensions and calculate the tensor size
    Eigen::array<Eigen::Index, TDim> getDimensions() const; ///< Get the tensor dimensions
    size_t getTensorBytes() { return tensor_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    size_t getTensorSize() { return tensor_size_; }; ///< Get the size of the tensor
    int getDims() { return dimensions_.size(); };  ///< TDims getter
    std::string getDeviceName() { return device_name_; }; ///< Device name getter
    bool getPinnedMemory() { return pinned_memory_; }
    TensorDataGpuPinnedFlags getPinnedFlag() { return pinned_flag_; }

    virtual void setData(const Eigen::Tensor<TensorT, TDim>& data) = 0; ///< data setter
    virtual void setData() = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data(h_data_.get(), this->getDimensions()); return data; } ///< data copy getter
    virtual std::shared_ptr<TensorT[]> getHDataPointer() = 0; ///< host data pointer getter
    virtual std::shared_ptr<TensorT[]> getDataPointer() = 0; ///< device data pointer getter
    
    virtual bool syncHAndDData(DeviceT& device) = 0;  ///< Sync the host and device data
    bool syncDData(DeviceT& device); ///< Transfer tensor data to the device (if not already)
    bool syncHData(DeviceT& device); ///< Transfer tensor data to the host (if not already)
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { h_data_updated_ = h_data_updated; d_data_updated_ = d_data_updated; } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return std::make_pair(h_data_updated_, d_data_updated_); };   ///< Get the status of the host and device data

    /*
    @brief Intialize the data from a Tensor with type std::string

    @param[out] data_new Eigen::Tensor of type std::string
    */
    virtual void convertFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, DeviceT& device) = 0;

  protected:
    std::shared_ptr<TensorT[]> h_data_ = nullptr;  ///< Shared pointer implementation of the host tensor data
    std::shared_ptr<TensorT[]> d_data_ = nullptr;  ///< Shared pointer implementation of the device (GPU) tensor data

    bool h_data_updated_ = false;  ///< boolean indicator if the host data is up to date
    bool d_data_updated_ = false;  ///< boolean indicator if the device data is up to date
    // MULTI-GPU: more advanced syncronization will need to be implemented when transfering data between different GPUs    

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>(); ///< Tensor dimensions (initialized to all zeros)
    size_t tensor_size_ = 0;  ///< Tensor size
    std::string device_name_ = "";
    bool pinned_memory_ = false; // true;
    TensorDataGpuPinnedFlags pinned_flag_ = TensorDataGpuPinnedFlags::HostAllocDefault;
  private:
    friend class cereal::access;
    //template<class Archive>
    //void serialize(Archive& archive) {
    //	archive(dimensions_, tensor_size_, device_name_, h_data_updated_, d_data_updated_,pinned_memory_, pinned_flag_);
    //}
    template<class Archive>
    void save(Archive& archive) const {
      std::array<long long int, TDim> dimensions;
      for (int i = 0; i < TDim; ++i) { dimensions.at(i) = dimensions_.at(i); }
      archive(dimensions, tensor_size_, device_name_, h_data_updated_, d_data_updated_, pinned_memory_, pinned_flag_);
    }
    template<class Archive>
    void load(Archive& archive) {
      std::array<long long int, TDim> dimensions;
      archive(dimensions, tensor_size_, device_name_, h_data_updated_, d_data_updated_, pinned_memory_, pinned_flag_);
      for (int i = 0; i < TDim; ++i) { dimensions_.at(i) = dimensions.at(i); }
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
    return dimensions_;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorData<TensorT, DeviceT, TDim>::syncDData(DeviceT& device)
  {
    bool synced = true;
    if (!getDataStatus().second)
      synced = syncHAndDData(device);
    return synced;
  }
  template<typename TensorT, typename DeviceT, int TDim>
  inline bool TensorData<TensorT, DeviceT, TDim>::syncHData(DeviceT& device)
  {
    bool synced = true;
    if (!getDataStatus().first)
      synced = syncHAndDData(device);
    return synced;
  }
}

#endif //TENSORBASE_TENSORDATA_H