/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORARRAY_H
#define TENSORBASE_TENSORARRAY_H

#include <TensorBase/ml/TensorData.h>

namespace TensorBase
{  
  /**
    @brief Base class for all fixed vector types
  */
  template<typename TensorT, typename DeviceT>
  class TensorArray
  {
  public:
    TensorArray() = default;
    TensorArray(TensorT* tensor_array, const size_t& array_size);
    virtual ~TensorArray() = default;

    size_t getArraySize() { return array_size_; } ///< array_size getter

    virtual void setTensorArray(TensorT* tensor_array, const size_t& array_size) = 0; ///< tensor_array setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> getTensorArray() { return tensor_array_->getData(); };  ///< tensor_array getter

    bool syncHAndDData(DeviceT& device) { return tensor_array_->syncHAndDData(device); };  ///< Sync the host and device tensor_array data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_array_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return tensor_array_->getDataStatus(); };   ///< Get the status of the host and device tensor_array data

    template<typename T>
    void getTensorArrayDataPointer(std::shared_ptr<T>& data_copy); ///< TensorAxisConcept labels getter

  protected:
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> tensor_array_; ///< dim=0: size of the array
    size_t array_size_ = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_);
    }
  };

  template<typename TensorT, typename DeviceT>
  inline TensorArray<TensorT, DeviceT>::TensorArray(TensorT* tensor_array, const size_t& array_size): array_size_(array_size) {
    setTensorArray(tensor_array, array_size);
  };

  template<typename TensorT, typename DeviceT>
  template<typename T>
  inline void TensorArray<TensorT, DeviceT>::getTensorArrayDataPointer(std::shared_ptr<T>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T>(tensor_array_->getDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT>
  class TensorArrayDefaultDevice : public TensorArray<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorArrayDefaultDevice() = default;  ///< Default constructor
    TensorArrayDefaultDevice(TensorT* tensor_array, const size_t& array_size) : TensorArray(tensor_array, array_size) {};
    ~TensorArrayDefaultDevice() = default; ///< Default destructor
    void setTensorArray(TensorT* tensor_array, const size_t& array_size) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT, Eigen::DefaultDevice>>(this));
    }
  };

  template<typename TensorT>
  inline void TensorArrayDefaultDevice<TensorT>::setTensorArray(TensorT* tensor_array, const size_t & array_size)
  {
    const Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_array_map(tensor_array, (int)array_size);
    this->tensor_array_.reset(new TensorDataDefaultDevice<TensorT, 1>(Eigen::array<Eigen::Index, 1>({(int)array_size})));
    this->tensor_array_->setData(tensor_array_map);
  }

  template<typename TensorT>
  class TensorArrayCpu : public TensorArray<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorArrayCpu() = default;  ///< Default constructor
    TensorArrayCpu(TensorT* tensor_array, const size_t& array_size) : TensorArray(tensor_array, array_size) {};
    ~TensorArrayCpu() = default; ///< Default destructor
    void setTensorArray(TensorT* tensor_array, const size_t& array_size) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };

  template<typename TensorT>
  inline void TensorArrayCpu<TensorT>::setTensorArray(TensorT * tensor_array, const size_t & array_size)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_array_map(tensor_array, (int)array_size);
    this->tensor_array_.reset(new TensorDataCpu<TensorT, 1>(Eigen::array<Eigen::Index, 1>({ (int)array_size })));
    this->tensor_array_->setData(tensor_array_map);
  }

};
#endif //TENSORBASE_TENSORARRAY_H