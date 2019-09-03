/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORARRAY_H
#define TENSORBASE_TENSORARRAY_H

#include <TensorBase/ml/TensorData.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{  
  /* Operators for DefaultDevice and Cpu classes
  */
  namespace TensorArrayOperatorsCpu {
    template<typename TensorT>
    int compare(const TensorT * s1, const TensorT * s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1[i] != s2[i]) break;
        if (i == size - 1) return 0;
      }
      return s1[i] - s2[i];
    }

    template<typename TensorT>
    bool isEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) == 0) return true;
      else return false;
    }

    template<typename TensorT>
    bool isNotEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) != 0) return true;
      else return false;
    }

    template<typename TensorT>
    bool isLessThan(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) < 0) return true;
      else return false;
    }

    template<typename TensorT>
    bool isGreaterThan(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) > 0) return true;
      else return false;
    }

    template<typename TensorT>
    bool isLessThanOrEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) <= 0) return true;
      else return false;
    }

    template<typename TensorT>
    bool isGreaterThanOrEqualTo(const TensorT * s1, const TensorT * s2, const int& size) {
      if (compare(s1, s2, size) >= 0) return true;
      else return false;
    }

    template<>
    int compare<char>(const char * s1, const char * s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1[i] != s2[i]) break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1[i] - (const unsigned char)s2[i];
    }
  };

  /**
    @brief Base class for all fixed vector types
  */
  template<typename TensorT, typename DeviceT>
  class TensorArray
  {
  public:
    TensorArray() = default;
    //virtual ~TensorArray() { if (tensor_array_ != nullptr) delete[] tensor_array_; };
    virtual ~TensorArray() = default;

    /// operators are defined on a DeviceT-basis and executed on the specific DeviceT
    virtual bool operator==(const TensorArray& other) = 0;
    virtual bool operator!=(const TensorArray& other) = 0;
    virtual bool operator<(const TensorArray& other) = 0;
    virtual bool operator<=(const TensorArray& other) = 0;
    virtual bool operator>(const TensorArray& other) = 0;
    virtual bool operator>=(const TensorArray& other) = 0;

    size_t getArraySize() const { return array_size_; } ///< array_size getter

    virtual void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) = 0; ///< tensor_array setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> getTensorArray() { Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data(this->tensor_array_, this->array_size_); return data; };  ///< tensor_array getter
    //Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> getTensorArray() { return tensor_array_->getData(); };  ///< tensor_array getter
    //std::shared_ptr<TensorData<TensorT, DeviceT, 1>> getTensorArraySharedPtr() const { return tensor_array_; };  ///< tensor_array getter

    //bool syncHAndDData(DeviceT& device) { return tensor_array_->syncHAndDData(device); };  ///< Sync the host and device tensor_array data
    //void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_array_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device data
    //std::pair<bool, bool> getDataStatus() { return tensor_array_->getDataStatus(); };   ///< Get the status of the host and device tensor_array data

    //template<typename T>
    //void getTensorArrayDataPointer(std::shared_ptr<T>& data_copy); ///< TensorAxisConcept labels getter

    TensorT* tensor_array_ = nullptr;
  protected:
    //std::shared_ptr<TensorData<TensorT, DeviceT, 1>> tensor_array_; ///< dim=0: size of the array
    size_t array_size_ = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_);
    }
  };

  //template<typename TensorT, typename DeviceT>
  //template<typename T>
  //inline void TensorArray<TensorT, DeviceT>::getTensorArrayDataPointer(std::shared_ptr<T>& data_copy) {
  //  if (std::is_same<T, TensorT>::value)
  //    data_copy = std::reinterpret_pointer_cast<T>(tensor_array_->getDataPointer()); // required for compilation: no conversion should be done
  //}

  template<typename TensorT>
  class TensorArrayDefaultDevice : public TensorArray<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorArrayDefaultDevice() = default;  ///< Default constructor
    TensorArrayDefaultDevice(const Eigen::Tensor<TensorT, 1>& tensor_array);
    ~TensorArrayDefaultDevice() = default; ///< Default destructor
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    bool operator==(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
    bool operator!=(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
    bool operator<(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
    bool operator<=(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
    bool operator>(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
    bool operator>=(const TensorArray<TensorT, Eigen::DefaultDevice>& other) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT, Eigen::DefaultDevice>>(this));
    }
  };

  template<typename TensorT>
  inline TensorArrayDefaultDevice<TensorT>::TensorArrayDefaultDevice(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    this->setTensorArray(tensor_array);
  }

  template<typename TensorT>
  inline void TensorArrayDefaultDevice<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // set the array size
    this->array_size_ = tensor_array.dimension(0);

    // copy the data
    this->tensor_array_ = new TensorT[this->array_size_];
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_copy(this->tensor_array_, this->array_size_);
    data_copy = tensor_array;

    //this->tensor_array_.reset(new TensorDataDefaultDevice<TensorT, 1>(tensor_array.dimensions()));
    //this->tensor_array_->setData(tensor_array);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator==(const TensorArray<TensorT, Eigen::DefaultDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator!=(const TensorArray<TensorT, Eigen::DefaultDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isNotEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator<(const TensorArray<TensorT, Eigen::DefaultDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isLessThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator<=(const TensorArray<TensorT, Eigen::DefaultDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isLessThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator>(const TensorArray<TensorT, Eigen::DefaultDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isGreaterThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayDefaultDevice<TensorT>::operator>=(const TensorArray<TensorT, Eigen::DefaultDevice>& other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  class TensorArrayCpu : public TensorArray<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorArrayCpu() = default;  ///< Default constructor
    TensorArrayCpu(const Eigen::Tensor<TensorT, 1>& tensor_array);
    ~TensorArrayCpu() = default; ///< Default destructor
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    bool operator==(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
    bool operator!=(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
    bool operator<(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
    bool operator<=(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
    bool operator>(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
    bool operator>=(const TensorArray<TensorT, Eigen::ThreadPoolDevice>& other) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };

  template<typename TensorT>
  inline TensorArrayCpu<TensorT>::TensorArrayCpu(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    this->setTensorArray(tensor_array);
  }

  template<typename TensorT>
  inline void TensorArrayCpu<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // set the array size
    this->array_size_ = tensor_array.dimension(0);

    // copy the data
    this->tensor_array_ = new TensorT[this->array_size_];
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_copy(this->tensor_array_, this->array_size_);
    data_copy = tensor_array;

    //this->tensor_array_.reset(new TensorDataCpu<TensorT, 1>(tensor_array.dimensions()));
    //this->tensor_array_->setData(tensor_array);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator==(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator!=(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isNotEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator<(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isLessThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator<=(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isLessThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator>(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isGreaterThan(this->tensor_array_, other.tensor_array_, this->array_size_);
  }

  template<typename TensorT>
  inline bool TensorArrayCpu<TensorT>::operator>=(const TensorArray<TensorT, Eigen::ThreadPoolDevice> & other)
  {
    assert(this->array_size_ == other.getArraySize());
    return TensorArrayOperatorsCpu::isGreaterThanOrEqualTo(this->tensor_array_, other.tensor_array_, this->array_size_);
  }
};
#endif //TENSORBASE_TENSORARRAY_H