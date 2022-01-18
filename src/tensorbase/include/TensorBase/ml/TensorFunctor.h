/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORFUNCTOR_H
#define TENSORBASE_TENSORFUNCTOR_H

#include <TensorBase/ml/TensorData.h>

namespace TensorBase
{
  /**
    @brief Class for applying user defined methods on TensorTable data
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorFunctor {
  public:
    TensorFunctor() = default;
    ~TensorFunctor() = default;

    /*
    @brief Template functor method to use for transforming a TensorTableData attribute

    @param[in] device
    */
    virtual void operator() (std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_data, DeviceT& device) = 0;
  };

  /// Example data copy functor
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorDataCopy : public TensorFunctor<TensorT, DeviceT, TDim> {
  public:
    TensorDataCopy(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& results): results_(results) {}
    void operator()(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_data, DeviceT& device) override {
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_data_values(tensor_data->getDataPointer().get(), tensor_data->getDimensions());
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> results_values(results_->getDataPointer().get(), results_->getDimensions());
      results_values.device(device) = tensor_data_values;
    }
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> results_; /// allocated and synced TensorData to capture the results
  };

  /// Example data capture without copy functor
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorDataPtrCapture : public TensorFunctor<TensorT, DeviceT, TDim> {
  public:
    void operator()(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_data, DeviceT& device) override {
      results_ = tensor_data;
    }
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> results_; /// any changes here will also change the Tensor Table data!
  };

  /// Example sum reduction functor
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorSumReduction : public TensorFunctor<TensorT, DeviceT, TDim> {
  public:
    TensorSumReduction(std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& results) : results_(results) {}
    void operator()(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_data, DeviceT& device) override {
      Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> tensor_data_values(tensor_data->getDataPointer().get(), tensor_data->getDimensions());
      Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> results_values(results_->getDataPointer().get());
      results_values.device(device) = tensor_data_values.sum();
    }
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> results_; /// allocated and synced TensorData to capture the results
  };
};
#endif //TENSORBASE_TENSORFUNCTOR_H