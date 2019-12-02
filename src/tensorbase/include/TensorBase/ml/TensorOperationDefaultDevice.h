/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATIONDEFAULTDEVICE_H
#define TENSORBASE_TENSOROPERATIONDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorOperation.h>

namespace TensorBase
{
  /**
    @brief DefaultDevice specialization of `TensorDeleteFromAxis`
  */
  template<typename LabelsT, typename TensorT, int TDim>
  class TensorDeleteFromAxisDefaultDevice : public TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorDeleteFromAxis<LabelsT, TensorT, Eigen::DefaultDevice, TDim>::TensorDeleteFromAxis;
    void allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice& device) override;
  };

  template<typename LabelsT, typename TensorT, int TDim>
  inline void TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, TDim>::allocateMemoryForValues(std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>& tensor_collection, Eigen::DefaultDevice & device)
  {
    // Determine the dimensions of the values that will be deleted
    Eigen::array<Eigen::Index, TDim> dimensions_new;
    for (auto& axis_map: tensor_collection->tables_.at(this->table_name_)->getAxes()) {
      dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(axis_map.second->getName())) = axis_map.second->getNLabels();
    }
    dimensions_new.at(tensor_collection->tables_.at(this->table_name_)->getDimFromAxisName(this->axis_name_)) = this->indices_->getTensorSize();

    // Allocate memory for the values
    TensorDataDefaultDevice<TensorT, TDim> values_tmp(dimensions_new);
    values_tmp.setData();
    this->values_ = std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(values_tmp);
  }
};
#endif //TENSORBASE_TENSOROPERATIONDEFAULTDEVICE_H