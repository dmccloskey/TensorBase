/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXISCPU_H
#define TENSORBASE_TENSORAXISCPU_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorDataCpu.h>
#include <TensorBase/ml/TensorAxis.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  template<typename TensorT>
  class TensorAxisCpu : public TensorAxis<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorAxisCpu() = default;  ///< Default constructor
    TensorAxisCpu(const std::string& name, const size_t& n_dimensions, const size_t& n_labels) : TensorAxis(name, n_dimensions, n_labels) { this->setLabels(); };
    TensorAxisCpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) : TensorAxis(name) { this->setDimensionsAndLabels(dimensions, labels); };
    ~TensorAxisCpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 2>& labels) override;
    void setLabels() override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels, Eigen::ThreadPoolDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& indices_sort, Eigen::ThreadPoolDevice& device) override;
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels_select, Eigen::ThreadPoolDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override;
    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice& device) override;
    void makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };
  template<typename TensorT>
  void TensorAxisCpu<TensorT>::setLabels(const Eigen::Tensor<TensorT, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_ = std::make_shared<TensorDataCpu<TensorT, 2>>(TensorDataCpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_ = std::make_shared<TensorDataCpu<TensorT, 2>>(TensorDataCpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // copy the original number of labels
    size_t n_labels_copy = this->n_labels_;

    // update the number of labels
    this->n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataCpu<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    labels_concat.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat.getDataPointer().get(), labels_concat.getDimensions());

    // Concatenate or assign the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    if (n_labels_copy > 0) {
      labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);
    }
    else {
      labels_concat_values.device(device) = new_labels_values;
    }

    // Move over the new labels
    this->tensor_dimension_labels_ = std::make_shared<TensorDataCpu<TensorT, 2>>(labels_concat);
  }

  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& indices_sort, Eigen::ThreadPoolDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataCpu<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();

    // create a dummy index along the dimension
    TensorDataCpu<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_dimension_reshape(indices_dimension.getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // normalize and broadcast the dummy indices across the tensor    
    auto indices_dimension_norm = indices_dimension_reshape - indices_dimension_reshape.constant(1);
    auto indices_dimension_bcast_values = indices_dimension_norm.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_reshape(indices->getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));
    auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(this->getNDimensions());
    auto indices_view_bcast_values = indices_view_norm.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // update the indices_sort_values
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());
    indices_sort_values.device(device) = indices_view_bcast_values + indices_dimension_bcast_values + indices_sort_values.constant(1);

    // move over the results
    indices_sort = std::make_shared<TensorDataCpu<int, 2>>(indices_sort_tmp);
  }

  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels_select, Eigen::ThreadPoolDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataCpu<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    axis_size_value.device(device) = indices_values.clip(0, 1).sum();

    // allocate memory for the new labels
    TensorDataCpu<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>> new_labels_ptr = std::make_shared<TensorDataCpu<TensorT, 2>>(new_labels);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataCpu<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), indices_select.getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices_select_ptr = std::make_shared<TensorDataCpu<int, 2>>(indices_select);

    // perform the reduction and move over the output
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    labels_select = new_labels_ptr;
  }
  template<typename TensorT>
  inline bool TensorAxisCpu<TensorT>::loadLabelsBinary(const std::string & filename, Eigen::ThreadPoolDevice & device)
  {
    // Read in the the labels
    this->setDataStatus(true, false);
    Eigen::Tensor<TensorT, 2> labels_data((int)this->n_dimensions_, (int)this->n_labels_);
    DataFile::loadDataBinary<TensorT, 2>(filename + ".ta", labels_data);
    this->getLabels() = labels_data;
    this->syncHAndDData(device); // H to D
    return true;
  }
  template<typename TensorT>
  inline bool TensorAxisCpu<TensorT>::storeLabelsBinary(const std::string & filename, Eigen::ThreadPoolDevice & device)
  {
    // Store the labels
    this->syncHAndDData(device); // D to H
    DataFile::storeDataBinary<TensorT, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataCpu<TensorT, 2> labels_converted(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    labels_converted.setData();
    labels_converted.syncHAndDData(device);
    labels_converted.convertFromStringToTensorT(labels, device);

    // Make the indices select
    TensorDataCpu<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    indices_select.setData();
    indices_select.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), (int)labels.dimension(0), (int)labels.dimension(1));

    // Determine the unique input axis labels
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_unique_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_unique_values_bcast = indices_unique_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v1(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_v1_bcast = labels_new_v1.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v2(labels_converted.getDataPointer().get(), 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1));
    auto labels_new_v2_bcast = labels_new_v2.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    // Select the overlapping labels
    auto labels_new_v1v2_select = (labels_new_v1_bcast == labels_new_v2_bcast).select(indices_unique_values_bcast.constant(1), indices_unique_values_bcast.constant(0));
    // Reduct along the axis dimensions and then cum sum along the axis labels
    auto labels_new_v1v2_prod = labels_new_v1v2_select.sum(Eigen::array<Eigen::Index, 2>({ 1, 3 })).clip(0, 1);
    auto labels_new_v1v2_cumsum = (labels_new_v1v2_prod.cumsum(1) * labels_new_v1v2_prod).cumsum(2) * labels_new_v1v2_prod;
    // Select the unique labels marked with a 1
    auto labels_unique_v1v2 = (labels_new_v1v2_cumsum == labels_new_v1v2_cumsum.constant(1)).select(labels_new_v1v2_cumsum.constant(1), labels_new_v1v2_cumsum.constant(0));
    // Collapse back to 1xnlabels by summing along one of the axis labels dimensions
    auto labels_unique = labels_unique_v1v2.sum(Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1);

    if (this->n_labels_ > 0) {
      // Determine the new labels to add to the axis 
      Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_select_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto indices_select_values_bcast = indices_select_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Broadcast along the labels and along the dimensions and for both the labels and the new labels
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_);
      auto labels_values_bcast = labels_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
      Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_values(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
      auto labels_new_values_bcast = labels_new_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
      // Select and sum along the labels and multiple along the dimensions
      auto labels_selected = (labels_values_bcast == labels_new_values_bcast).select(indices_select_values_bcast.constant(1), indices_select_values_bcast.constant(0)).sum(
        Eigen::array<Eigen::Index, 2>({ 3, 4 })).prod(Eigen::array<Eigen::Index, 1>({ 1 })); // not new > 1, new = 0
      // Invert the selection
      auto labels_new = (labels_selected > labels_selected.constant(0)).select(labels_selected.constant(0), labels_selected.constant(1)); // new = 1, not new = 0

      // Determine the new and unique labels to add to the axis
      auto labels_new_unique = labels_new * labels_unique;
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = labels_new_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    else {
      // Broadcast back to n_dimensions x labels
      auto labels_new_unique_bcast = labels_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

      // Store the selection indices
      indices_select_values.device(device) = labels_new_unique_bcast;
    }
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices_select_ptr = std::make_shared<TensorDataCpu<int, 2>>(indices_select);

    // Determine the number of new labels
    TensorDataCpu<int, 1> n_labels_new(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_labels_new.setData();
    n_labels_new.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_labels_new_values(n_labels_new.getDataPointer().get());
    n_labels_new_values.device(device) = indices_select_values.sum() / n_labels_new_values.constant((int)labels.dimension(0));
    n_labels_new.syncHAndDData(device); // NOTE: need sync for Gpu

    // Allocate memory for the new labels
    TensorDataCpu<TensorT, 2> labels_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, n_labels_new.getData()(0) }));
    labels_select.setData();
    labels_select.syncHAndDData(device);
    std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>> labels_select_ptr = std::make_shared<TensorDataCpu<TensorT, 2>>(labels_select);

    // Select the labels
    labels_converted.select(labels_select_ptr, indices_select_ptr, device);

    // Append the selected labels to the axis
    this->appendLabelsToAxis(labels_select_ptr, device);
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataCpu<TensorT, 2> select_labels(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    select_labels.setData();
    select_labels.syncHAndDData(device);
    select_labels.convertFromStringToTensorT(labels, device);

    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_names_selected_reshape(select_labels.getDataPointer().get(), 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1));
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1 }));

    // broadcast the axis labels the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_reshape(this->tensor_dimension_labels_->getDataPointer().get(), (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1);
    auto labels_bcast = labels_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // broadcast the tensor indices the size of the labels queried
    TensorDataCpu<int, 1> select_indices_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->getNLabels() }));
    select_indices_tmp.setData();
    select_indices_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 4>> indices_reshape(select_indices_tmp.getDataPointer().get(), 1, (int)this->getNLabels(), 1, 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast.constant(1), indices_bcast.constant(0)).sum(
      Eigen::array<Eigen::Index, 2>({ 2, 3 })).prod(Eigen::array<Eigen::Index, 1>({ 0 })).clip(0, 1); // matched = 1, not-matched = 0;

    // assign the select indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> select_indices_values(select_indices_tmp.getDataPointer().get(), (int)this->getNLabels());
    select_indices_values.device(device) = selected;
    select_indices = std::make_shared<TensorDataCpu<int, 1>>(select_indices_tmp);
  }
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray8<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray32<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray128<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray512<char>>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray2048<char>>);
#endif //TENSORBASE_TENSORAXISCPU_H