/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>
#include <set>

namespace TensorBase
{
  template<typename TensorT, typename DeviceT>
  class SelectClause {
  public:
    SelectClause() = default;
    ~SelectClause() = default;
    std::string table_name;
    std::string axis_name;
    std::string dimension_name;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels;
  };

  /// Class defining the `Where` clause statements
  // TODO use type erasure to make a vector of pointers to functors of the type shown
  template<typename TensorT, typename DeviceT, int TDim>
  class WhereClause {
  public:
    std::vector<std::function<void(TensorT* data_in, TensorT* data_out, DeviceT& device)>> tensor_predicate;
  };

  /// Class defining the `Order By` clause statements
  template<typename TensorT, typename DeviceT>
  class OrderByClause : public SelectClause<TensorT, DeviceT> {
  public:
    enum order { ASC, DESC };
    std::vector<order> order_by;
  };

  /**
    @brief Template class for all Tensor select operations
  */
  template<typename TensorT, typename DeviceT>
  class TensorSelect {
  public:
    /* Select the table/axis/dimension/labels that will be returned in the view.

    Behavior:
      - By default, If the table is a part of a select clause but the axis/dimension is not 
        specified, all labels for the axis/dimensions will be returned
      - By default, If the table is not a part of the select clause, the table will not
        be returned
    */
    virtual void selectClause(TensorCollection& tensor_collection, SelectClause<TensorT, DeviceT>& select_clause, DeviceT& device) = 0;

    /// Select the table/axis/dimension/labels by a boolean expression
    virtual void whereClause(TensorCollection& tensor_collection, DeviceT& device) = 0;

    /// Select group the dimensions by non-unique values
    virtual void groupByClause(TensorCollection& tensor_collectiont, SelectClause<TensorT, DeviceT>& group_by_clause, DeviceT& device) = 0;

    /// ?
    virtual void havingClause(TensorCollection& tensor_collection, SelectClause<TensorT, DeviceT>& having_clause, DeviceT& device) = 0;

    /// Order the selected table/axis/dimension/labels
    virtual void orderByClause(TensorCollection& tensor_collection, OrderByClause<TensorT, DeviceT>& order_by_clause, DeviceT& device) = 0;
  protected:
    std::set<std::string> selected_tables_;
    std::set<std::string> selected_axes_;
  };

  template<typename TensorT, typename DeviceT>
  void TensorSelect<TensorT, DeviceT>::selectClause(TensorCollection& tensor_collection, SelectClause<TensorT, DeviceT>& select_clause, DeviceT& device) {
    // iterate throgh each table axis
    for (auto& axis : tensor_collection.tables_.at(select_clause.table_name)->getAxes()) {
      if (axis.first == select_clause.axis_name) {
        // record the selected tables and axes
        selected_tables_.insert(select_clause.table_name);
        selected_axes_.insert(select_clause.axis_name);erate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == select_clause.dimension_name) {
            // zero the view for the axis (only once)
            if (selected_axes_.count(select_clause.axis_name) == 0) {
              // TODO: call to TensorTable
            }
            // copy over indices into the view that are in the select clause
            // TODO: call to TensorTable (const std::string& axis_name, const int& dimension_index, TensorT* select_labels_data, n_labels, const DeviceT& deviceT);
            // reshape to match the axis labels shape
            Eigen::TensorMap<Eigen::Tensor<std::string, 2>> labels_reshape(select_clause.labels->getHDataPointer(), 1, labels_selected.size());
            // broadcast the length of the labels
            auto labels_names_selected_bcast = labels_reshape.broadcast(Eigen::array<int, 2>({ (int)axis.second->getNLabels(), 1 }));
            // broadcast the axis labels the size of the labels queried
            Eigen::TensorMap<Eigen::Tensor<std::string, 3>> labels_reshape(axis.second->getLabels()->getHDataPointer().get(), (int)axis.second->getNDimensions(), (int)axis.second->getNLabels(), 1);
            auto labels_bcast = (labels_reshape.chip(d, 0)).broadcast(Eigen::array<int, 2>({ 1, (int)labels_selected.size() }));
            // broadcast the tensor indices the size of the labels queried
            Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(tensor_collection.tables_.at(select_clause.table_name)->getIndices().at(select_clause.axis_name)->data(),
              (int)axis.second->getNLabels(), 1);
            auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, (int)labels_selected.size() }));
            auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
            auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
            Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(tensor_collection.tables_.at(select_clause.table_name)->getIndicesView().at(select_clause.axis_name)->data(), (int)axis.second->getNLabels());
            indices_view.device(device) += selected_sum;
          }
        }
      }
    }
  };

  template<int TDim>
  class TensorSelectSlices {
  public:
    TensorSelectSlices(const std::string table_name,
      const Eigen::array<std::string, TDim>& axes_names,
      const Eigen::array<std::string, TDim>& dimension_names,
      const Eigen::array<std::string, TDim>& labels_start,
      const Eigen::array<std::string, TDim>& labels_stop) :
      table_name(table_name), axes_names(axes_names), dimension_names(dimension_names),
      labels_start(labels_start), labels_stop(labels_stop) {
      for (int iter = 0; iter < TDim; ++iter) {
        offsets.at(iter) = -1;
        exten-1;
      }
    };
    template<typename T>
    void operator()(T&& t){
      if (std::forward<decltype(t)>(t)->getName() == table_name) {
        for (auto& axis : std::forward<decltype(t)>(t)->getAxes()) {
          for (int iter = 0; iter < TDim; ++iter) {
            if (axis.first == axes_names.at(iter)) {
              for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
                if (axis.second->getDimensions()(d) == dimension_names.at(iter)) {
                  for (int l = 0; l < axis.second->getLabels().dimension(1); ++l) {
                    if (axis.second->getLabels()(d, l) == labels_start.at(iter)) {
                      offsets.at(iter) = l;
                    }
                    else if (axis.second->getLabels()(d, l) == labels_stop.at(iter)) {
                      extents.at(iter) = l - d + 1;
                    }
                    if (extents.at(iter) != -1 && offsets.at(iter) != -1)
                      break;
                  }
                  break;
                }
              }
              break;
            }
          }
        }
      }
    };
    std::string table_name;
    Eigen::array<std::string, TDim> axes_names;
    Eigen::array<std::string, TDim> dimension_names;
    Eigen::array<std::string, TDim> labels_start;
    Eigen::array<std::string, TDim> labels_stop;
    Eigen::array<int, TDim> offsets;
    Eigen::array<int, TDim> extents;
  };

  class TensorSelectUnion;
  class TensorSelectJoin;
};
#endif //TENSORBASE_TENSOROPERATION_H