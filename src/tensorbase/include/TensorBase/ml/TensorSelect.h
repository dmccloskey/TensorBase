/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorOperation.h>
#include <set>

namespace TensorBase
{
  /**
    @brief Template class for all Tensor select operations

    NOTES: order of execution
    1. Where: axes and data are reduced after all where clauses have been applied
    2. Select: axes and data are reduced after all select clauses have been applied
    3. Group By: 
    4. Aggregate: additional dimension label is and data are added after all aggregate clauses have been applied
    5. Having: axes and data are reduced after all having clauses have been applied
    6. Order By: axes and data are sorted after all order by clauses have ben applied
  */
  class TensorSelect {
  public:
    TensorSelect() = default;
    ~TensorSelect() = default;

    /*
    @brief reduce the selected Tables in the Tensor according to the selected indices view
      that have been modified through the `select` and `where` clauses
    */
    template<typename DeviceT>
    void applySelect(TensorCollection& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device);

    /*
    @brief sort the selected Tables in the Tensor according to the ordering of the indices view
      that have been modified through the `sort` clause
    */
    template<typename DeviceT>
    void applySort(TensorCollection& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device);

    /* @brief Select the table/axis/dimension/labels that will be returned in the view.

    Behavior:
      - By default, If the table is a part of a select clause but the axis/dimension is not 
        specified, all labels for the axis/dimensions will be returned
      - By default, If the table is not a part of the select clause, the table will not
        be returned
    */
    template<typename LabelsT, typename DeviceT>
    void selectClause(TensorCollection& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device);

    /// TODO Reduce the table by a reduction function
    template<typename DeviceT>
    void reductionClause(TensorCollection& tensor_collection, ReductionClause<DeviceT>& reduction_clause, DeviceT& device);

    /// Select the table/axis/dimension/labels by a boolean expression
    template<typename LabelsT, typename TensorT, typename DeviceT>
    void whereClause(TensorCollection& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device);

    /// Order the selected table/axis/dimension/labels
    template<typename LabelsT, typename DeviceT>
    void sortClause(TensorCollection& tensor_collection, SortClause<LabelsT, DeviceT>& sort_clause, DeviceT& device);
  };

  template<typename DeviceT>
  void TensorSelect::applySelect(TensorCollection& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device) {
    for (const std::string& table_name : table_names) {
      tensor_collection.tables_.at(table_name)->selectTensorData(device);
    }
  };

  template<typename DeviceT>
  void TensorSelect::applySort(TensorCollection& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device) {
    for (const std::string& table_name : table_names) {
      tensor_collection.tables_.at(select_clause.table_name)->sortTensorData(device);
    }
  };

  template<typename LabelsT, typename DeviceT>
  void TensorSelect::selectClause(TensorCollection& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(select_clause.table_name)->getAxes()) {
      if (axis.first == select_clause.axis_name) {
        // TODO: check for select_clause.axis_name == ""
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          // TODO: check for select_clause.dimension_name == ""
          if (axis.second->getDimensions()(d) == select_clause.dimension_name) {
            // copy over indices into the view that are in the select clause
            tensor_collection.tables_.at(select_clause.table_name)->selectIndicesView(
              select_clause.axis_name, d, select_clause.labels, device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect::whereClause(TensorCollection& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(where_clause.table_name)->getAxes()) {
      if (axis.first == where_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == where_clause.dimension_name) {
            // select axis indices based on the where clause critiera
            tensor_collection.tables_.at(where_clause.table_name)->whereIndicesView(
              where_clause.axis_name, d, where_clause.labels, 
              where_clause.values, where_clause.comparitor, where_clause.modifier, 
              where_clause.within_continuator, where_clause.prepend_continuator, device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename DeviceT>
  void TensorSelect::sortClause(TensorCollection& tensor_collection, SortClause<LabelsT, DeviceT>& sort_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection.tables_.at(sort_clause.table_name)->getAxes()) {
      if (axis.first == sort_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == sort_clause.dimension_name) {
            // order the indices view
            tensor_collection.tables_.at(sort_clause.table_name)->sortIndicesView(
              sort_clause.axis_name, d, sort_clause.label, sort_clause.order_by, device);
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