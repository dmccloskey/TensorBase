/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORSELECT_H
#define TENSORBASE_TENSORSELECT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <set>

namespace TensorBase
{
  /**
    @brief Class for all Tensor select operations
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
    void applySelect(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device);

    /*
    @brief sort the selected Tables in the Tensor according to the ordering of the indices view
      that have been modified through the `sort` clause
    */
    template<typename DeviceT>
    void applySort(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device);

    /* @brief Select the table/axis/dimension/labels that will be returned in the view.

    Behavior:
      - By default, If the table is a part of a select clause but the axis/dimension is not 
        specified, all labels for the axis/dimensions will be returned
      - By default, If the table is not a part of the select clause, the table will not
        be returned
    */
    template<typename LabelsT, typename DeviceT>
    void selectClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device);

    /// TODO Reduce the table by a reduction function
    template<typename DeviceT>
    void reductionClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, ReductionClause<DeviceT>& reduction_clause, DeviceT& device);

    /// Select the table/axis/dimension/labels by a boolean expression
    template<typename LabelsT, typename TensorT, typename DeviceT>
    void whereClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device);

    /// Order the selected table/axis/dimension/labels
    template<typename LabelsT, typename DeviceT>
    void sortClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SortClause<LabelsT, DeviceT>& sort_clause, DeviceT& device);
  };

  template<typename DeviceT>
  void TensorSelect::applySelect(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device) {
    for (const std::string& table_name : table_names) {
      tensor_collection->tables_.at(table_name)->selectTensorData(device);
    }
  };

  template<typename DeviceT>
  void TensorSelect::applySort(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, DeviceT& device) {
    for (const std::string& table_name : table_names) {
      tensor_collection->tables_.at(table_name)->sortTensorData(device);
    }
  };

  template<typename LabelsT, typename DeviceT>
  void TensorSelect::selectClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection->tables_.at(select_clause.table_name)->getAxes()) {
      if (axis.first == select_clause.axis_name) {
        // TODO: check for select_clause.axis_name == ""
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          // TODO: check for select_clause.dimension_name == ""
          if (axis.second->getDimensions()(d) == select_clause.dimension_name) {
            // copy over indices into the view that are in the select clause
            tensor_collection->tables_.at(select_clause.table_name)->selectIndicesView(
              select_clause.axis_name, d, select_clause.labels, device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect::whereClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection->tables_.at(where_clause.table_name)->getAxes()) {
      if (axis.first == where_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == where_clause.dimension_name) {
            // select axis indices based on the where clause critiera
            tensor_collection->tables_.at(where_clause.table_name)->whereIndicesView(
              where_clause.axis_name, d, where_clause.labels, 
              where_clause.values, where_clause.comparitor, where_clause.modifier, 
              where_clause.within_continuator, where_clause.prepend_continuator, device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename DeviceT>
  void TensorSelect::sortClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SortClause<LabelsT, DeviceT>& sort_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection->tables_.at(sort_clause.table_name)->getAxes()) {
      if (axis.first == sort_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == sort_clause.dimension_name) {
            // order the indices view
            tensor_collection->tables_.at(sort_clause.table_name)->sortIndicesView(
              sort_clause.axis_name, d, sort_clause.labels, sort_clause.order_by, device);
          }
        }
      }
    }
  };
  
  class TensorSelectUnion;
  class TensorSelectJoin;
};
#endif //TENSORBASE_TENSORSELECT_H