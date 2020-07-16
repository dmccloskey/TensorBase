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

    @param[in] tensor_collection The tensor collection to apply the reduction to
    @param[in] table_names List of table_names to apply the reduction to
    @param[in] table_names_new List of the new table_names after applying the reduction.
      WARNING: tables_names that match table_names_new will be modified in place.
    @param[in] device
    */
    template<typename DeviceT>
    void applySelect(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, const std::vector<std::string>& table_names_new, DeviceT& device);

    /*
    @brief reduce and concatenate the selected Tables in the Tensor according to the selected indices view
      that have been modified through the `select`, `where`, and `join` clauses

    @param[in] tensor_collection The tensor collection to apply the reduction to
    @param[in] table_LR_names List of pairs of table_names to apply the join to
    @param[in] table_names_new List of the new table_names after applying the reduction.
      WARNING: tables_names that match table_names_new will be modified in place.
    @param[in] device
    */
    template<typename DeviceT>
    void applyJoin(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::pair<std::string, std::string>>& table_LR_names, const std::vector<std::string>& table_names_new, DeviceT& device);

    /*
    @brief sort the selected Tables in the Tensor according to the ordering of the indices view
      that have been modified through the `sort` clause

    @param[in] tensor_collection The tensor collection to apply the reduction to
    @param[in] table_names List of table_names to apply the reduction to
    @param[in] table_names_new List of the new table_names after applying the reduction.
      WARNING: tables_names that match table_names_new will be modified in place.
    @param[in] device
    */
    template<typename DeviceT>
    void applySort(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, const std::vector<std::string>& table_names_new, DeviceT& device);

    /* @brief Select the table/axis/dimension/labels that will be returned in the view.

    Behavior:
      - By default, If the table is a part of a select clause but the axis/dimension is not 
        specified, all labels for the axis/dimensions will be returned
      - By default, If the table is not a part of the select clause, the table will not
        be returned
    */
    template<typename LabelsT, typename DeviceT>
    void selectClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device);

    /// TODO Aggregate table values by an aggregation function resulting in a new axis label with the aggregate value
    template<typename LabelsT, typename DeviceT>
    void applyAggregate(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, AggregateClause<LabelsT, DeviceT>& aggregate_clause, DeviceT& device);

    /// Reduce the table by a reduction function
    template<typename DeviceT>
    void applyReduction(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, ReductionClause<DeviceT>& reduction_clause, DeviceT& device);

    /// Update the values in the table using a scan function
    template<typename DeviceT>
    void applyScan(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, ScanClause<DeviceT>& scan_clause, DeviceT& device);

    /// TODO Merge two tables into one
    template<typename LabelsT, typename DeviceT>
    void joinClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, JoinClause<LabelsT, DeviceT>& join_clause, DeviceT& device);

    /// Select the table/axis/dimension/labels by a boolean expression
    template<typename LabelsT, typename TensorT, typename DeviceT>
    void whereClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device);

    /// Order the selected table/axis/dimension/labels
    template<typename LabelsT, typename DeviceT>
    void sortClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SortClause<LabelsT, DeviceT>& sort_clause, DeviceT& device);
  };

  template<typename DeviceT>
  void TensorSelect::applySelect(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, const std::vector<std::string>& table_names_new, DeviceT& device) {
    assert(table_names.size() == table_names_new.size());
    for (int i = 0; i < table_names.size(); ++i) {
      if (table_names.at(i) == table_names_new.at(i)) {
        tensor_collection->tables_.at(table_names.at(i))->syncAxesAndIndicesDData(device);
        tensor_collection->tables_.at(table_names.at(i))->selectTensorData(device);
      }
    }
  };

  template<typename DeviceT>
  void TensorSelect::applySort(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, const std::vector<std::string>& table_names, const std::vector<std::string>& table_names_new, DeviceT& device) {
    assert(table_names.size() == table_names_new.size());
    for (int i = 0; i < table_names.size(); ++i) {
      if (table_names.at(i) == table_names_new.at(i)) {
        tensor_collection->tables_.at(table_names.at(i))->syncAxesAndIndicesDData(device);
        tensor_collection->tables_.at(table_names.at(i))->sortTensorData(device);
      }
    }
  };

  template<typename LabelsT, typename DeviceT>
  void TensorSelect::selectClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection->tables_.at(select_clause.table_name)->getAxes()) {
      if (axis.first == select_clause.axis_name) {
        // TODO: check for select_clause.axis_name == ""
				if (select_clause.dimension_name == "" && select_clause.axis_labels != nullptr) { 
					// Select option 2
          if (!select_clause.axis_labels->getDataStatus().second) select_clause.axis_labels->syncHAndDData(device);
          tensor_collection->tables_.at(select_clause.table_name)->syncAxesAndIndicesDData(device);
					tensor_collection->tables_.at(select_clause.table_name)->selectIndicesView(
						select_clause.axis_name, select_clause.axis_labels, device);
				}				
				else { 
					// iterate through each axis dimensions
					for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
						if (axis.second->getDimensions()(d) == select_clause.dimension_name) {
							// Select option 1
              if (!select_clause.labels->getDataStatus().second) select_clause.labels->syncHAndDData(device);
              tensor_collection->tables_.at(select_clause.table_name)->syncAxesAndIndicesDData(device);
							tensor_collection->tables_.at(select_clause.table_name)->selectIndicesView(
								select_clause.axis_name, d, select_clause.labels, device);
						}
					}
				}
      }
    }
  }
  template<typename DeviceT>
  inline void TensorSelect::applyReduction(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, ReductionClause<DeviceT>& reduction_clause, DeviceT& device)
  {
    tensor_collection->tables_.at(reduction_clause.table_name)->reduceTensorDataConcept(reduction_clause.reduction_function, device);
  }
  template<typename DeviceT>
  inline void TensorSelect::applyScan(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, ScanClause<DeviceT>& scan_clause, DeviceT& device)
  {
    tensor_collection->tables_.at(scan_clause.table_name)->scanTensorDataConcept(scan_clause.axes_names, scan_clause.scan_function, device);
  }
  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect::whereClause(std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device) {
    // iterate through each table axis
    for (auto& axis : tensor_collection->tables_.at(where_clause.table_name)->getAxes()) {
      if (axis.first == where_clause.axis_name) {
        // TODO: check for select_clause.axis_name == ""
        if (where_clause.dimension_name == "" && where_clause.axis_labels != nullptr) {
          // Select option 2
          if (!where_clause.axis_labels->getDataStatus().second) where_clause.axis_labels->syncHAndDData(device);
          tensor_collection->tables_.at(where_clause.table_name)->syncAxesAndIndicesDData(device);
          tensor_collection->tables_.at(where_clause.table_name)->whereIndicesView(
            where_clause.axis_name, where_clause.axis_labels,
            where_clause.values, where_clause.comparitor, where_clause.modifier,
            where_clause.within_continuator, where_clause.prepend_continuator, device);
        }
        else {
          // iterate through each axis dimensions
          for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
            if (axis.second->getDimensions()(d) == where_clause.dimension_name) {
              // select axis indices based on the where clause critiera
              if (!where_clause.values->getDataStatus().second) where_clause.values->syncHAndDData(device);
              if (!where_clause.labels->getDataStatus().second) where_clause.labels->syncHAndDData(device);
              tensor_collection->tables_.at(where_clause.table_name)->syncAxesAndIndicesDData(device);
              tensor_collection->tables_.at(where_clause.table_name)->whereIndicesView(
                where_clause.axis_name, d, where_clause.labels,
                where_clause.values, where_clause.comparitor, where_clause.modifier,
                where_clause.within_continuator, where_clause.prepend_continuator, device);
            }
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
        if (sort_clause.dimension_name == "" && sort_clause.axis_labels != nullptr) {
          // Select option 2
          if (!sort_clause.axis_labels->getDataStatus().second) sort_clause.axis_labels->syncHAndDData(device);
          tensor_collection->tables_.at(sort_clause.table_name)->syncAxesAndIndicesDData(device);
          tensor_collection->tables_.at(sort_clause.table_name)->sortIndicesView(
            sort_clause.axis_name, sort_clause.axis_labels, sort_clause.order_by, device);
        }
        else {
          // iterate through each axis dimensions
          for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
            if (axis.second->getDimensions()(d) == sort_clause.dimension_name) {
              // order the indices view
              if (!sort_clause.labels->getDataStatus().second) sort_clause.labels->syncHAndDData(device);
              tensor_collection->tables_.at(sort_clause.table_name)->syncAxesAndIndicesDData(device);
              tensor_collection->tables_.at(sort_clause.table_name)->sortIndicesView(
                sort_clause.axis_name, d, sort_clause.labels, sort_clause.order_by, device);
            }
          }
        }
      }
    }
  };
  
  class TensorSelectUnion;
  class TensorSelectJoin;
};
#endif //TENSORBASE_TENSORSELECT_H