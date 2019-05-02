/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>
#include <set>

namespace TensorBase
{
  template<typename LabelsT, typename DeviceT>
  class SelectClause {
  public:
    SelectClause() = default;
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels) { };
    ~SelectClause() = default;
    std::string table_name;
    std::string axis_name;
    std::string dimension_name;
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels;
  };

  enum logicalComparitor {
    LESS_THAN_OR_EQUAL_TO,
    GREATER_THAN_OR_EQUAL_TO,
    LESS_THAN,
    GREATER_THAN,
    EQUAL_TO,
    NOT_EQUAL_TO
  };
  enum logicalModifier {
    NOT
  };
  enum logicalContinuator {
    AND,
    OR
  };
  /// Class defining the `Where` clause statements
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class WhereClause : public SelectClause<TensorT, DeviceT> {
  public:
    SelectClause() = default;
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const std::vector<logicalComparitor>& comparitors,
      const std::vector<logicalModifier>& modifiers, const logicalContinuator& prepend_continuator, const logicalContinuator& within_continuator) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels),
      modifiers(modifiers), prepend_continuator(prepend_continuator), within_continuator(within_continuator){ };
    ~SelectClause() = default;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> values;
    std::vector<logicalComparitor> comparitors;
    std::vector<logicalModifier> modifiers;
    logicalContinuator prepend_continuator;
    logicalContinuator within_continuator;
  };

  enum order { ASC, DESC };
  /// Class defining the `Order By` clause statements
  template<typename LabelsT, typename DeviceT>
  class OrderByClause : public SelectClause<LabelsT, DeviceT> {
  public:
    OrderByClause() = default;
    OrderByClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, 
      const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels, const std::vector<order>& order_by) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels), order_by(order_by) { };
    ~SelectClause() = default;
    std::vector<order> order_by;
  };

  /**
    @brief Template class for all Tensor select operations
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class TensorSelect {
  public:
    TensorSelect() = default;
    ~TensorSelect() = default;
    /* Select the table/axis/dimension/labels that will be returned in the view.

    Behavior:
      - By default, If the table is a part of a select clause but the axis/dimension is not 
        specified, all labels for the axis/dimensions will be returned
      - By default, If the table is not a part of the select clause, the table will not
        be returned
    */
    virtual void selectClause(TensorCollection& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device);

    /// Select the table/axis/dimension/labels by a boolean expression
    virtual void whereClause(TensorCollection& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device);

    /// Select group the dimensions by non-unique values
    virtual void groupByClause(TensorCollection& tensor_collectiont, SelectClause<LabelsT, DeviceT>& group_by_clause, DeviceT& device) {};

    /// Select the grouped table/axis/dimension/labels by a boolean expression
    virtual void havingClause(TensorCollection& tensor_collection, SelectClause<LabelsT, DeviceT>& having_clause, DeviceT& device) {};

    /// Order the selected table/axis/dimension/labels
    virtual void orderByClause(TensorCollection& tensor_collection, OrderByClause<LabelsT, DeviceT>& order_by_clause, DeviceT& device);
  protected:
    std::set<std::string> selected_tables_;
    std::set<std::string> selected_axes_;
  };

  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect<LabelsT, TensorT, DeviceT>::selectClause(TensorCollection& tensor_collection, SelectClause<LabelsT, DeviceT>& select_clause, DeviceT& device) {
    // iterate throgh each table axis
    for (auto& axis : tensor_collection.tables_.at(select_clause.table_name)->getAxes()) {
      if (axis.first == select_clause.axis_name) {
        // record the selected tables and axes
        selected_tables_.insert(select_clause.table_name);
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == select_clause.dimension_name) {
            // zero the view for the axis (only once)
            if (selected_axes_.count(select_clause.axis_name) == 0) {
              tensor_collection.tables_.at(select_clause.table_name)->zeroIndicesView(select_clause.axis_name, device);
              selected_axes_.insert(select_clause.axis_name);
            }
            // copy over indices into the view that are in the select clause
            tensor_collection.tables_.at(select_clause.table_name)->selectIndicesView(
              select_clause.axis_name, d, select_clause.labels->getDataPointer(), select_clause.labels->getData().size(), device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect<LabelsT, TensorT, DeviceT>::whereClause(TensorCollection& tensor_collection, WhereClause<LabelsT, TensorT, DeviceT>& where_clause, DeviceT& device) {
    // iterate throgh each table axis
    for (auto& axis : tensor_collection.tables_.at(where_clause.table_name)->getAxes()) {
      if (axis.first == where_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == where_clause.dimension_name) {
            // order the indices view
            tensor_collection.tables_.at(where_clause.table_name)->orderIndicesView( // TODO: update with where specific call
              where_clause.axis_name, d, where_clause.labels->getDataPointer(), where_clause.labels->getData().size(), device);
          }
        }
      }
    }
  };

  template<typename LabelsT, typename TensorT, typename DeviceT>
  void TensorSelect<LabelsT, TensorT, DeviceT>::orderByClause(TensorCollection& tensor_collection, OrderByClause<LabelsT, DeviceT>& order_by_clause, DeviceT& device) {
    // iterate throgh each table axis
    for (auto& axis : tensor_collection.tables_.at(order_by_clause.table_name)->getAxes()) {
      if (axis.first == order_by_clause.axis_name) {
        // iterate through each axis dimensions
        for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
          if (axis.second->getDimensions()(d) == order_by_clause.dimension_name) {
            // order the indices view
            tensor_collection.tables_.at(order_by_clause.table_name)->orderIndicesView(
              order_by_clause.axis_name, d, order_by_clause.labels->getDataPointer(), order_by_clause.labels->getData().size(), device);
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