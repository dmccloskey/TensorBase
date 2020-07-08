/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCLAUSE_H
#define TENSORBASE_TENSORCLAUSE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>

namespace TensorBase
{
  /*
  @brief Class defining the `Select` clause statements that selects particular axis, dimensions, and/or label indices.  

  The class provides two interfaces for querying according to "Option 1" and "Option 2"
    Option 1: labels (1D tensor) that match on the `table_name`, `axis_name`, and `dimension_name` are selected
    Option 2: labels (2D tensor) that match on the `table_name` and `axis_name` are selected

  [TODO]
  The class also allows the user to omit specifying the `labels`, `dimensions_name`, and/or `axis_name` attributes
    in order to more rapidly query all labels/dimensions/axes of a table.  The behavior is as followings:  
    - Specifying only an axis has the effect of selecting all dimensions and indices in the axis.
    - Specifying the axis and dimension has the effect of selecting all indices in the axis/dimension.
    - Specifying the axis, dimension, and labels has the effect of selecting only the axis indices corresponding to the
      named axis/dimension/labels.

  NOTES:
    - The user can execute multiple select and where statements in a defined order in order to select the regions of interest
  */
  template<typename LabelsT, typename DeviceT>
  class SelectClause {
  public:
    SelectClause() = default;
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels) { };
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& axis_labels) :
      table_name(table_name), axis_name(axis_name), axis_labels(axis_labels) { };
    ~SelectClause() = default;
    std::string table_name; ///< the table to select (option 1 and 2)
    std::string axis_name = ""; ///< the axis to select (option 1 and 2)
    std::string dimension_name = ""; ///< the dimension to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels = nullptr; ///< the labels to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> axis_labels = nullptr; ///< the labels to select (option 2)
  };

  struct aggregateFunctions {
    enum aggregateFunction {
      // Wrappers around Eigen::Tensor Reduction operations
      MIN,
      MAX,
      MEAN,
      COUNT,
      SUM,
      DINSTINCT, // unique values
      // TODO: other SQL standard aggregate functions
      CUSTOM, // TODO: example implementation for StDev and %RSD
      NONE
    };
  };
  /*
  @brief Class defining the `aggregate` clause statement that selects particular axis, dimensions, and/or label indices
    and performs and applies an aggregation function to the selected data resulting in a new entry with the aggregated data
    broadcasted to all other dimensions of the Tensor.
  */
  template<typename LabelsT, typename DeviceT>
  class AggregateClause: public SelectClause<LabelsT, DeviceT> {
  public:
    AggregateClause() = default;
    AggregateClause(const std::string & table_name, const std::string & axis_name, const std::string & dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> & labels, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& as_axis_labels, const aggregateFunctions::aggregateFunctions& aggregate_function) :
      SelectClause(table_name, axis_name, axis_labels), as_axis_labels(as_axis_labels), aggregate_function(aggregate_function){ };
    AggregateClause(const std::string & table_name, const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> & axis_labels, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& as_axis_labels, const aggregateFunctions::aggregateFunctions& aggregate_function) :
      SelectClause(table_name, axis_name, axis_labels), as_axis_labels(as_axis_labels), aggregate_function(aggregate_function) { };
    AggregateClause(const std::string& table_name, const std::vector<std::string>& axes_names,
      const aggregateFunctions::aggregateFunctions& reduction_function) :
      table_name(table_name), axes_names(axes_names),
      reduction_function(reduction_function) { };
    ~ReductionClause() = default;
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> as_axis_labels = nullptr; ///< the labels to select (option 2)
    aggregateFunctions::aggregateFunctions aggregate_function;
  };

  struct joinTypes {
    enum joinType {
      INNER, // Default
      LEFT,
      RIGHT,
      FULL
    };
  };
  /*
  @brief Class defining the `join` clause statement that joins two tables along a particular axis, dimensions, and/or label indices.

  NOTES:
    - The table dimensions must be equal 
    - The method will match ON the user supplied table axis, dimensions, and/or label indices
    - It is assumed that the user has already selected other axes, dimensions, and/or labels to consider during execution prior to running the Join clause
  */
  template<typename LabelsT, typename DeviceT>
  class JoinClause {
  public:
    JoinClause() = default;
    SelectClause(const std::string & table_name, const std::string & axis_name, const std::string & dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> & labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels) { };
    SelectClause(const std::string & table_name, const std::string & axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> & axis_labels) :
      table_name(table_name), axis_name(axis_name), axis_labels(axis_labels) { };
    JoinClause(const std::string& table_name, const std::vector<std::string>& axes_names,
      const aggregateFunctions::aggregateFunctions& reduction_function) :
      table_name(table_name), axes_names(axes_names),
      reduction_function(reduction_function) { };
    ~JoinClause() = default;
    std::string table_name_l; ///< the table to select (option 1 and 2)
    std::string axis_name_l = ""; ///< the axis to select (option 1 and 2)
    std::string dimension_name_l = ""; ///< the dimension to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels_l = nullptr; ///< the labels to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> axis_labels_l = nullptr; ///< the labels to select (option 2)
    std::string table_name_r; ///< the table to select (option 1 and 2)
    std::string axis_name_r = ""; ///< the axis to select (option 1 and 2)
    std::string dimension_name_r = ""; ///< the dimension to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels_r = nullptr; ///< the labels to select (option 1)
    std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> axis_labels_r = nullptr; ///< the labels to select (option 2)
    joinTypes::joinTypes join_type;
  };

  struct reductionFunctions {
    enum reductionFunction {
      // Wrappers around Eigen::Tensor Reduction operations
      MIN,
      MAX,
      MEAN,
      COUNT,
      SUM,
      PROD,
      CUSTOM, // TODO: example implementation for StDev and %RSD
      NONE
    };
  };
  /*
  @brief Class defining the `reduction` clause statements.  Specified axis from the same
    table will be reduced using the reduction function for all selected indices resulting in 
    a tensor with dimensions = TDim - axes_names.size()
  */
  template<typename DeviceT>
  class ReductionClause {
  public:
    ReductionClause() = default;
    ReductionClause(const std::string& table_name, const std::vector<std::string>& axes_names,
      const reductionFunctions::reductionFunction& reduction_function) :
      table_name(table_name), axes_names(axes_names),
      reduction_function(reduction_function) { };
    ~ReductionClause() = default;
    std::string table_name;
    std::vector<std::string> axes_names;
    reductionFunctions::reductionFunction reduction_function; ///< the reduction_function to apply across each of the labels
  };

  struct scanFunctions {
    enum scanFunction {
      // Wrappers around Eigen::Tensor Scan operations
      CUMSUM,
      CUMPROD,
      CUSTOM,
      NONE
    };
  };
  /*
  @brief Class defining the `scan` clause statements.  Specified axis from the same
    table will be reduced using the reduction function and replaced in-place using
    the running total of the scan operation
  */
  template<typename DeviceT>
  class ScanClause {
  public:
    ScanClause() = default;
    ScanClause(const std::string& table_name, const std::vector<std::string>& axes_names,
      const scanFunctions::scanFunction& scan_function) :
      table_name(table_name), axes_names(axes_names),
      scan_function(scan_function) { };
    ~ScanClause() = default;
    std::string table_name;
    std::vector<std::string> axes_names;
    scanFunctions::scanFunction scan_function; ///< the reduction_function to apply across each of the labels
  };

  struct logicalComparitors {
    enum logicalComparitor {
      // Wrappers around Eigen::Tensor select operations
      LESS_THAN_OR_EQUAL_TO,
      GREATER_THAN_OR_EQUAL_TO,
      LESS_THAN,
      GREATER_THAN,
      EQUAL_TO,
      NOT_EQUAL_TO
    };
  };
  struct logicalModifiers {
    enum logicalModifier {
      NOT,
      NONE
    };
  };
  struct logicalContinuators {
    enum logicalContinuator {
      AND,
      OR
    };
  };
  /*
  @param Class defining the `Where` clause statements that filters axis indices based on the
    selection criteria that is applied across all selected indices.  If the Tensor is of TDim > 2
    an `OR` clause will be applied to aggregate all other non-target selection axis

  TODO: Missing support for using another table's values as the RHS comparison.

  NOTES:
    - The user can execute multiple select and where statements in a defined order in order to select the regions of interest
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class WhereClause : public SelectClause<LabelsT, DeviceT> {
  public:
    WhereClause() = default;
    WhereClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor,
      const logicalModifiers::logicalModifier& modifier, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator) :
      SelectClause(table_name, axis_name, dimension_name, labels),
      values(values), comparitor(comparitor),
      modifier(modifier), within_continuator(within_continuator), prepend_continuator(prepend_continuator) { };
    WhereClause(const std::string& table_name, const std::string& axis_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& axis_labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor,
      const logicalModifiers::logicalModifier& modifier, const logicalContinuators::logicalContinuator& within_continuator, const logicalContinuators::logicalContinuator& prepend_continuator) :
      SelectClause(table_name, axis_name, axis_labels),
      values(values), comparitor(comparitor),
      modifier(modifier), within_continuator(within_continuator), prepend_continuator(prepend_continuator) { };
    ~WhereClause() = default;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> values;
    logicalComparitors::logicalComparitor comparitor;
    logicalModifiers::logicalModifier modifier;
    logicalContinuators::logicalContinuator prepend_continuator;
    logicalContinuators::logicalContinuator within_continuator;
  };

  struct sortOrder {
    enum order { ASC, DESC };
  };
  /*
  @brief Class defining the `Order by` clause that orders the tensor according to the
    specified axis and label.  
    
    If the tensor is of TDims > 2, then the values in the first index of 
    all non-target sort axis will be used.
  */
  template<typename LabelsT, typename DeviceT>
  class SortClause: public SelectClause<LabelsT, DeviceT> {
  public:
    SortClause() = default;
    SortClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name,
      const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels, const sortOrder::order& order_by) :
      SelectClause(table_name, axis_name, dimension_name, labels), order_by(order_by) { };
    SortClause(const std::string& table_name, const std::string& axis_name,
      const std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& axis_labels, const sortOrder::order& order_by) :
      SelectClause(table_name, axis_name, axis_labels), order_by(order_by) { };
    ~SortClause() = default;
    sortOrder::order order_by = sortOrder::ASC;
  };
};
#endif //TENSORBASE_TENSORCLAUSE_H