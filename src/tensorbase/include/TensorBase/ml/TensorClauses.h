/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCLAUSE_H
#define TENSORBASE_TENSORCLAUSE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

namespace TensorBase
{
  /*
  @brief Class defining the `Select` clause statements that selects particular axis, dimensions, and/or label indices
    to be returned or further used.  Specifying only an axis has the effect of selecting all dimensions and indices in the axis.
    Specifying the axis and dimension has the effect of selecting all indices in the axis/dimension.
    Specifying the axis, dimension, and labels has the effect of selecting only the axis indices corresponding to the
    named axis/dimension/labels.
  */
  template<typename LabelsT, typename DeviceT>
  class SelectClause {
  public:
    SelectClause() = default;
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels) { };
    ~SelectClause() = default;
    std::string table_name; ///< the table to select
    std::string axis_name = ""; ///< the axis to select
    std::string dimension_name = ""; ///< the dimension to select
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels = nullptr; ///< the labels to select
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
      CUSTOM,
      NONE
    };
  };
  /*
  @brief Class defining the `reduction` clause statements.  Specified axis from the same
    table will be reduced using the reduction function for all selected indices resulting in 
    a tensor with dimensions = TDim - axes_names.siz()
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
    enum logicalModifier {
      NOT,
      NONE
    };
    enum logicalContinuator {
      AND,
      OR
    };
  };
  /*
  @param Class defining the `Where` clause statements that filters axis indices based on the
    selection criteria that is applied across all selected indices.  If the Tensor is of TDim > 2
    an `OR` clause will be applied to aggregate all other non-target selection axis
  */
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class WhereClause : public SelectClause<TensorT, DeviceT> {
  public:
    WhereClause() = default;
    WhereClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitors::logicalComparitor& comparitor,
      const logicalComparitors::logicalModifier& modifier, const logicalComparitors::logicalContinuator& prepend_continuator, const logicalComparitors::logicalContinuator& within_continuator) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels),
      values(values), comparitor(comparitor),
      modifier(modifier), prepend_continuator(prepend_continuator), within_continuator(within_continuator) { };
    ~WhereClause() = default;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> values;
    logicalComparitors::logicalComparitor comparitor;
    logicalComparitors::logicalModifier modifier;
    logicalComparitors::logicalContinuator prepend_continuator;
    logicalComparitors::logicalContinuator within_continuator;
  };

  struct sortOrder {
    enum order { ASC, DESC };
  };
  /*
  @brief Class defining the `Order by` clause that orders the tensor according to the
    specified axis and label.  If the tensor is of TDims > 2, then the values in the first index of 
    all non-target sort axis will be used
  */
  template<typename LabelsT, typename DeviceT>
  class SortClause {
  public:
    SortClause() = default;
    SortClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name,
      const LabelsT& label, const sortOrder::order& order_by) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), label(label), order_by(order_by) { };
    ~SortClause() = default;
    std::string table_name;
    std::string axis_name;
    std::string dimension_name;
    LabelsT label;
    sortOrder::order order_by;
  };
};
#endif //TENSORBASE_TENSORCLAUSE_H