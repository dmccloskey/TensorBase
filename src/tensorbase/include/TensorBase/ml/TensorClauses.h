/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCLAUSE_H
#define TENSORBASE_TENSORCLAUSE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

namespace TensorBase
{
  template<typename LabelsT, typename DeviceT>
  class SelectClause {
  public:
    SelectClause() = default;
    SelectClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels) { };
    ~SelectClause() = default;
    std::string table_name;
    std::string axis_name;
    std::string dimension_name;
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> labels;
  };

  enum aggregateFunction {
    // Wrappers around Eigen::Tensor Reduction and Scan operations
    MIN,
    MAX,
    MEAN,
    COUNT,
    SUM,
    PROD,
    CUMSUM,
    CUMPROD,
    CUSTOM
  };
  /// Class defining the `aggregate` clause statements
  template<typename LabelsT, typename TensorT, typename DeviceT>
  class AggregateClause : public SelectClause<TensorT, DeviceT> {
  public:
    AggregateClause() = default;
    AggregateClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels,
      const aggregateFunction& agg_functions, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& agg_labels) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels),
      agg_functions(agg_functions), agg_labels(agg_labels) { };
    ~AggregateClause() = default;
    aggregateFunction agg_functions; ///< the agg_function to apply to each label
    std::shared_ptr<TensorData<LabelsT, DeviceT, 1>> agg_labels;  ///< the resulting agg_function labels
  };

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
    WhereClause() = default;
    WhereClause(const std::string& table_name, const std::string& axis_name, const std::string& dimension_name, const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels,
      const std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& values, const logicalComparitor& comparitor,
      const logicalModifier& modifier, const logicalContinuator& prepend_continuator, const logicalContinuator& within_continuator) :
      table_name(table_name), axis_name(axis_name), dimension_name(dimension_name), labels(labels),
      values(values), comparitor(comparitor),
      modifier(modifier), prepend_continuator(prepend_continuator), within_continuator(within_continuator) { };
    ~WhereClause() = default;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> values;
    logicalComparitor comparitor;
    logicalModifier modifier;
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
      const std::shared_ptr<TensorData<LabelsT, DeviceT, 1>>& labels, const std::vector<order>& order_by) :
      SelectClause(table_name, axis_name, dimension_name, labels), order_by(order_by) { };
    ~OrderByClause() = default;
    std::vector<order> order_by;
  };
};
#endif //TENSORBASE_TENSORCLAUSE_H