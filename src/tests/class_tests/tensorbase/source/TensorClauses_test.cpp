/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorClauses test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorClauses.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorClauses1)

/*SelectClause DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorSelectClauseDefaultDevice)
{
  SelectClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  SelectClause<int, Eigen::DefaultDevice>* nullPointer = nullptr;
  ptr = new SelectClause<int, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorSelectClauseDefaultDevice)
{
  SelectClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  ptr = new SelectClause<int, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersSelectClauseDefaultDevice)
{
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values(2);
  labels_values.setValues({ 0, 1 });
  select_labels->setData(labels_values);
  SelectClause<int, Eigen::DefaultDevice> select_clause("1", "2", "3", select_labels);
  BOOST_CHECK_EQUAL(select_clause.table_name, "1");
  BOOST_CHECK_EQUAL(select_clause.axis_name, "2");
  BOOST_CHECK_EQUAL(select_clause.dimension_name, "3");
  BOOST_CHECK_EQUAL(select_clause.labels->getData()(0), 0);
  BOOST_CHECK_EQUAL(select_clause.labels->getData()(1), 1);
}

/*SortClause DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorSortClauseDefaultDevice)
{
  SortClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  SortClause<int, Eigen::DefaultDevice>* nullPointer = nullptr;
  ptr = new SortClause<int, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorSortClauseDefaultDevice)
{
  SortClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  ptr = new SortClause<int, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersSortClauseDefaultDevice)
{
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values(1);
  labels_values.setValues({ 1 });
  select_labels->setData(labels_values);
  SortClause<int, Eigen::DefaultDevice> order_by_clause("1", "2", "3", select_labels, sortOrder::order::ASC);
  BOOST_CHECK_EQUAL(order_by_clause.table_name, "1");
  BOOST_CHECK_EQUAL(order_by_clause.axis_name, "2");
  BOOST_CHECK_EQUAL(order_by_clause.dimension_name, "3");
  BOOST_CHECK_EQUAL(order_by_clause.labels->getData()(0), 1);
  BOOST_CHECK_EQUAL(order_by_clause.order_by, sortOrder::ASC);
}

/*WhereClause DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorWhereClauseDefaultDevice)
{
  WhereClause<int, float, Eigen::DefaultDevice>* ptr = nullptr;
  WhereClause<int, float, Eigen::DefaultDevice>* nullPointer = nullptr;
  ptr = new WhereClause<int, float, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorWhereClauseDefaultDevice)
{
  WhereClause<int, float, Eigen::DefaultDevice>* ptr = nullptr;
  ptr = new WhereClause<int, float, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersWhereClauseDefaultDevice)
{
  // Setup the labels
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values(2);
  labels_values.setValues({ 0, 1 });
  select_labels->setData(labels_values);

  // Setup the values
  std::shared_ptr<TensorDataDefaultDevice<float, 1>> select_values = std::make_shared<TensorDataDefaultDevice<float, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<float, 1> values_values(2);
  values_values.setValues({ 9.0, 2.0 });
  select_values->setData(values_values);

  WhereClause<int, float, Eigen::DefaultDevice> where_clause("1", "2", "3", select_labels, select_values, logicalComparitors::EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::OR);
  BOOST_CHECK_EQUAL(where_clause.table_name, "1");
  BOOST_CHECK_EQUAL(where_clause.axis_name, "2");
  BOOST_CHECK_EQUAL(where_clause.dimension_name, "3");
  BOOST_CHECK_EQUAL(where_clause.labels->getData()(0), 0);
  BOOST_CHECK_EQUAL(where_clause.labels->getData()(1), 1);
  BOOST_CHECK_EQUAL(where_clause.values->getData()(0), 9);
  BOOST_CHECK_EQUAL(where_clause.values->getData()(1), 2);
  BOOST_CHECK_EQUAL(where_clause.comparitor, logicalComparitors::EQUAL_TO);
  BOOST_CHECK_EQUAL(where_clause.modifier, logicalModifiers::NONE);
  BOOST_CHECK_EQUAL(where_clause.within_continuator, logicalContinuators::AND);
  BOOST_CHECK_EQUAL(where_clause.prepend_continuator, logicalContinuators::OR);
}

BOOST_AUTO_TEST_SUITE_END()