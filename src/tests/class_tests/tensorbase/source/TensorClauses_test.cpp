/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorOperation test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorOperation.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorOperation1)

//BOOST_AUTO_TEST_CASE(constructor) 
//{
//  TensorOperation* ptr = nullptr;
//  TensorOperation* nullPointer = nullptr;
//	ptr = new TensorOperation();
//  BOOST_CHECK_NE(ptr, nullPointer);
//}
//
//BOOST_AUTO_TEST_CASE(destructor) 
//{
//  TensorOperation* ptr = nullptr;
//	ptr = new TensorOperation();
//  delete ptr;
//}

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

/*OrderByClause DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorOrderByClauseDefaultDevice)
{
  OrderByClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  OrderByClause<int, Eigen::DefaultDevice>* nullPointer = nullptr;
  ptr = new OrderByClause<int, Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorOrderByClauseDefaultDevice)
{
  OrderByClause<int, Eigen::DefaultDevice>* ptr = nullptr;
  ptr = new OrderByClause<int, Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersOrderByClauseDefaultDevice)
{
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values(2);
  labels_values.setValues({ 0, 1 });
  select_labels->setData(labels_values);
  OrderByClause<int, Eigen::DefaultDevice> order_by_clause("1", "2", "3", select_labels,
    { order::ASC, order::DESC }
  );
  BOOST_CHECK_EQUAL(order_by_clause.table_name, "1");
  BOOST_CHECK_EQUAL(order_by_clause.axis_name, "2");
  BOOST_CHECK_EQUAL(order_by_clause.dimension_name, "3");
  BOOST_CHECK_EQUAL(order_by_clause.labels->getData()(0), 0);
  BOOST_CHECK_EQUAL(order_by_clause.labels->getData()(1), 1);
  BOOST_CHECK_EQUAL(order_by_clause.order_by.at(0), order::ASC);
  BOOST_CHECK_EQUAL(order_by_clause.order_by.at(1), order::DESC);
}

BOOST_AUTO_TEST_SUITE_END()