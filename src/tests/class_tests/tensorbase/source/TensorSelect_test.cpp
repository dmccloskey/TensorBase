/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorSelect test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorSelect.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorSelect1)

/*TensorSelect DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorSelectDefaultDevice)
{
  TensorSelect* ptr = nullptr;
  TensorSelect* nullPointer = nullptr;
	ptr = new TensorSelect();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorSelectDefaultDevice)
{
  TensorSelect* ptr = nullptr;
	ptr = new TensorSelect();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(TensorSelectDefaultDevice) 
{
  Eigen::DefaultDevice device;

  // Set up the tables
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollection collection_1;
  collection_1.addTensorTable(tensorTable1_ptr);
  collection_1.addTensorTable(tensorTable2_ptr);

  // Set up the SelectClause
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values1(2);
  labels_values1.setValues({ 0, 1 });
  select_labels1->setData(labels_values1);
  SelectClause<int, Eigen::DefaultDevice> select_clause1("1", "1", "x", select_labels1);
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values2(2);
  labels_values2.setValues({ 0, 2 });
  select_labels2->setData(labels_values2);
  SelectClause<int, Eigen::DefaultDevice> select_clause2("2", "2", "y", select_labels2);

  // Test the unchanged values
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.selectClause(collection_1, select_clause1, device);
  tensorSelect.selectClause(collection_1, select_clause2, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);
}

BOOST_AUTO_TEST_SUITE_END()