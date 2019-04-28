/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorSelect test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorSelect.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorSelect1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorSelect* ptr = nullptr;
  TensorSelect* nullPointer = nullptr;
	ptr = new TensorSelect();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  TensorSelect* ptr = nullptr;
	ptr = new TensorSelect();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(TensorTableSelectClause)
{
  Eigen::Tensor<std::string, 1> table_names(4), axis_names(4), dimension_names(4), label_names(4);
  table_names.setValues({"1", "2", "1", "2"});
  axis_names.setValues({ "1", "2", "1", "2" });
  dimension_names.setValues({ "x", "y", "x", "y" });
  label_names.setValues({ "x-axis-0", "y-axis-0", "x-axis-1", "y-axis-2" });

  SelectClause selectClause(table_names, axis_names, dimension_names, label_names);

  // Check get labels
  Eigen::DefaultDevice device;
  Eigen::Tensor<std::string, 1> table_1 = selectClause.getLabels("1", "1", "x", device);
  BOOST_CHECK_EQUAL(table_1.size(), 2);
  BOOST_CHECK_EQUAL(table_1(0), "x-axis-1");
  BOOST_CHECK_EQUAL(table_1(1), "x-axis-0");
  Eigen::Tensor<std::string, 1> table_2 = selectClause.getLabels("2", "2", "y", device);
  BOOST_CHECK_EQUAL(table_2.size(), 2);
  BOOST_CHECK_EQUAL(table_2(0), "y-axis-2");
  BOOST_CHECK_EQUAL(table_2(1), "y-axis-0");
}

BOOST_AUTO_TEST_CASE(TensorTableSelect) 
{
  // Set up the tables
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { "x-axis-0", "x-axis-1"} });
  labels2.setValues({ { "y-axis-0", "y-axis-1", "y-axis-2" } });
  labels3.setValues({ { "z-axis-0", "z-axis-1", "z-axis-2", "z-axis-3", "z-axis-4" } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    });
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollection collection_1;
  collection_1.addTensorTable(tensorTable1_ptr);
  collection_1.addTensorTable(tensorTable2_ptr);

  // Set up the SelectClause
  Eigen::Tensor<std::string, 1> table_names(4), axis_names(4), dimension_names(4), label_names(4);
  table_names.setValues({ "1", "2", "1", "2" });
  axis_names.setValues({ "1", "2", "1", "2" });
  dimension_names.setValues({ "x", "y", "x", "y" });
  label_names.setValues({ "x-axis-0", "y-axis-0", "x-axis-1", "y-axis-2" });
  SelectClause selectClause(table_names, axis_names, dimension_names, label_names);

  TensorSelect tensorSelect;
  Eigen::DefaultDevice device;
  // Test the expected view indices after the select command
  tensorSelect.selectClause(collection_1, selectClause, device);
  BOOST_CHECK_EQUAL(collection_1.tensor_tables_.at("1")->getIndicesView().at("1")->operator()(0), 1);
  BOOST_CHECK_EQUAL(collection_1.tensor_tables_.at("1")->getIndicesView().at("1")->operator()(1), 2);
  BOOST_CHECK_EQUAL(collection_1.tensor_tables_.at("2")->getIndicesView().at("2")->operator()(0), 1);
  BOOST_CHECK_EQUAL(collection_1.tensor_tables_.at("2")->getIndicesView().at("2")->operator()(1), 0);
  BOOST_CHECK_EQUAL(collection_1.tensor_tables_.at("2")->getIndicesView().at("2")->operator()(2), 3);

}

BOOST_AUTO_TEST_SUITE_END()