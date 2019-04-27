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

BOOST_AUTO_TEST_CASE(TensorTableSelectClause)
{
  Eigen::Tensor<std::string, 1> table_names(4), axis_names(4), dimension_names(4), label_names(4);
  table_names.setValues({"1", "2", "1", "2"});
  axis_names.setValues({ "1", "2", "1", "2" });
  dimension_names.setValues({ "x", "y", "x", "y" });
  label_names.setValues({ "x-axis-0", "y-axis-0", "x-axis-1", "y-axis-2" });
  SelectClause selectClause(table_names, axis_names, dimension_names, label_names);

  // Check get labels
  Eigen::Tensor<std::string, 1> table_1 = selectClause.getLabels("1", "1", "x");
  BOOST_CHECK_EQUAL(table_1.size(), 2);
  BOOST_CHECK_EQUAL(table_1(0), "x-axis-0");
  BOOST_CHECK_EQUAL(table_1(1), "x-axis-1");
  

}
BOOST_AUTO_TEST_CASE(TensorTableSelect) 
{
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

  TensorCollection collection_1;
  collection_1.addTensorTable(tensorTable1_ptr);
  collection_1.addTensorTable(tensorTable2_ptr);

}

BOOST_AUTO_TEST_SUITE_END()