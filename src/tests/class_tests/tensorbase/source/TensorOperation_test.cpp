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

BOOST_AUTO_TEST_CASE(scratch) 
{
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant("x-axis");
  labels2.setConstant("y-axis");
  labels3.setConstant("z-axis");

  TensorTableDefaultDevice<float, 3> tensorTable1("1", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorCollection<std::shared_ptr<TensorTableDefaultDevice<float, 3>>> collection_1(tensorTable1_ptr);

  // add a TensorTable to the collection
  TensorTableDefaultDevice<int, 2> tensorTable2("2", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    });
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable1);

  TensorCollection<
    std::shared_ptr<TensorTableDefaultDevice<float, 3>>,
    std::shared_ptr<TensorTableDefaultDevice<int, 2>>> collection_2;
  TensorCreateTable(collection_1, collection_2, tensorTable2_ptr);

}

BOOST_AUTO_TEST_SUITE_END()