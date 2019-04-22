/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollection test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorCollection.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorCollection)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollection<TensorTable<float,Eigen::DefaultDevice,3>>* ptr = nullptr;
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* nullPointer = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* ptr = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
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
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr(new TensorTableDefaultDevice<float, 3>(tensorTable1));

  TensorTableDefaultDevice<int, 2> tensorTable2("2", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    });
  std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> tensorTable2_ptr(new TensorTableDefaultDevice<int, 2>(tensorTable2));

  TensorTableDefaultDevice<char, 3> tensorTable3("3", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });
  std::shared_ptr<TensorTable<char, Eigen::DefaultDevice, 3>> tensorTable3_ptr(new TensorTableDefaultDevice<char, 3>(tensorTable3));

  TensorCollection<
    std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>>,
    std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>>,
    std::shared_ptr<TensorTable<char, Eigen::DefaultDevice, 3>>
  > tensorCollection(tensorTable1_ptr, tensorTable2_ptr, tensorTable3_ptr);

  // name setters and getters
  //BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "tfloat", "tint", "tchar" }));
}

BOOST_AUTO_TEST_SUITE_END()