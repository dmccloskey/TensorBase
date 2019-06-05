/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollectionFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/TensorCollectionFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorCollectionFile1)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
  TensorCollectionFileDefaultDevice* ptr = nullptr;
  TensorCollectionFileDefaultDevice* nullPointer = nullptr;
  ptr = new TensorCollectionFileDefaultDevice();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice) 
{
  TensorCollectionFileDefaultDevice* ptr = nullptr;
	ptr = new TensorCollectionFileDefaultDevice();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinaryDefaultDevice) 
{
  // Setup the tensor collection
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

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

  TensorTableDefaultDevice<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<char, 3>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 3>>(tensorTable3);

  TensorCollectionDefaultDevice tensorCollection("1");
  tensorCollection.addTensorTable(tensorTable1_ptr);
  tensorCollection.addTensorTable(tensorTable2_ptr);
  tensorCollection.addTensorTable(tensorTable3_ptr);

  // Store the Tensor Collection
  TensorCollectionFileDefaultDevice data;
  std::string filename = "TensorCollectionFileDefaultDeviceTest.dat";
  data.storeTensorCollectionBinary(filename, tensorCollection);

  // Load the Tensor Collection
  TensorCollectionDefaultDevice tensorCollection_test;
  data.loadTensorCollectionBinary(filename, tensorCollection_test);
  BOOST_CHECK(tensorCollection == tensorCollection_test);
}

BOOST_AUTO_TEST_SUITE_END()