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

  // Setup Table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup Table 1 data
  Eigen::Tensor<float, 3> tensor1_values(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor1_values(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable1_ptr->setData(tensor1_values);

  // Setup Table 2 axes
  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Setup Table 2 data
  Eigen::Tensor<int, 2> tensor2_values(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor2_values(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2_ptr->setData(tensor2_values);

  // Setup Table 3 axes
  TensorTableDefaultDevice<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<char, 3>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 3>>(tensorTable3);

  // Setup Table 3 data
  Eigen::Tensor<char, 3> tensor3_values(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  tensor3_values.setConstant('a');
  tensorTable3_ptr->setData(tensor3_values);

  TensorCollectionDefaultDevice tensorCollection("1");
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // Store the Tensor Collection
  Eigen::DefaultDevice device;
  TensorCollectionFileDefaultDevice data;
  std::string filename = "TensorCollectionFileDefaultDeviceTest.dat";
  data.storeTensorCollectionBinary(filename, tensorCollection, device);

  // Load the Tensor Collection
  TensorCollectionDefaultDevice tensorCollection_test;
  data.loadTensorCollectionBinary(filename, tensorCollection_test, device);

  // Test the serialized metadata
  BOOST_CHECK(tensorCollection == tensorCollection_test);

  // Test the binarized tensor axes data and tensor data
  // NOTE: tests only work on the DefaultDevice
  {// Table 1 Axis 1
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("1")->getAxes().at("1")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels1);
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels1(0, j));
    }
  }
  {// Table 1 Axis 2
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("1")->getAxes().at("2")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels2);
    for (int j = 0; j < nlabels2; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels2(0, j));
    }
  }
  {// Table 1 Axis 3
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("1")->getAxes().at("3")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels3);
    for (int j = 0; j < nlabels3; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels3(0, j));
    }
  }
  {// Table 1 Data
    std::shared_ptr<float> labels_ptr;
    tensorCollection_test.tables_.at("1")->getDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<float, 3>> data_values(labels_ptr.get(), Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
    for (int k = 0; k < nlabels3; ++k) {
      for (int j = 0; j < nlabels2; ++j) {
        for (int i = 0; i < nlabels1; ++i) {
          BOOST_CHECK_EQUAL(data_values(i, j, k), tensor1_values(i, j, k));
        }
      }
    }
  }

  {// Table 2 Axis 1
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("2")->getAxes().at("1")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels1);
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels1(0, j));
    }
  }
  {// Table 2 Axis 2
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("2")->getAxes().at("2")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels2);
    for (int j = 0; j < nlabels2; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels2(0, j));
    }
  }
  {// Table 2 Data
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("2")->getDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> data_values(labels_ptr.get(), Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        BOOST_CHECK_EQUAL(data_values(i, j), tensor2_values(i, j));
      }
    }
  }

  {// Table 3 Axis 1
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("3")->getAxes().at("1")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels1);
    for (int j = 0; j < nlabels1; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels1(0, j));
    }
  }
  {// Table 3 Axis 2
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("3")->getAxes().at("2")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels2);
    for (int j = 0; j < nlabels2; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels2(0, j));
    }
  }
  {// Table 3 Axis 3
    std::shared_ptr<int> labels_ptr;
    tensorCollection_test.tables_.at("3")->getAxes().at("3")->getLabelsDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels_ptr.get(), 1, nlabels3);
    for (int j = 0; j < nlabels3; ++j) {
      BOOST_CHECK_EQUAL(labels_values(0, j), labels3(0, j));
    }
  }
  {// Table 3 Data
    std::shared_ptr<char> labels_ptr;
    tensorCollection_test.tables_.at("3")->getDataPointer(labels_ptr);
    Eigen::TensorMap<Eigen::Tensor<char, 3>> data_values(labels_ptr.get(), Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
    for (int k = 0; k < nlabels3; ++k) {
      for (int j = 0; j < nlabels2; ++j) {
        for (int i = 0; i < nlabels1; ++i) {
          BOOST_CHECK_EQUAL(data_values(i, j, k), tensor3_values(i, j, k));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(getTensorTableHeadersDefaultDevice)
{
  // Setup the tensor collection
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ {0, 1} });
  labels2.setConstant(2);
  labels3.setConstant(3);

  // Setup Table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup Table 1 data
  Eigen::Tensor<float, 3> tensor1_values(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor1_values(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable1_ptr->setData(tensor1_values);

  // Setup Table 2 axes
  TensorTableDefaultDevice<int, 3> tensorTable2("2");
  labels1.setValues({ {2, 3} });
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable2.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<int, 3>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 3>>(tensorTable2);

  // Setup Table 2 data
  Eigen::Tensor<int, 3> tensor2_values(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor2_values(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable2_ptr->setData(tensor2_values);

  // Setup Table 3 axes
  TensorTableDefaultDevice<char, 3> tensorTable3("3");
  labels1.setValues({ {4, 5} });
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes();
  std::shared_ptr<TensorTableDefaultDevice<char, 3>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 3>>(tensorTable3);

  // Setup Table 3 data
  Eigen::Tensor<char, 3> tensor3_values(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  tensor3_values.setConstant('a');
  tensorTable3_ptr->setData(tensor3_values);

  TensorCollectionDefaultDevice tensorCollection("1");
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // Make the output header names
  Eigen::DefaultDevice device;
  TensorCollectionFileDefaultDevice data;
  std::pair<std::vector<std::string>, std::vector<std::string>> headers = data.getTensorTableHeaders("1", tensorCollection, device);
  BOOST_CHECK(headers.first == std::vector<std::string>({ "y", "z" }));
  BOOST_CHECK(headers.second == std::vector<std::string>({ "0", "1", "2", "3", "4", "5" }));
}

BOOST_AUTO_TEST_SUITE_END()