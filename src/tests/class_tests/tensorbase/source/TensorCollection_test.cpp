/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollection test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorCollection)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice)
{
  TensorCollectionDefaultDevice* ptr = nullptr;
  TensorCollectionDefaultDevice* nullPointer = nullptr;
	ptr = new TensorCollectionDefaultDevice();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
  TensorCollectionDefaultDevice* ptr = nullptr;
	ptr = new TensorCollectionDefaultDevice();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(commparisonDefaultDevice)
{
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

  // Test collection
  TensorCollectionDefaultDevice tensorCollection_test("1");
  tensorCollection_test.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection_test.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection_test.addTensorTable(tensorTable3_ptr, "1");

  // Expected collection
  TensorCollectionDefaultDevice tensorCollection1("1");
  tensorCollection1.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection1.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection1.addTensorTable(tensorTable3_ptr, "1");

  BOOST_CHECK(tensorCollection_test == tensorCollection1); // Control
  tensorCollection1.setName("3");
  BOOST_CHECK(tensorCollection_test != tensorCollection1); // Different names but same data
  tensorCollection1.setName("1");
  tensorCollection1.removeTensorTable("1");
  BOOST_CHECK(tensorCollection_test != tensorCollection1); // Different data but same names
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
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

  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // name getter
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "1", "2", "3" }));
  BOOST_CHECK(tensorCollection.getTableNamesFromUserName("1") == std::set<std::string>({ "1", "2", "3" }));

  // table concept getter
  auto tt1_ptr = tensorCollection.getTensorTableConcept("1");
  BOOST_CHECK_EQUAL(tt1_ptr->getName(), tensorTable1_ptr->getName());
  BOOST_CHECK(tt1_ptr->getAxes() == tensorTable1_ptr->getAxes());
  BOOST_CHECK(tt1_ptr->getIndices() == tensorTable1_ptr->getIndices());
  BOOST_CHECK(tt1_ptr->getIndicesView() == tensorTable1_ptr->getIndicesView());
  BOOST_CHECK(tt1_ptr->getIsModified() == tensorTable1_ptr->getIsModified());
  BOOST_CHECK(tt1_ptr->getNotInMemory() == tensorTable1_ptr->getNotInMemory());
  BOOST_CHECK(tt1_ptr->getShardId() == tensorTable1_ptr->getShardId());

  // remove tensor tables
  tensorCollection.removeTensorTable("2");
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "1", "3" }));
  BOOST_CHECK(tensorCollection.getTableNamesFromUserName("1") == std::set<std::string>({ "1", "3" }));

  // clear the collection
  tensorCollection.clear();
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>());
  BOOST_CHECK(tensorCollection.getTableNamesFromUserName("1") == std::set<std::string>());
}

BOOST_AUTO_TEST_CASE(addTensorTableConceptDefaultDevice)
{
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

  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // table concept getter
  const std::shared_ptr<TensorTableConcept<Eigen::DefaultDevice>> tt1_ptr = tensorCollection.getTensorTableConcept("1");
  BOOST_CHECK_EQUAL(tt1_ptr->getName(), tensorTable1_ptr->getName());
  BOOST_CHECK(tt1_ptr->getAxes() == tensorTable1_ptr->getAxes());
  BOOST_CHECK(tt1_ptr->getIndices() == tensorTable1_ptr->getIndices());
  BOOST_CHECK(tt1_ptr->getIndicesView() == tensorTable1_ptr->getIndicesView());
  BOOST_CHECK(tt1_ptr->getIsModified() == tensorTable1_ptr->getIsModified());
  BOOST_CHECK(tt1_ptr->getNotInMemory() == tensorTable1_ptr->getNotInMemory());
  BOOST_CHECK(tt1_ptr->getShardId() == tensorTable1_ptr->getShardId());

  // table concept adder
  tensorCollection.removeTensorTable("1");
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "2", "3" }));
  BOOST_CHECK(tensorCollection.getTableNamesFromUserName("1") == std::set<std::string>({ "2", "3" }));
  tensorCollection.addTensorTableConcept(tt1_ptr, "1");
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "1", "2", "3" }));
  BOOST_CHECK(tensorCollection.getTableNamesFromUserName("1") == std::set<std::string>({ "1", "2", "3" }));
}

BOOST_AUTO_TEST_SUITE_END()