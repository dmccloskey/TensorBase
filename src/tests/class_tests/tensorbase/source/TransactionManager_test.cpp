/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TransactionManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TransactionManager1)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice)
{
  TransactionManager<Eigen::DefaultDevice>* ptr = nullptr;
  TransactionManager<Eigen::DefaultDevice>* nullPointer = nullptr;
	ptr = new TransactionManager<Eigen::DefaultDevice>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
  TransactionManager<Eigen::DefaultDevice>* ptr = nullptr;
	ptr = new TransactionManager<Eigen::DefaultDevice>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  // Setup the tensor tables
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

  // Setup the tensor collection
  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr);
  tensorCollection.addTensorTable(tensorTable2_ptr);
  tensorCollection.addTensorTable(tensorTable3_ptr);
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> tensorCollection_ptr = std::make_shared<TensorCollectionDefaultDevice>(tensorCollection);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transactionManager;
  transactionManager.setTensorCollection(tensorCollection_ptr);
  transactionManager.setMaxOperations(10);

  // Test getters
  BOOST_CHECK(transactionManager.getTensorCollection()->getTableNames() == std::vector<std::string>({ "1", "2", "3" }));
  BOOST_CHECK_EQUAL(transactionManager.getMaxOperations(), 10);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 0);
}

BOOST_AUTO_TEST_CASE(executeOperationDefaultDevice)
{
  // Setup the axes and tables
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

  // Setup the tensor collection
  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr);
  tensorCollection.addTensorTable(tensorTable2_ptr);
  tensorCollection.addTensorTable(tensorTable3_ptr);
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> tensorCollection_ptr = std::make_shared<TensorCollectionDefaultDevice>(tensorCollection);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transactionManager;
  transactionManager.setTensorCollection(tensorCollection_ptr);

  // Execute several transactions

  // Undo two of them

  // Redo one of them

  // Rollback all of them

  // Redo all of them

  // Commit
}

BOOST_AUTO_TEST_SUITE_END()