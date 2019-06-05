/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TransactionManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TransactionManager1)

/// The delete select Functor for table 1
struct DeleteTable2 {
  template<typename DeviceT>
  void operator() (TensorCollection<DeviceT> & tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 2 and axis 2 where labels=1
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels_t2a2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t2a2(1);
    labels_values_t2a2.setValues({ 1 });
    select_labels_t2a2->setData(labels_values_t1a2);
    SelectClause<int, DeviceT> select_clause2("2", "2", "y", select_labels_t2a2);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause2, device);
  }
};

/// The select Functor for table 3
struct SelectTable3 {
  template<typename DeviceT>
  void operator() (TensorCollection<DeviceT> & tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 3: 
    std::shared_ptr<TensorDataDefaultDevice<char, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<char, 1> labels_values(1);
    labels_values.setValues({ 'b' });
    select_labels->setData(labels_values);
    SelectClause<int, DeviceT> select_clause1("3", "1", "x", select_labels);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause1, device);

    // Apply the select clause
    tensorSelect.applySelect(tensor_collection, { "3" }, device);
  }
};

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

BOOST_AUTO_TEST_CASE(undoRedoAndRollbackDefaultDevice)
{
  // Setup the device
  Eigen::DefaultDevice device;

  // Setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  // Setup the tables
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes();

  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable1.getData()->setData(tensor_values1);
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes();

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.getData()->setData(tensor_values2);
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorTableDefaultDevice<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.setAxes();

  Eigen::Tensor<char, 1> tensor_values3(Eigen::array<Eigen::Index, 1>({ nlabels1 }));
  tensor_values3.setValues({ 'a', 'b' });
  tensorTable3.getData()->setData(tensor_values3);
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

  // Operation #1: add
  // Set up the new labels
  Eigen::Tensor<int, 2> labels_new_values(1, nlabels2);
  labels_new_values.setValues({ {3, 4, 5} });
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, 3 }));
  labels_new.setData(labels_new_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_new_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new);

  // Set up the new values
  Eigen::Tensor<float, 3> tensor_values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values_new(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2 + nlabels1 + nlabels2 * nlabels1 + nlabels3 * nlabels1*nlabels2;
      }
    }
  }
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  values_new.setData(tensor_values_new);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new);

  // Test the AppendToAxis execution
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3> appendToAxis("1", "2", labels_new_ptr, values_new_ptr);
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>>(appendToAxis);
  transactionManager.executeOperation(appendToAxis_ptr, device);

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 0);
  // TODO: other tests from tensorOperation...

  // Operation #2: delete
  TensorDeleteFromAxisDefaultDevice<int, float, 2> deleteFromAxis("2", "2", DeleteTable2());
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> deleteFromAxis_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, float, 2>>(deleteFromAxis);
  transactionManager.executeOperation(deleteFromAxis_ptr, device);

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 1);
  // TODO: test for the missing data in Tensor collection

  // operation #3: update
  // Set up the update values
  Eigen::Tensor<char, 1> values_new_values(1);
  values_new_values.setValues({'c'});
  TensorDataDefaultDevice<char, 1> values_update(Eigen::array<Eigen::Index, 1>({ 1 }));
  values_update.setData(values_new_values);

  // Set up the update
  TensorUpdate<char, Eigen::DefaultDevice, 1> tensorUpdate("3", SelectTable3(), std::make_shared<TensorDataDefaultDevice<char, 1>>(values_update));
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorUpdate_ptr = std::make_shared<TensorUpdate<char, Eigen::DefaultDevice, 1>>(tensorUpdate);
  transactionManager.executeOperation(tensorUpdate_ptr, device);

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 2);
  // TODO: test for the changed value

  // Undo two of them

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 0);

  // Redo one of them

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 1);

  // Rollback all of them

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), -1);
}

BOOST_AUTO_TEST_CASE(CommitDefaultDevice)
{
  // Setup the device
  Eigen::DefaultDevice device;

  // Execute an operation

  // Commit
}

BOOST_AUTO_TEST_SUITE_END()