/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TransactionManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>
#include <TensorBase/ml/TensorSelect.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TransactionManager1)

/// The delete select Functor for table 2
struct DeleteTable2 {
  template<typename DeviceT>
  void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 2 and axis 2 where labels=1
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels_t2a2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t2a2(1);
    labels_values_t2a2.setValues({ 1 });
    select_labels_t2a2->setData(labels_values_t2a2);
    SelectClause<int, DeviceT> select_clause2("2", "2", "y", select_labels_t2a2);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause2, device);
  }
};

/// The select Functor for table 3
struct SelectTable3 {
  template<typename DeviceT>
  void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 3:
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values(1);
    labels_values.setValues({ 1 });
    select_labels->setData(labels_values);
    SelectClause<int, DeviceT> select_clause1("3", "1", "x", select_labels);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause1, device);
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
  // Set up the device
  Eigen::DefaultDevice device;

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
  tensorTable1.setAxes(device);
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorTableDefaultDevice<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes(device);
  std::shared_ptr<TensorTableDefaultDevice<char, 3>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 3>>(tensorTable3);

  // Setup the tensor collection
  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> tensorCollection_ptr = std::make_shared<TensorCollectionDefaultDevice>(tensorCollection);

  // Setup the transaction manager
  TransactionManager<Eigen::DefaultDevice> transactionManager;
  transactionManager.setTensorCollection(tensorCollection_ptr);
  transactionManager.setMaxOperations(10);

  // Test getters
  BOOST_CHECK(transactionManager.getTensorCollection()->getTableNames() == std::vector<std::string>({ "1", "2", "3" }));
  BOOST_CHECK_EQUAL(transactionManager.getMaxOperations(), 10);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), -1);
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
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  // Setup the tables
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(table_1_axis_1_ptr);
  tensorTable1.addTensorAxis(table_1_axis_2_ptr);
  tensorTable1.addTensorAxis(table_1_axis_3_ptr);
  tensorTable1.setAxes(device);

  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  auto table_2_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto table_2_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  tensorTable2.addTensorAxis(table_2_axis_1_ptr);
  tensorTable2.addTensorAxis(table_2_axis_2_ptr);
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorTableDefaultDevice<char, 1> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.setAxes(device);

  Eigen::Tensor<char, 1> tensor_values3(Eigen::array<Eigen::Index, 1>({ nlabels1 }));
  tensor_values3.setValues({ 'a', 'b' });
  tensorTable3.setData(tensor_values3);
  std::shared_ptr<TensorTableDefaultDevice<char, 1>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 1>>(tensorTable3);

  // Setup the tensor collection
  TensorCollectionDefaultDevice tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");
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

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getNLabels(), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    if (i < nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i - nlabels2 + 1);
    }
  }

  // Operation #2: delete
  TensorDeleteFromAxisDefaultDevice<int, int, 2> deleteFromAxis("2", "2", DeleteTable2());
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> deleteFromAxis_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<int, int, 2>>(deleteFromAxis);
  transactionManager.executeOperation(deleteFromAxis_ptr, device);

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 1);

  // Make the expected tensor data
  Eigen::Tensor<int, 2> expected_tensor_values(nlabels1, nlabels2 - 1);
  for (int i = 0; i < nlabels1; ++i) {
    int iter = 0;
    for (int j = 0; j < nlabels2; ++j) {
      if (j != 1) {
        expected_tensor_values(i, iter) = i + j * nlabels1;
        ++iter;
      }
    }
  }

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2 - 1; ++j) {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(i, j), expected_tensor_values(i, j));
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getNLabels(), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1)
      BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getLabels()(0, i), i);
    else
      BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getLabels()(0, i), i + 1);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(tensorTable2_ptr->getDimFromAxisName("2")), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1) {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndices().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndices().at("2")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardIndices().at("2")->getData()(i), i + 2);
    }
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardId().at("2")->getData()(i), 1);
  }

  // operation #3: update
  // Set up the update values
  Eigen::Tensor<char, 1> values_new_values(1);
  values_new_values.setValues({'c'});
  TensorDataDefaultDevice<char, 1> values_update(Eigen::array<Eigen::Index, 1>({ 1 }));
  values_update.setData(values_new_values);

  // Set up the update
  TensorUpdateValues<char, Eigen::DefaultDevice, 1> tensorUpdate("3", SelectTable3(), std::make_shared<TensorDataDefaultDevice<char, 1>>(values_update));
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<char, Eigen::DefaultDevice, 1>>(tensorUpdate);
  transactionManager.executeOperation(tensorUpdate_ptr, device);

  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 2);

  // Test for the changed value
  Eigen::Tensor<char, 1> values_update_expected(Eigen::array<Eigen::Index, 1>({ nlabels1 }));
  values_update_expected.setValues({ 'a', 'c' });
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(tensorTable3_ptr->getData()(i), values_update_expected(i));
  }

  // Undo #1
  transactionManager.undo(device);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 1);

  // Test that table 3 update has been reverted
  for (int i = 0; i < nlabels1; ++i) {
    BOOST_CHECK_EQUAL(tensorTable3_ptr->getData()(i), tensor_values3(i));
  }

  // Undo #2
  transactionManager.undo(device);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 0);

  // Test that table 2 deletion as been reverted
  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2 - 1; ++j) {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(i, j), tensor_values2(i, j));
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getNLabels(), nlabels2);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(tensorTable2_ptr->getDimFromAxisName("2")), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardId().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }

  // Redo one of them
  transactionManager.redo(device);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), 1);

  // Test that tensor 2 has the deletion
  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2 - 1; ++j) {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(i, j), expected_tensor_values(i, j));
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getNLabels(), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1)
      BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getLabels()(0, i), i);
    else
      BOOST_CHECK_EQUAL(table_2_axis_2_ptr->getLabels()(0, i), i + 1);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(tensorTable2_ptr->getDimFromAxisName("2")), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1) {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndices().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(i), i + 1);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndices().at("2")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(i), i + 2);
      BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardIndices().at("2")->getData()(i), i + 2);
    }
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getIsModified().at("2")->getData()(i), 1); 
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getShardId().at("2")->getData()(i), 1);
  }

  // Rollback all of them
  transactionManager.rollback(device);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), -1);

  // Test that tensors 1, 2, and 3 have their original data
  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getNLabels(), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }
}

BOOST_AUTO_TEST_CASE(CommitDefaultDevice)
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
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  // Setup the tables
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(table_1_axis_1_ptr);
  tensorTable1.addTensorAxis(table_1_axis_2_ptr);
  tensorTable1.addTensorAxis(table_1_axis_3_ptr);
  tensorTable1.setAxes(device);

  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  auto table_2_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto table_2_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  tensorTable2.addTensorAxis(table_2_axis_1_ptr);
  tensorTable2.addTensorAxis(table_2_axis_2_ptr);
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorTableDefaultDevice<char, 1> tensorTable3("3");
  tensorTable3.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable3.setAxes(device);

  Eigen::Tensor<char, 1> tensor_values3(Eigen::array<Eigen::Index, 1>({ nlabels1 }));
  tensor_values3.setValues({ 'a', 'b' });
  tensorTable3.setData(tensor_values3);
  std::shared_ptr<TensorTableDefaultDevice<char, 1>> tensorTable3_ptr = std::make_shared<TensorTableDefaultDevice<char, 1>>(tensorTable3);

  // Setup the tensor collection
  TensorCollectionDefaultDevice tensorCollection("tensorCollection1");
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");
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

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getNLabels(), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    if (i < nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i - nlabels2 + 1);
    }
  }

  // Test Commit
  transactionManager.commit(device);
  BOOST_CHECK_EQUAL(transactionManager.getCurrentIndex(), -1);

  // Re-test that the is modified attribute was changed
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
  }

  // Make a new tensor collection
  TensorCollectionDefaultDevice tensorCollectionCommit;
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> tensorCollectionCommit_ptr = std::make_shared<TensorCollectionDefaultDevice>(tensorCollection);

  // Reload the committed data into a new tensor collection
  TensorCollectionFile<Eigen::DefaultDevice> data;
  data.loadTensorCollectionBinary(tensorCollection_ptr->getName() + ".TensorCollection", tensorCollectionCommit_ptr, device);
  
  // Test for the expected metadata 
  BOOST_CHECK(*(tensorCollection_ptr.get()) == *(tensorCollectionCommit_ptr.get()));

  // Test for the expected table data
  std::shared_ptr<float[]> table_1_data;
  tensorCollectionCommit_ptr->tables_.at("1")->getDataPointer(table_1_data);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data.get(), nlabels1, 2 * nlabels2, nlabels3);
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(table_1_values(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(table_1_values(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  std::shared_ptr<int[]> table_1_axis2_data;
  tensorCollectionCommit_ptr->tables_.at("1")->getAxes().at("2")->getLabelsDataPointer(table_1_axis2_data);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> table_1_axis2_values(table_1_axis2_data.get(), labels1.dimensions());
  BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getNLabels(), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(table_1_axis2_values(0, i), i);
  }

  // Test for the expected indices data
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getIsModified().at("2")->getData()(i), 0); // This is correct (i.e., should not be modified)
    BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getNotInMemory().at("2")->getData()(i), 0);
    if (i < nlabels2) {
      BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getShardId().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getShardId().at("2")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorCollectionCommit_ptr->tables_.at("1")->getShardIndices().at("2")->getData()(i), i - nlabels2 + 1);
    }
  }

  // test re-initializing the TensorTable data and adding additional data
  transactionManager.initTensorCollectionTensorData(device);

  // Operation #2: add
  // Set up the new labels
  Eigen::Tensor<int, 2> labels_new_values2(1, nlabels2);
  labels_new_values2.setValues({ {6, 7, 8} });
  TensorDataDefaultDevice<int, 2> labels_new2(Eigen::array<Eigen::Index, 2>({ 1, 3 }));
  labels_new2.setData(labels_new_values2);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> labels_new2_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new2);

  // Set up the new values
  Eigen::Tensor<float, 3> tensor_values_new2(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values_new2(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2 + 2*(nlabels1 + nlabels2 * nlabels1 + nlabels3 * nlabels1 * nlabels2);
      }
    }
  }
  TensorDataDefaultDevice<float, 3> values_new2(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  values_new2.setData(tensor_values_new2);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> values_new2_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new2);

  // Test the AppendToAxis execution
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3> appendToAxis2("1", "2", labels_new2_ptr, values_new2_ptr);
  std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> appendToAxis2_ptr = std::make_shared<TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>>(appendToAxis2);
  transactionManager.executeOperation(appendToAxis2_ptr, device);

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()(i, j + 2*nlabels2, k), tensor_values_new2(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getNLabels(), 3 * nlabels2);
  for (int i = 0; i < 3 * nlabels2; ++i) {
    BOOST_CHECK_EQUAL(table_1_axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), 3 * nlabels2);
  //std::cout << "getIsModified\n"<< tensorTable1_ptr->getIsModified().at("2")->getData() <<std::endl;
  //std::cout << "getNotInMemory\n" << tensorTable1_ptr->getNotInMemory().at("2")->getData() << std::endl;
  //std::cout << "getShardId\n" << tensorTable1_ptr->getShardId().at("2")->getData() << std::endl;
  //std::cout << "getShardIndices\n" << tensorTable1_ptr->getShardIndices().at("2")->getData() << std::endl;
  for (int i = 0; i < 3*nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    // Is modified
    if (i >= 2 * nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
    // Shard ID and Shard Indices
    if (i < nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else if (i >= nlabels2 && i<2* nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 2);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i - nlabels2 + 1);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardId().at("2")->getData()(i), 3);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i - 2*nlabels2 + 1);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()