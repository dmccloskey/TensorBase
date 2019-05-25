/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorOperation1 test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorOperation.h>
#include <TensorBase/ml/TensorSelect.h>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorOperation1)

/// The select Functor for table 1
struct SelectTable1 {
  template<typename DeviceT>
  void operator() (TensorCollection<DeviceT> & tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 1:  all values in dim 0 for labels = 0 in dims 1 and 2
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels_t1a1 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
    Eigen::Tensor<int, 1> labels_values_t1a1(2);
    labels_values_t1a1.setValues({ 0, 1 });
    select_labels_t1a1->setData(labels_values_t1a1);
    SelectClause<int, Eigen::DefaultDevice> select_clause1("1", "1", "x", select_labels_t1a1);
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels_t1a2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t1a2(1);
    labels_values_t1a2.setValues({ 0 });
    select_labels_t1a2->setData(labels_values_t1a2);
    SelectClause<int, Eigen::DefaultDevice> select_clause2("1", "2", "y", select_labels_t1a2);
    std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels_t1a3 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t1a3(1);
    labels_values_t1a3.setValues({ 0 });
    select_labels_t1a3->setData(labels_values_t1a3);
    SelectClause<int, Eigen::DefaultDevice> select_clause3("1", "3", "z", select_labels_t1a3);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause1, device);
    tensorSelect.selectClause(tensor_collection, select_clause2, device);
    tensorSelect.selectClause(tensor_collection, select_clause3, device);

    // Apply the select clause
    tensorSelect.applySelect(tensor_collection, { "1" }, device);
  }
};

/*TensorAppendToAxis Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorAppendToAxis)
{
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorAppendToAxis)
{
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoAndUndoTensorAppendToAxis)
{
  Eigen::DefaultDevice device;

  // Set up the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  // Set up table 1
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  auto axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(axis_1_ptr);
  tensorTable1.addTensorAxis(axis_2_ptr);
  tensorTable1.addTensorAxis(axis_3_ptr);
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

  // set up table 2
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

  // Set up the collection
  TensorCollection<Eigen::DefaultDevice> collection_1;
  collection_1.addTensorTable(tensorTable1_ptr);
  collection_1.addTensorTable(tensorTable2_ptr);

  // Set up the new labels
  Eigen::Tensor<int, 2> labels_new_values(1, nlabels2);
  labels_new_values.setValues({ {3, 4, 5} });
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({1, 3}));
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

  // Test redo to append the new values
  TensorAppendToAxis<int, float, Eigen::DefaultDevice, 3> appendToAxis("1", "2", labels_new_ptr, values_new_ptr);
  appendToAxis.redo(collection_1, device);

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()->getData()(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(axis_2_ptr->getNLabels(), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    if (i < nlabels2) {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getInMemory().at("2")->getData()(i), 0);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsShardable().at("2")->getData()(i), 0);
    }
    else {
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getInMemory().at("2")->getData()(i), 1);
      BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsShardable().at("2")->getData()(i), 1); // TODO...
    }
  }

  // Test undo to remove the appended values
  appendToAxis.undo(collection_1, device);

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  BOOST_CHECK_EQUAL(axis_2_ptr->getNLabels(), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsShardable().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getInMemory().at("2")->getData()(i), 0);
  }
}

/*TensorDeleteFromAxis Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorDeleteFromAxis)
{
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
  ptr = new TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorDeleteFromAxis)
{
  TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  ptr = new TensorDeleteFromAxis<int, float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoAndTensorDeleteFromAxis)
{
  // TODO
}

/*TensorUpdate Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorUpdate)
{
  TensorUpdate<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  TensorUpdate<float, Eigen::DefaultDevice, 3>* nullPointer = nullptr;
  ptr = new TensorUpdate<float, Eigen::DefaultDevice, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorUpdate)
{
  TensorUpdate<float, Eigen::DefaultDevice, 3>* ptr = nullptr;
  ptr = new TensorUpdate<float, Eigen::DefaultDevice, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(redoAndUndoTensorUpdate)
{
  Eigen::DefaultDevice device;

  // Set up the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  // Set up table 1
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

  // set up table 2
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

  // Set up the collection
  TensorCollection<Eigen::DefaultDevice> collection_1;
  collection_1.addTensorTable(tensorTable1_ptr);
  collection_1.addTensorTable(tensorTable2_ptr);

  // Set up the update values
  Eigen::Tensor<float, 3> values_new_values(nlabels1, 1, 1);
  for (int i = 0; i < nlabels1; ++i)
    values_new_values(i, 0, 0) = (float)((i + 1) * 10);
  TensorDataDefaultDevice<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, 1, 1 }));
  values_new.setData(values_new_values);

  // Set up the update
  TensorUpdate<float, Eigen::DefaultDevice, 3> tensorUpdate("1", SelectTable1(), std::make_shared<TensorDataDefaultDevice<float, 3>>(values_new));

  // Test redo
  tensorUpdate.redo(collection_1, device);
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 1; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()->getData()(i, j, k), values_new_values(i, j, k));
        BOOST_CHECK_EQUAL(tensorUpdate.getValuesOld()->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test undo
  tensorUpdate.undo(collection_1, device);
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 1; ++k) {
        BOOST_CHECK_EQUAL(tensorTable1_ptr->getData()->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()