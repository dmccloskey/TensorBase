/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorSelect1 test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorSelect.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorSelect1)

/*TensorSelect DefaultDevice Tests*/
BOOST_AUTO_TEST_CASE(constructorTensorSelectDefaultDevice)
{
  TensorSelect* ptr = nullptr;
  TensorSelect* nullPointer = nullptr;
	ptr = new TensorSelect();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorTensorSelectDefaultDevice)
{
  TensorSelect* ptr = nullptr;
	ptr = new TensorSelect();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(selectClause1DefaultDevice)
{
  Eigen::DefaultDevice device;

  // Set up the tables
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1 * nlabels2;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // Set up the SelectClause
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);
  SelectClause<int, Eigen::DefaultDevice> select_clause1("1", "1", "x", select_labels1);
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values2(2);
  labels_values2.setValues({ 0, 2 });
  select_labels2->setData(labels_values2);
  SelectClause<int, Eigen::DefaultDevice> select_clause2("2", "2", "y", select_labels2);

  // Test the unchanged values
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.selectClause(collection_1_ptr, select_clause1, device);
  tensorSelect.selectClause(collection_1_ptr, select_clause2, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5); // unchanged

  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  // Test the expected data sizes after the apply commend
  tensorSelect.applySelect(collection_1_ptr, {"1","2"}, {"1","2a"}, device);
  BOOST_CHECK(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize(), (nlabels1 - 1) * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), (nlabels1 - 1), nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 1; i < nlabels1; ++i) {
        BOOST_CHECK_EQUAL(table_1_values(i-1,j,k), tensor_values1(i, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize(), nlabels1 * nlabels2); // unchanged
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize(), nlabels1 * (nlabels2 - 1));
  std::shared_ptr<int[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2-1);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      if (j < 1) BOOST_CHECK_EQUAL(table_2a_values(i, j), tensor_values2(i, j));
      else if (j > 1) BOOST_CHECK_EQUAL(table_2a_values(i, j-1), tensor_values2(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(selectClause2DefaultDevice)
{
	Eigen::DefaultDevice device;

	// Set up the tables
	Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
	dimensions1(0) = "x";
	dimensions2(0) = "y";
	dimensions3(0) = "z";
	int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
	Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
	labels1.setValues({ { 0, 1} });
	labels2.setValues({ { 0, 1, 2 } });
	labels3.setValues({ { 0, 1, 2, 3, 4 } });

	TensorTableDefaultDevice<float, 3> tensorTable1("1");
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
	tensorTable1.setAxes(device);
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1 * nlabels2;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
	std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

	TensorTableDefaultDevice<int, 2> tensorTable2("2");
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
	tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
	std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

	// Set up the collection
	TensorCollectionDefaultDevice collection_1;
	collection_1.addTensorTable(tensorTable1_ptr, "1");
	collection_1.addTensorTable(tensorTable2_ptr, "1");
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

	// Set up the SelectClause
	std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
	Eigen::Tensor<int, 2> labels_values1(1, 1);
	labels_values1.setValues({ {1} });
	select_labels1->setData(labels_values1);
	SelectClause<int, Eigen::DefaultDevice> select_clause1("1", "1", select_labels1);
	std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 2 }));
	Eigen::Tensor<int, 2> labels_values2(1, 2);
	labels_values2.setValues({ {0, 2} });
	select_labels2->setData(labels_values2);
	SelectClause<int, Eigen::DefaultDevice> select_clause2("2", "2", select_labels2);

	// Test the unchanged values
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

	TensorSelect tensorSelect;
	// Test the expected view indices after the select command
	tensorSelect.selectClause(collection_1_ptr, select_clause1, device);
	tensorSelect.selectClause(collection_1_ptr, select_clause2, device);
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 0);
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4); // unchanged
	BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5); // unchanged

	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 0);
	BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  // Test the expected data indices after the apply commend
  tensorSelect.applySelect(collection_1_ptr, { "1","2" }, { "1","2a" }, device);
  BOOST_CHECK(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize(), (nlabels1 - 1) * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), (nlabels1 - 1), nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 1; i < nlabels1; ++i) {
        BOOST_CHECK_EQUAL(table_1_values(i - 1, j, k), tensor_values1(i, j, k));
      }
    }
  }
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize(), nlabels1 * nlabels2); // unchanged
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize(), nlabels1 * (nlabels2 - 1));
  std::shared_ptr<int[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2 - 1);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      if (j < 1) BOOST_CHECK_EQUAL(table_2a_values(i, j), tensor_values2(i, j));
      else if (j > 1) BOOST_CHECK_EQUAL(table_2a_values(i, j - 1), tensor_values2(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(whereClause1DefaultDevice)
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

  // Setup table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  // setup the table 1 tensor data
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  int iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableDefaultDevice<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  // setup the table 2 tensor data
  Eigen::Tensor<double, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = double(iter);
      ++iter;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<double, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // Set up the WhereClauses
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);

  std::shared_ptr<TensorDataDefaultDevice<float, 1>> select_values1 = std::make_shared<TensorDataDefaultDevice<float, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<float, 1> values_values1(1);
  values_values1.setValues({ 17 });
  select_values1->setData(values_values1);
  WhereClause<int, float, Eigen::DefaultDevice> where_clause1("1", "1", "x", select_labels1, select_values1, logicalComparitors::LESS_THAN, logicalModifiers::NONE, logicalContinuators::OR, logicalContinuators::AND);

  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values2(2);
  labels_values2.setValues({ 0, 2 });
  select_labels2->setData(labels_values2);

  std::shared_ptr<TensorDataDefaultDevice<double, 1>> select_values2 = std::make_shared<TensorDataDefaultDevice<double, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<double, 1> values_values2(2);
  values_values2.setValues({ 3, 3 });
  select_values2->setData(values_values2);
  WhereClause<int, double, Eigen::DefaultDevice> where_clause2("2", "2", "y", select_labels2, select_values2, logicalComparitors::LESS_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);

  // Test the indices views
  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.whereClause(collection_1_ptr, where_clause1, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 0);

  tensorSelect.whereClause(collection_1_ptr, where_clause2, device);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged

  // Apply the select clause
  tensorSelect.applySelect(collection_1_ptr, { "1", "2" }, { "1", "2" }, device);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { 2, 1, 2 };
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(i), dimensions1_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDataDimensions().at(i), dimensions1_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);

  Eigen::array<Eigen::Index, 2> dimensions2_test = { 1, 3 };
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(i), dimensions2_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDataDimensions().at(i), dimensions2_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1); 
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);
}

BOOST_AUTO_TEST_CASE(whereClause2DefaultDevice)
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

  // Setup table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  // setup the table 1 tensor data
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  int iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableDefaultDevice<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  // setup the table 2 tensor data
  Eigen::Tensor<double, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = double(iter);
      ++iter;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<double, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // Set up the WhereClauses
  std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values1(1, 1);
  labels_values1.setValues({ { 1 } });
  select_labels1->setData(labels_values1);

  std::shared_ptr<TensorDataDefaultDevice<float, 1>> select_values1 = std::make_shared<TensorDataDefaultDevice<float, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<float, 1> values_values1(1);
  values_values1.setValues({ 17 });
  select_values1->setData(values_values1);
  WhereClause<int, float, Eigen::DefaultDevice> where_clause1("1", "1", select_labels1, select_values1, logicalComparitors::LESS_THAN, logicalModifiers::NONE, logicalContinuators::OR, logicalContinuators::AND);

  std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 2 }));
  Eigen::Tensor<int, 2> labels_values2(1, 2);
  labels_values2.setValues({ { 0, 2 } });
  select_labels2->setData(labels_values2);

  std::shared_ptr<TensorDataDefaultDevice<double, 1>> select_values2 = std::make_shared<TensorDataDefaultDevice<double, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<double, 1> values_values2(2);
  values_values2.setValues({ 3, 3 });
  select_values2->setData(values_values2);
  WhereClause<int, double, Eigen::DefaultDevice> where_clause2("2", "2", select_labels2, select_values2, logicalComparitors::LESS_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);

  // Test the indices views
  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.whereClause(collection_1_ptr, where_clause1, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 0);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 0);

  tensorSelect.whereClause(collection_1_ptr, where_clause2, device);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 0);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged

  // Apply the select clause
  tensorSelect.applySelect(collection_1_ptr, { "1", "2" }, { "1", "2" }, device);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { 2, 1, 2 };
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(i), dimensions1_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDataDimensions().at(i), dimensions1_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);

  Eigen::array<Eigen::Index, 2> dimensions2_test = { 1, 3 };
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(i), dimensions2_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDataDimensions().at(i), dimensions2_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);
}

BOOST_AUTO_TEST_CASE(sortClause1DefaultDevice)
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

  // Setup table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  // setup the table 1 tensor data
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  int iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableDefaultDevice<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  // setup the table 2 tensor data
  Eigen::Tensor<double, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = double(iter);
      ++iter;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<double, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // setup the sort clauses
  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);
  SortClause<int, Eigen::DefaultDevice> sort_clause_1("1", "2", "y", select_labels1, sortOrder::order::DESC);

  std::shared_ptr<TensorDataDefaultDevice<int, 1>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values2(1);
  labels_values2.setValues({ 0 });
  select_labels2->setData(labels_values2);
  SortClause<int, Eigen::DefaultDevice> sort_clause_2("2", "1", "x", select_labels2, sortOrder::order::DESC);

  // Test the expected view indices after the select command
  TensorSelect tensorSelect;
  tensorSelect.sortClause(collection_1_ptr, sort_clause_1, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 5);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 4);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 1);

  tensorSelect.sortClause(collection_1_ptr, sort_clause_2, device);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 3);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 1);

  // Apply the select clause
  tensorSelect.applySort(collection_1_ptr, { "1", "2" }, { "1", "2a" }, device);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { nlabels1, nlabels2, nlabels3 };
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(i), dimensions1_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDataDimensions().at(i), dimensions1_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5);
  BOOST_CHECK(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize(), nlabels1 * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), nlabels1, nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        BOOST_CHECK_EQUAL(table_1_values(i, j, k), tensor_values1(nlabels1 - i - 1, j, nlabels3 - k - 1));
      }
    }
  }

  Eigen::array<Eigen::Index, 2> dimensions2_test = { nlabels1, nlabels2 };
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(i), dimensions2_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDataDimensions().at(i), dimensions2_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 3);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize(), nlabels1 * nlabels2);

  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(2), 3);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize(), nlabels1 * nlabels2);
  std::shared_ptr<double[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      BOOST_CHECK_EQUAL(table_2a_values(i, j), tensor_values2(i, nlabels2 - j -1));
    }
  }
}

BOOST_AUTO_TEST_CASE(sortClause2DefaultDevice)
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

  // Setup table 1 axes
  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);

  // setup the table 1 tensor data
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  int iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values1(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableDefaultDevice<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  // setup the table 2 tensor data
  Eigen::Tensor<double, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  iter = 0;
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = double(iter);
      ++iter;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<double, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // setup the sort clauses
  std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels1 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values1(1, 1);
  labels_values1.setValues({ {1} });
  select_labels1->setData(labels_values1);
  SortClause<int, Eigen::DefaultDevice> sort_clause_1("1", "2", select_labels1, sortOrder::order::DESC);

  std::shared_ptr<TensorDataDefaultDevice<int, 2>> select_labels2 = std::make_shared<TensorDataDefaultDevice<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values2(1, 1);
  labels_values2.setValues({ {0} });
  select_labels2->setData(labels_values2);
  SortClause<int, Eigen::DefaultDevice> sort_clause_2("2", "1", select_labels2, sortOrder::order::DESC);

  // Test the expected view indices after the select command
  TensorSelect tensorSelect;
  tensorSelect.sortClause(collection_1_ptr, sort_clause_1, device);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 5);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 4);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 1);

  tensorSelect.sortClause(collection_1_ptr, sort_clause_2, device);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 3);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 1);

  // Apply the select clause
  tensorSelect.applySort(collection_1_ptr, { "1", "2" }, { "1", "2a" }, device);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { nlabels1, nlabels2, nlabels3 };
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDimensions().at(i), dimensions1_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable1_ptr->getDataDimensions().at(i), dimensions1_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5);
  BOOST_CHECK(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize(), nlabels1 * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), nlabels1, nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        BOOST_CHECK_EQUAL(table_1_values(i, j, k), tensor_values1(nlabels1 - i - 1, j, nlabels3 - k - 1));
      }
    }
  }

  Eigen::array<Eigen::Index, 2> dimensions2_test = { nlabels1, nlabels2 };
  for (int i = 0; i < 2; ++i) {
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDimensions().at(i), dimensions2_test.at(i));
    BOOST_CHECK_EQUAL(tensorTable2_ptr->getDataDimensions().at(i), dimensions2_test.at(i));
  }
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 3);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize(), nlabels1 * nlabels2);

  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(2), 3);
  BOOST_CHECK_EQUAL(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize(), nlabels1 * nlabels2);
  std::shared_ptr<double[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      BOOST_CHECK_EQUAL(table_2a_values(i, j), tensor_values2(i, nlabels2 - j - 1));
    }
  }
}

BOOST_AUTO_TEST_CASE(reductionClause1DefaultDevice)
{
  Eigen::DefaultDevice device;

  // Set up the tables
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  float tensor_values_sum1 = 0;
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1 * nlabels2;
        tensor_values_sum1 += tensor_values1(i, j, k);
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  int tensor_values_sum2 = 0;
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
      tensor_values_sum2 += tensor_values2(i, j);
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // Set up the reductionClause  
  ReductionClause<Eigen::DefaultDevice> reduction_clause1("1", reductionFunctions::SUM);

  // Test the unchanged values
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.applyReduction(collection_1_ptr, reduction_clause1, device);
  BOOST_CHECK_CLOSE(tensorTable1_ptr->getData()(0, 0, 0), tensor_values_sum1, 1e-3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(0, 0), tensor_values2(0,0));
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
}

BOOST_AUTO_TEST_CASE(scanClause1DefaultDevice)
{
  Eigen::DefaultDevice device;

  // Set up the tables
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { 0, 1} });
  labels2.setValues({ { 0, 1, 2 } });
  labels3.setValues({ { 0, 1, 2, 3, 4 } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  float tensor_values_sum1 = 0;
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1 * nlabels2;
        tensor_values_sum1 += tensor_values1(i, j, k);
      }
    }
  }
  tensorTable1.setData(tensor_values1);
  std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::DefaultDevice>>)std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  int tensor_values_sum2 = 0;
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
      tensor_values_sum2 += tensor_values2(i, j);
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<int, Eigen::DefaultDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionDefaultDevice collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(collection_1);

  // Set up the reductionClause  
  ScanClause<Eigen::DefaultDevice> scan_clause1("1", {"1", "2", "3"}, scanFunctions::CUMSUM);

  // Test the unchanged values
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2);
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  tensorSelect.applyScan(collection_1_ptr, scan_clause1, device);
  BOOST_CHECK_CLOSE(tensorTable1_ptr->getData()(0, 0, 0), tensor_values1(0,0,0), 1e-3);
  BOOST_CHECK_CLOSE(tensorTable1_ptr->getData()(nlabels1 - 1, nlabels2 - 1, nlabels3 - 1), tensor_values_sum1, 1e-3);
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(2), 3); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(3), 4); // unchanged
  BOOST_CHECK_EQUAL(tensorTable1_ptr->getIndicesView().at("3")->getData()(4), 5); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(0, 0), tensor_values2(0, 0));
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getData()(nlabels1 - 1, nlabels2 - 1), tensor_values2(nlabels1 - 1, nlabels2 - 1));
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("1")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(0), 1); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(1), 2); // unchanged
  BOOST_CHECK_EQUAL(tensorTable2_ptr->getIndicesView().at("2")->getData()(2), 3); // unchanged
}

BOOST_AUTO_TEST_SUITE_END()