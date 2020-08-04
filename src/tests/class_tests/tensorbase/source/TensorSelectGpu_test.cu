/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorSelect.h>
#include <TensorBase/ml/TensorCollectionGpu.h>

using namespace TensorBase;
using namespace std;

/*TensorSelect DefaultDevice Tests*/
void test_constructorTensorSelectGpu()
{
  TensorSelect* ptr = nullptr;
  TensorSelect* nullPointer = nullptr;
	ptr = new TensorSelect();
  assert(ptr != nullPointer);
}

void test_destructorTensorSelectGpu()
{
  TensorSelect* ptr = nullptr;
	ptr = new TensorSelect();
  delete ptr;
}

void test_selectClause1Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Set up the SelectClause
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);
  SelectClause<int, Eigen::GpuDevice> select_clause1("1", "1", "x", select_labels1);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values2(2);
  labels_values2.setValues({ 0, 2 });
  select_labels2->setData(labels_values2);
  SelectClause<int, Eigen::GpuDevice> select_clause2("2", "2", "y", select_labels2);

  // Test the unchanged values
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.selectClause(collection_1_ptr, select_clause1, device);
  tensorSelect.selectClause(collection_1_ptr, select_clause2, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5); // unchanged

  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 0);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  // Test the expected data sizes after the apply command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySelect(collection_1_ptr, { "1","2" }, { "1","2a" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  assert(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize() == (nlabels1 - 1) * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getHDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), (nlabels1 - 1), nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 1; i < nlabels1; ++i) {
        assert(table_1_values(i-1,j,k) == tensor_values1(i, j, k));
      }
    }
  }
  assert(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize() == nlabels1 * nlabels2); // unchanged
  assert(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize() == nlabels1 * (nlabels2 - 1));
  std::shared_ptr<int[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getHDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2-1);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      if (j < 1) assert(table_2a_values(i, j) == tensor_values2(i, j));
      else if (j > 1) assert(table_2a_values(i, j-1) == tensor_values2(i, j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectClause2Gpu()
{
	cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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

	TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
	std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

	TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
	tensorTable2.setAxes(device);
  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
	std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

	// Set up the collection
	TensorCollectionGpu collection_1;
	collection_1.addTensorTable(tensorTable1_ptr, "1");
	collection_1.addTensorTable(tensorTable2_ptr, "1");
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

	// Set up the SelectClause
	std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
	Eigen::Tensor<int, 2> labels_values1(1, 1);
	labels_values1.setValues({ {1} });
	select_labels1->setData(labels_values1);
	SelectClause<int, Eigen::GpuDevice> select_clause1("1", "1", select_labels1);
	std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 2 }));
	Eigen::Tensor<int, 2> labels_values2(1, 2);
	labels_values2.setValues({ {0, 2} });
	select_labels2->setData(labels_values2);
	SelectClause<int, Eigen::GpuDevice> select_clause2("2", "2", select_labels2);

	// Test the unchanged values
	assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
	assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

	TensorSelect tensorSelect;
	// Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.selectClause(collection_1_ptr, select_clause1, device);
  tensorSelect.selectClause(collection_1_ptr, select_clause2, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
	assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 0);
	assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
	assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4); // unchanged
	assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5); // unchanged

	assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
	assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 0);
	assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  // Test the expected data indices after the apply commend
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySelect(collection_1_ptr, { "1","2" }, { "1","2a" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  assert(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize() == (nlabels1 - 1) * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getHDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), (nlabels1 - 1), nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 1; i < nlabels1; ++i) {
        assert(table_1_values(i - 1, j, k) == tensor_values1(i, j, k));
      }
    }
  }
  assert(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize() == nlabels1 * nlabels2); // unchanged
  assert(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize() == nlabels1 * (nlabels2 - 1));
  std::shared_ptr<int[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getHDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2 - 1);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      if (j < 1) assert(table_2a_values(i, j) == tensor_values2(i, j));
      else if (j > 1) assert(table_2a_values(i, j - 1) == tensor_values2(i, j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_whereClause1Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableGpuPrimitiveT<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<double, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Set up the WhereClauses
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);

  std::shared_ptr<TensorDataGpuPrimitiveT<float, 1>> select_values1 = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<float, 1> values_values1(1);
  values_values1.setValues({ 17 });
  select_values1->setData(values_values1);
  WhereClause<int, float, Eigen::GpuDevice> where_clause1("1", "1", "x", select_labels1, select_values1, logicalComparitors::LESS_THAN, logicalModifiers::NONE, logicalContinuators::OR, logicalContinuators::AND);

  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<int, 1> labels_values2(2);
  labels_values2.setValues({ 0, 2 });
  select_labels2->setData(labels_values2);

  std::shared_ptr<TensorDataGpuPrimitiveT<double, 1>> select_values2 = std::make_shared<TensorDataGpuPrimitiveT<double, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<double, 1> values_values2(2);
  values_values2.setValues({ 3, 3 });
  select_values2->setData(values_values2);
  WhereClause<int, double, Eigen::GpuDevice> where_clause2("2", "2", "y", select_labels2, select_values2, logicalComparitors::LESS_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);

  // Test the indices views
  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.whereClause(collection_1_ptr, where_clause1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 0);

  tensorSelect.whereClause(collection_1_ptr, where_clause2, device);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 0);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged

  // Apply the select clause
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySelect(collection_1_ptr, { "1", "2" }, { "1", "2" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { 2, 1, 2 };
  for (int i = 0; i < 3; ++i) {
    assert(tensorTable1_ptr->getDimensions().at(i) == dimensions1_test.at(i));
    assert(tensorTable1_ptr->getDataDimensions().at(i) == dimensions1_test.at(i));
  }
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);

  Eigen::array<Eigen::Index, 2> dimensions2_test = { 1, 3 };
  for (int i = 0; i < 2; ++i) {
    assert(tensorTable2_ptr->getDimensions().at(i) == dimensions2_test.at(i));
    assert(tensorTable2_ptr->getDataDimensions().at(i) == dimensions2_test.at(i));
  }
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1); 
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_whereClause2Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableGpuPrimitiveT<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<double, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Set up the WhereClauses
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values1(1, 1);
  labels_values1.setValues({ { 1 } });
  select_labels1->setData(labels_values1);

  std::shared_ptr<TensorDataGpuPrimitiveT<float, 1>> select_values1 = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<float, 1> values_values1(1);
  values_values1.setValues({ 17 });
  select_values1->setData(values_values1);
  WhereClause<int, float, Eigen::GpuDevice> where_clause1("1", "1", select_labels1, select_values1, logicalComparitors::LESS_THAN, logicalModifiers::NONE, logicalContinuators::OR, logicalContinuators::AND);

  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 2 }));
  Eigen::Tensor<int, 2> labels_values2(1, 2);
  labels_values2.setValues({ { 0, 2 } });
  select_labels2->setData(labels_values2);

  std::shared_ptr<TensorDataGpuPrimitiveT<double, 1>> select_values2 = std::make_shared<TensorDataGpuPrimitiveT<double, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
  Eigen::Tensor<double, 1> values_values2(2);
  values_values2.setValues({ 3, 3 });
  select_values2->setData(values_values2);
  WhereClause<int, double, Eigen::GpuDevice> where_clause2("2", "2", select_labels2, select_values2, logicalComparitors::LESS_THAN_OR_EQUAL_TO, logicalModifiers::NONE, logicalContinuators::AND, logicalContinuators::AND);

  // Test the indices views
  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.whereClause(collection_1_ptr, where_clause1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 0);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 0);

  tensorSelect.whereClause(collection_1_ptr, where_clause2, device);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 0);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged

  // Apply the select clause
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySelect(collection_1_ptr, { "1", "2" }, { "1", "2" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { 2, 1, 2 };
  for (int i = 0; i < 3; ++i) {
    assert(tensorTable1_ptr->getDimensions().at(i) == dimensions1_test.at(i));
    assert(tensorTable1_ptr->getDataDimensions().at(i) == dimensions1_test.at(i));
  }
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);

  Eigen::array<Eigen::Index, 2> dimensions2_test = { 1, 3 };
  for (int i = 0; i < 2; ++i) {
    assert(tensorTable2_ptr->getDimensions().at(i) == dimensions2_test.at(i));
    assert(tensorTable2_ptr->getDataDimensions().at(i) == dimensions2_test.at(i));
  }
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortClause1Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableGpuPrimitiveT<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<double, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // setup the sort clauses
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values1(1);
  labels_values1.setValues({ 1 });
  select_labels1->setData(labels_values1);
  SortClause<int, Eigen::GpuDevice> sort_clause_1("1", "2", "y", select_labels1, sortOrder::order::DESC);

  std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> labels_values2(1);
  labels_values2.setValues({ 0 });
  select_labels2->setData(labels_values2);
  SortClause<int, Eigen::GpuDevice> sort_clause_2("2", "1", "x", select_labels2, sortOrder::order::DESC);

  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  TensorSelect tensorSelect;
  tensorSelect.sortClause(collection_1_ptr, sort_clause_1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 5);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 4);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 1);

  tensorSelect.sortClause(collection_1_ptr, sort_clause_2, device);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 3);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 1);

  // Apply the select clause
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySort(collection_1_ptr, { "1", "2" }, { "1", "2a" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { nlabels1, nlabels2, nlabels3 };
  for (int i = 0; i < 3; ++i) {
    assert(tensorTable1_ptr->getDimensions().at(i) == dimensions1_test.at(i));
    assert(tensorTable1_ptr->getDataDimensions().at(i) == dimensions1_test.at(i));
  }
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5);
  assert(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  assert(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize() == nlabels1 * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getHDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), nlabels1, nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        assert(table_1_values(i, j, k) == tensor_values1(nlabels1 - i - 1, j, nlabels3 - k - 1));
      }
    }
  }

  Eigen::array<Eigen::Index, 2> dimensions2_test = { nlabels1, nlabels2 };
  for (int i = 0; i < 2; ++i) {
    assert(tensorTable2_ptr->getDimensions().at(i) == dimensions2_test.at(i));
    assert(tensorTable2_ptr->getDataDimensions().at(i) == dimensions2_test.at(i));
  }
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 3);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize() == nlabels1 * nlabels2);

  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(0) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(1) == 2);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(0) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(1) == 2);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(2) == 3);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize() == nlabels1 * nlabels2);
  std::shared_ptr<double[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getHDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      assert(table_2a_values(i, j) == tensor_values2(i, nlabels2 - j -1));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortClause2Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // Setup table 2 axes
  TensorTableGpuPrimitiveT<double, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<double, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<double, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // setup the sort clauses
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels1 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values1(1, 1);
  labels_values1.setValues({ {1} });
  select_labels1->setData(labels_values1);
  SortClause<int, Eigen::GpuDevice> sort_clause_1("1", "2", select_labels1, sortOrder::order::DESC);

  std::shared_ptr<TensorDataGpuPrimitiveT<int, 2>> select_labels2 = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  Eigen::Tensor<int, 2> labels_values2(1, 1);
  labels_values2.setValues({ {0} });
  select_labels2->setData(labels_values2);
  SortClause<int, Eigen::GpuDevice> sort_clause_2("2", "1", select_labels2, sortOrder::order::DESC);

  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  TensorSelect tensorSelect;
  tensorSelect.sortClause(collection_1_ptr, sort_clause_1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 5);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 4);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 1);

  tensorSelect.sortClause(collection_1_ptr, sort_clause_2, device);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 3);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 1);

  // Apply the select clause
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applySort(collection_1_ptr, { "1", "2" }, { "1", "2a" }, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test for the expected table attributes
  Eigen::array<Eigen::Index, 3> dimensions1_test = { nlabels1, nlabels2, nlabels3 };
  for (int i = 0; i < 3; ++i) {
    assert(tensorTable1_ptr->getDimensions().at(i) == dimensions1_test.at(i));
    assert(tensorTable1_ptr->getDataDimensions().at(i) == dimensions1_test.at(i));
  }
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4);
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5);
  assert(collection_1_ptr->getTableNames() == std::vector<std::string>({ "1","2","2a" }));
  assert(collection_1_ptr->getTensorTableConcept("1")->getDataTensorSize() == nlabels1 * nlabels2 * nlabels3);
  std::shared_ptr<float[]> table_1_data_ptr;
  collection_1_ptr->getTensorTableConcept("1")->getHDataPointer(table_1_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<float, 3>> table_1_values(table_1_data_ptr.get(), nlabels1, nlabels2, nlabels3);
  for (int k = 0; k < nlabels3; ++k) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int i = 0; i < nlabels1; ++i) {
        assert(table_1_values(i, j, k) == tensor_values1(nlabels1 - i - 1, j, nlabels3 - k - 1));
      }
    }
  }

  Eigen::array<Eigen::Index, 2> dimensions2_test = { nlabels1, nlabels2 };
  for (int i = 0; i < 2; ++i) {
    assert(tensorTable2_ptr->getDimensions().at(i) == dimensions2_test.at(i));
    assert(tensorTable2_ptr->getDataDimensions().at(i) == dimensions2_test.at(i));
  }
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 3);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2")->getDataTensorSize() == nlabels1 * nlabels2);

  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(0) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("1")->getData()(1) == 2);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(0) == 1);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(1) == 2);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getIndicesView().at("2")->getData()(2) == 3);
  assert(collection_1_ptr->getTensorTableConcept("2a")->getDataTensorSize() == nlabels1 * nlabels2);
  std::shared_ptr<double[]> table_2a_data_ptr;
  collection_1_ptr->getTensorTableConcept("2a")->getHDataPointer(table_2a_data_ptr);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> table_2a_values(table_2a_data_ptr.get(), nlabels1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    for (int i = 0; i < nlabels1; ++i) {
      assert(table_2a_values(i, j) == tensor_values2(i, nlabels2 - j - 1));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_reductionClause1Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Set up the reductionClause  
  ReductionClause<Eigen::GpuDevice> reduction_clause1("1", reductionFunctions::SUM);

  // Test the unchanged values
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applyReduction(collection_1_ptr, reduction_clause1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getData()(0, 0, 0) == tensor_values_sum1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5); // unchanged
  assert(tensorTable2_ptr->getData()(0, 0) == tensor_values2(0,0));
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_scanClause1Gpu()
{
  cudaStream_t stream; assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
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
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  collection_1.addTensorTable(tensorTable2_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Set up the reductionClause  
  ScanClause<Eigen::GpuDevice> scan_clause1("1", {"1", "2", "3"}, scanFunctions::CUMSUM);

  // Test the unchanged values
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2);
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3);

  TensorSelect tensorSelect;
  // Test the expected view indices after the select command
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorSelect.applyScan(collection_1_ptr, scan_clause1, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable1_ptr->getData()(0, 0, 0) == tensor_values1(0,0,0));
  assert(tensorTable1_ptr->getData()(nlabels1 - 1, nlabels2 - 1, nlabels3 - 1) == tensor_values_sum1);
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(0) == 1); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(1) == 2); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(2) == 3); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(3) == 4); // unchanged
  assert(tensorTable1_ptr->getIndicesView().at("3")->getData()(4) == 5); // unchanged
  assert(tensorTable2_ptr->getData()(0, 0) == tensor_values2(0, 0));
  assert(tensorTable2_ptr->getData()(nlabels1 - 1, nlabels2 - 1) == tensor_values2(nlabels1 - 1, nlabels2 - 1));
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("1")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(0) == 1); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(1) == 2); // unchanged
  assert(tensorTable2_ptr->getIndicesView().at("2")->getData()(2) == 3); // unchanged

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{
  test_constructorTensorSelectGpu();
  test_destructorTensorSelectGpu();
  test_selectClause1Gpu();
  test_selectClause2Gpu();
  test_whereClause1Gpu();
  test_whereClause2Gpu();
  test_sortClause1Gpu();
  test_sortClause2Gpu();
  test_reductionClause1Gpu();
  test_scanClause1Gpu();
  return 0;
}
#endif