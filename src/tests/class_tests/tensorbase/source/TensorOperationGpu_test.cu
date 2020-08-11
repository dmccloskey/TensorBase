/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <TensorBase/ml/TensorSelect.h>
#include <TensorBase/ml/TensorOperationGpu.h>
#include <TensorBase/ml/TensorCollectionGpu.h>

using namespace TensorBase;
using namespace std;

/// The select Functor for table 1
struct SelectTable1 {
  template<typename DeviceT>
  void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 1:  all values in dim 0 for labels = 0 in dims 1 and 2
    std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels_t1a1 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 2 }));
    Eigen::Tensor<int, 1> labels_values_t1a1(2);
    labels_values_t1a1.setValues({ 0, 1 });
    select_labels_t1a1->setData(labels_values_t1a1);
    SelectClause<int, DeviceT> select_clause1("1", "1", "x", select_labels_t1a1);
    std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels_t1a2 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t1a2(1);
    labels_values_t1a2.setValues({ 0 });
    select_labels_t1a2->setData(labels_values_t1a2);
    SelectClause<int, DeviceT> select_clause2("1", "2", "y", select_labels_t1a2);
    std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels_t1a3 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t1a3(1);
    labels_values_t1a3.setValues({ 0 });
    select_labels_t1a3->setData(labels_values_t1a3);
    SelectClause<int, DeviceT> select_clause3("1", "3", "z", select_labels_t1a3);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause1, device);
    tensorSelect.selectClause(tensor_collection, select_clause2, device);
    tensorSelect.selectClause(tensor_collection, select_clause3, device);

    // Apply the select clause
    if (apply_select)
      tensorSelect.applySelect(tensor_collection, { "1" }, { "1" }, device);
  }
  bool apply_select = true;
};

/// The delete select Functor for table 1
struct DeleteTable1 {
  template<typename DeviceT>
  void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
    // Set up the SelectClauses for table 1 and axis 2 where labels=1
    std::shared_ptr<TensorDataGpuPrimitiveT<int, 1>> select_labels_t1a2 = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(Eigen::array<Eigen::Index, 1>({ 1 }));
    Eigen::Tensor<int, 1> labels_values_t1a2(1);
    labels_values_t1a2.setValues({ 1 });
    select_labels_t1a2->setData(labels_values_t1a2);
    SelectClause<int, DeviceT> select_clause2("1", "2", "y", select_labels_t1a2);

    TensorSelect tensorSelect;

    // Select the axes
    tensorSelect.selectClause(tensor_collection, select_clause2, device);
  }
};

/*TensorAppendToAxis Tests*/
void test_constructorTensorAppendToAxis()
{
  TensorAppendToAxis<int, float, Eigen::GpuDevice, 3>* ptr = nullptr;
  TensorAppendToAxis<int, float, Eigen::GpuDevice, 3>* nullPointer = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::GpuDevice, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorAppendToAxis()
{
  TensorAppendToAxis<int, float, Eigen::GpuDevice, 3>* ptr = nullptr;
	ptr = new TensorAppendToAxis<int, float, Eigen::GpuDevice, 3>();
  delete ptr;
}

void test_redoAndUndoTensorAppendToAxis()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(axis_1_ptr);
  tensorTable1.addTensorAxis(axis_2_ptr);
  tensorTable1.addTensorAxis(axis_3_ptr);
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
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

  // Set up the new labels
  Eigen::Tensor<int, 2> labels_new_values(1, nlabels2);
  labels_new_values.setValues({ {3, 4, 5} });
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({1, 3}));
  labels_new.setData(labels_new_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // Set up the new values
  Eigen::Tensor<float, 3> tensor_values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_values_new(i, j, k) = i + j * nlabels1 + k * nlabels1*nlabels2 + nlabels1 + nlabels2 * nlabels1 + nlabels3 * nlabels1*nlabels2;
      }
    }
  }
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  values_new.setData(tensor_values_new);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);

  // Reset `is_modified` to test for the new additions after the call to `setData`
  for (auto& tensor_table_map : collection_1_ptr->tables_) {
    for (auto& is_modified_map : tensor_table_map.second->getIsModified()) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
    }
  }

  // Test redo to append the new values
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  TensorAppendToAxis<int, float, Eigen::GpuDevice, 3> appendToAxis("1", "2", labels_new_ptr, values_new_ptr);
  appendToAxis.redo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j + nlabels2, k), tensor_values_new(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  gpuCheckEqual(axis_2_ptr->getNLabels(), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  gpuCheckEqual(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2 + nlabels2);
  for (int i = 0; i < nlabels2 + nlabels2; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    if (i < nlabels2) {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
      gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
      gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
      gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 2);
      gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i - nlabels2 + 1);
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_inserted_, 3);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_inserted_, 3);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_inserted_, 30);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_inserted_, 0);

  // Test undo to remove the appended values
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  appendToAxis.undo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  gpuCheckEqual(axis_2_ptr->getNLabels(), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  gpuCheckEqual(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_labels_inserted_, 3);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_indices_inserted_, 3);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_updated_, 0);
  gpuCheckEqual(appendToAxis.getRedoLog().n_data_inserted_, 30);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_deleted_, 3);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_deleted_, 3);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_deleted_, 30);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(appendToAxis.getUndoLog().n_data_inserted_, 0);

  gpuErrchk(cudaStreamDestroy(stream));
}

/*TensorDeleteFromAxis Tests*/
void test_constructorTensorDeleteFromAxis()
{
  TensorDeleteFromAxisGpuPrimitiveT<int, float, 3>* ptr = nullptr;
  TensorDeleteFromAxisGpuPrimitiveT<int, float, 3>* nullPointer = nullptr;
  ptr = new TensorDeleteFromAxisGpuPrimitiveT<int, float, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorDeleteFromAxis()
{
  TensorDeleteFromAxisGpuPrimitiveT<int, float, 3>* ptr = nullptr;
  ptr = new TensorDeleteFromAxisGpuPrimitiveT<int, float, 3>();
  delete ptr;
}

void test_redoAndTensorDeleteFromAxis()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(axis_1_ptr);
  tensorTable1.addTensorAxis(axis_2_ptr);
  tensorTable1.addTensorAxis(axis_3_ptr);
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
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

  // Make the expected tensor data
  Eigen::Tensor<float, 3> expected_tensor_values(nlabels1, nlabels2 - 1, nlabels3);  
  for (int i = 0; i < nlabels1; ++i) {
    int iter = 0;
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        if (j != 1)
          expected_tensor_values(i, iter, k) = i + j * nlabels1 + k * nlabels1*nlabels2;
      }
      if (j != 1) ++iter;
    }
  }

  // Reset `is_modified` to test for the new additions after the call to `setData`
  for (auto& tensor_table_map : collection_1_ptr->tables_) {
    for (auto& is_modified_map : tensor_table_map.second->getIsModified()) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
    }
  }

  // Test redo to delete the specified values
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  TensorDeleteFromAxisGpuPrimitiveT<int, float, 3> deleteFromAxis("1", "2", DeleteTable1());
  deleteFromAxis.redo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2 - 1; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), expected_tensor_values(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  gpuCheckEqual(axis_2_ptr->getNLabels(), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1)
    {
      gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i);
    }
    else
    {
      gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i + 1);
    }
  }

  // Test for the expected indices data
  gpuCheckEqual(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2 - 1);
  for (int i = 0; i < nlabels2 - 1; ++i) {
    if (i < 1) {
      gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 2);
    }
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_deleted_, 1);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_deleted_, 1);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_deleted_, 10);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_inserted_, 0);

  // Test redo to restore the deleted values
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  deleteFromAxis.undo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test for the expected table data
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test for the expected axis data
  gpuCheckEqual(axis_2_ptr->getNLabels(), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i);
  }

  // Test for the expected indices data
  gpuCheckEqual(tensorTable1_ptr->getDimensions().at(tensorTable1_ptr->getDimFromAxisName("2")), nlabels2);
  for (int i = 0; i < nlabels2; ++i) {
    if (i == 1) {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_deleted_, 1);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_deleted_, 1);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_deleted_, 10);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_labels_inserted_, 1);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_indices_inserted_, 1);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(deleteFromAxis.getUndoLog().n_data_inserted_, 10);

  gpuErrchk(cudaStreamDestroy(stream));
}

/*TensorUpdateSelectValues Tests*/
void test_constructorTensorUpdateSelectValues()
{
  TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* ptr = nullptr;
  TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* nullPointer = nullptr;
  ptr = new TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorUpdateSelectValues()
{
  TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* ptr = nullptr;
  ptr = new TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>();
  delete ptr;
}

void test_redoAndUndoTensorUpdateSelectValues()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
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

  // Set up the update values
  Eigen::Tensor<float, 3> values_new_values(nlabels1, 1, 1);
  for (int i = 0; i < nlabels1; ++i)
    values_new_values(i, 0, 0) = (float)((i + 1) * 10);
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, 1, 1 }));
  values_new.setData(values_new_values);

  // Set up the update
  TensorUpdateSelectValues<float, Eigen::GpuDevice, 3> tensorUpdate("1", SelectTable1(), std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new));

  // Reset `is_modified` to test for the new additions after the call to `setData`
  for (auto& tensor_table_map : collection_1_ptr->tables_) {
    for (auto& is_modified_map : tensor_table_map.second->getIsModified()) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
    }
  }

  // Test redo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.redo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  tensorUpdate.getValuesOld()->syncHAndDData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 1; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), values_new_values(i, j, k));
        gpuCheckEqual(tensorUpdate.getValuesOld()->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test the changed indices
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

  // Test undo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.undo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 1; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }

  // Test the changed indices
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1); // TODO: should be reverted to 0
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

  gpuErrchk(cudaStreamDestroy(stream));
}

/*TensorUpdateValues Tests*/
void test_constructorTensorUpdateValues()
{
	TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* ptr = nullptr;
	TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* nullPointer = nullptr;
	ptr = new TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>();
	gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorUpdateValues()
{
	TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>* ptr = nullptr;
	ptr = new TensorUpdateSelectValues<float, Eigen::GpuDevice, 3>();
	delete ptr;
}

void test_redoAndUndoTensorUpdateValues()
{
	cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
	TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
	tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
	tensorTable1.setAxes(device);

	Eigen::Tensor<float, 3> tensor_values1(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
	for (int i = 0; i < nlabels1; ++i) {
		for (int j = 0; j < nlabels2; ++j) {
			for (int k = 0; k < nlabels3; ++k) {
				tensor_values1(i, j, k) = i + j * nlabels1 + k * nlabels1 * nlabels2;
			}
		}
	}
	tensorTable1.setData(tensor_values1);
	std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

	// set up table 2
	TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
	tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
	tensorTable2.setAxes(device);

	Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
	for (int i = 0; i < nlabels1; ++i) {
		for (int j = 0; j < nlabels2; ++j) {
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

	// Set up the update values
	Eigen::Tensor<float, 3> values_new_values(nlabels1, 1, 1);
	for (int i = 0; i < nlabels1; ++i)
		values_new_values(i, 0, 0) = (float)((i + 1) * 10);
	TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels1, 1, 1 }));
	values_new.setData(values_new_values);

	// Set up the update
	SelectTable1 selectTable1;
	selectTable1.apply_select = false;
	TensorUpdateValues<float, Eigen::GpuDevice, 3> tensorUpdate("1", selectTable1, std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new));

	// Reset `is_modified` to test for the new additions after the call to `setData`
	for (auto& tensor_table_map : collection_1_ptr->tables_) {
		for (auto& is_modified_map : tensor_table_map.second->getIsModified()) {
			is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
		}
	}

	// Test redo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.redo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  tensorUpdate.getValuesOld()->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
	for (int i = 0; i < nlabels1; ++i) {
		gpuCheckEqual(tensorUpdate.getValuesOld()->getData()(i), tensor_values1(i, 0, 0));
		for (int j = 0; j < nlabels2; ++j) {
			for (int k = 0; k < nlabels3; ++k) {
        if (j == 0 && k == 0)
        {
          gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), values_new_values(i, j, k));
        }
        else
        {
          gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
        }
			}
		}
	}
	for (int i = 0; i < nlabels1; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
	}
	for (int i = 0; i < nlabels2; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
	}
	for (int i = 0; i < nlabels3; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 0);
    }
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
	}

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

	// Test undo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.undo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));
	for (int i = 0; i < nlabels1; ++i) {
		for (int j = 0; j < nlabels2; ++j) {
			for (int k = 0; k < nlabels3; ++k) {
				gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
			}
		}
	}
	for (int i = 0; i < nlabels1; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
	}
	for (int i = 0; i < nlabels2; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
	}
	for (int i = 0; i < nlabels3; ++i) {
		gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
		gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 0);
    }
		gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
		gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
		gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
	}

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

  gpuErrchk(cudaStreamDestroy(stream));
}

/*TensorUpdateConstant Tests*/
void test_constructorTensorUpdateConstant()
{
  TensorUpdateConstant<float, Eigen::GpuDevice>* ptr = nullptr;
  TensorUpdateConstant<float, Eigen::GpuDevice>* nullPointer = nullptr;
  ptr = new TensorUpdateConstant<float, Eigen::GpuDevice>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorUpdateConstant()
{
  TensorUpdateConstant<float, Eigen::GpuDevice>* ptr = nullptr;
  ptr = new TensorUpdateConstant<float, Eigen::GpuDevice>();
  delete ptr;
}

void test_redoAndUndoTensorUpdateConstant()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
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

  // Set up the update values
  Eigen::Tensor<float, 1> values_new_values(1);
  values_new_values.setConstant(100);
  TensorDataGpuPrimitiveT<float, 1> values_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  values_new.setData(values_new_values);
  auto values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(values_new);

  // Set up the update
  SelectTable1 selectTable1;
  selectTable1.apply_select = false;
  TensorUpdateConstant<float, Eigen::GpuDevice> tensorUpdate("1", selectTable1, values_new_ptr);

  // Reset `is_modified` to test for the new additions after the call to `setData`
  for (auto& tensor_table_map : collection_1_ptr->tables_) {
    for (auto& is_modified_map : tensor_table_map.second->getIsModified()) {
      is_modified_map.second->getData() = is_modified_map.second->getData().constant(0);
    }
  }

  // Test redo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.redo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  tensorUpdate.getValuesOld()->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorUpdate.getValuesOld()->getData()(i), tensor_values1(i, 0, 0));
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        if (j == 0 && k == 0)
        {
          gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), 100);
        }
        else
        {
          gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
        }
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 0);
    }
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

  // Test undo
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesDData(device);
    table_map.second->syncDData(device);
  }
  tensorUpdate.undo(collection_1_ptr, device);
  for (auto& table_map : collection_1_ptr->tables_) {
    table_map.second->syncAxesAndIndicesHData(device);
    table_map.second->syncHData(device);
  }
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        gpuCheckEqual(tensorTable1_ptr->getData()(i, j, k), tensor_values1(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("1")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("2")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("2")->getData()(i), 0);
    }
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("2")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable1_ptr->getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable1_ptr->getIndicesView().at("3")->getData()(i), i + 1);
    if (i == 0)
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 1);
    }
    else
    {
      gpuCheckEqual(tensorTable1_ptr->getIsModified().at("3")->getData()(i), 0);
    }
    gpuCheckEqual(tensorTable1_ptr->getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable1_ptr->getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable1_ptr->getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test for the expected redo/undo log data
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getRedoLog().n_data_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_labels_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_updated_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_indices_inserted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_deleted_, 0);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_updated_, 2);
  gpuCheckEqual(tensorUpdate.getUndoLog().n_data_inserted_, 0);

  gpuErrchk(cudaStreamDestroy(stream));
}

/*TensorAddTable Tests*/
void test_constructorTensorAddTable()
{
  TensorAddTable<TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>* ptr = nullptr;
  TensorAddTable<TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>* nullPointer = nullptr;
  ptr = new TensorAddTable<TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorAddTable()
{
  TensorAddTable<TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>* ptr = nullptr;
  ptr = new TensorAddTable<TensorTable<float, Eigen::GpuDevice, 3>, Eigen::GpuDevice>();
  delete ptr;
}

void test_redoAndUndoTensorAddTable()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      tensor_values2(i, j) = i + j * nlabels1;
    }
  }
  tensorTable2.setData(tensor_values2);
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  // Set up the collection
  TensorCollectionGpu collection_1;
  collection_1.addTensorTable(tensorTable1_ptr, "1");
  std::shared_ptr<TensorCollection<Eigen::GpuDevice>> collection_1_ptr = std::make_shared<TensorCollectionGpu>(collection_1);

  // Test Redo
  TensorAddTable<TensorTable<int, Eigen::GpuDevice, 2>, Eigen::GpuDevice> tensorAddTable(tensorTable2_ptr, "1");
  tensorAddTable.redo(collection_1_ptr, device);
  gpuCheckEqual(collection_1_ptr->getTableNames(), std::vector<std::string>({ "1", "2" }));

  // Test Undo
  tensorAddTable.undo(collection_1_ptr, device);
  gpuCheckEqual(collection_1_ptr->getTableNames(), std::vector<std::string>({ "1" }));
}

/*TensorDropTable Tests*/
void test_constructorTensorDropTable()
{
  TensorDropTable<Eigen::GpuDevice>* ptr = nullptr;
  TensorDropTable<Eigen::GpuDevice>* nullPointer = nullptr;
  ptr = new TensorDropTable<Eigen::GpuDevice>();
  gpuCheckNotEqual(ptr, nullPointer);
}

void test_destructorTensorDropTable()
{
  TensorDropTable<Eigen::GpuDevice>* ptr = nullptr;
  ptr = new TensorDropTable<Eigen::GpuDevice>();
  delete ptr;
}

void test_redoAndUndoTensorDropTable()
{
  cudaStream_t stream; gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); Eigen::GpuStreamDevice stream_device(&stream, 0); Eigen::GpuDevice device(&stream_device);

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
  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  // set up table 2
  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);

  Eigen::Tensor<int, 2> tensor_values2(Eigen::array<Eigen::Index, 2>({ nlabels1, nlabels2 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
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

  // Test redo drop table
  TensorDropTable<Eigen::GpuDevice> tensorDropTable("1");
  tensorDropTable.redo(collection_1_ptr, device);
  gpuCheckEqual(collection_1_ptr->getTableNames(), std::vector<std::string>({ "2" }));

  // Test undo drop table
  tensorDropTable.undo(collection_1_ptr, device);
  gpuCheckEqual(collection_1_ptr->getTableNames(), std::vector<std::string>({ "1", "2" }));
}

int main(int argc, char** argv)
{
  test_constructorTensorAppendToAxis();
  test_destructorTensorAppendToAxis();
  test_redoAndUndoTensorAppendToAxis();
  test_constructorTensorDeleteFromAxis();
  test_destructorTensorDeleteFromAxis();
  test_redoAndTensorDeleteFromAxis();
  test_constructorTensorUpdateSelectValues();
  test_destructorTensorUpdateSelectValues();
  test_redoAndUndoTensorUpdateSelectValues();
  test_constructorTensorUpdateValues();
  test_destructorTensorUpdateValues();
  test_redoAndUndoTensorUpdateValues();
  test_constructorTensorUpdateConstant();
  test_destructorTensorUpdateConstant();
  test_redoAndUndoTensorUpdateConstant();
  test_constructorTensorAddTable();
  test_destructorTensorAddTable();
  test_redoAndUndoTensorAddTable();
  test_constructorTensorDropTable();
  test_destructorTensorDropTable();
  test_redoAndUndoTensorDropTable();
  return 0;
}
#endif