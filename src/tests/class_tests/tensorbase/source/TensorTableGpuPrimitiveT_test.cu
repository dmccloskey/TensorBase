/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorTableGpuPrimitiveT.h>
#include <string>

using namespace TensorBase;
using namespace std;

void test_constructorGpu()
{
  TensorTableGpuPrimitiveT<float, 3>* ptr = nullptr;
  TensorTableGpuPrimitiveT<float, 3>* nullPointer = nullptr;
  ptr = new TensorTableGpuPrimitiveT<float, 3>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorTableGpuPrimitiveT<float, 3>* ptr = nullptr;
  ptr = new TensorTableGpuPrimitiveT<float, 3>();
  delete ptr;
}

void test_constructorNameAndAxesGpu()
{
  TensorTableGpuPrimitiveT<float, 3> tensorTable("1");

  assert(tensorTable.getId() == -1);
  assert(tensorTable.getName() == "1");
  assert(tensorTable.getDir() == "");

  TensorTableGpuPrimitiveT<float, 3> tensorTable2("1", "dir");

  assert(tensorTable2.getId() == -1);
  assert(tensorTable2.getName() == "1");
  assert(tensorTable2.getDir() == "dir");
}

void test_gettersAndSettersGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  TensorTableGpuPrimitiveT<float, 3> tensorTable;
  // Check defaults
  assert(tensorTable.getId() == -1);
  assert(tensorTable.getName() == "");
  assert(tensorTable.getAxes().size() == 0);
  assert(tensorTable.getDir() == "");
  assert(tensorTable.getTensorSize() == 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");
  std::map<std::string, int> shard_span = {
    {"1", 2}, {"2", 2}, {"3", 3} };
  tensorTable.setShardSpans(shard_span);
  tensorTable.setDir("dir");

  assert(tensorTable.getId() == 1);
  assert(tensorTable.getName() == "1");
  assert(tensorTable.getShardSpans() == shard_span);
  assert(tensorTable.getDir() == "dir");

  // SetAxes associated getters/setters
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  //Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  //labels1.setConstant("x-axis");
  //labels2.setConstant("y-axis");
  //labels3.setConstant("z-axis");
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setShardSpans(std::map<std::string, int>()); // reset the shard spans to zero
  tensorTable.setAxes(device);

  // Test expected axes values
  assert(tensorTable.getAxes().at("1")->getName() == "1");
  //assert(tensorTable.getAxes().at("1")->getLabels()(0, 0) == 1);
  ////assert(tensorTable.getAxes().at("1")->getLabels()(0,0) == "x-axis");
  assert(tensorTable.getAxes().at("1")->getNLabels() == nlabels1);
  assert(tensorTable.getAxes().at("1")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("1")->getDimensions()(0) == "x");
  assert(tensorTable.getIndices().at("1")->getData()(0) == 1);
  assert(tensorTable.getIndices().at("1")->getData()(nlabels1 - 1) == nlabels1);
  assert(tensorTable.getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("1")->getData()(nlabels1 - 1) == nlabels1);
  assert(tensorTable.getIsModified().at("1")->getData()(0) == 0);
  assert(tensorTable.getNotInMemory().at("1")->getData()(0) == 1);
  assert(tensorTable.getShardId().at("1")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("1")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("1")->getData()(nlabels1 - 1) == nlabels1);

  assert(tensorTable.getAxes().at("2")->getName() == "2");
  //assert(tensorTable.getAxes().at("2")->getLabels()(0, 0) == 2);
  ////assert(tensorTable.getAxes().at("2")->getLabels()(0, 0) == "y-axis");
  assert(tensorTable.getAxes().at("2")->getNLabels() == nlabels2);
  assert(tensorTable.getAxes().at("2")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("2")->getDimensions()(0) == "y");
  assert(tensorTable.getIndices().at("2")->getData()(0) == 1);
  assert(tensorTable.getIndices().at("2")->getData()(nlabels2 - 1) == nlabels2);
  assert(tensorTable.getIndicesView().at("2")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("2")->getData()(nlabels2 - 1) == nlabels2);
  assert(tensorTable.getIsModified().at("2")->getData()(0) == 0);
  assert(tensorTable.getNotInMemory().at("2")->getData()(0) == 1);
  assert(tensorTable.getShardId().at("2")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("2")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("2")->getData()(nlabels2 - 1) == nlabels2);

  assert(tensorTable.getAxes().at("3")->getName() == "3");
  //assert(tensorTable.getAxes().at("3")->getLabels()(0, 0) == 3);
  ////assert(tensorTable.getAxes().at("3")->getLabels()(0, 0) == "z-axis");
  assert(tensorTable.getAxes().at("3")->getNLabels() == nlabels3);
  assert(tensorTable.getAxes().at("3")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("3")->getDimensions()(0) == "z");
  assert(tensorTable.getIndices().at("3")->getData()(0) == 1);
  assert(tensorTable.getIndices().at("3")->getData()(nlabels3 - 1) == nlabels3);
  assert(tensorTable.getIndicesView().at("3")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("3")->getData()(nlabels3 - 1) == nlabels3);
  assert(tensorTable.getIsModified().at("3")->getData()(0) == 0);
  assert(tensorTable.getNotInMemory().at("3")->getData()(0) == 1);
  assert(tensorTable.getShardId().at("3")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("3")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("3")->getData()(nlabels3 - 1) == nlabels3);

  // Test expected axis to dims mapping
  assert(tensorTable.getDimFromAxisName("1") == 0);
  assert(tensorTable.getDimFromAxisName("2") == 1);
  assert(tensorTable.getDimFromAxisName("3") == 2);

  // Test expected tensor shard spans
  assert(tensorTable.getShardSpans().at("1") == 2);
  assert(tensorTable.getShardSpans().at("2") == 3);
  assert(tensorTable.getShardSpans().at("3") == 5);

  // Test expected tensor dimensions
  assert(tensorTable.getDimensions().at(0) == 2);
  assert(tensorTable.getDimensions().at(1) == 3);
  assert(tensorTable.getDimensions().at(2) == 5);
  assert(tensorTable.getTensorSize() == 0);

  // Test expected tensor data values
  assert(tensorTable.getDataDimensions().at(0) == 2);
  assert(tensorTable.getDataDimensions().at(1) == 3);
  assert(tensorTable.getDataDimensions().at(2) == 5);
  size_t test = 2 * 3 * 5 * sizeof(float);
  assert(tensorTable.getDataTensorBytes() == test);

  // Test setting the data
  Eigen::Tensor<float, 3> tensor_data(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_data(i, j, k) = i + j + k;
      }
    }
  }
  tensorTable.setData(tensor_data);
  for (int i = 0; i < nlabels1; ++i) {
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
  }
  for (int i = 0; i < nlabels2; ++i) {
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
  }
  for (int i = 0; i < nlabels3; ++i) {
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
  }

  // Test setting the data
  tensorTable.setData();
  for (int i = 0; i < nlabels1; ++i) {
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 1);
  }
  for (int i = 0; i < nlabels2; ++i) {
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 1);
  }

  // Test clear
  tensorTable.clear();
  assert(tensorTable.getAxes().size() == 0);
  assert(tensorTable.getIndices().size() == 0);
  assert(tensorTable.getIndicesView().size() == 0);
  assert(tensorTable.getIsModified().size() == 0);
  assert(tensorTable.getNotInMemory().size() == 0);
  assert(tensorTable.getShardId().size() == 0);
  assert(tensorTable.getShardIndices().size() == 0);
  assert(tensorTable.getDimensions().at(0) == 0);
  assert(tensorTable.getDimensions().at(1) == 0);
  assert(tensorTable.getDimensions().at(2) == 0);
  assert(tensorTable.getShardSpans().size() == 0);
}

void test_reShardIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test the default shard span
  assert(tensorTable.getShardSpans().at("1") == 4);
  assert(tensorTable.getShardSpans().at("2") == 4);
  assert(tensorTable.getShardSpans().at("3") == 4);

  // Reset the shard span
  int shard_span = 3;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  assert(tensorTable.getShardSpans().at("1") == 3);
  assert(tensorTable.getShardSpans().at("2") == 3);
  assert(tensorTable.getShardSpans().at("3") == 3);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.reShardIndices(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i < shard_span) {
      assert(tensorTable.getShardId().at("1")->getData()(i) == 1);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 1);
      assert(tensorTable.getShardId().at("2")->getData()(i) == 1);
      assert(tensorTable.getShardIndices().at("2")->getData()(i) == i + 1);
      assert(tensorTable.getShardId().at("3")->getData()(i) == 1);
      assert(tensorTable.getShardIndices().at("3")->getData()(i) == i + 1);
    }
    else {
      assert(tensorTable.getShardId().at("1")->getData()(i) == 2);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i - shard_span + 1);
      assert(tensorTable.getShardId().at("2")->getData()(i) == 2);
      assert(tensorTable.getShardIndices().at("2")->getData()(i) == i - shard_span + 1);
      assert(tensorTable.getShardId().at("3")->getData()(i) == 2);
      assert(tensorTable.getShardIndices().at("3")->getData()(i) == i - shard_span + 1);
    }
  }
}

void test_tensorDataWrappersGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // Set up the tensor table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test indices wrappers
  std::map<std::string, std::pair<bool, bool>> statuses;
  tensorTable.setIndicesDataStatus(false, false);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setIndicesDataStatus(true, false);
  tensorTable.syncIndicesHAndDData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setIndicesViewDataStatus(false, false);
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewHAndDData(device);
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test in memory wrappers
  tensorTable.setIsModifiedDataStatus(false, false);
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setNotInMemoryDataStatus(false, false);
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setShardIdDataStatus(false, false);
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setShardIdDataStatus(true, false);
  tensorTable.syncShardIdHAndDData(device);
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test indices view wrappers
  tensorTable.setShardIndicesDataStatus(false, false);
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(!index_status.second.second);
  }
  tensorTable.setShardIndicesDataStatus(true, false);
  tensorTable.syncShardIndicesHAndDData(device);
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }

  // Test axes wrappers
  tensorTable.setAxesDataStatus(false, false);
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    assert(!axis_status.second.first);
    assert(!axis_status.second.second);
  }
  tensorTable.setAxesDataStatus(true, false);
  tensorTable.syncAxesHAndDData(device);
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    assert(!axis_status.second.first);
    assert(axis_status.second.second);
  }

  // Test bulk host/device transfers
  tensorTable.setIndicesDataStatus(true, false);
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.setShardIdDataStatus(true, false);
  tensorTable.setShardIndicesDataStatus(true, false);
  tensorTable.setAxesDataStatus(true, false);

  // Test to Device
  tensorTable.syncAxesAndIndicesDData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(!index_status.second.first);
    assert(index_status.second.second);
  }
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    assert(!axis_status.second.first);
    assert(axis_status.second.second);
  }

  // Test to host
  tensorTable.setIndicesDataStatus(false, true);
  tensorTable.setIndicesViewDataStatus(false, true);
  tensorTable.setIsModifiedDataStatus(false, true);
  tensorTable.setNotInMemoryDataStatus(false, true);
  tensorTable.setShardIdDataStatus(false, true);
  tensorTable.setShardIndicesDataStatus(false, true);
  tensorTable.setAxesDataStatus(false, true);
  tensorTable.syncAxesAndIndicesHData(device);
  statuses = tensorTable.getIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getIndicesViewDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getIsModifiedDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getNotInMemoryDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getShardIdDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getShardIndicesDataStatus();
  for (auto& index_status : statuses) {
    assert(index_status.second.first);
    assert(!index_status.second.second);
  }
  statuses = tensorTable.getAxesDataStatus();
  for (auto& axis_status : statuses) {
    assert(axis_status.second.first);
    assert(!axis_status.second.second);
  }
}

void test_zeroIndicesViewAndResetIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable indices
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);

  // test null
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
  }

  // test zero
  tensorTable.zeroIndicesView("1", device);
  tensorTable.getIndicesView().at("1")->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == 0);
  }

  // test reset
  tensorTable.getIndicesView().at("1")->setDataStatus(false, true);
  tensorTable.resetIndicesView("1", device);
  tensorTable.getIndicesView().at("1")->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectIndicesView1Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(nlabels / 2);
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0) {
      select_labels_values(iter) = i;
      ++iter;
    }
  }
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ nlabels / 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // test the updated view
  select_labels_ptr->syncHAndDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  tensorTable.syncIndicesViewHAndDData(device);
  select_labels_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0)
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == 0);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectIndicesView2Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(2), dimensions2(1), dimensions3(1);
  dimensions1(0) = "a"; dimensions1(1) = "b";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(2, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3}, {4, 5, 6, 7} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(2, 2);
  select_labels_values.setValues({ {0, 2}, {4, 6} });
  TensorDataGpuPrimitiveT<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 2, 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(select_labels);

  // test the updated view
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  select_labels_ptr->syncHAndDData(device);
  tensorTable.selectIndicesView("1", select_labels_ptr, device);
  tensorTable.getIndicesView().at("1")->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0)
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == 0);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_broadcastSelectIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable indices
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // setup the indices test
  Eigen::Tensor<int, 3> indices_test(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = i + 1;
      }
    }
  }

  // test the broadcast indices values
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_view_bcast;
  tensorTable.broadcastSelectIndicesView(indices_view_bcast, "1", device);
  indices_view_bcast->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        //std::cout << "Test broadcastSelectIndicesView i,j,k :" << i << "," << j << "," << k << "; Labels: " << indices_view_bcast->getData()(i, j, k) << "; Expected: " << indices_test(i, j, k) << std::endl;
        assert(indices_view_bcast->getData()(i, j, k) == indices_test(i, j, k));
      }
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_extractTensorDataGpuPrimitiveT()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data, selection indices, and test selection data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> tensor_test(Eigen::array<Eigen::Index, 3>({ nlabels / 2, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        float value = i * nlabels + j * nlabels + k;
        tensor_values(i, j, k) = value;
        if (i % 2 == 0) {
          indices_values(i, j, k) = 1;
          tensor_test(i / 2, j, k) = value;
        }
        else {
          indices_values(i, j, k) = 0;
        }
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataGpuPrimitiveT<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_values);
  auto indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices_select);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // test
  indices_select_ptr->syncHAndDData(device);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select;
  tensorTable.reduceTensorDataToSelectIndices(indices_select_ptr,
    tensor_select, "1", nlabels / 2, device);
  tensor_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels / 2; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(tensor_select->getData()(i, j, k) == tensor_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectTensorIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 2;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor select and values select data
  Eigen::Tensor<float, 3> tensor_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 1> values_select_values(Eigen::array<Eigen::Index, 1>({ nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    values_select_values(i) = 2.0;
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_select_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  TensorDataGpuPrimitiveT<float, 3> tensor_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  tensor_select.setData(tensor_select_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(tensor_select);
  TensorDataGpuPrimitiveT<float, 1> values_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  values_select.setData(values_select_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> values_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(values_select);

  // Sync the data
  tensor_select_ptr->syncHAndDData(device);
  values_select_ptr->syncHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // test inequality
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::NOT_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          assert(indices_select->getData()(i, j, k) == 0);
        else
          assert(indices_select->getData()(i, j, k) == 1);
      }
    }
  }

  // test equality
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          assert(indices_select->getData()(i, j, k) == 1);
        else
          assert(indices_select->getData()(i, j, k) == 0);
      }
    }
  }

  // test less than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) < 2.0)
          assert(indices_select->getData()(i, j, k) == 1);
        else
          assert(indices_select->getData()(i, j, k) == 0);
      }
    }
  }

  // test less than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) <= 2.0)
          assert(indices_select->getData()(i, j, k) == 1);
        else
          assert(indices_select->getData()(i, j, k) == 0);
      }
    }
  }

  // test greater than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) > 2.0)
          assert(indices_select->getData()(i, j, k) == 1);
        else
          assert(indices_select->getData()(i, j, k) == 0);
      }
    }
  }

  // test greater than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) >= 2.0)
          assert(indices_select->getData()(i, j, k) == 1);
        else
          assert(indices_select->getData()(i, j, k) == 0);
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_applyIndicesSelectToIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // setup the indices select
  Eigen::Tensor<int, 3> indices_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == j && j == k && k == i
          && i < nlabels - 1 && j < nlabels - 1 && k < nlabels - 1) // the first 2 diagonal elements
          indices_select_values(i, j, k) = 1;
        else
          indices_select_values(i, j, k) = 0;
      }
    }
  }
  TensorDataGpuPrimitiveT<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices_select);
  indices_select_ptr->syncHAndDData(device);

  // test using the second indices view  
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.getIndicesView().at("2")->getData()(nlabels - 1) = 0;
  tensorTable.syncIndicesViewHAndDData(device);

  // test for OR within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::OR, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i == nlabels - 1)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
  }

  // reset and modify the indices view
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  tensorTable.syncIndicesViewHAndDData(device);

  // test for AND within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::OR, device);  
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    //std::cout << "Test applyIndicesSelectToIndicesView i " << i << "; Indices View: " << tensorTable.getIndicesView().at("2")->getData()(i) << std::endl;
    if (i == 0)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
  }

  // Reset and modify the indices view
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.resetIndicesView("2", device);  
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  tensorTable.syncIndicesViewHAndDData(device);

  // test for OR within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i != 0 && i < nlabels - 1)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
  }

  // Reset the indices view
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.resetIndicesView("2", device);

  // and update the indices_select_ptr
  indices_select_ptr->setDataStatus(true, false);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == j && j == k && k == i
          && i < nlabels - 1 && j < nlabels - 1 && k < nlabels - 1) // the first 2 diagonal elements
          indices_select_ptr->getData()(i, j, k) = 1;
        else if (j == 0)
          indices_select_ptr->getData()(i, j, k) = 1; // all elements along the first index of the selection dim
        else
          indices_select_ptr->getData()(i, j, k) = 0;
      }
    }
  }
  indices_select_ptr->syncHAndDData(device);

  // test for AND within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
  }

  // TODO: lacking code coverage for the case of TDim = 2
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_whereIndicesViewDataGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 4;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3} });
  labels2.setValues({ {0, 1, 2, 3} });
  labels3.setValues({ {0, 1, 2, 3} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(2);
  select_labels_values(0) = 0; select_labels_values(1) = 2;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);

  // set up the selection values
  Eigen::Tensor<float, 1> select_values_values(2);
  select_values_values(0) = 9; select_values_values(1) = 9;
  TensorDataGpuPrimitiveT<float, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<float, 1>> select_values_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(select_values);
  select_values_ptr->syncHAndDData(device);

  // test
  tensorTable.whereIndicesView("1", 0, select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  std::cout << "test_whereIndicesViewDataGpu Failing:" << std::endl;
  std::cout << "tensorTable.getIndicesView().at(2)->getData()\n" << tensorTable.getIndicesView().at("2")->getData() << std::endl;
  std::cout << "tensorTable.getIndicesView().at(3)->getData()\n" << tensorTable.getIndicesView().at("3")->getData() << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1); // Unchanged

    //// indices view 2
    //if (i == 2) // FIXME: i==0?
    //  assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    //else
    //  assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);

    //// indices view 3
    //if (i == 1) // FIXME: i==3?
    //  assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    //else
    //  assert(tensorTable.getIndicesView().at("3")->getData()(i) == 0);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // test
  tensorTable.whereIndicesView("1", 0, select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  std::cout << "test_whereIndicesViewDataGpu Failing:" << std::endl;
  std::cout << "tensorTable.getIndicesView().at(2)->getData()\n" << tensorTable.getIndicesView().at("2")->getData() << std::endl;
  std::cout << "tensorTable.getIndicesView().at(3)->getData()\n" << tensorTable.getIndicesView().at("3")->getData() << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1); // Unchanged

    //// indices view 2
    //if (i == 2) // FIXME: i==0?
    //  assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    //else
    //  assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);

    //// indices view 3
    //if (i == 1) // FIXME: i==3?
    //  assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    //else
    //  assert(tensorTable.getIndicesView().at("3")->getData()(i) == 0);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sliceTensorForSortGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // test sliceTensorForSort for axis 2
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> tensor_sort;
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "2", device); 
  tensor_sort->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<float> tensor_slice_2_test = { 9, 12, 15 };
  for (int i = 0; i < nlabels; ++i) {
    assert(tensor_sort->getData()(i) == tensor_slice_2_test.at(i), 1e-3);
  }

  // test sliceTensorForSort for axis 2
  tensor_sort.reset();
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "3", device);
  tensor_sort->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  std::vector<float> tensor_slice_3_test = { 9, 10, 11 };
  for (int i = 0; i < nlabels; ++i) {
    assert(tensor_sort->getData()(i) == tensor_slice_3_test.at(i), 1e-3);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortIndicesViewData1Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 1;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);

  // test sort ASC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::ASC, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
  }

  // test sort DESC
  tensorTable.setIndicesViewDataStatus(false, true);
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == nlabels - i);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == nlabels - i);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortIndicesViewData2Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(1,1);
  select_labels_values(0,0) = 1;
  TensorDataGpuPrimitiveT<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 1,1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(select_labels);
  select_labels_ptr->syncHAndDData(device);

  // test sort ASC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::ASC, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
  }

  // test sort DESC
  tensorTable.setIndicesViewDataStatus(false, true);
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::DESC, device);
  tensorTable.syncIndicesViewHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == nlabels - i);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == nlabels - i);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeSelectIndicesFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // Test null
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_select->getData()(i, j, k) == 1);
      }
    }
  }

  // make the expected indices tensor
  Eigen::Tensor<int, 3> indices_select_test(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == 1)
          indices_select_test(i, j, k) = 1;
        else
          indices_select_test(i, j, k) = 0;
      }
    }
  }

  // select
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test selected
  indices_select.reset();
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  indices_select->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_select->getData()(i, j, k) == indices_select_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_getSelectTensorDataFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // select label 1 from axis 1
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // make the expected dimensions
  Eigen::array<Eigen::Index, 3> select_dimensions = { 1, 3, 3 };

  // make the indices_select
  Eigen::Tensor<float, 3> tensor_select_test(select_dimensions);
  Eigen::Tensor<int, 3> indices_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == 1) {
          indices_select_values(i, j, k) = 1;
          tensor_select_test(0, j, k) = float(iter);
        }
        else {
          indices_select_values(i, j, k) = 0;
        }
        ++iter;
      }
    }
  }
  TensorDataGpuPrimitiveT<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);
  std::shared_ptr<TensorDataGpuPrimitiveT<int, 3>> indices_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 3>>(indices_select);
  indices_select_ptr->syncHAndDData(device);

  // test for the selected data
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select_ptr;
  tensorTable.getSelectTensorDataFromIndicesView(tensor_select_ptr, indices_select_ptr, device);
  tensor_select_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensor_select_ptr->getDimensions() == select_dimensions);
  for (int j = 0; j < nlabels; ++j) {
    for (int k = 0; k < nlabels; ++k) {
      assert(tensor_select_ptr->getData()(0, j, k) == tensor_select_test(0, j, k), 1e-3);
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectTensorDataGpuPrimitiveT()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // select label 1 from axis 1
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test `selectTensorData`
  tensorTable.selectTensorData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test expected axes values
  assert(tensorTable.getAxes().at("1")->getName() == "1");
  assert(tensorTable.getAxes().at("1")->getNLabels() == 1);
  assert(tensorTable.getAxes().at("1")->getDimensions()(0) == "x");
  assert(tensorTable.getIndices().at("1")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable.getIsModified().at("1")->getData()(0) == 1);
  assert(tensorTable.getNotInMemory().at("1")->getData()(0) == 0);
  assert(tensorTable.getShardId().at("1")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("1")->getData()(0) == 1);

  assert(tensorTable.getAxes().at("2")->getName() == "2");
  assert(tensorTable.getAxes().at("2")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("2")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("2")->getDimensions()(0) == "y");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getShardId().at("2")->getData()(i) == 1);
    assert(tensorTable.getShardIndices().at("2")->getData()(i) == i + 1);
  }

  assert(tensorTable.getAxes().at("3")->getName() == "3");
  assert(tensorTable.getAxes().at("3")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("3")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("3")->getDimensions()(0) == "z");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getShardId().at("3")->getData()(i) == 1);
    assert(tensorTable.getShardIndices().at("3")->getData()(i) == i + 1);
  }

  // Test expected axis to dims mapping
  assert(tensorTable.getDimFromAxisName("1") == 0);
  assert(tensorTable.getDimFromAxisName("2") == 1);
  assert(tensorTable.getDimFromAxisName("3") == 2);

  // Test expected tensor dimensions
  assert(tensorTable.getDimensions().at(0) == 1);
  assert(tensorTable.getDimensions().at(1) == 3);
  assert(tensorTable.getDimensions().at(2) == 3);

  // Test expected tensor data values
  assert(tensorTable.getDataDimensions().at(0) == 1);
  assert(tensorTable.getDataDimensions().at(1) == 3);
  assert(tensorTable.getDataDimensions().at(2) == 3);
  size_t test = 1 * 3 * 3 * sizeof(float);
  assert(tensorTable.getDataTensorBytes() == test);

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test selectTensorData
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  tensorTable.selectTensorData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test expected axes values
  assert(tensorTable.getAxes().at("1")->getName() == "1");
  assert(tensorTable.getAxes().at("1")->getNLabels() == 1);
  assert(tensorTable.getAxes().at("1")->getDimensions()(0) == "x");
  assert(tensorTable.getIndices().at("1")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable.getIsModified().at("1")->getData()(0) == 1);
  assert(tensorTable.getNotInMemory().at("1")->getData()(0) == 0);
  assert(tensorTable.getShardId().at("1")->getData()(0) == 1);
  assert(tensorTable.getShardIndices().at("1")->getData()(0) == 1);

  assert(tensorTable.getAxes().at("2")->getName() == "2");
  assert(tensorTable.getAxes().at("2")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("2")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("2")->getDimensions()(0) == "y");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getShardId().at("2")->getData()(i) == 1);
    assert(tensorTable.getShardIndices().at("2")->getData()(i) == i + 1);
  }

  assert(tensorTable.getAxes().at("3")->getName() == "3");
  assert(tensorTable.getAxes().at("3")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("3")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("3")->getDimensions()(0) == "z");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getShardId().at("3")->getData()(i) == 1);
    assert(tensorTable.getShardIndices().at("3")->getData()(i) == i + 1);
  }

  // Test expected axis to dims mapping
  assert(tensorTable.getDimFromAxisName("1") == 0);
  assert(tensorTable.getDimFromAxisName("2") == 1);
  assert(tensorTable.getDimFromAxisName("3") == 2);

  // Test expected tensor dimensions
  assert(tensorTable.getDimensions().at(0) == 1);
  assert(tensorTable.getDimensions().at(1) == 3);
  assert(tensorTable.getDimensions().at(2) == 3);

  // Test expected tensor data values
  assert(tensorTable.getDataDimensions().at(0) == 1);
  assert(tensorTable.getDataDimensions().at(1) == 3);
  assert(tensorTable.getDataDimensions().at(2) == 3);
  assert(tensorTable.getDataTensorBytes() == test);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeSortIndicesViewFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // make the expected tensor indices
  Eigen::Tensor<int, 3> indices_test(nlabels, nlabels, nlabels);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = i + j * nlabels + k * nlabels*nlabels + 1;
      }
    }
  }

  // Test for the sort indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_sort_ptr;
  tensorTable.makeSortIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_sort_ptr, device);
  indices_sort_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_sort_ptr->getData()(i, j, k) == indices_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortTensorDataGpuPrimitiveT()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 0;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncHAndDData(device);

  // make the expected sorted tensor
  float sorted_data[] = { 24, 25, 26, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2 };
  Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor_sorted_values(sorted_data, Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  //std::cout << "test_sortTensorDataGpuPrimitiveT Expected:" << std::endl;
  //std::cout << "axis_1_ptr->getLabels()\n" << axis_1_ptr->getLabels() << std::endl;
  //std::cout << "axis_2_ptr->getLabels()\n" << axis_2_ptr->getLabels() << std::endl;
  //std::cout << "axis_3_ptr->getLabels()\n" << axis_3_ptr->getLabels() << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    assert(axis_1_ptr->getLabels()(0, i) == i);
    assert(axis_2_ptr->getLabels()(0, i) == nlabels - i - 1);
    assert(axis_3_ptr->getLabels()(0, i) == nlabels - i - 1);
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()(i, j, k) == tensor_sorted_values(i, j, k));
      }
    }
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    assert(axis_1_ptr->getLabels()(0, i) == i);
    assert(axis_2_ptr->getLabels()(0, i) == nlabels - i - 1);
    assert(axis_3_ptr->getLabels()(0, i) == nlabels - i - 1);
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()(i, j, k) == tensor_sorted_values(i, j, k));
      }
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_updateSelectTensorDataValues1Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        update_values(i, j, k) = 100;
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);
  
  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_new_ptr->syncHAndDData(device);

  // Test update
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_old_ptr;
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  values_old_ptr->syncHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i, j, k) == float(iter));
        assert(tensorTable.getData()(i, j, k) == 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test update
  values_old_ptr.reset();
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  values_old_ptr->syncHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i, j, k) == float(iter));
        assert(tensorTable.getData()(i, j, k) == 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_updateSelectTensorDataValues2Gpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        update_values(i, j, k) = 100;
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_new_ptr->syncHAndDData(device);

  // Test update
  TensorDataGpuPrimitiveT<float, 3> values_old(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_old.setData();
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_old_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_old);
  values_old_ptr->syncHAndDData(device);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  values_old_ptr->syncHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i, j, k) == float(iter));
        assert(tensorTable.getData()(i, j, k) == 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test update
  values_old_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_old);
  values_old_ptr->syncHAndDData(device);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  values_old_ptr->syncHAndDData(device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i, j, k) == float(iter));
        assert(tensorTable.getData()(i, j, k) == 100);
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_updateTensorDataValuesGpu()
{
	// setup the table
	TensorTableGpuPrimitiveT<float, 3> tensorTable;

	// Initialize the device
	cudaStream_t stream;
	assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
	Eigen::GpuStreamDevice stream_device(&stream, 0);
	Eigen::GpuDevice device(&stream_device);

	// setup the axes
	Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
	dimensions1(0) = "x";
	dimensions2(0) = "y";
	dimensions3(0) = "z";
	int nlabels = 3;
	Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
	labels1.setValues({ {0, 1, 2} });
	labels2.setValues({ {0, 1, 2} });
	labels3.setValues({ {0, 1, 2} });
	auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
	auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
	auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
	tensorTable.addTensorAxis(axis_1_ptr);
	tensorTable.addTensorAxis(axis_2_ptr);
	tensorTable.addTensorAxis(axis_3_ptr);
	tensorTable.setAxes(device);

	// setup the tensor data and the update values
	Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	int iter = 0;
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				tensor_values(i, j, k) = float(iter);
				update_values(i, j, k) = 100;
				++iter;
			}
		}
	}
	tensorTable.setData(tensor_values);
	TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	values_new.setData(update_values);
	std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);

	// sync the tensorTable
	tensorTable.syncIndicesHAndDData(device);
	tensorTable.syncIndicesViewHAndDData(device);
	tensorTable.syncNotInMemoryHAndDData(device);
	tensorTable.syncIsModifiedHAndDData(device);
	tensorTable.syncShardIdHAndDData(device);
	tensorTable.syncShardIndicesHAndDData(device);
	tensorTable.syncAxesHAndDData(device);
	tensorTable.syncHAndDData(device);
	values_new_ptr->syncHAndDData(device);

	// Test update
	std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 2>> values_old_ptr;
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	values_old_ptr->syncHAndDData(device);
	tensorTable.syncIndicesHAndDData(device);
	tensorTable.syncIndicesViewHAndDData(device);
	tensorTable.syncNotInMemoryHAndDData(device);
	tensorTable.syncIsModifiedHAndDData(device);
	tensorTable.syncShardIdHAndDData(device);
	tensorTable.syncShardIndicesHAndDData(device);
	tensorTable.syncAxesHAndDData(device);
	tensorTable.syncHAndDData(device);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				assert(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels) == tensor_values(i, j, k));
				assert(tensorTable.getData()(i, j, k) == 100);
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
		assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
		assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
		assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
		assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
		assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
	}

	// Write the original data to disk, clear the data, and repeat the tests
	tensorTable.clear();
	axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
	axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
	axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
	tensorTable.addTensorAxis(axis_1_ptr);
	tensorTable.addTensorAxis(axis_2_ptr);
	tensorTable.addTensorAxis(axis_3_ptr);
	tensorTable.setAxes(device);
	tensorTable.setData(tensor_values);
	tensorTable.syncIndicesHAndDData(device);
	tensorTable.syncIndicesViewHAndDData(device);
	tensorTable.syncNotInMemoryHAndDData(device);
	tensorTable.syncIsModifiedHAndDData(device);
	tensorTable.syncShardIdHAndDData(device);
	tensorTable.syncShardIndicesHAndDData(device);
	tensorTable.syncAxesHAndDData(device);
	tensorTable.syncHAndDData(device);
	tensorTable.storeTensorTableBinary("", device);
	tensorTable.setData();
	tensorTable.setNotInMemoryDataStatus(true, false);
	tensorTable.syncNotInMemoryHAndDData(device);
	tensorTable.setIsModifiedDataStatus(true, false);
	tensorTable.syncIsModifiedHAndDData(device);
	tensorTable.syncHAndDData(device);

	// Test update
	values_old_ptr.reset();
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	values_old_ptr->syncHAndDData(device);
	tensorTable.syncIndicesHAndDData(device);
	tensorTable.syncIndicesViewHAndDData(device);
	tensorTable.syncNotInMemoryHAndDData(device);
	tensorTable.syncIsModifiedHAndDData(device);
	tensorTable.syncShardIdHAndDData(device);
	tensorTable.syncShardIndicesHAndDData(device);
	tensorTable.syncAxesHAndDData(device);
	tensorTable.syncHAndDData(device);
	assert(cudaStreamSynchronize(stream) == cudaSuccess);
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				assert(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels) == tensor_values(i, j, k));
				assert(tensorTable.getData()(i, j, k) == 100);
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
		assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
		assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
		assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
		assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
		assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
	}

	assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeAppendIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", 1, 0));
  axis_3_ptr->setDimensions(dimensions3);
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // test the making the append indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_ptr;
  tensorTable.makeAppendIndices("1", nlabels, indices_ptr, device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    assert(indices_ptr->getData()(i) == nlabels + i + 1);
  }

  // test the making the append indices on a zero axis
  indices_ptr.reset();
  tensorTable.makeAppendIndices("3", 1, indices_ptr, device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < 1; ++i) {
    assert(indices_ptr->getData()(i) == i + 1);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendToIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the new indices
  Eigen::Tensor<int, 1> indices_new_values(nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    indices_new_values(i) = nlabels + i + 1;
  }
  TensorDataGpuPrimitiveT<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ nlabels - 1 }));
  indices_new.setData(indices_new_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_new);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  indices_new_ptr->syncHAndDData(device);

  // test appendToIndices
  tensorTable.appendToIndices("1", indices_new_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")) == nlabels + nlabels - 1);
  for (int i = 0; i < nlabels + nlabels - 1; ++i) {
    assert(tensorTable.getIndices().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getShardId().at("1")->getData()(i) == 1);
    if (i < nlabels) {
      assert(tensorTable.getIsModified().at("1")->getData()(i) == 0);
      assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 1);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 1);
    }
    else {
      assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
      assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == 0);
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_appendToAxisGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = float(iter);
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new tensor data
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j) = i;
    }
  }
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);

  // setup the new axis labels
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_values(0, 0) = 3;
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // setup the new indices
  TensorDataGpuPrimitiveT<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_new.setData();
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_new);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  labels_new_ptr->syncHAndDData(device);
  values_new_ptr->syncHAndDData(device);
  indices_new_ptr->syncHAndDData(device);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  indices_new_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    assert(axis_1_ptr->getLabels()(0, i) == labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(tensorTable.getData()(i, j, k) == tensor_values(i, j, k));
      }
    }
  }
  assert(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      assert(tensorTable.getData()(nlabels, i, j) == update_values(0, i, j));
    }
  }
  assert(indices_new_ptr->getData()(0) == nlabels + 1);

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);
  indices_new_ptr->setDataStatus(false, true);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  indices_new_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    assert(axis_1_ptr->getLabels()(0, i) == labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(tensorTable.getData()(i, j, k) == tensor_values(i, j, k));
      }
    }
  }
  assert(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      assert(tensorTable.getData()(nlabels, i, j) == update_values(0, i, j));
    }
  }
  assert(indices_new_ptr->getData()(0) == nlabels + 1);

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeIndicesViewSelectFromIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  indices_to_select_values.setValues({ 1, 2 });
  TensorDataGpuPrimitiveT<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 2 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_to_select);
  
  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  indices_to_select_ptr->syncHAndDData(device);

  // test makeIndicesViewSelectFromIndices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_select_ptr;
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, true, device);
  indices_select_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i > 1)
      assert(indices_select_ptr->getData()(i) == 1);
    else
      assert(indices_select_ptr->getData()(i) == 0);
  }
  indices_select_ptr.reset();
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, false, device);
  indices_select_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    if (i <= 1)
      assert(indices_select_ptr->getData()(i) == 1);
    else
      assert(indices_select_ptr->getData()(i) == 0);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_deleteFromIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select_values.setValues({ 2 });
  TensorDataGpuPrimitiveT<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_to_select);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  indices_to_select_ptr->syncHAndDData(device);

  // test deleteFromIndices
  tensorTable.deleteFromIndices("1", indices_to_select_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")) == nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      assert(tensorTable.getIndices().at("1")->getData()(i) == i + 1);
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 1);
    }
    else {
      assert(tensorTable.getIndices().at("1")->getData()(i) == i + 2);
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 2);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 2);
    }
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 1);
    assert(tensorTable.getShardId().at("1")->getData()(i) == 1);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeSelectIndicesFromIndicesGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0) indices_to_select_values(i) = i + 1;
    else indices_to_select_values(i) = 0;
  }
  TensorDataGpuPrimitiveT<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_to_select);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  indices_to_select_ptr->syncHAndDData(device);

  // test the selection indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select_ptr;
  tensorTable.makeSelectIndicesFromIndices("1", indices_to_select_ptr, indices_select_ptr, device);
  indices_select_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i % 2 == 0)
          assert(indices_select_ptr->getData()(i, j, k) == 1);
        else
          assert(indices_select_ptr->getData()(i, j, k) == 0);
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_deleteFromAxisGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<float, 3> new_values(Eigen::array<Eigen::Index, 3>({ nlabels - 1, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
        if (i != 1) {
          new_values(iter, j, k) = i + j * nlabels + k * nlabels*nlabels;
        }
      }
    }
    if (i != 1) ++iter;
  }
  tensorTable.setData(tensor_values);

  // setup the selection indices
  Eigen::Tensor<int, 1> indices_to_select_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select_values.setValues({ 2 });
  TensorDataGpuPrimitiveT<int, 1> indices_to_select(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_to_select.setData(indices_to_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_to_select_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_to_select);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  indices_to_select_ptr->syncHAndDData(device);

  // test deleteFromAxis
  TensorDataGpuPrimitiveT<float, 3> values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values.setData();
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values);
  values_ptr->syncHAndDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_ptr;
  tensorTable.deleteFromAxis("1", indices_to_select_ptr, labels_ptr, values_ptr->getDataPointer(), device);

  // test the expected indices sizes and values
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_ptr->syncHAndDData(device);
  labels_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")) == nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      assert(tensorTable.getIndices().at("1")->getData()(i) == i + 1);
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 1);
    }
    else {
      assert(tensorTable.getIndices().at("1")->getData()(i) == i + 2);
      assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 2);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 2);
    }
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getShardId().at("1")->getData()(i) == 1);
  }

  // Test the expected data values
  for (int i = 0; i < nlabels - 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(tensorTable.getData()(i, j, k) == new_values(i, j, k));
      }
    }
  }

  // Test the expected axis values
  std::vector<int> expected_labels = { 0, 2 };
  for (int i = 0; i < nlabels - 1; ++i) {
    assert(axis_1_ptr->getLabels()(0, i) == expected_labels.at(i));
  }

  // Test the expected returned labels
  assert(labels_ptr->getData()(0, 0) == 1);

  // Test the expected returned data
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(values_ptr->getData()(i, j, k) == tensor_values(1, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeIndicesFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // modify the indices view for axis 1
  tensorTable.getIndicesView().at("1")->getData()(0) = 0;

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncAxesHAndDData(device);

  // test makeIndicesFromIndicesView
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_ptr;
  tensorTable.makeIndicesFromIndicesView("1", indices_ptr, device);
  indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels - 1; ++i) {
    assert(indices_ptr->getData()(i) == i + 2);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_insertIntoAxisGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new tensor data
  Eigen::Tensor<float, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j) = 100;
    }
  }
  TensorDataGpuPrimitiveT<float, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(values_new);

  // setup the new axis labels
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_values(0, 0) = 100;
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  labels_new.setData(labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // setup the new indices
  Eigen::Tensor<int, 1> indices_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_values(0) = 3;
  TensorDataGpuPrimitiveT<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  indices_new.setData(indices_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_new);

  // Change the indices and indices view to simulate a deletion
  tensorTable.getIndices().at("1")->getData()(nlabels - 1) = 4;
  tensorTable.getIndicesView().at("1")->getData()(nlabels - 1) = 4;

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_new_ptr->syncHAndDData(device);
  labels_new_ptr->syncHAndDData(device);
  indices_new_ptr->syncHAndDData(device);

  // test insertIntoAxis
  tensorTable.insertIntoAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  //std::cout << "insertIntoAxis Failing:" << std::endl;
  //std::cout << "[TEST = {1 2 0 4} ]tensorTable.getIsModified().at(1)->getData()\n" << tensorTable.getIsModified().at("1")->getData() << std::endl;
  int iter = 0;
  for (int i = 0; i < nlabels + 1; ++i) {
    // check the axis
    if (i == 2)
      assert(axis_1_ptr->getLabels()(0, i) == 100);
    else
      assert(axis_1_ptr->getLabels()(0, i) == labels1(iter));

    // check the indices
    assert(tensorTable.getIndices().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    if (i >= nlabels) {
      assert(tensorTable.getShardId().at("1")->getData()(i) == 2);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i - nlabels + 1);
    }
    else {
      assert(tensorTable.getShardId().at("1")->getData()(i) == 1);
      assert(tensorTable.getShardIndices().at("1")->getData()(i) == i + 1);
    }

    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        // check the tensor data
        if (i == 2)
          assert(tensorTable.getData()(i, j, k) == 100);
        else
          assert(tensorTable.getData()(i, j, k) == tensor_values(iter, j, k));
      }
    }
    if (i != 2) ++iter;
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeSparseAxisLabelsFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<int, 2> expected_values(Eigen::array<Eigen::Index, 2>({ 3, nlabels*nlabels*nlabels }));
  expected_values.setValues({
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_ptr;
  tensorTable.makeSparseAxisLabelsFromIndicesView(labels_ptr, device);
  labels_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(labels_ptr->getDimensions().at(0) == 3);
  assert(labels_ptr->getDimensions().at(1) == nlabels*nlabels*nlabels);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < nlabels*nlabels*nlabels; ++j) {
      assert(labels_ptr->getData()(i, j) == expected_values(i, j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeSparseTensorTableGpu()
{
  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the expected axes
  Eigen::Tensor<std::string, 1> dimensions1(3), dimensions2(1);
  dimensions1.setValues({ "0", "1", "2" });
  dimensions2(0) = "Values";

  // setup the expected labels
  int nlabels1 = 27;
  Eigen::Tensor<int, 2> labels1(3, nlabels1), labels2(1, 1);
  labels1.setValues({
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });
  TensorDataGpuPrimitiveT<int, 2> sparse_labels(Eigen::array<Eigen::Index, 2>({ 3, nlabels1 }));
  sparse_labels.setData(labels1);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> sparse_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(sparse_labels);

  labels2.setConstant(0);

  // setup the expected data
  int nlabels = 3;
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  TensorDataGpuPrimitiveT<float, 3> sparse_data(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  sparse_data.setData(tensor_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> sparse_data_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 3>>(sparse_data);

  // Test
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 2>> sparse_table_ptr;
  sparse_labels_ptr->syncHAndDData(device);
  sparse_data_ptr->syncHAndDData(device);
  sparse_labels_ptr->syncHAndDData(device);
  TensorTableGpuPrimitiveT<float, 3> tensorTable;
  tensorTable.makeSparseTensorTable(dimensions1, sparse_labels_ptr, sparse_data_ptr, sparse_table_ptr, device);
  sparse_labels_ptr->syncHAndDData(device);
  sparse_data_ptr->syncHAndDData(device);
  sparse_table_ptr->syncIndicesHAndDData(device);
  sparse_table_ptr->syncIndicesViewHAndDData(device);
  sparse_table_ptr->syncNotInMemoryHAndDData(device);
  sparse_table_ptr->syncIsModifiedHAndDData(device);
  sparse_table_ptr->syncShardIdHAndDData(device);
  sparse_table_ptr->syncShardIndicesHAndDData(device);
  sparse_table_ptr->syncAxesHAndDData(device);
  sparse_table_ptr->syncHAndDData(device);
  sparse_labels_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Check for the correct dimensions
  assert(sparse_table_ptr->getDimensions().at(0) == nlabels1);
  assert(sparse_table_ptr->getDimensions().at(1) == 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels) == tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  assert(sparse_table_ptr->getAxes().at("Indices")->getName() == "Indices");
  assert(sparse_table_ptr->getAxes().at("Indices")->getNLabels() == nlabels1);
  assert(sparse_table_ptr->getAxes().at("Indices")->getNDimensions() == 3);
  // TODO: transfer to host
  //std::shared_ptr<int> labels1_ptr;
  //sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    assert(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i) == std::to_string(i));
    //for (int j = 0; j < nlabels1; ++j) {
    //  assert(labels_values(i, j) == labels1(i, j));
    //}
  }

  // Check the Values axes
  assert(sparse_table_ptr->getAxes().at("Values")->getName() == "Values");
  assert(sparse_table_ptr->getAxes().at("Values")->getNLabels() == 1);
  assert(sparse_table_ptr->getAxes().at("Values")->getNDimensions() == 1);
  // TODO: transfer to host
  //std::shared_ptr<int> labels2_ptr;
  //sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  //assert(labels2_values(0, 0), 0);
  assert(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0) == "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i) == i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Values")->getData()(i) == i + 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_getSelectTensorDataAsSparseTensorTableGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the expected labels
  int nlabels1 = 27;
  Eigen::Tensor<int, 2> labels1_expected(3, nlabels1);
  labels1_expected.setValues({
    {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3 },
    {1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3,1,1,1,2,2,2,3,3,3 },
    {1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3 } });

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 2>> sparse_table_ptr;
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);
  sparse_table_ptr->syncIndicesHAndDData(device);
  sparse_table_ptr->syncIndicesViewHAndDData(device);
  sparse_table_ptr->syncNotInMemoryHAndDData(device);
  sparse_table_ptr->syncIsModifiedHAndDData(device);
  sparse_table_ptr->syncShardIdHAndDData(device);
  sparse_table_ptr->syncShardIndicesHAndDData(device);
  sparse_table_ptr->syncAxesHAndDData(device);
  sparse_table_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Check for the correct dimensions
  assert(sparse_table_ptr->getDimensions().at(0) == nlabels1);
  assert(sparse_table_ptr->getDimensions().at(1) == 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels) == tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  assert(sparse_table_ptr->getAxes().at("Indices")->getName() == "Indices");
  assert(sparse_table_ptr->getAxes().at("Indices")->getNLabels() == nlabels1);
  assert(sparse_table_ptr->getAxes().at("Indices")->getNDimensions() == 3);
  // TODO: transfer to host
  //std::shared_ptr<int> labels1_ptr;
  //sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    assert(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i) == std::to_string(i + 1));
    //for (int j = 0; j < nlabels1; ++j) {
    //  assert(labels_values(i, j) == labels1_expected(i, j));
    //}
  }

  // Check the Values axes
  assert(sparse_table_ptr->getAxes().at("Values")->getName() == "Values");
  assert(sparse_table_ptr->getAxes().at("Values")->getNLabels() == 1);
  assert(sparse_table_ptr->getAxes().at("Values")->getNDimensions() == 1);
  // TODO: transfer to host
  //std::shared_ptr<int> labels2_ptr;
  //sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  //assert(labels2_values(0, 0) == 0);
  assert(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0) == "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i) == i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Values")->getData()(i) == i + 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Re-test getSelectTensorDataAsSparseTensorTable
  sparse_table_ptr.reset();
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);
  sparse_table_ptr->syncIndicesHAndDData(device);
  sparse_table_ptr->syncIndicesViewHAndDData(device);
  sparse_table_ptr->syncNotInMemoryHAndDData(device);
  sparse_table_ptr->syncIsModifiedHAndDData(device);
  sparse_table_ptr->syncShardIdHAndDData(device);
  sparse_table_ptr->syncShardIndicesHAndDData(device);
  sparse_table_ptr->syncAxesHAndDData(device);
  sparse_table_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Check for the correct dimensions
  assert(sparse_table_ptr->getDimensions().at(0) == nlabels1);
  assert(sparse_table_ptr->getDimensions().at(1) == 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels) == tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  assert(sparse_table_ptr->getAxes().at("Indices")->getName() == "Indices");
  assert(sparse_table_ptr->getAxes().at("Indices")->getNLabels() == nlabels1);
  assert(sparse_table_ptr->getAxes().at("Indices")->getNDimensions() == 3);
  // TODO: transfer to host
  //std::shared_ptr<int> labels1_ptr;
  //sparse_table_ptr->getAxes().at("Indices")->getLabelsDataPointer(labels1_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    assert(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i) == std::to_string(i + 1));
    //for (int j = 0; j < nlabels1; ++j) {
    //  assert(labels_values(i, j) == labels1_expected(i, j));
    //}
  }

  // Check the Values axes
  assert(sparse_table_ptr->getAxes().at("Values")->getName() == "Values");
  assert(sparse_table_ptr->getAxes().at("Values")->getNLabels() == 1);
  assert(sparse_table_ptr->getAxes().at("Values")->getNDimensions() == 1);
  // TODO: transfer to host
  //std::shared_ptr<int> labels2_ptr;
  //sparse_table_ptr->getAxes().at("Values")->getLabelsDataPointer(labels2_ptr);
  //Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  //assert(labels2_values(0, 0) == 0);
  assert(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0) == "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Indices")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i) == i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    assert(sparse_table_ptr->getIndices().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIndicesView().at("Values")->getData()(i) == i + 1);
    assert(sparse_table_ptr->getIsModified().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i) == 0);
    assert(sparse_table_ptr->getShardId().at("Values")->getData()(i) == 1);
    assert(sparse_table_ptr->getShardIndices().at("Values")->getData()(i) == i + 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_updateTensorDataConstantGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the update values
  TensorDataGpuPrimitiveT<float, 1> values_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  values_new.setData();
  values_new.getData()(0) = 100;
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> values_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<float, 1>>(values_new);

  // sync the tensorTable
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_new_ptr->syncHAndDData(device);

  // Test update
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 2>> values_old_ptr;
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_old_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels) == tensor_values(i, j, k));
        assert(tensorTable.getData()(i, j, k) == 100);
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  // reset is_modified attribute
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(1);
  }

  // Revert the operation and test
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_old_ptr->syncHAndDData(device);
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()(i, j, k) == tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  // TODO: Test after a selection (see test for TensorOperation TensorUpdateConstant)

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Test update
  values_old_ptr.reset();
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  values_old_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test the data
  //std::cout << "test_updateTensorDataConstantGpu Failing:" << std::endl;
  //std::cout << "values_old_ptr->getData()\n" << values_old_ptr->getData() << std::endl;
  //std::cout << "[TEST] tensor_values\n" << tensor_values << std::endl;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels) == tensor_values(i, j, k));
        assert(tensorTable.getData()(i, j, k) == 100);
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  // clear the data
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncHAndDData(device);

  // Revert the operation and test
  values_old_ptr->setDataStatus(false, true);
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()(i, j, k) == tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeShardIndicesFromShardIDsGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 6;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2, 3, 4, 5} });
  labels2.setValues({ {0, 1, 2, 3, 4, 5} });
  labels3.setValues({ {0, 1, 2, 3, 4, 5} });
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);

  // Test for the shard indices
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_shard_ptr;
  tensorTable.makeShardIndicesFromShardIDs(indices_shard_ptr, device);
  indices_shard_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_shard_ptr->getData()(i, j, k) == 1);
      }
    }
  }

  // make the expected tensor indices
  int shard_n_indices = 3;
  std::vector<int> shard_id_indices = { 0, 0, 1, 1, 2, 2 };
  Eigen::Tensor<int, 3> indices_test(nlabels, nlabels, nlabels);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = shard_id_indices.at(i) + shard_id_indices.at(j) * shard_n_indices + shard_id_indices.at(k) * shard_n_indices*shard_n_indices + 1;
      }
    }
  }

  // Test for the shard indices
  tensorTable.reShardIndices(device);
  indices_shard_ptr.reset();
  tensorTable.makeShardIndicesFromShardIDs(indices_shard_ptr, device);
  indices_shard_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_shard_ptr->getData()(i, j, k) == indices_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeModifiedShardIDTensorGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // Reshard indices
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test the unmodified case
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  shard_id_indices_ptr->syncHAndDData(device);
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  assert(slice_indices.size() == 0);

  // Test the fully modified case
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(1);
  }
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    assert(shard_id_indices_ptr->getData()(i) == i + 1);
  }

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    assert(slice_indices_map.first == iter);
    assert(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    assert(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }

  // Test the partially modified case
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        is_modified_map.second->getData()(i) = 1;
      else
        is_modified_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    assert(shard_id_indices_ptr->getData()(i) == i + 1);
  }

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    assert(slice_indices_map.first == iter);
    assert(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    assert(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_makeNotInMemoryShardIDTensorGpu()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels), labels3(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  labels3.setValues({ {0, 1, 2} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // Reshard indices
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test all in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  assert(slice_indices.size() == 0);

  // Test not all in memory case and none selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    indices_view_map.second->getData() = indices_view_map.second->getData().constant(0);
  }
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 0);

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  assert(slice_indices.size() == 0);

  // Test all not in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.resetIndicesView("1", device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.resetIndicesView("3", device);
  //tensorTable.setIndicesViewDataStatus(true, false);
  //tensorTable.syncIndicesViewHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    assert(shard_id_indices_ptr->getData()(i) == i + 1);
  }

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    assert(slice_indices_map.first == iter);
    assert(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    assert(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }

  // Test the partially in memory case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        in_memory_map.second->getData()(i) = 1;
      else
        in_memory_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    assert(shard_id_indices_ptr->getData()(i) == i + 1);
  }

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    assert(slice_indices_map.first == iter);
    assert(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    assert(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  
  // Test the partially in memory case and partially selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        in_memory_map.second->getData()(i) = 1;
      else
        in_memory_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryHAndDData(device);
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < 1)
        indices_view_map.second->getData()(i) = i + 1;
      else
        indices_view_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewHAndDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);
  assert(shard_id_indices_ptr->getTensorSize() == 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    assert(shard_id_indices_ptr->getData()(i) == i + 1);
  }

  slice_indices.clear();
  tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    assert(slice_indices_map.first == iter);
    assert(slice_indices_map.second.first == slice_indices_test.at(slice_indices_map.first).first);
    assert(slice_indices_map.second.second == slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_getCsvDataRowGpuPrimitiveT()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, nlabels);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0, 1, 2} });
  Eigen::Tensor<char, 2> labels3(1, nlabels);
  labels3.setValues({ {'0', '1', '2'} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<char>>(TensorAxisGpuPrimitiveT<char>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  std::vector<std::string> row_0_test, row_1_test, row_4_test;
  int row_iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        float value = i + j * nlabels + k * nlabels*nlabels;
        tensor_values(i, j, k) = value;
        if (row_iter == 0) {
          row_0_test.push_back(std::to_string(value));
        }
        else if (row_iter == 1) {
          row_1_test.push_back(std::to_string(value));
        }
        else if (row_iter == 4) {
          row_4_test.push_back(std::to_string(value));
        }
      }
      ++row_iter;
    }
  }
  tensorTable.setData(tensor_values);

  // Test getCsvDataRow
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  // TODO: also test char and tensorArray types
  std::vector<std::string> row_0 = tensorTable.getCsvDataRow(0);
  std::vector<std::string> row_1 = tensorTable.getCsvDataRow(1);
  std::vector<std::string> row_4 = tensorTable.getCsvDataRow(4);
  assert(row_0.size() == nlabels);
  assert(row_1.size() == nlabels);
  assert(row_4.size() == nlabels);
  for (int i = 0; i < nlabels; ++i) {
    assert(row_0.at(i) == row_0_test.at(i));
    assert(row_1.at(i) == row_1_test.at(i));
    assert(row_4.at(i) == row_4_test.at(i));
  }

  // Make the expected labels row values
  std::map<std::string, std::vector<std::string>> labels_row_0_test = { {"2", {"0"}}, {"3", {"a"}} };
  std::map<std::string, std::vector<std::string>> labels_row_1_test = { {"2", {"1"}}, {"3", {"a"}} };
  std::map<std::string, std::vector<std::string>> labels_row_4_test = { {"2", {"1"}}, {"3", {"b"}} };

  // Test getCsvAxesLabelsRow
  std::map<std::string, std::vector<std::string>> labels_row_0 = tensorTable.getCsvAxesLabelsRow(0);
  std::map<std::string, std::vector<std::string>> labels_row_1 = tensorTable.getCsvAxesLabelsRow(1);
  std::map<std::string, std::vector<std::string>> labels_row_4 = tensorTable.getCsvAxesLabelsRow(4);
  assert(labels_row_0.size(), 2);
  assert(labels_row_1.size(), 2);
  assert(labels_row_4.size(), 2);
  std::cout << "GPU getCsvAxesLabelsRow Failing char conversion." << std::endl;
  for (int i = 2; i < 4; ++i) {
    std::string axis_name = std::to_string(i);
    for (int j = 0; j < 1; ++j) {
      std::cout << "labels_row_0.at(axis_name).at(" << j << "): "<< labels_row_0.at(axis_name).at(j) << std::endl;
      std::cout << "labels_row_1.at(axis_name).at(" << j << "): " << labels_row_1.at(axis_name).at(j) << std::endl;
      std::cout << "labels_row_4.at(axis_name).at(" << j << "): " << labels_row_4.at(axis_name).at(j) << std::endl;
      //assert(labels_row_0.at(axis_name).at(j) == labels_row_0_test.at(axis_name).at(j)); //FIXME
      //assert(labels_row_1.at(axis_name).at(j) == labels_row_1_test.at(axis_name).at(j));
      //assert(labels_row_4.at(axis_name).at(j) == labels_row_4_test.at(axis_name).at(j));
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_insertIntoTableFromCsvGpuPrimitiveT()
{
  // setup the table
  TensorTableGpuPrimitiveT<float, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  assert(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels = 3;
  Eigen::Tensor<int, 2> labels1(1, nlabels), labels2(1, 1), labels3(1, 1);
  labels1.setValues({ {0, 1, 2} });
  labels2.setValues({ {0} });
  labels3.setValues({ {0} });
  auto axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  auto axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  auto axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data, the new tensor data from csv, and the new axes labels from csv
  Eigen::Tensor<float, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, 1, 1 }));
  Eigen::Tensor<std::string, 2> new_values_str(Eigen::array<Eigen::Index, 2>({ nlabels, 8 }));
  Eigen::Tensor<std::string, 2> labels_2_str(1, 8);
  Eigen::Tensor<std::string, 2> labels_3_str(1, 8);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        if (j == 0 && k == 0) {
          tensor_values(i, j, k) = i + j * nlabels + k * nlabels*nlabels;
        }
        else {
          int index = j + k * nlabels - 1;
          new_values_str(i, index) = std::to_string(i + j * nlabels + k * nlabels*nlabels);
          labels_2_str(0, index) = std::to_string(j);
          labels_3_str(0, index) = std::to_string(k);
        }
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new axis labels from csv
  std::map<std::string, Eigen::Tensor<std::string, 2>> labels_new_str;
  labels_new_str.emplace("2", labels_2_str);
  labels_new_str.emplace("3", labels_3_str);

  // Test
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  tensorTable.insertIntoTableFromCsv(labels_new_str, new_values_str, device);
  tensorTable.syncIndicesHAndDData(device);
  tensorTable.syncIndicesViewHAndDData(device);
  tensorTable.syncIsModifiedHAndDData(device);
  tensorTable.syncNotInMemoryHAndDData(device);
  tensorTable.syncShardIdHAndDData(device);
  tensorTable.syncShardIndicesHAndDData(device);
  tensorTable.syncAxesHAndDData(device);
  tensorTable.syncHAndDData(device);
  assert(cudaStreamSynchronize(stream) == cudaSuccess);

  // Test for the tensor data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()(i, j, k) == i + j * nlabels + k * nlabels*nlabels);
      }
    }
  }

  // Test for the axis labels
  for (int i = 0; i < nlabels; ++i) {
    assert(axis_1_ptr->getLabels()(0, i) == i);
    assert(axis_2_ptr->getLabels()(0, i) == i);
    assert(axis_3_ptr->getLabels()(0, i) == i);
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getNotInMemory().at("1")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getNotInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsModified().at("1")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{	
  test_constructorGpu();
  test_destructorGpu(); 
  test_constructorNameAndAxesGpu();
  test_gettersAndSettersGpu();
  test_reShardIndicesGpu();
  test_tensorDataWrappersGpu();
  test_zeroIndicesViewAndResetIndicesViewGpu();
  test_selectIndicesView1Gpu();
  test_selectIndicesView2Gpu();
  test_broadcastSelectIndicesViewGpu();
  test_extractTensorDataGpuPrimitiveT();
  test_selectTensorIndicesGpu();
  test_applyIndicesSelectToIndicesViewGpu();
  test_whereIndicesViewDataGpu();
  test_sliceTensorForSortGpu();
  test_sortIndicesViewData1Gpu();
  test_sortIndicesViewData2Gpu();
  test_makeSelectIndicesFromIndicesViewGpu();
  test_getSelectTensorDataFromIndicesViewGpu();
  test_selectTensorDataGpuPrimitiveT();
  test_makeSortIndicesViewFromIndicesViewGpu();
  test_sortTensorDataGpuPrimitiveT();
  test_updateSelectTensorDataValues1Gpu();
  test_updateSelectTensorDataValues2Gpu();
	test_updateTensorDataValuesGpu();
  test_makeAppendIndicesGpu();
  test_appendToIndicesGpu(); // Failing to launch Gpu kernal???
  test_appendToAxisGpu();
  test_makeIndicesViewSelectFromIndicesGpu();
  test_deleteFromIndicesGpu();
  test_makeSelectIndicesFromIndicesGpu();
  test_deleteFromAxisGpu();
  test_makeIndicesFromIndicesViewGpu();
  test_insertIntoAxisGpu();
  test_makeSparseAxisLabelsFromIndicesViewGpu();
  test_makeSparseTensorTableGpu();
  test_getSelectTensorDataAsSparseTensorTableGpu();
  test_updateTensorDataConstantGpu(); // Fix broken test
  test_makeShardIndicesFromShardIDsGpu();
  test_makeModifiedShardIDTensorGpu();
  test_makeNotInMemoryShardIDTensorGpu();
  test_getCsvDataRowGpuPrimitiveT();
  test_insertIntoTableFromCsvGpuPrimitiveT();

  return 0;
}

#endif