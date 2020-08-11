/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorTableGpuClassT.h>
#include <string>

using namespace TensorBase;
using namespace std;

void test_constructorGpu()
{
  TensorTableGpuClassT<TensorArrayGpu8, char, 3>* ptr = nullptr;
  TensorTableGpuClassT<TensorArrayGpu8, char, 3>* nullPointer = nullptr;
  ptr = new TensorTableGpuClassT<TensorArrayGpu8, char, 3>();
  gpuCheckNotEqual(ptr, nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorTableGpuClassT<TensorArrayGpu8, char, 3>* ptr = nullptr;
  ptr = new TensorTableGpuClassT<TensorArrayGpu8, char, 3>();
  delete ptr;
}

void test_constructorNameAndAxesGpu()
{
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable("1");

  gpuCheckEqual(tensorTable.getId(), -1);
  gpuCheckEqual(tensorTable.getName(), "1");
  gpuCheckEqual(tensorTable.getDir(), "");

  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable2("1", "dir");

  gpuCheckEqual(tensorTable2.getId(), -1);
  gpuCheckEqual(tensorTable2.getName(), "1");
  gpuCheckEqual(tensorTable2.getDir(), "dir");
}

void test_gettersAndSettersGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;
  // Check defaults
  gpuCheckEqual(tensorTable.getId(), -1);
  gpuCheckEqual(tensorTable.getName(), "");
  gpuCheckEqual(tensorTable.getAxes().size(), 0);
  gpuCheckEqual(tensorTable.getDir(), "");
  gpuCheckEqual(tensorTable.getTensorSize(), 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");
  tensorTable.setDir("dir");

  gpuCheckEqual(tensorTable.getId(), 1);
  gpuCheckEqual(tensorTable.getName(), "1");
  gpuCheckEqual(tensorTable.getDir(), "dir");

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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test expected axes values
  gpuCheckEqual(tensorTable.getAxes().at("1")->getName(), "1");
  //gpuCheckEqual(tensorTable.getAxes().at("1")->getLabels()(0, 0), 1);
  ////gpuCheckEqual(tensorTable.getAxes().at("1")->getLabels()(0,0), "x-axis");
  gpuCheckEqual(tensorTable.getAxes().at("1")->getNLabels(), nlabels1);
  gpuCheckEqual(tensorTable.getAxes().at("1")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(nlabels1 - 1), nlabels1);
  gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(nlabels1 - 1), nlabels1);
  gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(0), 0);
  gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(nlabels1 - 1), nlabels1);

  gpuCheckEqual(tensorTable.getAxes().at("2")->getName(), "2");
  //gpuCheckEqual(tensorTable.getAxes().at("2")->getLabels()(0, 0), 2);
  ////gpuCheckEqual(tensorTable.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNLabels(), nlabels2);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  gpuCheckEqual(tensorTable.getIndices().at("2")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndices().at("2")->getData()(nlabels2 - 1), nlabels2);
  gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(nlabels2 - 1), nlabels2);
  gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(0), 0);
  gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(nlabels2 - 1), nlabels2);

  gpuCheckEqual(tensorTable.getAxes().at("3")->getName(), "3");
  //gpuCheckEqual(tensorTable.getAxes().at("3")->getLabels()(0, 0), 3);
  ////gpuCheckEqual(tensorTable.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  gpuCheckEqual(tensorTable.getIndices().at("3")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndices().at("3")->getData()(nlabels3 - 1), nlabels3);
  gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(nlabels3 - 1), nlabels3);
  gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(0), 0);
  gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(nlabels3 - 1), nlabels3);

  // Test expected axis to dims mapping
  gpuCheckEqual(tensorTable.getDimFromAxisName("1"), 0);
  gpuCheckEqual(tensorTable.getDimFromAxisName("2"), 1);
  gpuCheckEqual(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor shard spans
  gpuCheckEqual(tensorTable.getShardSpans().at("1"), 2);
  gpuCheckEqual(tensorTable.getShardSpans().at("2"), 3);
  gpuCheckEqual(tensorTable.getShardSpans().at("3"), 5);

  // Test expected tensor dimensions
  gpuCheckEqual(tensorTable.getDimensions().at(0), 2);
  gpuCheckEqual(tensorTable.getDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDimensions().at(2), 5);
  gpuCheckEqual(tensorTable.getTensorSize(), 30);

  // Test expected maximum dimensions
  gpuCheckEqual(tensorTable.getMaximumDimensions().at(0), 36);
  gpuCheckEqual(tensorTable.getMaximumDimensions().at(1), 54);
  gpuCheckEqual(tensorTable.getMaximumDimensions().at(2), 90);

  // Test expected tensor data values
  gpuCheckEqual(tensorTable.getDataDimensions().at(0), 2);
  gpuCheckEqual(tensorTable.getDataDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDataDimensions().at(2), 5);
  size_t test = 2 * 3 * 5 * sizeof(TensorArrayGpu8<char>);
  gpuCheckEqual(tensorTable.getDataTensorBytes(), test);

  // Test setting the data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_data(Eigen::array<Eigen::Index, 3>({ nlabels1, nlabels2, nlabels3 }));
  for (int i = 0; i < nlabels1; ++i) {
    for (int j = 0; j < nlabels2; ++j) {
      for (int k = 0; k < nlabels3; ++k) {
        tensor_data(i, j, k).setTensorArray(std::to_string(i + j + k));
      }
    }
  }
  tensorTable.setData(tensor_data);
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
  }
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
  }

  // Test setting the data
  tensorTable.setData();
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
  }
  for (int i = 0; i < nlabels2; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
  }
  for (int i = 0; i < nlabels3; ++i) {
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }

  // Test clear
  tensorTable.clear();
  gpuCheckEqual(tensorTable.getAxes().size(), 0);
  gpuCheckEqual(tensorTable.getIndices().size(), 0);
  gpuCheckEqual(tensorTable.getIndicesView().size(), 0);
  gpuCheckEqual(tensorTable.getIsModified().size(), 0);
  gpuCheckEqual(tensorTable.getNotInMemory().size(), 0);
  gpuCheckEqual(tensorTable.getShardId().size(), 0);
  gpuCheckEqual(tensorTable.getShardIndices().size(), 0);
  gpuCheckEqual(tensorTable.getDimensions().at(0), 0);
  gpuCheckEqual(tensorTable.getDimensions().at(1), 0);
  gpuCheckEqual(tensorTable.getDimensions().at(2), 0);
  gpuCheckEqual(tensorTable.getShardSpans().size(), 0);
}

void test_initDataGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // Check the dimensions and expected not_in_memory values
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  gpuCheckEqual(tensorTable.getDataTensorSize(), nlabels * nlabels*nlabels);

  // Reset the not_in_memory to false
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }
  tensorTable.syncNotInMemoryDData(device);

  // Resize the tensor data
  Eigen::array<Eigen::Index, 3> new_dimensions = { 2, 2, 2 };
  tensorTable.initData(new_dimensions, device);

  // Check the dimensions and expected not_in_memory values
  tensorTable.syncNotInMemoryHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  gpuCheckEqual(tensorTable.getDataTensorSize(), 8);

  // Reset the not_in_memory to false
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }
  tensorTable.syncNotInMemoryDData(device);

  // Resize the tensor data to 0
  tensorTable.initData(device);

  // Check the dimensions and expected not_in_memory values
  tensorTable.syncNotInMemoryHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
  }
  gpuCheckEqual(tensorTable.getDataTensorSize(), 0);
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_reShardIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Test the default shard span
  gpuCheckEqual(tensorTable.getShardSpans().at("1"), 4);
  gpuCheckEqual(tensorTable.getShardSpans().at("2"), 4);
  gpuCheckEqual(tensorTable.getShardSpans().at("3"), 4);

  // Reset the shard span
  int shard_span = 3;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  gpuCheckEqual(tensorTable.getShardSpans().at("1"), 3);
  gpuCheckEqual(tensorTable.getShardSpans().at("2"), 3);
  gpuCheckEqual(tensorTable.getShardSpans().at("3"), 3);
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.reShardIndices(device);
  tensorTable.syncAxesAndIndicesHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i < shard_span) {
      gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i - shard_span + 1);
      gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(i), 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(i), i - shard_span + 1);
      gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(i), 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(i), i - shard_span + 1);
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_zeroIndicesViewAndResetIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable indices
  tensorTable.syncIndicesDData(device);
  tensorTable.syncIndicesViewDData(device);

  // test null
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
  }

  // test zero
  tensorTable.zeroIndicesView("1", device);
  tensorTable.getIndicesView().at("1")->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }

  // test reset
  tensorTable.getIndicesView().at("1")->setDataStatus(false, true);
  tensorTable.resetIndicesView("1", device);
  tensorTable.getIndicesView().at("1")->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
  }

  // test zero
  tensorTable.getIndicesView().at("1")->setDataStatus(false, true);
  tensorTable.zeroIndicesView("1", device);
  tensorTable.getIndicesView().at("1")->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }

  // test replace
  tensorTable.getIndicesView().at("1")->setDataStatus(false, true);
  tensorTable.replaceIndicesView("1", tensorTable.getIndicesView().at("2"), device);
  tensorTable.getIndicesView().at("1")->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_selectIndicesView1Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test the updated view
  select_labels_ptr->syncDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  tensorTable.syncIndicesViewHData(device);
  select_labels_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0)
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}
void test_selectIndicesView2Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(2, 2);
  select_labels_values.setValues({ {0, 2}, {4, 6} });
  TensorDataGpuPrimitiveT<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 2, 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(select_labels);

  // test the updated view
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  select_labels_ptr->syncDData(device);
  tensorTable.selectIndicesView("1", select_labels_ptr, device);
  tensorTable.getIndicesView().at("1")->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0)
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_broadcastSelectIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable indices
  tensorTable.syncAxesAndIndicesDData(device);

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
  indices_view_bcast->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        //std::cout << "Test broadcastSelectIndicesView i,j,k :" << i << "," << j << "," << k << "; Labels: " << indices_view_bcast->getData()(i, j, k) << "; Expected: " << indices_test(i, j, k) << std::endl;
        gpuCheckEqual(indices_view_bcast->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_extractTensorDataGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data, selection indices, and test selection data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<int, 3> indices_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_test(Eigen::array<Eigen::Index, 3>({ nlabels / 2, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        int value = i * nlabels + j * nlabels + k;
        tensor_values(i, j, k).setTensorArray(std::to_string(value));
        if (i % 2 == 0) {
          indices_values(i, j, k) = 1;
          tensor_test(i / 2, j, k).setTensorArray(std::to_string(value));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test
  indices_select_ptr->syncDData(device);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensor_select;
  tensorTable.reduceTensorDataToSelectIndices(indices_select_ptr,
    tensor_select, "1", nlabels / 2, device);
  tensor_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels / 2; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensor_select->getData()(i, j, k), tensor_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_selectTensorIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor select and values select data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<TensorArrayGpu8<char>, 1> values_select_values(Eigen::array<Eigen::Index, 1>({ nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    values_select_values(i).setTensorArray("2");
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_select_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> tensor_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  tensor_select.setData(tensor_select_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensor_select_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(tensor_select);
  TensorDataGpuClassT<TensorArrayGpu8, char, 1> values_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  values_select.setData(values_select_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 1>> values_select_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(values_select);

  // Sync the data
  tensor_select_ptr->syncDData(device);
  values_select_ptr->syncDData(device);
  tensorTable.syncAxesAndIndicesDData(device);

  // test inequality
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::NOT_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
      }
    }
  }

  // test equality
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) < TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) <= TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) > TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) >= TensorArrayGpu8<char>("2"))
          gpuCheckEqual(indices_select->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select->getData()(i, j, k), 0);
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_applyIndicesSelectToIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

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
  indices_select_ptr->syncDData(device);

  // test using the second indices view
  tensorTable.getIndicesView().at("2")->getData()(nlabels - 1) = 0;
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewDData(device);

  // test for OR within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::OR, device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i == nlabels - 1)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
  }

  // reset and modify the indices view
  tensorTable.syncIndicesViewDData(device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  tensorTable.syncIndicesViewDData(device);

  // test for AND within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::OR, device);  
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
  }

  // Reset and modify the indices view
  tensorTable.syncIndicesViewDData(device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  tensorTable.syncIndicesViewDData(device);

  // test for OR within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i != 0 && i < nlabels - 1)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
  }

  // Reset the indices view
  tensorTable.syncIndicesViewDData(device);
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
  indices_select_ptr->syncDData(device);

  // test for AND within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
  }

  // TODO: lacking code coverage for the case of TDim = 2
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_whereIndicesViewData1Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(2);
  select_labels_values(0) = 0; select_labels_values(1) = 2;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncDData(device);

  // set up the selection values
  Eigen::Tensor<TensorArrayGpu8<char>, 1> select_values_values(2);
  select_values_values(0).setTensorArray("9"); select_values_values(1).setTensorArray("9");
  TensorDataGpuClassT<TensorArrayGpu8, char, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);
  std::shared_ptr<TensorDataGpuClassT<TensorArrayGpu8, char, 1>> select_values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(select_values);
  select_values_ptr->syncDData(device);

  // test
  tensorTable.whereIndicesView("1", 0, select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2) 
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 3
    if (i == 1)
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesDData(device);
  tensorTable.syncIndicesViewDData(device);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncShardIdDData(device);
  tensorTable.syncShardIndicesDData(device);
  tensorTable.syncAxesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // test  // FIXME: call to whereIndicesView is causing assertion failures for TensorArrayGpu8 size
  tensorTable.whereIndicesView("1", 0, select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2) // FIXME: i==0?
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 3
    if (i == 1) // FIXME: i==3?
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_whereIndicesViewData2Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(1, 2);
  select_labels_values(0, 0) = 0; select_labels_values(0, 1) = 2;
  TensorDataGpuPrimitiveT<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 1,2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(select_labels);
  select_labels_ptr->syncDData(device);

  // set up the selection values
  Eigen::Tensor<TensorArrayGpu8<char>, 1> select_values_values(2);
  select_values_values(0).setTensorArray("9"); select_values_values(1).setTensorArray("9");
  TensorDataGpuClassT<TensorArrayGpu8, char, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);
  std::shared_ptr<TensorDataGpuClassT<TensorArrayGpu8, char, 1>> select_values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(select_values);
  select_values_ptr->syncDData(device);

  // test
  tensorTable.whereIndicesView("1", select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 3
    if (i == 1)
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesDData(device);
  tensorTable.syncIndicesViewDData(device);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncShardIdDData(device);
  tensorTable.syncShardIndicesDData(device);
  tensorTable.syncAxesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // FIXME: call to whereIndicesView is causing assertion failures for TensorArrayGpu8 size
  tensorTable.whereIndicesView("1", select_labels_ptr, select_values_ptr,
    logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);

    // indices view 3
    if (i == 1)
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    else
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sliceTensorForSortGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test sliceTensorForSort for axis 2
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 1>> tensor_sort;
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "2", device); 
  tensor_sort->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  std::vector<TensorArrayGpu8<char>> tensor_slice_2_test = { TensorArrayGpu8<char>("9"),  TensorArrayGpu8<char>("12"),  TensorArrayGpu8<char>("15") };
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensor_sort->getData()(i), tensor_slice_2_test.at(i));
  }

  // test sliceTensorForSort for axis 2
  tensor_sort.reset();
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "3", device);
  tensor_sort->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  std::vector<TensorArrayGpu8<char>> tensor_slice_3_test = { TensorArrayGpu8<char>("9"),  TensorArrayGpu8<char>("10"),  TensorArrayGpu8<char>("11") };
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensor_sort->getData()(i), tensor_slice_3_test.at(i));
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sortIndicesViewData1Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 1;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncDData(device);

  // test sort ASC
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::ASC, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  std::cout << "\nFailing test_sortIndicesViewData1Gpu tests:" << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    std::cout << "Predicted IndicesView2: " << tensorTable.getIndicesView().at("2")->getData()(i) << " Expected: " << i + 1 << std::endl;
    std::cout << "Predicted IndicesView3: " << tensorTable.getIndicesView().at("3")->getData()(i) << " Expected: " << i + 1 << std::endl;
    //gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1); // FIXME
    //gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1); // FIXME
  }

  // test sort DESC
  tensorTable.setIndicesViewDataStatus(false, true);
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    std::cout << "Predicted IndicesView2: " << tensorTable.getIndicesView().at("2")->getData()(i) << " Expected: " << nlabels - i << std::endl;
    std::cout << "Predicted IndicesView3: " << tensorTable.getIndicesView().at("3")->getData()(i) << " Expected: " << nlabels - i << std::endl;
    //gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i); // FIXME
    //gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i); // FIXME
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sortIndicesViewData2Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 2> select_labels_values(1, 1);
  select_labels_values(0,0) = 1;
  TensorDataGpuPrimitiveT<int, 2> select_labels(Eigen::array<Eigen::Index, 2>({ 1, 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(select_labels);
  select_labels_ptr->syncDData(device);

  // test sort ASC
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::ASC, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    std::cout << "Predicted IndicesView2: " << tensorTable.getIndicesView().at("2")->getData()(i) << " Expected: " << i + 1 << std::endl;
    std::cout << "Predicted IndicesView3: " << tensorTable.getIndicesView().at("3")->getData()(i) << " Expected: " << i + 1 << std::endl;
    //gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1); // FIXME
    //gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1); // FIXME
  }

  // test sort DESC
  tensorTable.setIndicesViewDataStatus(false, true);
  tensorTable.sortIndicesView("1", select_labels_ptr, sortOrder::DESC, device);
  tensorTable.syncHData(device);
  tensorTable.syncIndicesViewHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    std::cout << "Predicted IndicesView2: " << tensorTable.getIndicesView().at("2")->getData()(i) << " Expected: " << nlabels - i << std::endl;
    std::cout << "Predicted IndicesView3: " << tensorTable.getIndicesView().at("3")->getData()(i) << " Expected: " << nlabels - i << std::endl;
    //gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), nlabels - i); // FIXME
    //gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), nlabels - i); // FIXME
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeSelectIndicesFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // Test null
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(indices_select->getData()(i, j, k), 1);
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
  select_labels_ptr->syncDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test selected
  indices_select.reset();
  tensorTable.makeSelectIndicesFromTensorIndicesComponent(tensorTable.getIndicesView(), indices_select, device);
  indices_select->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(indices_select->getData()(i, j, k), indices_select_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_getSelectTensorDataFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // select label 1 from axis 1
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // make the expected dimensions
  Eigen::array<Eigen::Index, 3> select_dimensions = { 1, 3, 3 };

  // make the indices_select
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_select_test(select_dimensions);
  Eigen::Tensor<int, 3> indices_select_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == 1) {
          indices_select_values(i, j, k) = 1;
          tensor_select_test(0, j, k).setTensorArray(std::to_string(iter));
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
  indices_select_ptr->syncDData(device);

  // test for the selected data
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> tensor_select_ptr;
  tensorTable.getSelectTensorDataFromIndicesView(tensor_select_ptr, indices_select_ptr, device);
  tensor_select_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensor_select_ptr->getDimensions(), select_dimensions);
  for (int j = 0; j < nlabels; ++j) {
    for (int k = 0; k < nlabels; ++k) {
      gpuCheckEqual(tensor_select_ptr->getData()(0, j, k), tensor_select_test(0, j, k), 1e-3);
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_selectTensorDataGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // select label 1 from axis 1
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncDData(device);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test `selectTensorData`
  tensorTable.selectTensorData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test expected axes values
  gpuCheckEqual(tensorTable.getAxes().at("1")->getName(), "1");
  gpuCheckEqual(tensorTable.getAxes().at("1")->getNLabels(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(0), 0);
  gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(0), 1);

  gpuCheckEqual(tensorTable.getAxes().at("2")->getName(), "2");
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNLabels(), nlabels);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }

  gpuCheckEqual(tensorTable.getAxes().at("3")->getName(), "3");
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNLabels(), nlabels);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test expected axis to dims mapping
  gpuCheckEqual(tensorTable.getDimFromAxisName("1"), 0);
  gpuCheckEqual(tensorTable.getDimFromAxisName("2"), 1);
  gpuCheckEqual(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  gpuCheckEqual(tensorTable.getDimensions().at(0), 1);
  gpuCheckEqual(tensorTable.getDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDimensions().at(2), 3);

  // Test expected tensor data values
  gpuCheckEqual(tensorTable.getDataDimensions().at(0), 1);
  gpuCheckEqual(tensorTable.getDataDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDataDimensions().at(2), 3);
  size_t test = 1 * 3 * 3 * sizeof(TensorArrayGpu8<char>);
  gpuCheckEqual(tensorTable.getDataTensorBytes(), test);

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.clear();
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);
  tensorTable.setData(tensor_values);
  tensorTable.syncIndicesDData(device);
  tensorTable.syncIndicesViewDData(device);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncShardIdDData(device);
  tensorTable.syncShardIndicesDData(device);
  tensorTable.syncAxesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Test selectTensorData
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  tensorTable.selectTensorData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test expected axes values
  gpuCheckEqual(tensorTable.getAxes().at("1")->getName(), "1");
  gpuCheckEqual(tensorTable.getAxes().at("1")->getNLabels(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(0), 0);
  gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(0), 1);
  gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(0), 1);

  gpuCheckEqual(tensorTable.getAxes().at("2")->getName(), "2");
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNLabels(), nlabels);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }

  gpuCheckEqual(tensorTable.getAxes().at("3")->getName(), "3");
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNLabels(), nlabels);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  gpuCheckEqual(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Test expected axis to dims mapping
  gpuCheckEqual(tensorTable.getDimFromAxisName("1"), 0);
  gpuCheckEqual(tensorTable.getDimFromAxisName("2"), 1);
  gpuCheckEqual(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  gpuCheckEqual(tensorTable.getDimensions().at(0), 1);
  gpuCheckEqual(tensorTable.getDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDimensions().at(2), 3);

  // Test expected tensor data values
  gpuCheckEqual(tensorTable.getDataDimensions().at(0), 1);
  gpuCheckEqual(tensorTable.getDataDimensions().at(1), 3);
  gpuCheckEqual(tensorTable.getDataDimensions().at(2), 3);
  gpuCheckEqual(tensorTable.getDataTensorBytes(), test);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeSortIndicesViewFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

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
  indices_sort_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(indices_sort_ptr->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_sortTensorDataGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(1);
  select_labels_values(0) = 0;
  TensorDataGpuPrimitiveT<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(select_labels);
  select_labels_ptr->syncDData(device);

  // make the expected sorted tensor
  TensorArrayGpu8<char> sorted_data[] = { TensorArrayGpu8<char>("24"),  TensorArrayGpu8<char>("25"),  TensorArrayGpu8<char>("26"),  TensorArrayGpu8<char>("21"),  TensorArrayGpu8<char>("22"),  TensorArrayGpu8<char>("23"),  TensorArrayGpu8<char>("18"),  TensorArrayGpu8<char>("19"),  TensorArrayGpu8<char>("20"),  TensorArrayGpu8<char>("15"),  TensorArrayGpu8<char>("16"),  TensorArrayGpu8<char>("17"),  TensorArrayGpu8<char>("12"),  TensorArrayGpu8<char>("13"),  TensorArrayGpu8<char>("14"),  TensorArrayGpu8<char>("9"),  TensorArrayGpu8<char>("10"),  TensorArrayGpu8<char>("11"),  TensorArrayGpu8<char>("6"),  TensorArrayGpu8<char>("7"),  TensorArrayGpu8<char>("8"),  TensorArrayGpu8<char>("3"),  TensorArrayGpu8<char>("4"),  TensorArrayGpu8<char>("5"),  TensorArrayGpu8<char>("0"),  TensorArrayGpu8<char>("1"),  TensorArrayGpu8<char>("2") };
  Eigen::TensorMap<Eigen::Tensor<TensorArrayGpu8<char>, 3>> tensor_sorted_values(sorted_data, Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  std::cout << "test_sortTensorDataGpuClassT\n" << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), i);
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), nlabels - i - 1);
    std::cout << "axis_3_ptr->getLabels() Predicted: " << axis_3_ptr->getLabels()(0, i) << " Expected: " << nlabels - i - 1 << std::endl;
    //gpuCheckEqual(axis_3_ptr->getLabels()(0, i), nlabels - i - 1); //FIXME
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        std::cout << "Predicted: " << tensorTable.getData()(i, j, k) << " Expected: " << tensor_sorted_values(i, j, k) << std::endl;
        //gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_sorted_values(i, j, k)); //FIXME
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, select_labels_ptr, sortOrder::DESC, device);

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), i);
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), nlabels - i - 1);
    std::cout << "axis_3_ptr->getLabels() Predicted: " << axis_3_ptr->getLabels()(0, i) << " Expected: " << nlabels - i - 1 << std::endl;
    //gpuCheckEqual(axis_3_ptr->getLabels()(0, i), nlabels - i - 1); // FIXME
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        std::cout << "Predicted: " << tensorTable.getData()(i, j, k) << " Expected: " << tensor_sorted_values(i, j, k) << std::endl;
        //gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_sorted_values(i, j, k)); // FIXME
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_updateSelectTensorDataValues1Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        update_values(i, j, k).setTensorArray(std::to_string(100));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);
  
  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  values_new_ptr->syncDData(device);

  // Test update
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_old_ptr;
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(iter)));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Test update
  values_old_ptr.reset();
  tensorTable.updateSelectTensorDataValues(values_new_ptr, values_old_ptr, device);
  values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(iter)));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_updateSelectTensorDataValues2Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data and the update values
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        update_values(i, j, k).setTensorArray(std::to_string(100));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  values_new_ptr->syncDData(device);

  // Test update
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_old(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_old.setData();
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_old_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_old);
  values_old_ptr->syncDData(device);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(iter)));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Test update
  values_old_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_old);
  values_old_ptr->syncDData(device);
  tensorTable.updateSelectTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr->getDataPointer(), device);
  values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(iter)));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
        ++iter;
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_updateTensorDataValuesGpu()
{
	// setup the table
	TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

	// Initialize the device
	cudaStream_t stream;
	gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
	std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
	std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
	std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
	tensorTable.addTensorAxis(axis_1_ptr);
	tensorTable.addTensorAxis(axis_2_ptr);
	tensorTable.addTensorAxis(axis_3_ptr);
	tensorTable.setAxes(device);

	// setup the tensor data and the update values
	Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	Eigen::Tensor<TensorArrayGpu8<char>, 3> update_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	int iter = 0;
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				tensor_values(i, j, k).setTensorArray(std::to_string(iter));
				update_values(i, j, k).setTensorArray(std::to_string(100));
				++iter;
			}
		}
	}
	tensorTable.setData(tensor_values);
	TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
	values_new.setData(update_values);
	std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);

	// sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
	values_new_ptr->syncDData(device);

	// Test update
	std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_old_ptr;
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
	gpuErrchk(cudaStreamSynchronize(stream));
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				gpuCheckEqual(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels), tensor_values(i, j, k));
				gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
		gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
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
	tensorTable.syncAxesAndIndicesDData(device);
	tensorTable.syncDData(device);
	tensorTable.storeTensorTableBinary("", device);
	tensorTable.setData();
	tensorTable.setNotInMemoryDataStatus(true, false);
	tensorTable.syncNotInMemoryDData(device);
	tensorTable.setIsModifiedDataStatus(true, false);
	tensorTable.syncIsModifiedDData(device);
	tensorTable.syncDData(device);

	// Test update
	values_old_ptr.reset();
	tensorTable.updateTensorDataValues(values_new_ptr->getDataPointer(), values_old_ptr, device);
	values_old_ptr->syncHData(device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
	gpuErrchk(cudaStreamSynchronize(stream));
	for (int k = 0; k < nlabels; ++k) {
		for (int j = 0; j < nlabels; ++j) {
			for (int i = 0; i < nlabels; ++i) {
				gpuCheckEqual(values_old_ptr->getData()(i + j * nlabels + k * nlabels * nlabels), tensor_values(i, j, k));
				gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
			}
		}
	}

	// Test for the in_memory and is_modified attributes
	for (int i = 0; i < nlabels; ++i) {
		gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
		gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
		gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
		gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
	}

	gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeAppendIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  //std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", 1, 0));
  axis_3_ptr->setDimensions(dimensions3);
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);

  // test the making the append indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_ptr;
  tensorTable.makeAppendIndices("1", nlabels, indices_ptr, device);
  indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(indices_ptr->getData()(i), nlabels + i + 1);
  }

  // test the making the append indices
  indices_ptr.reset();
  tensorTable.makeAppendIndices("3", nlabels, indices_ptr, device);
  indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(indices_ptr->getData()(i), i + 1);
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_appendToIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  indices_new_ptr->syncDData(device);

  // test appendToIndices
  tensorTable.appendToIndices("1", indices_new_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels + nlabels - 1);
  for (int i = 0; i < nlabels + nlabels - 1; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 1);
    if (i < nlabels) {
      gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), 0);
    }
  }

  // check the existing indices
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getShardId().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardIndices().at("2")->getData()(i), i + 1);
  }
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getIndices().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getShardId().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardIndices().at("3")->getData()(i), i + 1);
  }

  // Check the dimensions and tensor size
  Eigen::array<Eigen::Index, 3> dimensions_test = { nlabels + nlabels - 1, nlabels, nlabels };
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), (nlabels + nlabels - 1) * nlabels * nlabels);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_appendToAxis1Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j).setTensorArray(std::to_string(i));
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);

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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  labels_new_ptr->syncDData(device);
  values_new_ptr->syncDData(device);
  indices_new_ptr->syncDData(device);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesHData(device);
  tensorTable.syncHData(device);
  indices_new_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  gpuCheck(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      gpuCheckEqual(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }
  gpuCheckEqual(indices_new_ptr->getData()(0), nlabels + 1);

  // test the expected dimensions
  Eigen::array<Eigen::Index, 3> dimensions_test = { nlabels + 1, nlabels, nlabels };
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), (nlabels + 1) * nlabels * nlabels);

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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);
  indices_new_ptr->setDataStatus(false, true);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesHData(device);
  tensorTable.syncHData(device);
  indices_new_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), labels1(i));
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  gpuCheck(axis_1_ptr->getLabels()(0, nlabels), 3);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      gpuCheckEqual(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }
  gpuCheckEqual(indices_new_ptr->getData()(0), nlabels + 1);

  // test the expected dimensions
  dimensions_test = Eigen::array<Eigen::Index, 3>({ nlabels + 1, nlabels, nlabels });
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), (nlabels + 1) * nlabels * nlabels);

  // Check that the binarized data was written correctly
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.syncDData(device);

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);

  tensorTable.loadTensorTableBinary("", device);
  // Test the new TensorTable
  tensorTable.syncHData(device);
  indices_new_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      gpuCheckEqual(tensorTable.getData()(nlabels, i, j), update_values(0, i, j));
    }
  }

  // test the expected dimensions
  dimensions_test = Eigen::array<Eigen::Index, 3>({ nlabels + 1, nlabels, nlabels });
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), (nlabels + 1) * nlabels * nlabels);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_appendToAxis2Gpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", 1, 0));
  axis_1_ptr->setDimensions(dimensions1);
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);
  tensorTable.setData();

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  int iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(iter));
        ++iter;
      }
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  values_new.setData(tensor_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);

  // setup the new axis labels
  TensorDataGpuPrimitiveT<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ 1, nlabels }));
  labels_new.setData(labels1);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 2>>(labels_new);

  // setup the new indices
  TensorDataGpuPrimitiveT<int, 1> indices_new(Eigen::array<Eigen::Index, 1>({ nlabels }));
  indices_new.setData();
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_new_ptr = std::make_shared<TensorDataGpuPrimitiveT<int, 1>>(indices_new);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  labels_new_ptr->syncDData(device);
  values_new_ptr->syncDData(device);
  indices_new_ptr->syncDData(device);

  // test appendToAxis
  tensorTable.appendToAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesHData(device);
  tensorTable.syncHData(device);
  indices_new_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), labels1(i));
    gpuCheckEqual(indices_new_ptr->getData()(i), i + 1);
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // test the expected dimensions
  Eigen::array<Eigen::Index, 3> dimensions_test = { nlabels, nlabels, nlabels };
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), nlabels * nlabels * nlabels);

  // Check that the binarized data was written correctly
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.syncDData(device);

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);

  tensorTable.loadTensorTableBinary("", device);
  // Test the new TensorTable  
  tensorTable.syncAxesHData(device);
  tensorTable.syncHData(device);
  indices_new_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), labels1(i));
    gpuCheckEqual(indices_new_ptr->getData()(i), i + 1);
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // test the expected dimensions
  dimensions_test = Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels });
  gpuCheckEqual(tensorTable.getDimensions(), dimensions_test);
  gpuCheckEqual(tensorTable.getTensorSize(), nlabels * nlabels * nlabels);


  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeIndicesViewSelectFromIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  indices_to_select_ptr->syncDData(device);

  // test makeIndicesViewSelectFromIndices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_select_ptr;
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, true, device);
  indices_select_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i > 1)
      gpuCheckEqual(indices_select_ptr->getData()(i), 1);
    else
      gpuCheckEqual(indices_select_ptr->getData()(i), 0);
  }
  indices_select_ptr.reset();
  tensorTable.makeIndicesViewSelectFromIndices("1", indices_select_ptr, indices_to_select_ptr, false, device);
  indices_select_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i <= 1)
      gpuCheckEqual(indices_select_ptr->getData()(i), 1);
    else
      gpuCheckEqual(indices_select_ptr->getData()(i), 0);
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_deleteFromIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
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
  tensorTable.syncAxesAndIndicesDData(device);
  indices_to_select_ptr->syncDData(device);

  // test deleteFromIndices
  tensorTable.deleteFromIndices("1", indices_to_select_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 2);
    }
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 1);
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeSelectIndicesFromIndicesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  indices_to_select_ptr->syncDData(device);

  // test the selection indices
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select_ptr;
  tensorTable.makeSelectIndicesFromIndices("1", indices_to_select_ptr, indices_select_ptr, device);
  indices_select_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i % 2 == 0)
          gpuCheckEqual(indices_select_ptr->getData()(i, j, k), 1);
        else
          gpuCheckEqual(indices_select_ptr->getData()(i, j, k), 0);
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_deleteFromAxisGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  Eigen::Tensor<TensorArrayGpu8<char>, 3> new_values(Eigen::array<Eigen::Index, 3>({ nlabels - 1, nlabels, nlabels }));
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
        if (i != 1) {
          new_values(iter, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  indices_to_select_ptr->syncDData(device);

  // test deleteFromAxis
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values.setData();
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values);
  values_ptr->syncDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_ptr;
  tensorTable.deleteFromAxis("1", indices_to_select_ptr, labels_ptr, values_ptr->getDataPointer(), device);

  // test the expected indices sizes and values
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  values_ptr->syncHData(device);
  labels_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensorTable.getDimensions().at(tensorTable.getDimFromAxisName("1")), nlabels - 1);
  for (int i = 0; i < nlabels - 1; ++i) {
    if (i == 0) {
      gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 2);
    }
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 1);
  }

  // Test the expected data values
  for (int i = 0; i < nlabels - 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), new_values(i, j, k));
      }
    }
  }

  // Test the expected axis values
  std::vector<int> expected_labels = { 0, 2 };
  for (int i = 0; i < nlabels - 1; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), expected_labels.at(i));
  }

  // Test the expected returned labels
  gpuCheckEqual(labels_ptr->getData()(0, 0), 1);

  // Test the expected returned data
  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(values_ptr->getData()(i, j, k), tensor_values(1, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeIndicesFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // modify the indices view for axis 1
  tensorTable.getIndicesView().at("1")->getData()(0) = 0;

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);

  // test makeIndicesFromIndicesView
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> indices_ptr;
  tensorTable.makeIndicesFromIndicesView("1", indices_ptr, device);
  indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels - 1; ++i) {
    gpuCheckEqual(indices_ptr->getData()(i), i + 2);
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_insertIntoAxisGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the new tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> update_values(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      update_values(0, i, j).setTensorArray(std::to_string(100));
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> values_new(Eigen::array<Eigen::Index, 3>({ 1, nlabels, nlabels }));
  values_new.setData(update_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(values_new);

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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  values_new_ptr->syncDData(device);
  labels_new_ptr->syncDData(device);
  indices_new_ptr->syncDData(device);

  // test insertIntoAxis
  tensorTable.insertIntoAxis("1", labels_new_ptr, values_new_ptr->getDataPointer(), indices_new_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  int iter = 0;
  for (int i = 0; i < nlabels + 1; ++i) {
    // check the axis
    if (i == 2)
      gpuCheckEqual(axis_1_ptr->getLabels()(0, i), 100);
    else
      gpuCheckEqual(axis_1_ptr->getLabels()(0, i), labels1(iter));

    // check the indices
    gpuCheckEqual(tensorTable.getIndices().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    if (i >= nlabels) {
      gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 2);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i - nlabels + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getShardId().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getShardIndices().at("1")->getData()(i), i + 1);
    }

    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        // check the tensor data
        if (i == 2)
          gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
        else
          gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(iter, j, k));
      }
    }
    if (i != 2) ++iter;
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeSparseAxisLabelsFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // Test
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 2>> labels_ptr;
  tensorTable.makeSparseAxisLabelsFromIndicesView(labels_ptr, device);
  labels_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(labels_ptr->getDimensions().at(0), 3);
  gpuCheckEqual(labels_ptr->getDimensions().at(1), nlabels*nlabels*nlabels);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < nlabels*nlabels*nlabels; ++j) {
      gpuCheckEqual(labels_ptr->getData()(i, j), expected_values(i, j));
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeSparseTensorTableGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
      }
    }
  }
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> sparse_data(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  sparse_data.setData(tensor_values);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> sparse_data_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(sparse_data);

  // Test
  std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> sparse_table_ptr;
  sparse_labels_ptr->syncDData(device);
  sparse_data_ptr->syncDData(device);
  sparse_labels_ptr->syncDData(device);
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;
  tensorTable.makeSparseTensorTable(dimensions1, sparse_labels_ptr, sparse_data_ptr, sparse_table_ptr, device);
  sparse_labels_ptr->syncHData(device);
  sparse_data_ptr->syncHData(device);
  sparse_table_ptr->syncAxesAndIndicesHData(device);
  sparse_table_ptr->syncHData(device);
  sparse_labels_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Check for the correct dimensions
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(0), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);

  std::shared_ptr<int[]> labels1_ptr;
  sparse_table_ptr->getAxes().at("Indices")->getLabelsHDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i));
    for (int j = 0; j < nlabels1; ++j) {
      gpuCheckEqual(labels_values(i, j), labels1(i, j));
    }
  }

  // Check the Values axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);

  std::shared_ptr<int[]> labels2_ptr;
  sparse_table_ptr->getAxes().at("Values")->getLabelsHDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  gpuCheckEqual(labels2_values(0, 0), 0);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_getSelectTensorDataAsSparseTensorTableGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // Test
  std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> sparse_table_ptr;
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);
  sparse_table_ptr->syncAxesAndIndicesHData(device);
  sparse_table_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Check for the correct dimensions
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(0), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);

  std::shared_ptr<int[]> labels1_ptr;
  sparse_table_ptr->getAxes().at("Indices")->getLabelsHDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i + 1));
    for (int j = 0; j < nlabels1; ++j) {
      gpuCheckEqual(labels_values(i, j), labels1_expected(i, j));
    }
  }

  // Check the Values axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);

  std::shared_ptr<int[]> labels2_ptr;
  sparse_table_ptr->getAxes().at("Values")->getLabelsHDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, 1);
  gpuCheckEqual(labels2_values(0, 0), 0);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Re-test getSelectTensorDataAsSparseTensorTable
  sparse_table_ptr.reset();
  tensorTable.getSelectTensorDataAsSparseTensorTable(sparse_table_ptr, device);
  sparse_table_ptr->syncAxesAndIndicesHData(device);
  sparse_table_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Check for the correct dimensions
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(0), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getDimensions().at(1), 1);

  // Check the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(sparse_table_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
      }
    }
  }

  // Check the Indices axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getName(), "Indices");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNLabels(), nlabels1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getNDimensions(), 3);
  labels1_ptr.reset();
  sparse_table_ptr->getAxes().at("Indices")->getLabelsHDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels_values3(labels1_ptr.get(), 3, nlabels1);
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(sparse_table_ptr->getAxes().at("Indices")->getDimensions()(i), std::to_string(i + 1));
    for (int j = 0; j < nlabels1; ++j) {
      gpuCheckEqual(labels_values3(i, j), labels1_expected(i, j));
    }
  }

  // Check the Values axes
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getName(), "Values");
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNLabels(), 1);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getNDimensions(), 1);

  labels2_ptr.reset();
  sparse_table_ptr->getAxes().at("Values")->getLabelsHDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values2(labels2_ptr.get(), 1, 1);
  gpuCheckEqual(labels2_values2(0, 0), 0);
  gpuCheckEqual(sparse_table_ptr->getAxes().at("Values")->getDimensions()(0), "Values");

  // Check the indices axis indices
  for (int i = 0; i < nlabels1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Indices")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Indices")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Indices")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Indices")->getData()(i), i + 1);
  }

  // Check the values axis indices
  for (int i = 0; i < 1; ++i) {
    gpuCheckEqual(sparse_table_ptr->getIndices().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIndicesView().at("Values")->getData()(i), i + 1);
    gpuCheckEqual(sparse_table_ptr->getIsModified().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getNotInMemory().at("Values")->getData()(i), 0);
    gpuCheckEqual(sparse_table_ptr->getShardId().at("Values")->getData()(i), 1);
    gpuCheckEqual(sparse_table_ptr->getShardIndices().at("Values")->getData()(i), i + 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_updateTensorDataConstantGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels*nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // setup the update values
  TensorDataGpuClassT<TensorArrayGpu8, char, 1> values_new(Eigen::array<Eigen::Index, 1>({ 1 }));
  values_new.setData();
  values_new.getData()(0).setTensorArray(std::to_string(100));
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 1>> values_new_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 1>>(values_new);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  values_new_ptr->syncDData(device);

  // Test update
  std::shared_ptr<TensorTable<TensorArrayGpu8<char>, Eigen::GpuDevice, 2>> values_old_ptr;
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  values_old_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // reset is_modified attribute
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(1);
  }

  // Revert the operation and test
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  values_old_ptr->syncDData(device);
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // TODO: Test after a selection (see test for TensorOperation TensorUpdateConstant)

  // Write the original data to disk, clear the data, and repeat the tests
  tensorTable.setData(tensor_values);
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Test update
  values_old_ptr.reset();
  tensorTable.updateTensorDataConstant(values_new_ptr, values_old_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  values_old_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test the data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(values_old_ptr->getData()(i + j * nlabels + k * nlabels*nlabels), tensor_values(i, j, k));
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(100)));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  // clear the data
  tensorTable.setData();
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncDData(device);

  // Revert the operation and test
  values_old_ptr->setDataStatus(false, true);
  tensorTable.updateTensorDataFromSparseTensorTable(values_old_ptr, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeShardIndicesFromShardIDsGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable.setAxes(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ nlabels , nlabels , nlabels }));

  // Test for the shard indices
  tensorTable.syncShardIdDData(device);
  tensorTable.syncShardIndicesDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_shard_ptr;
  tensorTable.makeShardIndicesFromShardIDs(indices_shard_ptr, device);
  indices_shard_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(indices_shard_ptr->getData()(i, j, k), 1);
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
  indices_shard_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        gpuCheckEqual(indices_shard_ptr->getData()(i, j, k), indices_test(i, j, k));
      }
    }
  }
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeModifiedShardIDTensorGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // Reshard indices
  tensorTable.syncAxesAndIndicesDData(device);
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);
  tensorTable.setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ nlabels , nlabels , nlabels }));

  // Test the unmodified case
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  Eigen::array<Eigen::Index, 3> shard_data_dimensions;
  int shard_data_size = 0;
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  gpuCheckEqual(slice_indices.size(), 0);
  gpuCheckEqual(shard_data_size, 0);

  // Test the fully modified case
  for (auto& is_modified_map : tensorTable.getIsModified()) {
    is_modified_map.second->getData() = is_modified_map.second->getData().constant(1);
  }
  tensorTable.setIsModifiedDataStatus(true, false);
  tensorTable.syncIsModifiedDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    gpuCheckEqual(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  Eigen::array<Eigen::Index, 3> shard_data_dimensions_test = { nlabels, nlabels, nlabels };
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    gpuCheckEqual(slice_indices_map.first, iter);
    gpuCheckEqual(slice_indices_map.second.first, slice_indices_test.at(slice_indices_map.first).first);
    gpuCheckEqual(slice_indices_map.second.second, slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  gpuCheckEqual(shard_data_size, nlabels * nlabels * nlabels);

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
  tensorTable.syncIsModifiedDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeModifiedShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    gpuCheckEqual(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    gpuCheckEqual(slice_indices_map.first, iter);
    gpuCheckEqual(slice_indices_map.second.first, slice_indices_test.at(slice_indices_map.first).first);
    gpuCheckEqual(slice_indices_map.second.second, slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  gpuCheckEqual(shard_data_size, 8);
  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeNotInMemoryShardIDTensorGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // Reshard indices
  tensorTable.syncAxesAndIndicesDData(device);
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);
  tensorTable.setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ nlabels , nlabels , nlabels }));

  // Test all in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(0);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> shard_id_indices_ptr;
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 0);

  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices;
  Eigen::array<Eigen::Index, 3> shard_data_dimensions;
  int shard_data_size = 0;
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  gpuCheckEqual(slice_indices.size(), 0);
  gpuCheckEqual(shard_data_size, 0);

  // Test not all in memory case and none selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    indices_view_map.second->getData() = indices_view_map.second->getData().constant(0);
  }
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 0);

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  gpuCheckEqual(slice_indices.size(), 0);
  gpuCheckEqual(shard_data_size, 0);

  // Test all not in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.resetIndicesView("1", device);
  tensorTable.resetIndicesView("2", device);
  tensorTable.resetIndicesView("3", device);
  //tensorTable.setIndicesViewDataStatus(true, false);
  //tensorTable.syncIndicesViewDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 8);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    gpuCheckEqual(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  std::map<int, std::pair<Eigen::array<Eigen::Index, 3>, Eigen::array<Eigen::Index, 3>>> slice_indices_test;
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  slice_indices_test.emplace(2, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,0 }), Eigen::array<Eigen::Index, 3>({ 1,2,2 })));
  slice_indices_test.emplace(3, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,0 }), Eigen::array<Eigen::Index, 3>({ 2,1,2 })));
  slice_indices_test.emplace(4, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,0 }), Eigen::array<Eigen::Index, 3>({ 1,1,2 })));
  slice_indices_test.emplace(5, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,2 }), Eigen::array<Eigen::Index, 3>({ 2,2,1 })));
  slice_indices_test.emplace(6, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,0,2 }), Eigen::array<Eigen::Index, 3>({ 1,2,1 })));
  slice_indices_test.emplace(7, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,2,2 }), Eigen::array<Eigen::Index, 3>({ 2,1,1 })));
  slice_indices_test.emplace(8, std::make_pair(Eigen::array<Eigen::Index, 3>({ 2,2,2 }), Eigen::array<Eigen::Index, 3>({ 1,1,1 })));
  Eigen::array<Eigen::Index, 3> shard_data_dimensions_test = { nlabels, nlabels, nlabels };
  int iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    gpuCheckEqual(slice_indices_map.first, iter);
    gpuCheckEqual(slice_indices_map.second.first, slice_indices_test.at(slice_indices_map.first).first);
    gpuCheckEqual(slice_indices_map.second.second, slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  gpuCheckEqual(shard_data_size, nlabels * nlabels * nlabels);

  // Test the partially in memory case and all selected case
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < shard_span)
        in_memory_map.second->getData()(i) = 1;
      else
        in_memory_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setNotInMemoryDataStatus(true, false);
  tensorTable.syncNotInMemoryDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    gpuCheckEqual(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    gpuCheckEqual(slice_indices_map.first, iter);
    gpuCheckEqual(slice_indices_map.second.first, slice_indices_test.at(slice_indices_map.first).first);
    gpuCheckEqual(slice_indices_map.second.second, slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  gpuCheckEqual(shard_data_size, 8);
  
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
  tensorTable.syncNotInMemoryDData(device);
  for (auto& indices_view_map : tensorTable.getIndicesView()) {
    for (int i = 0; i < nlabels; ++i) {
      if (i < 1)
        indices_view_map.second->getData()(i) = i + 1;
      else
        indices_view_map.second->getData()(i) = 0;
    }
  }
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewDData(device);
  shard_id_indices_ptr.reset();
  tensorTable.makeNotInMemoryShardIDTensor(shard_id_indices_ptr, device);
  shard_id_indices_ptr->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(shard_id_indices_ptr->getTensorSize(), 1);
  for (int i = 0; i < shard_id_indices_ptr->getTensorSize(); ++i) {
    gpuCheckEqual(shard_id_indices_ptr->getData()(i), i + 1);
  }

  slice_indices.clear();
  shard_data_dimensions = Eigen::array<Eigen::Index, 3>();
  shard_id_indices_ptr->syncDData(device);
  shard_data_size = tensorTable.makeSliceIndicesFromShardIndices(shard_id_indices_ptr, slice_indices, shard_data_dimensions, device);
  slice_indices_test.clear();
  slice_indices_test.emplace(1, std::make_pair(Eigen::array<Eigen::Index, 3>({ 0,0,0 }), Eigen::array<Eigen::Index, 3>({ 2,2,2 })));
  shard_data_dimensions_test = Eigen::array<Eigen::Index, 3>({ 2, 2, 2 });
  iter = 1;
  for (const auto& slice_indices_map : slice_indices) {
    gpuCheckEqual(slice_indices_map.first, iter);
    gpuCheckEqual(slice_indices_map.second.first, slice_indices_test.at(slice_indices_map.first).first);
    gpuCheckEqual(slice_indices_map.second.second, slice_indices_test.at(slice_indices_map.first).second);
    ++iter;
  }
  for (int i = 0; i < 3; ++i) {
    gpuCheckEqual(shard_data_dimensions.at(i), shard_data_dimensions_test.at(i));
  }
  gpuCheckEqual(shard_data_size, 8);

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_makeTensorTableShardFilenameGpu()
{
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;
  gpuCheckEqual(tensorTable.makeTensorTableShardFilename("dir/", "table1", 1), "dir/table1_1.tts");
}

void test_storeAndLoadBinaryGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k).setTensorArray(std::to_string(i + j * nlabels + k * nlabels * nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // Reshard indices
  int shard_span = 2;
  std::map<std::string, int> shard_span_new = { {"1", shard_span}, {"2", shard_span}, {"3", shard_span} };
  tensorTable.setShardSpans(shard_span_new);
  tensorTable.reShardIndices(device);

  // Test store/load for the case of all `is_modified`, all not `not_in_memory`, and selected `indices_view`
  tensorTable.storeTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  tensorTable.syncIsModifiedHData(device);
  tensorTable.syncNotInMemoryHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Reset the in_memory values
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncNotInMemoryDData(device);

  // Load the data
  tensorTable.loadTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  tensorTable.syncIsModifiedHData(device);
  tensorTable.syncNotInMemoryHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 0);
  }

  // Test for the original data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  // Test store/load for the case of partially `is_modified`, all not `not_in_memory`, and partially selected `indices_view`
  for (auto& is_modified_map : tensorTable.getIsModified()) { // Shards 1-7
    for (int i = 0; i < nlabels; ++i) {
      if (i < 2)
        is_modified_map.second->getData()(i) = 1;
      else
        is_modified_map.second->getData()(i) = 0;
    }
  }
  for (auto& indices_view_map : tensorTable.getIndicesView()) { // Shard 1
    for (int i = 0; i < nlabels; ++i) {
      if (i < 1)
        indices_view_map.second->getData()(i) = i + 1;
      else
        indices_view_map.second->getData()(i) = 0;
    }
  }
  tensorTable.syncIsModifiedDData(device);
  tensorTable.setIndicesViewDataStatus(true, false);
  tensorTable.syncIndicesViewDData(device);
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.syncDData(device);

  // Test for the in_memory, is_modified, and indices_view attributes before store
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    if (i < 2) {
      gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 0);
    }
    if (i < 1) {
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
    }
  }

  // Test for the in_memory, is_modified, and indices_view attributes after store
  tensorTable.storeTensorTableBinary("", device);
  tensorTable.syncIsModifiedHData(device);
  tensorTable.syncNotInMemoryHData(device);
  tensorTable.syncIndicesViewHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 0);
    if (i < 1) {
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), i + 1);
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), i + 1);
    }
    else {
      gpuCheckEqual(tensorTable.getIndicesView().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIndicesView().at("2")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getIndicesView().at("3")->getData()(i), 0);
    }
  }

  // Reset the in_memory values and Zero the TensorData
  for (auto& in_memory_map : tensorTable.getNotInMemory()) {
    in_memory_map.second->getData() = in_memory_map.second->getData().constant(1);
  }
  tensorTable.syncNotInMemoryDData(device);
  tensorTable.syncIsModifiedDData(device);
  tensorTable.syncIndicesViewDData(device);
  tensorTable.getData() = tensorTable.getData().constant(TensorArrayGpu8<char>("0"));
  tensorTable.syncDData(device);

  // Load the data
  tensorTable.loadTensorTableBinary("", device);

  // Test for the in_memory and is_modified attributes
  tensorTable.syncIsModifiedHData(device);
  tensorTable.syncNotInMemoryHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int i = 0; i < nlabels; ++i) {
    if (i < shard_span) {
      gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
      gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    }
    else {
      gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 1);
      gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 1);
    }
  }

  // Test for the original data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        if (i < shard_span && j < shard_span && k < shard_span)
          gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k));
        else
          gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>("0"));
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_storeAndLoadTensorTableAxesGpu()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable1;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the axes
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable1.addTensorAxis(axis_1_ptr);
  tensorTable1.addTensorAxis(axis_2_ptr);
  tensorTable1.addTensorAxis(axis_3_ptr);
  tensorTable1.setAxes(device);

  // sync the tensorTable
  tensorTable1.syncAxesAndIndicesDData(device);

  // Store the axes
  tensorTable1.storeTensorTableAxesBinary("", device);

  // Remake empty axes
  tensorTable1.clear();
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", 1, nlabels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", 1, nlabels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", 1, nlabels3)));
  tensorTable1.setAxes(device);

  // sync the tensorTable
  tensorTable1.syncAxesAndIndicesDData(device);

  // Load the axes
  tensorTable1.loadTensorTableAxesBinary("", device);

  // Test for the correct axes data
  std::shared_ptr<int[]> labels1_ptr;
  tensorTable1.getAxes().at("1")->getLabelsHDataPointer(labels1_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels1_values(labels1_ptr.get(), 1, nlabels1);
  for (int j = 0; j < nlabels1; ++j) {
    gpuCheckEqual(labels1_values(0, j), labels1(0, j));
  }

  std::shared_ptr<int[]> labels2_ptr;
  tensorTable1.getAxes().at("2")->getLabelsHDataPointer(labels2_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels2_values(labels2_ptr.get(), 1, nlabels2);
  for (int j = 0; j < nlabels2; ++j) {
    gpuCheckEqual(labels2_values(0, j), labels2(0, j));
  }

  std::shared_ptr<int[]> labels3_ptr;
  tensorTable1.getAxes().at("3")->getLabelsHDataPointer(labels3_ptr);
  Eigen::TensorMap<Eigen::Tensor<int, 2>> labels3_values(labels3_ptr.get(), 1, nlabels3);
  for (int j = 0; j < nlabels3; ++j) {
    gpuCheckEqual(labels3_values(0, j), labels3(0, j));
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_getCsvDataRowGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  std::vector<std::string> row_0_test, row_1_test, row_4_test;
  int row_iter = 0;
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        int value = i + j * nlabels + k * nlabels*nlabels;
        tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(value));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  // TODO: also test char and tensorArray types
  std::vector<std::string> row_0 = tensorTable.getCsvDataRow(0);
  std::vector<std::string> row_1 = tensorTable.getCsvDataRow(1);
  std::vector<std::string> row_4 = tensorTable.getCsvDataRow(4);
  gpuCheckEqual(row_0.size(), nlabels);
  gpuCheckEqual(row_1.size(), nlabels);
  gpuCheckEqual(row_4.size(), nlabels);
  std::cout << "Passing/Failing test_getCsvDataRowGpuClassT\n" << std::endl;
  for (int i = 0; i < nlabels; ++i) {
    std::cout << "row_0.at(" << i << "): [Test] " << row_0.at(i) << " [Expected] " << row_0_test.at(i) << std::endl;
    std::cout << "row_1.at(" << i << "): [Test] " << row_1.at(i) << " [Expected] " << row_1_test.at(i) << std::endl;
    std::cout << "row_4.at(" << i << "): [Test] " << row_4.at(i) << " [Expected] " << row_4_test.at(i) << std::endl;
    //gpuCheckEqual(row_0.at(i), row_0_test.at(i)); // FIXME: issue with \0?
    //gpuCheckEqual(row_1.at(i), row_1_test.at(i));
    //gpuCheckEqual(row_4.at(i), row_4_test.at(i));
  }

  // Make the expected labels row values
  std::map<std::string, std::vector<std::string>> labels_row_0_test = { {"2", {"0"}}, {"3", {"0"}} };
  std::map<std::string, std::vector<std::string>> labels_row_1_test = { {"2", {"1"}}, {"3", {"0"}} };
  std::map<std::string, std::vector<std::string>> labels_row_4_test = { {"2", {"1"}}, {"3", {"1"}} };

  // Test getCsvAxesLabelsRow
  std::map<std::string, std::vector<std::string>> labels_row_0 = tensorTable.getCsvAxesLabelsRow(0);
  std::map<std::string, std::vector<std::string>> labels_row_1 = tensorTable.getCsvAxesLabelsRow(1);
  std::map<std::string, std::vector<std::string>> labels_row_4 = tensorTable.getCsvAxesLabelsRow(4);
  gpuCheck(labels_row_0.size(), 2);
  gpuCheck(labels_row_1.size(), 2);
  gpuCheck(labels_row_4.size(), 2);
  for (int i = 2; i < 4; ++i) {
    std::string axis_name = std::to_string(i);
    for (int j = 0; j < 1; ++j) {
      gpuCheckEqual(labels_row_0.at(axis_name).at(j), labels_row_0_test.at(axis_name).at(j));
      gpuCheckEqual(labels_row_1.at(axis_name).at(j), labels_row_1_test.at(axis_name).at(j));
      gpuCheckEqual(labels_row_4.at(axis_name).at(j), labels_row_4_test.at(axis_name).at(j));
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_insertIntoTableFromCsvGpuClassT()
{
  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data, the new tensor data from csv, and the new axes labels from csv
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, 1, 1 }));
  Eigen::Tensor<std::string, 2> new_values_str(Eigen::array<Eigen::Index, 2>({ nlabels, 8 }));
  Eigen::Tensor<std::string, 2> labels_2_str(1, 8);
  Eigen::Tensor<std::string, 2> labels_3_str(1, 8);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        if (j == 0 && k == 0) {
          tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels*nlabels));
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
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);
  tensorTable.insertIntoTableFromCsv(labels_new_str, new_values_str, device);
  tensorTable.syncAxesAndIndicesHData(device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));

  // Test for the tensor data
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels*nlabels)));
      }
    }
  }

  // Test for the axis labels
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(axis_1_ptr->getLabels()(0, i), i);
    gpuCheckEqual(axis_2_ptr->getLabels()(0, i), i);
    gpuCheckEqual(axis_3_ptr->getLabels()(0, i), i);
  }

  // Test for the in_memory and is_modified attributes
  for (int i = 0; i < nlabels; ++i) {
    gpuCheckEqual(tensorTable.getNotInMemory().at("1")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("2")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getNotInMemory().at("3")->getData()(i), 0);
    gpuCheckEqual(tensorTable.getIsModified().at("1")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("2")->getData()(i), 1);
    gpuCheckEqual(tensorTable.getIsModified().at("3")->getData()(i), 1);
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_applyFunctorGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels * nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test using the copy functor
  TensorDataGpuClassT<TensorArrayGpu8, char, 3> results(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  results.setData();
  results.syncDData(device);
  std::shared_ptr<TensorData<TensorArrayGpu8<char>, Eigen::GpuDevice, 3>> results_ptr = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 3>>(results);
  TensorDataCopy<TensorArrayGpu8<char>, Eigen::GpuDevice, 3> sumReduction(results_ptr);
  tensorTable.applyFunctor(sumReduction, device);
  results.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(results.getData()(i, j, k), tensor_values(i, j, k)); // no change
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_reduceTensorDataGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels * nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test using the different reduction functions
  tensorTable.reduceTensorData(reductionFunctions::COUNT, device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(tensorTable.getData()(0, 0, 0), tensor_values(0,0,0)); // no change
  gpuCheckEqual(tensorTable.getData()(nlabels - 1, nlabels - 1, nlabels - 1), tensor_values(nlabels - 1, nlabels - 1, nlabels - 1)); // no change

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_scanTensorDataGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels * nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // sync the tensorTable
  tensorTable.syncAxesAndIndicesDData(device);
  tensorTable.syncDData(device);

  // test using the different reduction functions
  tensorTable.scanTensorData({ "1", "2" }, scanFunctions::CUMSUM, device);
  tensorTable.syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTable.getData()(i, j, k), tensor_values(i, j, k)); // no change
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

void test_copyGpu()
{
  // Initialize the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  // setup the table
  TensorTableGpuClassT<TensorArrayGpu8, char, 3> tensorTable;

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
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2));
  std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>> axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3));
  tensorTable.addTensorAxis(axis_1_ptr);
  tensorTable.addTensorAxis(axis_2_ptr);
  tensorTable.addTensorAxis(axis_3_ptr);
  tensorTable.setAxes(device);

  // setup the tensor data
  Eigen::Tensor<TensorArrayGpu8<char>, 3> tensor_values(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        tensor_values(i, j, k) = TensorArrayGpu8<char>(std::to_string(i + j * nlabels + k * nlabels * nlabels));
      }
    }
  }
  tensorTable.setData(tensor_values);

  // test copy
  auto tensorTableCopy = tensorTable.copyToHost(device);
  tensorTableCopy->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(*(tensorTableCopy.get()), tensorTable);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTableCopy->getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }
  auto tensorTableCopy2 = tensorTable.copyToDevice(device);
  tensorTableCopy2->syncHData(device);
  gpuErrchk(cudaStreamSynchronize(stream));
  gpuCheckEqual(*(tensorTableCopy2.get()), tensorTable);
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        gpuCheckEqual(tensorTableCopy2->getData()(i, j, k), tensor_values(i, j, k));
      }
    }
  }

  gpuErrchk(cudaStreamDestroy(stream));
}

int main(int argc, char** argv)
{	
  test_constructorGpu();
  test_destructorGpu(); 
  test_constructorNameAndAxesGpu();
  test_gettersAndSettersGpu();
  test_initDataGpuClassT();
  test_reShardIndicesGpu();
  test_zeroIndicesViewAndResetIndicesViewGpu();
  test_selectIndicesView1Gpu();
  test_selectIndicesView2Gpu();
  test_broadcastSelectIndicesViewGpu();
  test_extractTensorDataGpuClassT();
  test_selectTensorIndicesGpu();
  test_applyIndicesSelectToIndicesViewGpu();
  test_whereIndicesViewData1Gpu();
  test_whereIndicesViewData2Gpu();
  test_sliceTensorForSortGpu();
  test_sortIndicesViewData1Gpu();
  test_sortIndicesViewData2Gpu();
  test_makeSelectIndicesFromIndicesViewGpu();
  test_getSelectTensorDataFromIndicesViewGpu();
  test_selectTensorDataGpuClassT();
  test_makeSortIndicesViewFromIndicesViewGpu();
  test_sortTensorDataGpuClassT();
  test_updateSelectTensorDataValues1Gpu();
  test_updateSelectTensorDataValues2Gpu();
	test_updateTensorDataValuesGpu();
  test_makeAppendIndicesGpu();
  test_appendToIndicesGpu();
  test_appendToAxis1Gpu();
  test_appendToAxis2Gpu();
  test_makeIndicesViewSelectFromIndicesGpu();
  test_deleteFromIndicesGpu();
  test_makeSelectIndicesFromIndicesGpu();
  test_deleteFromAxisGpu();
  test_makeIndicesFromIndicesViewGpu();
  test_insertIntoAxisGpu();
  test_makeSparseAxisLabelsFromIndicesViewGpu();
  test_makeSparseTensorTableGpu();
  test_getSelectTensorDataAsSparseTensorTableGpu();
  test_updateTensorDataConstantGpu();
  test_makeShardIndicesFromShardIDsGpu();
  test_makeModifiedShardIDTensorGpu();
  test_makeNotInMemoryShardIDTensorGpu();
  test_makeTensorTableShardFilenameGpu();
  test_storeAndLoadBinaryGpu();
  test_storeAndLoadTensorTableAxesGpu();
  test_getCsvDataRowGpuClassT();
  test_insertIntoTableFromCsvGpuClassT();
  test_applyFunctorGpu();
  test_reduceTensorDataGpu();
  test_scanTensorDataGpu();
  test_copyGpu();
  return 0;
}

#endif