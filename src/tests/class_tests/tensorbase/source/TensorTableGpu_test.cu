/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorTableGpu.h>

using namespace TensorBase;
using namespace std;

void test_constructorGpu()
{
  TensorTableGpu<float, 3>* ptr = nullptr;
  TensorTableGpu<float, 3>* nullPointer = nullptr;
  ptr = new TensorTableGpu<float, 3>();
  assert(ptr != nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorTableGpu<float, 3>* ptr = nullptr;
  ptr = new TensorTableGpu<float, 3>();
  delete ptr;
}

void test_constructorNameAndAxesGpu()
{
  TensorTableGpu<float, 3> tensorTable("1");

  assert(tensorTable.getId() == -1);
  assert(tensorTable.getName() == "1");
}

void test_gettersAndSettersGpu()
{
  TensorTableGpu<float, 3> tensorTable;
  // Check defaults
  assert(tensorTable.getId() == -1);
  assert(tensorTable.getName() == "");
  assert(tensorTable.getAxes().size() == 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");

  assert(tensorTable.getId() == 1);
  assert(tensorTable.getName() == "1");

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  assert(tensorTable.getInMemory().at("1")->getData()(0) == 0);
  assert(tensorTable.getIsShardable().at("1")->getData()(0) == 1);

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
  assert(tensorTable.getInMemory().at("2")->getData()(0) == 0);
  assert(tensorTable.getIsShardable().at("2")->getData()(0) == 0);

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
  assert(tensorTable.getInMemory().at("3")->getData()(0) == 0);
  assert(tensorTable.getIsShardable().at("3")->getData()(0) == 0);

  // Test expected axis to dims mapping
  assert(tensorTable.getDimFromAxisName("1") == 0);
  assert(tensorTable.getDimFromAxisName("2") == 1);
  assert(tensorTable.getDimFromAxisName("3") == 2);

  // Test expected tensor dimensions
  assert(tensorTable.getDimensions().at(0) == 2);
  assert(tensorTable.getDimensions().at(1) == 3);
  assert(tensorTable.getDimensions().at(2) == 5);

  // Test expected tensor data values
  assert(tensorTable.getData()->getDimensions().at(0) == 2);
  assert(tensorTable.getData()->getDimensions().at(1) == 3);
  assert(tensorTable.getData()->getDimensions().at(2) == 5);
  size_t test = 2 * 3 * 5 * sizeof(float);
  assert(tensorTable.getData()->getTensorBytes() == test);

  // Test clear
  tensorTable.clear();
  assert(tensorTable.getAxes().size() == 0);
  assert(tensorTable.getIndices().size() == 0);
  assert(tensorTable.getIndicesView().size() == 0);
  assert(tensorTable.getIsModified().size() == 0);
  assert(tensorTable.getInMemory().size() == 0);
  assert(tensorTable.getIsShardable().size() == 0);
  assert(tensorTable.getDimensions().at(0) == 0);
  assert(tensorTable.getDimensions().at(1) == 0);
  assert(tensorTable.getDimensions().at(2) == 0);
  assert(tensorTable.getData() == nullptr);
}

void test_zeroIndicesViewAndResetIndicesViewGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // test null
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
  }

  // test zero
  tensorTable.zeroIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == 0);
  }
  // test reset
  tensorTable.resetIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectIndicesViewGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(nlabels / 2);
  int iter = 0;
  for (int i = 0; i < nlabels; ++i) {
    if (i % 2 == 0) {
      select_labels_values(iter) = i;
      ++iter;
    }
  }
  TensorDataGpu<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ nlabels / 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpu<int, 1>>(select_labels);

  // test the updated view
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
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
  TensorTableGpu<float, 3> tensorTable;
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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // setup the indices test
  Eigen::Tensor<int, 3> indices_test(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        indices_test(i, j, k) = i;
      }
    }
  }

  // test the broadcast indices values
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_view_bcast;
  tensorTable.broadcastSelectIndicesView(indices_view_bcast, "1", device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_view_bcast->getData()(i, j, k) == indices_test(i, j, k));
      }
    }
  }

  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_extractTensorDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);
  TensorDataGpu<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_values);

  // test
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select;
  tensorTable.reduceTensorDataToSelectIndices(std::make_shared<TensorDataGpu<int, 3>>(indices_select),
    tensor_select, "1", nlabels / 2, device);
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
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  TensorDataGpu<float, 3> tensor_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  tensor_select.setData(tensor_select_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select_ptr = std::make_shared<TensorDataGpu<float, 3>>(tensor_select);
  TensorDataGpu<float, 1> values_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  values_select.setData(values_select_values);
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> values_select_ptr = std::make_shared<TensorDataGpu<float, 1>>(values_select);

  // test inequality
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.selectTensorIndicesOnReducedTensorData(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::NOT_EQUAL_TO, logicalModifiers::logicalModifier::NONE, device);
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
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  TensorDataGpu<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select_ptr = std::make_shared<TensorDataGpu<int, 3>>(indices_select);

  // test using the second indices view
  tensorTable.getIndicesView().at("2")->getData()(nlabels - 1) = 0;
  // test for OR within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == nlabels - 1)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  // test for AND within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  tensorTable.getIndicesView().at("2")->getData()(0) = 0;
  // test for OR within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i != 0 && i < nlabels - 1)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);
  }

  tensorTable.resetIndicesView("2", device);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> indices_select_values2(indices_select_ptr->getDataPointer().get(), indices_select_ptr->getDimensions());
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
  // test for AND within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalContinuators::logicalContinuator::AND, logicalContinuators::logicalContinuator::AND, device);
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
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // set up the selection labels
  Eigen::Tensor<int, 1> select_labels_values(2);
  select_labels_values(0) = 0; select_labels_values(1) = 2;
  TensorDataGpu<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpu<int, 1>>(select_labels);

  // set up the selection values
  Eigen::Tensor<float, 1> select_values_values(2);
  select_values_values(0) = 9; select_values_values(1) = 9;
  TensorDataGpu<float, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);

  // test
  tensorTable.whereIndicesView("1", 0, select_labels_ptr,
    std::make_shared<TensorDataGpu<float, 1>>(select_values), logicalComparitors::logicalComparitor::EQUAL_TO, logicalModifiers::logicalModifier::NONE,
    logicalContinuators::logicalContinuator::OR, logicalContinuators::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("2")->getData()(i) == 0);

    // indices view 2
    if (i == 1)
      assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    else
      assert(tensorTable.getIndicesView().at("3")->getData()(i) == 0);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sliceTensorForSortGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // test sliceTensorForSort for axis 2
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 1>> tensor_sort;
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "2", device);
  std::vector<float> tensor_slice_2_test = { 9, 12, 15 };
  for (int i = 0; i < nlabels; ++i) {
    assert(tensor_sort->getData()(i) == tensor_slice_2_test.at(i), 1e-3);
  }

  // test sliceTensorForSort for axis 2
  tensor_sort.reset();
  tensorTable.sliceTensorDataForSort(tensor_sort, "1", 1, "3", device);
  std::vector<float> tensor_slice_3_test = { 9, 10, 11 };
  for (int i = 0; i < nlabels; ++i) {
    assert(tensor_sort->getData()(i) == tensor_slice_3_test.at(i), 1e-3);
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortIndicesViewDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // test sort ASC
  tensorTable.sortIndicesView("1", 0, 1, sortOrder::ASC, device);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", 0, 1, sortOrder::DESC, device);
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
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // Test null
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 3>> indices_select;
  tensorTable.makeSelectIndicesFromIndicesView(indices_select, device);
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
  TensorDataGpu<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpu<int, 1>>(select_labels);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test selected
  indices_select.reset();
  tensorTable.makeSelectIndicesFromIndicesView(indices_select, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_select->getData()(i, j, k) == indices_select_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_getSelectTensorDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // select label 1 from axis 1
  TensorDataGpu<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpu<int, 1>>(select_labels);
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
  TensorDataGpu<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);

  // test for the selected data
  std::shared_ptr<TensorData<float, Eigen::GpuDevice, 3>> tensor_select_ptr;
  tensorTable.getSelectTensorData(tensor_select_ptr, std::make_shared<TensorDataGpu<int, 3>>(indices_select), device);
  assert(tensor_select_ptr->getDimensions() == select_dimensions);
  for (int j = 0; j < nlabels; ++j) {
    for (int k = 0; k < nlabels; ++k) {
      assert(tensor_select_ptr->getData()(0, j, k) == tensor_select_test(0, j, k), 1e-3);
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_selectTensorDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // select label 1 from axis 1
  TensorDataGpu<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 1 }));
  Eigen::Tensor<int, 1> select_labels_values(Eigen::array<Eigen::Index, 1>({ 1 }));
  select_labels_values.setValues({ 1 });
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>> select_labels_ptr = std::make_shared<TensorDataGpu<int, 1>>(select_labels);
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);

  // Test `selectTensorData`
  tensorTable.selectTensorData(device);

  // Test expected axes values
  assert(tensorTable.getAxes().at("1")->getName() == "1");
  assert(tensorTable.getAxes().at("1")->getNLabels() == 1);
  assert(tensorTable.getAxes().at("1")->getDimensions()(0) == "x");
  assert(tensorTable.getIndices().at("1")->getData()(0) == 1);
  assert(tensorTable.getIndicesView().at("1")->getData()(0) == 1);
  assert(tensorTable.getIsModified().at("1")->getData()(0) == 0);
  assert(tensorTable.getInMemory().at("1")->getData()(0) == 0);
  assert(tensorTable.getIsShardable().at("1")->getData()(0) == 1);

  assert(tensorTable.getAxes().at("2")->getName() == "2");
  assert(tensorTable.getAxes().at("2")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("2")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("2")->getDimensions()(0) == "y");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("2")->getData()(i) == 0);
    assert(tensorTable.getInMemory().at("2")->getData()(i) == 0);
    assert(tensorTable.getIsShardable().at("2")->getData()(i) == 0);
  }

  assert(tensorTable.getAxes().at("3")->getName() == "3");
  assert(tensorTable.getAxes().at("3")->getNLabels() == nlabels);
  assert(tensorTable.getAxes().at("3")->getNDimensions() == 1);
  assert(tensorTable.getAxes().at("3")->getDimensions()(0) == "z");
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndices().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
    assert(tensorTable.getIsModified().at("3")->getData()(i) == 0);
    assert(tensorTable.getInMemory().at("3")->getData()(i) == 0);
    assert(tensorTable.getIsShardable().at("3")->getData()(i) == 0);
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
  assert(tensorTable.getData()->getDimensions().at(0) == 1);
  assert(tensorTable.getData()->getDimensions().at(1) == 3);
  assert(tensorTable.getData()->getDimensions().at(2) == 3);
  size_t test = 1 * 3 * 3 * sizeof(float);
  assert(tensorTable.getData()->getTensorBytes() == test);
}

void test_makeSortIndicesViewFromIndicesViewGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.makeSortIndicesViewFromIndicesView(indices_sort_ptr, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(indices_sort_ptr->getData()(i, j, k) == indices_test(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

void test_sortTensorDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisGpu<int>>(TensorAxisGpu<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

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
  tensorTable.getData()->setData(tensor_values);

  // sort each of the axes
  tensorTable.sortIndicesView("1", 0, 0, sortOrder::DESC, device);

  // make the expected sorted tensor
  float sorted_data[] = { 24, 25, 26, 21, 22, 23, 18, 19, 20, 15, 16, 17, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2 };
  Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor_sorted_values(sorted_data, Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));

  // Test for sorted tensor data and reset indices view
  tensorTable.sortTensorData(device);
  for (int i = 0; i < nlabels; ++i) {
    assert(tensorTable.getIndicesView().at("1")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("2")->getData()(i) == i + 1);
    assert(tensorTable.getIndicesView().at("3")->getData()(i) == i + 1);
  }
  for (int k = 0; k < nlabels; ++k) {
    for (int j = 0; j < nlabels; ++j) {
      for (int i = 0; i < nlabels; ++i) {
        assert(tensorTable.getData()->getData()(i, j, k) == tensor_sorted_values(i, j, k));
      }
    }
  }
  assert(cudaStreamDestroy(stream) == cudaSuccess);
}

int main(int argc, char** argv)
{	
  test_constructorGpu();
  test_destructorGpu(); 
  test_constructorNameAndAxesGpu();
  test_zeroIndicesViewAndResetIndicesViewGpu();
  test_selectIndicesViewGpu();
  test_broadcastSelectIndicesViewGpu();
  test_extractTensorDataGpu();
  test_selectTensorIndicesGpu();
  test_applyIndicesSelectToIndicesViewGpu();
  test_whereIndicesViewDataGpu();
  test_sliceTensorForSortGpu();
  test_sortIndicesViewDataGpu();
  test_makeSelectIndicesFromIndicesViewGpu();
  test_getSelectTensorDataGpu();
  test_selectTensorDataGpu();
  test_makeSortIndicesViewFromIndicesViewGpu();
  test_sortTensorDataGpu();
  return 0;
}

#endif