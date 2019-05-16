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
  assert(tensorTable.getIndices().at("1")->getData()(nlabels1 -1) == nlabels1);
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
  assert(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
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
  assert(tensorTable.getData()->getTensorSize() == test);

  // Test clear
  tensorTable.clear();
  assert(tensorTable.getAxes().size() == 0);
  assert(tensorTable.getIndices().size() == 0);
  assert(tensorTable.getIndicesView().size() == 0);
  assert(tensorTable.getIsModified().size() == 0);
  assert(tensorTable.getInMemory().size() == 0);
  assert(tensorTable.getIsShardable().size() == 0);
  assert(tensorTable.getDimensions().at(0) == 0);
  assert(tensorTable.getDimensions().at(1)== 0);
  assert(tensorTable.getDimensions().at(2)== 0);
  assert(tensorTable.getData()== nullptr);
}

void test_broadcastSelectIndicesViewGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;
  Eigen::GpuDevice device;

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
        assert(indices_view_bcast->getData()(i,j,k) == indices_test(i, j, k));
      }
    }
  }
}

void test_selectTensorDataGpu()
{
  // setup the table
  TensorTableGpu<float, 3> tensorTable;
  Eigen::GpuDevice device;

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
          tensor_test(i/2, j, k) = value;
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
  tensorTable.selectTensorData(std::make_shared<TensorDataGpu<int, 3>>(indices_select), 
    tensor_select, "1", nlabels / 2, device);
  for (int i = 0; i < nlabels/2; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        assert(tensor_select->getData()(i, j, k) == tensor_test(i, j, k));
      }
    }
  }
}

int main(int argc, char** argv)
{	
  test_constructorGpu();
  test_destructorGpu();
  test_constructorNameAndAxesGpu();
  test_gettersAndSettersGpu();
  test_broadcastSelectIndicesViewGpu();
  test_selectTensorDataGpu();
  return 0;
}

#endif