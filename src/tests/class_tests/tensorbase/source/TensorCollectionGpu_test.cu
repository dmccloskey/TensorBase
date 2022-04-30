/**TODO:  Add copyright*/

#if COMPILE_WITH_CUDA
#include <TensorBase/ml/TensorCollectionGpu.h>

using namespace TensorBase;
using namespace std;

/* TensorCollectionGpu Tests
*/
void test_constructorGpu() 
{
	TensorCollectionGpu* ptr = nullptr;
	TensorCollectionGpu* nullPointer = nullptr;
	ptr = new TensorCollectionGpu();
  gpuCheckNotEqual(ptr, nullPointer);
  delete ptr;
}

void test_destructorGpu()
{
  TensorCollectionGpu* ptr = nullptr;
	ptr = new TensorCollectionGpu();
  delete ptr;
}

void test_comparisonGpu()
{
  // Set up the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  TensorTableGpuPrimitiveT<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes(device);
  std::shared_ptr<TensorTable<char, Eigen::GpuDevice, 3>> tensorTable3_ptr = std::make_shared<TensorTableGpuPrimitiveT<char, 3>>(tensorTable3);

  // Test collection
  TensorCollectionGpu tensorCollection_test("1");
  tensorCollection_test.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection_test.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection_test.addTensorTable(tensorTable3_ptr, "1");

  // Expected collection
  TensorCollectionGpu tensorCollection1("1");
  tensorCollection1.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection1.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection1.addTensorTable(tensorTable3_ptr, "1");

  gpuCheckEqualNoLhsRhsPrint(tensorCollection_test, tensorCollection1); // Control
  tensorCollection1.setName("3");
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection_test, tensorCollection1); // Different names but same data
  tensorCollection1.setName("1");
  tensorCollection1.removeTensorTable("1");
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection_test, tensorCollection1); // Different data but same names
}

void test_gettersAndSettersGpu()
{
  // Set up the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  TensorTableGpuPrimitiveT<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes(device);
  std::shared_ptr<TensorTable<char, Eigen::GpuDevice, 3>> tensorTable3_ptr = std::make_shared<TensorTableGpuPrimitiveT<char, 3>>(tensorTable3);

  TensorCollectionGpu tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // name getter
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNames(), std::vector<std::string>({ "1", "2", "3" }));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNamesFromUserName("1"), std::set<std::string>({ "1", "2", "3" }));

  // table concept getter
  auto tt1_ptr = tensorCollection.getTensorTableConcept("1");
  gpuCheckEqual(tt1_ptr->getName(), tensorTable1_ptr->getName());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getAxes(), tensorTable1_ptr->getAxes());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIndices(), tensorTable1_ptr->getIndices());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIndicesView(), tensorTable1_ptr->getIndicesView());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIsModified(), tensorTable1_ptr->getIsModified());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getNotInMemory(), tensorTable1_ptr->getNotInMemory());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getShardId(), tensorTable1_ptr->getShardId());

  // remove tensor tables
  tensorCollection.removeTensorTable("2");
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNames(), std::vector<std::string>({ "1", "3" }));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNamesFromUserName("1"), std::set<std::string>({ "1", "3" }));

  // clear the collection
  tensorCollection.clear();
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNames(), std::vector<std::string>());
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNamesFromUserName("1"), std::set<std::string>());
}

void test_addTensorTableConceptGpu()
{
  // Set up the device
  cudaStream_t stream;
  gpuErrchk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Eigen::GpuStreamDevice stream_device(&stream, 0);
  Eigen::GpuDevice device(&stream_device);

  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<int, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setConstant(1);
  labels2.setConstant(2);
  labels3.setConstant(3);

  TensorTableGpuPrimitiveT<float, 3> tensorTable1("1");
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable1.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable1.setAxes(device);
  std::shared_ptr<TensorTable<float, Eigen::GpuDevice, 3>> tensorTable1_ptr = std::make_shared<TensorTableGpuPrimitiveT<float, 3>>(tensorTable1);

  TensorTableGpuPrimitiveT<int, 2> tensorTable2("2");
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable2.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable2.setAxes(device);
  std::shared_ptr<TensorTable<int, Eigen::GpuDevice, 2>> tensorTable2_ptr = std::make_shared<TensorTableGpuPrimitiveT<int, 2>>(tensorTable2);

  TensorTableGpuPrimitiveT<char, 3> tensorTable3("3");
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("1", dimensions1, labels1)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("2", dimensions2, labels2)));
  tensorTable3.addTensorAxis((std::shared_ptr<TensorAxis<int, Eigen::GpuDevice>>)std::make_shared<TensorAxisGpuPrimitiveT<int>>(TensorAxisGpuPrimitiveT<int>("3", dimensions3, labels3)));
  tensorTable3.setAxes(device);
  std::shared_ptr<TensorTable<char, Eigen::GpuDevice, 3>> tensorTable3_ptr = std::make_shared<TensorTableGpuPrimitiveT<char, 3>>(tensorTable3);

  TensorCollectionGpu tensorCollection;
  tensorCollection.addTensorTable(tensorTable1_ptr, "1");
  tensorCollection.addTensorTable(tensorTable2_ptr, "1");
  tensorCollection.addTensorTable(tensorTable3_ptr, "1");

  // table concept getter
  const std::shared_ptr<TensorTableConcept<Eigen::GpuDevice>> tt1_ptr = tensorCollection.getTensorTableConcept("1");
  gpuCheckEqual(tt1_ptr->getName(), tensorTable1_ptr->getName());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getAxes(), tensorTable1_ptr->getAxes());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIndices(), tensorTable1_ptr->getIndices());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIndicesView(), tensorTable1_ptr->getIndicesView());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getIsModified(), tensorTable1_ptr->getIsModified());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getNotInMemory(), tensorTable1_ptr->getNotInMemory());
  gpuCheckEqualNoLhsRhsPrint(tt1_ptr->getShardId(), tensorTable1_ptr->getShardId());

  // table concept adder
  tensorCollection.removeTensorTable("1");
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNames(), std::vector<std::string>({ "2", "3" }));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNamesFromUserName("1"), std::set<std::string>({ "2", "3" }));
  tensorCollection.addTensorTableConcept(tt1_ptr, "1");
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNames(), std::vector<std::string>({ "1", "2", "3" }));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTableNamesFromUserName("1"), std::set<std::string>({ "1", "2", "3" }));;

  // test default axes and indices linkage
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("3"), tensorTable1.getAxes().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("3"), tensorTable1.getAxes().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("3"), tensorTable1.getIndices().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("3"), tensorTable1.getIndices().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("3"), tensorTable1.getIndicesView().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("3"), tensorTable1.getIndicesView().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("3"), tensorTable1.getIsModified().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("3"), tensorTable1.getIsModified().at("3"));

  // test linkAxesAndIndicesByUserTableName
  tensorCollection.linkAxesAndIndicesByUserTableName("1", "1");
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("3"), tensorTable1.getAxes().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("2"), tensorTable1.getAxes().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("3"), tensorTable1.getAxes().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("3"), tensorTable1.getIndices().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("2"), tensorTable1.getIndices().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("3"), tensorTable1.getIndices().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("3"), tensorTable1.getIndicesView().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("2"), tensorTable1.getIndicesView().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("3"), tensorTable1.getIndicesView().at("3"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("3"), tensorTable1.getIsModified().at("3"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckNotEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("2"), tensorTable1.getIsModified().at("2"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("3"), tensorTable1.getIsModified().at("3"));

  // test linkAxesAndIndicesByAxisName
  tensorCollection.linkAxesAndIndicesByAxisName({ "1" });
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getAxes().at("1"), tensorTable1.getAxes().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndices().at("1"), tensorTable1.getIndices().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIndicesView().at("1"), tensorTable1.getIndicesView().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("1")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("2")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
  gpuCheckEqualNoLhsRhsPrint(tensorCollection.getTensorTableConcept("3")->getIsModified().at("1"), tensorTable1.getIsModified().at("1"));
}

int main(int argc, char** argv)
{
  test_constructorGpu();
  test_destructorGpu();
  test_comparisonGpu();
  test_gettersAndSettersGpu();
  test_addTensorTableConceptGpu();
  return 0;
}
#endif