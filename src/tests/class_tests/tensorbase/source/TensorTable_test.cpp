/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTable test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorTableDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorTable)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
  TensorTableDefaultDevice<float, 3>* nullPointer = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
  TensorTableDefaultDevice<float, 3>* ptr = nullptr;
	ptr = new TensorTableDefaultDevice<float, 3>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructorNameAndAxesDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable("1");

  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorTableDefaultDevice<float, 3> tensorTable;
  // Check defaults
  BOOST_CHECK_EQUAL(tensorTable.getId(), -1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);

  // Check getters/setters
  tensorTable.setId(1);
  tensorTable.setName("1");

  BOOST_CHECK_EQUAL(tensorTable.getId(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getName(), "1");

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // Test expected axes values
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getName(), "1");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0, 0), 1);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getLabels()(0,0), "x-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNLabels(), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("1")->getDimensions()(0), "x");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("1")->getData()(nlabels1 -1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("1")->getData()(nlabels1 - 1), nlabels1);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("1")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("1")->getData()(0), 1);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getName(), "2");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), 2);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getLabels()(0, 0), "y-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNLabels(), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("2")->getDimensions()(0), "y");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("2")->getData()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("2")->getData()(nlabels2 - 1), nlabels2);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("2")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("2")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("2")->getData()(0), 0);

  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getName(), "3");
  //BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), 3);
  ////BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getLabels()(0, 0), "z-axis");
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNLabels(), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensorTable.getAxes().at("3")->getDimensions()(0), "z");
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().at("3")->getData()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(0), 1);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().at("3")->getData()(nlabels3 - 1), nlabels3);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().at("3")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().at("3")->getData()(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().at("3")->getData()(0), 0);

  // Test expected axis to dims mapping
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("1"), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("2"), 1);
  BOOST_CHECK_EQUAL(tensorTable.getDimFromAxisName("3"), 2);

  // Test expected tensor dimensions
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 5);

  // Test expected tensor data values
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(0), 2);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(1), 3);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getDimensions().at(2), 5);
  size_t test = 2 * 3 * 5 * sizeof(float);
  BOOST_CHECK_EQUAL(tensorTable.getData()->getTensorSize(), test);

  // Test clear
  tensorTable.clear();
  BOOST_CHECK_EQUAL(tensorTable.getAxes().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndices().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIndicesView().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsModified().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getInMemory().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getIsShardable().size(), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(0), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(1), 0);
  BOOST_CHECK_EQUAL(tensorTable.getDimensions().at(2), 0);
  BOOST_CHECK_EQUAL(tensorTable.getData(), nullptr);
}

BOOST_AUTO_TEST_CASE(zeroIndicesViewAndResetIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
  tensorTable.setAxes();

  // test null
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_1(tensorTable.getIndicesView().at("1")->getDataPointer().get(), tensorTable.getIndicesView().at("1")->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_view_1(i), i+1);
  }

  // test zero
  tensorTable.zeroIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_view_1(i), 0);
  }
  // test reset
  tensorTable.resetIndicesView("1", device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_view_1(i), i+1);
  }
}

BOOST_AUTO_TEST_CASE(selectIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ nlabels/2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // test the updated view
  tensorTable.selectIndicesView("1", 0, select_labels_ptr, device);
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_1(tensorTable.getIndicesView().at("1")->getDataPointer().get(), tensorTable.getIndicesView().at("1")->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    if (i%2==0)
      BOOST_CHECK_EQUAL(indices_view_1(i), i + 1);
    else
      BOOST_CHECK_EQUAL(indices_view_1(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(broadcastSelectIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_view_bcast;
  tensorTable.broadcastSelectIndicesView(indices_view_bcast, "1", device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK(indices_view_bcast->getData()(i,j,k), indices_test(i, j, k));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(extractTensorDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  TensorDataDefaultDevice<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_values);
  
  // test
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensor_select;
  tensorTable.extractTensorData(std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_select), 
    tensor_select, "1", nlabels / 2, device);
  for (int i = 0; i < nlabels/2; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        BOOST_CHECK_CLOSE(tensor_select->getData()(i, j, k), tensor_test(i, j, k), 1e-3);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(selectTensorIndicesDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  TensorDataDefaultDevice<float, 3> tensor_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  tensor_select.setData(tensor_select_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 3>> tensor_select_ptr = std::make_shared<TensorDataDefaultDevice<float, 3>>(tensor_select);
  TensorDataDefaultDevice<float, 1> values_select(Eigen::array<Eigen::Index, 1>({ nlabels }));
  values_select.setData(values_select_values);
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> values_select_ptr = std::make_shared<TensorDataDefaultDevice<float, 1>>(values_select);

  // test inequality
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select;
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::NOT_EQUAL_TO, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
      }
    }
  }

  // test equality
  indices_select.reset();
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::EQUAL_TO, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) == 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than
  indices_select.reset();
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) < 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test less than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::LESS_THAN_OR_EQUAL_TO, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) <= 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than
  indices_select.reset();
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) > 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }

  // test greater than or equal to
  indices_select.reset();
  tensorTable.selectTensorIndices(indices_select, values_select_ptr, tensor_select_ptr,
    "1", nlabels, logicalComparitors::logicalComparitor::GREATER_THAN_OR_EQUAL_TO, logicalComparitors::logicalModifier::NONE, device);
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (tensor_select_values(i, j, k) >= 2.0)
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 1);
        else
          BOOST_CHECK_EQUAL(indices_select->getData()(i, j, k), 0);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(applyIndicesSelectToIndicesViewDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  TensorDataDefaultDevice<int, 3> indices_select(Eigen::array<Eigen::Index, 3>({ nlabels, nlabels, nlabels }));
  indices_select.setData(indices_select_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 3>> indices_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 3>>(indices_select);

  // test using the second indices view
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_2(tensorTable.getIndicesView().at("2")->getDataPointer().get(), tensorTable.getIndicesView().at("2")->getDimensions());
  
  indices_view_2(nlabels - 1) = 0;
  // test for OR within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalComparitors::logicalComparitors::logicalContinuator::OR, logicalComparitors::logicalComparitors::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == nlabels - 1)
      BOOST_CHECK_EQUAL(indices_view_2(i), 0);
    else
      BOOST_CHECK_EQUAL(indices_view_2(i), i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  indices_view_2(0) = 0;
  // test for AND within continuator and OR prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalComparitors::logicalComparitors::logicalContinuator::AND, logicalComparitors::logicalComparitors::logicalContinuator::OR, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i == 0)
      BOOST_CHECK_EQUAL(indices_view_2(i), 0);
    else
      BOOST_CHECK_EQUAL(indices_view_2(i), i + 1);
  }

  tensorTable.resetIndicesView("2", device);
  indices_view_2(0) = 0;
  // test for OR within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalComparitors::logicalComparitors::logicalContinuator::OR, logicalComparitors::logicalComparitors::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i != 0 && i < nlabels - 1)
      BOOST_CHECK_EQUAL(indices_view_2(i), i + 1);
    else
      BOOST_CHECK_EQUAL(indices_view_2(i), 0);
  }

  tensorTable.resetIndicesView("2", device);
  Eigen::TensorMap<Eigen::Tensor<int, 3>> indices_select_values2(indices_select_ptr->getDataPointer().get(), indices_select_ptr->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    for (int j = 0; j < nlabels; ++j) {
      for (int k = 0; k < nlabels; ++k) {
        if (i == j && j == k && k == i
          && i < nlabels - 1 && j < nlabels - 1 && k < nlabels - 1) // the first 2 diagonal elements
          indices_select_values2(i, j, k) = 1;
        else if (j == 0)
          indices_select_values2(i, j, k) = 1; // all elements along the first index of the selection dim
        else
          indices_select_values2(i, j, k) = 0;
      }
    }
  }
  // test for AND within continuator and AND prepend continuator
  tensorTable.applyIndicesSelectToIndicesView(indices_select_ptr, "1", "2", logicalComparitors::logicalComparitors::logicalContinuator::AND, logicalComparitors::logicalComparitors::logicalContinuator::AND, device);
  for (int i = 0; i < nlabels; ++i) {
    if (i==0)
      BOOST_CHECK_EQUAL(indices_view_2(i), i+1);
    else
      BOOST_CHECK_EQUAL(indices_view_2(i), 0);
  }

  // TODO: lacking code coverage for the case of TDim = 2
}

BOOST_AUTO_TEST_CASE(whereIndicesViewDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  TensorDataDefaultDevice<int, 1> select_labels(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_labels.setData(select_labels_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_labels);

  // set up the selection values
  Eigen::Tensor<float, 1> select_values_values(2);
  select_values_values(0) = 9; select_values_values(1) = 9;
  TensorDataDefaultDevice<float, 1> select_values(Eigen::array<Eigen::Index, 1>({ 2 }));
  select_values.setData(select_values_values);

  // test
  tensorTable.whereIndicesView("1", 0, select_labels_ptr,
    std::make_shared<TensorDataDefaultDevice<float, 1>>(select_values), logicalComparitors::logicalComparitor::EQUAL_TO, logicalComparitors::logicalModifier::NONE,
    logicalComparitors::logicalComparitors::logicalContinuator::OR, logicalComparitors::logicalComparitors::logicalContinuator::AND, device);
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_1(tensorTable.getIndicesView().at("1")->getDataPointer().get(), tensorTable.getIndicesView().at("1")->getDimensions());
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_2(tensorTable.getIndicesView().at("2")->getDataPointer().get(), tensorTable.getIndicesView().at("2")->getDimensions());
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_3(tensorTable.getIndicesView().at("3")->getDataPointer().get(), tensorTable.getIndicesView().at("3")->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    // indices view 1
    BOOST_CHECK_EQUAL(indices_view_1(i), i + 1); // Unchanged

    // indices view 2
    if (i == 2)
      BOOST_CHECK_EQUAL(indices_view_2(i), i + 1);
    else
      BOOST_CHECK_EQUAL(indices_view_2(i), 0);

    // indices view 2
    if (i == 1)
      BOOST_CHECK_EQUAL(indices_view_3(i), i + 1);
    else
      BOOST_CHECK_EQUAL(indices_view_3(i), 0);
  }
}

BOOST_AUTO_TEST_CASE(sliceTensorForSortDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 1>> tensor_sort;
  tensorTable.sliceTensorForSort(tensor_sort, "1", 1, "2", device);
  Eigen::TensorMap<Eigen::Tensor<float, 1>> tensor_sort_2(tensor_sort->getDataPointer().get(), tensor_sort->getDimensions());
  std::vector<float> tensor_slice_2_test = {9, 12, 15};
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_CLOSE(tensor_sort_2(i), tensor_slice_2_test.at(i), 1e-3);
  }

  // test sliceTensorForSort for axis 2
  tensor_sort.reset();
  tensorTable.sliceTensorForSort(tensor_sort, "1", 1, "3", device);
  Eigen::TensorMap<Eigen::Tensor<float, 1>> tensor_sort_3(tensor_sort->getDataPointer().get(), tensor_sort->getDimensions());
  std::vector<float> tensor_slice_3_test = { 9, 10, 11 };
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_CLOSE(tensor_sort_3(i), tensor_slice_3_test.at(i), 1e-3);
  }
}

BOOST_AUTO_TEST_CASE(sortIndicesViewDataDefaultDevice)
{
  // setup the table
  TensorTableDefaultDevice<float, 3> tensorTable;
  Eigen::DefaultDevice device;

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
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("1", dimensions1, labels1)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("2", dimensions2, labels2)));
  tensorTable.addTensorAxis(std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("3", dimensions3, labels3)));
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
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_1(tensorTable.getIndicesView().at("1")->getDataPointer().get(), tensorTable.getIndicesView().at("1")->getDimensions());
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_2(tensorTable.getIndicesView().at("2")->getDataPointer().get(), tensorTable.getIndicesView().at("2")->getDimensions());
  Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view_3(tensorTable.getIndicesView().at("3")->getDataPointer().get(), tensorTable.getIndicesView().at("3")->getDimensions());
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_view_1(i), i + 1);
    BOOST_CHECK_EQUAL(indices_view_3(i), i + 1);
    BOOST_CHECK_EQUAL(indices_view_1(i), i + 1);
  }

  // test sort DESC
  tensorTable.sortIndicesView("1", 0, 1, sortOrder::DESC, device);
  for (int i = 0; i < nlabels; ++i) {
    BOOST_CHECK_EQUAL(indices_view_1(i), i + 1);
    BOOST_CHECK_EQUAL(indices_view_2(i), nlabels - i);
    BOOST_CHECK_EQUAL(indices_view_3(i), nlabels - i);
  }
}

BOOST_AUTO_TEST_SUITE_END()