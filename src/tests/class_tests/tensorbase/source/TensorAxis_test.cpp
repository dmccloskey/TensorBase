/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorAxis test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorAxis.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorAxis)

BOOST_AUTO_TEST_CASE(constructorDefaultDevice) 
{
	TensorAxisDefaultDevice<int>* ptr = nullptr;
	TensorAxisDefaultDevice<int>* nullPointer = nullptr;
	ptr = new TensorAxisDefaultDevice<int>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorDefaultDevice)
{
	TensorAxisDefaultDevice<int>* ptr = nullptr;
	ptr = new TensorAxisDefaultDevice<int>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor1DefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_CASE(gettersAndSettersDefaultDevice)
{
  TensorAxisDefaultDevice<int> tensoraxis;
  // Check defaults
  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "");
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 0);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 0);

  // Check getters/setters
  tensoraxis.setId(1);
  tensoraxis.setName("1");
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  tensoraxis.setDimensionsAndLabels(dimensions, labels);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_CASE(getLabelsDataPointerDefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // Test data copy
  std::shared_ptr<int> data_int = nullptr;
  tensoraxis.getLabelsDataPointer<int>(data_int);
  BOOST_CHECK_EQUAL(data_int.get()[0], 1);
  BOOST_CHECK_EQUAL(data_int.get()[14], 1);

  // Test that no data is reinterpreted
  std::shared_ptr<char> data_char = nullptr;
  tensoraxis.getLabelsDataPointer<char>(data_char);
  BOOST_CHECK_EQUAL(data_char, nullptr);
  std::shared_ptr<float> data_float = nullptr;
  tensoraxis.getLabelsDataPointer<float>(data_float);
  BOOST_CHECK_EQUAL(data_float, nullptr);
  std::shared_ptr<double> data_double = nullptr;
  tensoraxis.getLabelsDataPointer<double>(data_double);
  BOOST_CHECK_EQUAL(data_double, nullptr);
}

BOOST_AUTO_TEST_CASE(deleteFromAxisDefaultDevice)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // Setup the selection indices and the expected labels
  int n_select_labels = 3;
  Eigen::Tensor<int, 1> indices_values(n_labels);
  Eigen::Tensor<int, 2> labels_test(n_dimensions, n_select_labels);
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      if (j % 2 == 0) {
        indices_values(j) = j + 1;
        labels_test(i, j/2) = iter;
      }
      else {
        indices_values(j) = 0;
      }
      ++iter;
    }
  }
  TensorDataDefaultDevice<int, 1> indices(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices.setData(indices_values);

  // Test
  Eigen::DefaultDevice device;
  tensoraxis.deleteFromAxis(std::make_shared<TensorDataDefaultDevice<int, 1>>(indices), device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_select_labels);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i,j), labels_test(i,j));
    }
  }
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxisDefaultDevice)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 2;
  Eigen::Tensor<int, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      labels_values(i, j) = iter;
      ++iter;
    }
  }
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  labels_new.setData(labels_values);

  // Test
  Eigen::DefaultDevice device;
  tensoraxis.appendLabelsToAxis(std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new), device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_labels + n_new_labels);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels(i, j));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels_values(i, j - n_labels));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()