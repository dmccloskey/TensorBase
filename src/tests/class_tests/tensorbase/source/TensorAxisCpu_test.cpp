/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorAxisCpu test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorAxisCpu.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorAxisCpu)

/*TensorAxisCpu Tests*/
BOOST_AUTO_TEST_CASE(constructorCpu)
{
  TensorAxisCpu<int>* ptr = nullptr;
  TensorAxisCpu<int>* nullPointer = nullptr;
  ptr = new TensorAxisCpu<int>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
  TensorAxisCpu<int>* ptr = nullptr;
  ptr = new TensorAxisCpu<int>();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor1Cpu)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

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

BOOST_AUTO_TEST_CASE(constructor2Cpu)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisCpu<int> tensoraxis("1", 1, 1);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 1);

  tensoraxis.setDimensions(dimensions);
  tensoraxis.setLabels(labels);

  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3");
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_CASE(gettersAndSettersCpu)
{
  TensorAxisCpu<int> tensoraxis;
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

BOOST_AUTO_TEST_CASE(copyCpu)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisCpu<int> tensoraxis1("1", dimensions, labels);

  // Test expected
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  auto tensoraxis_copy = tensoraxis1.copyToHost(device);
  BOOST_CHECK(*(tensoraxis_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis_copy->getLabels()(i, j), labels(i, j));
    }
  }
  auto tensoraxis2_copy = tensoraxis1.copyToDevice(device);
  BOOST_CHECK(*(tensoraxis2_copy.get()) == tensoraxis1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis_copy->getLabels()(i, j), labels(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(getLabelsDataPointerCpu)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

  // Test data copyToHost
  std::shared_ptr<int[]> data_int = nullptr;
  tensoraxis.getLabelsDataPointer<int>(data_int);
  BOOST_CHECK_EQUAL(data_int.get()[0], 1);
  BOOST_CHECK_EQUAL(data_int.get()[14], 1);

  // Test that no data is reinterpreted
  std::shared_ptr<char[]> data_char = nullptr;
  tensoraxis.getLabelsDataPointer<char>(data_char);
  BOOST_CHECK_EQUAL(data_char, nullptr);
  std::shared_ptr<float[]> data_float = nullptr;
  tensoraxis.getLabelsDataPointer<float>(data_float);
  BOOST_CHECK_EQUAL(data_float, nullptr);
  std::shared_ptr<double[]> data_double = nullptr;
  tensoraxis.getLabelsDataPointer<double>(data_double);
  BOOST_CHECK_EQUAL(data_double, nullptr);
}

BOOST_AUTO_TEST_CASE(deleteFromAxisCpu)
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
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

  // Setup the selection indices and the expected labels
  int n_select_labels = 3;
  Eigen::Tensor<int, 1> indices_values(n_labels);
  Eigen::Tensor<int, 2> labels_test(n_dimensions, n_select_labels);
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      if (j % 2 == 0) {
        indices_values(j) = j + 1;
        labels_test(i, j / 2) = iter;
      }
      else {
        indices_values(j) = 0;
      }
      ++iter;
    }
  }
  TensorDataCpu<int, 1> indices(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices.setData(indices_values);

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis.deleteFromAxis(std::make_shared<TensorDataCpu<int, 1>>(indices), device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_select_labels);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_select_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels_test(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxis1Cpu)
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
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

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
  TensorDataCpu<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  labels_new.setData(labels_values);

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis.appendLabelsToAxis(std::make_shared<TensorDataCpu<int, 2>>(labels_new), device);
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

BOOST_AUTO_TEST_CASE(appendLabelsToAxis2Cpu)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisCpu<int> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = iter;
      ++iter;
    }
  }
  TensorDataCpu<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  labels_new.setData(labels);

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis.appendLabelsToAxis(std::make_shared<TensorDataCpu<int, 2>>(labels_new), device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_labels);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(makeSortIndicesCpu)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = j;
    }
  }
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

  // setup the device
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataCpu<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> indices_view_ptr = std::make_shared<TensorDataCpu<int, 1>>(indices_view);

  // make the expected indices
  Eigen::Tensor<int, 2> indices_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      indices_sort_test(i, j) = i + j * n_dimensions + 1;
    }
  }

  // test making the sort indices
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices_sort;
  tensoraxis.makeSortIndices(indices_view_ptr, indices_sort, device);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(indices_sort->getData()(i, j), indices_sort_test(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(sortLabelsCpu)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels(i, j) = j;
    }
  }
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

  // setup the device
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataCpu<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>> indices_view_ptr = std::make_shared<TensorDataCpu<int, 1>>(indices_view);

  // test sorting ASC
  tensoraxis.sortLabels(indices_view_ptr, device);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels(i, j));
    }
  }

  // make the expected labels
  Eigen::Tensor<int, 2> labels_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_sort_test(i, j) = n_labels - j - 1;
    }
  }
  for (int i = 0; i < n_labels; ++i)
    indices_view_ptr->getData()(i) = n_labels - i;

  // test sorting DESC
  tensoraxis.sortLabels(indices_view_ptr, device);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels_sort_test(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(storeAndLoadLabelsCpu)
{
  // Setup the axis
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisCpu<int> tensoraxis_io("1", dimensions, labels);

  // Store the axis data
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis_io.storeLabelsBinary("axis", device);

  // Load the axis data
  TensorAxisCpu<int> tensoraxis("1", 3, 5);
  tensoraxis.loadLabelsBinary("axis", device);

  BOOST_CHECK_EQUAL(tensoraxis.getId(), -1);
  BOOST_CHECK_EQUAL(tensoraxis.getName(), "1");
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), 3);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), 5);
  //BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1"); // Not loaded
  //BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2"); // Not loaded
  //BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(2), "TensorDimension3"); // Not loaded
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(0, 0), 1);
  BOOST_CHECK_EQUAL(tensoraxis.getLabels()(2, 4), 1);
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxisFromCsv1Cpu)
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
  TensorAxisCpu<int> tensoraxis("1", dimensions, labels);

  // Setup the new labels
  int n_new_labels = 4;
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_new_labels; ++j) {
      if (j < 2)
        labels_values(i, j) = std::to_string(i + j * n_dimensions + iter);
      else
        labels_values(i, j) = std::to_string(i + (j - 2) * n_dimensions + iter); // duplicates
    }
  }

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_labels + 2);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels(i, j));
    }
  }
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = n_labels; j < tensoraxis.getNLabels(); ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), std::stoi(labels_values(i, j - n_labels)));
    }
  }
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxisFromCsv2Cpu)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisCpu<int> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_values(i, j) = std::to_string(i + j * n_dimensions);
    }
  }

  // Test
  Eigen::ThreadPool pool(1);
  Eigen::ThreadPoolDevice device(&pool, 1);
  tensoraxis.appendLabelsToAxisFromCsv(labels_values, device);
  BOOST_CHECK_EQUAL(tensoraxis.getNDimensions(), n_dimensions);
  BOOST_CHECK_EQUAL(tensoraxis.getNLabels(), n_labels);
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(0), "TensorDimension1");
  BOOST_CHECK_EQUAL(tensoraxis.getDimensions()(1), "TensorDimension2");
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), i + j * n_dimensions);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()