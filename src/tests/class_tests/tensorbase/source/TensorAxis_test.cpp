/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorAxis test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorAxisDefaultDevice.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorAxis)

/*TensorAxisDefaultDevice Tests*/
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

BOOST_AUTO_TEST_CASE(constructor2DefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis("1", 1, 1);

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

BOOST_AUTO_TEST_CASE(comparatorDefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis_test("1", dimensions, labels);
  TensorAxisDefaultDevice<int> tensoraxis1("1", dimensions, labels);

  // Test expected
  BOOST_CHECK(tensoraxis_test == tensoraxis1);

  // Test with different names but same data
  tensoraxis1.setName("2");
  BOOST_CHECK(tensoraxis_test != tensoraxis1);

  Eigen::Tensor<std::string, 1> dimensions2(2);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  Eigen::Tensor<int, 2> labels2(2, 3);
  labels2.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis2("1", dimensions2, labels2);

  // Test with same names but different data
  BOOST_CHECK(tensoraxis_test != tensoraxis2);
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

  // Test char type
  Eigen::Tensor<char, 2> labels_char(3, 5);
  labels_char.setConstant('a');
  TensorAxisDefaultDevice<char> tensoraxis_char("1", dimensions, labels_char);

  // Test data copy
  tensoraxis_char.getLabelsDataPointer<char>(data_char);
  BOOST_CHECK_EQUAL(data_char.get()[0], 'a');
  BOOST_CHECK_EQUAL(data_char.get()[14], 'a');
}

BOOST_AUTO_TEST_CASE(deleteFromAxisDefaultDevice)
{// Test also covers selectFromAxis
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
    for (int j = 0; j < n_select_labels; ++j) {
      BOOST_CHECK_EQUAL(tensoraxis.getLabels()(i, j), labels_test(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxis1DefaultDevice)
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

BOOST_AUTO_TEST_CASE(appendLabelsToAxis2DefaultDevice)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisDefaultDevice<int> tensoraxis("1", n_dimensions, 0);
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
  TensorDataDefaultDevice<int, 2> labels_new(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  labels_new.setData(labels);

  // Test
  Eigen::DefaultDevice device;
  tensoraxis.appendLabelsToAxis(std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_new), device);
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

BOOST_AUTO_TEST_CASE(makeSortIndicesDefaultDevice)
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
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // setup the device
  Eigen::DefaultDevice device;

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataDefaultDevice<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({n_labels}));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_view_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_view);

  // make the expected indices
  Eigen::Tensor<int, 2> indices_sort_test(n_dimensions, n_labels);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      indices_sort_test(i, j) = i + j * n_dimensions + 1;
    }
  }
  
  // test making the sort indices
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices_sort;
  tensoraxis.makeSortIndices(indices_view_ptr, indices_sort, device);
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      BOOST_CHECK_EQUAL(indices_sort->getData()(i,j), indices_sort_test(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(sortLabelsDefaultDevice)
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
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // setup the device
  Eigen::DefaultDevice device;

  // setup the sort indices
  Eigen::Tensor<int, 1> indices_view_values(n_labels);
  for (int i = 0; i < n_labels; ++i)
    indices_view_values(i) = i + 1;
  TensorDataDefaultDevice<int, 1> indices_view(Eigen::array<Eigen::Index, 1>({ n_labels }));
  indices_view.setData(indices_view_values);
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> indices_view_ptr = std::make_shared<TensorDataDefaultDevice<int, 1>>(indices_view);

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

BOOST_AUTO_TEST_CASE(storeAndLoadLabelsDefaultDevice)
{
  // Setup the axis
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis_io("1", dimensions, labels);

  // Store the axis data
  Eigen::DefaultDevice device;
  tensoraxis_io.storeLabelsBinary("axis", device);

  // Load the axis data
  TensorAxisDefaultDevice<int> tensoraxis("1", 3, 5);
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

BOOST_AUTO_TEST_CASE(getLabelsAsStringsDefaultDevice)
{
  Eigen::Tensor<std::string, 1> dimensions(3);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  dimensions(2) = "TensorDimension3";
  Eigen::Tensor<int, 2> labels(3, 5);
  labels.setConstant(1);
  TensorAxisDefaultDevice<int> tensoraxis("1", dimensions, labels);

  // Test getLabelsAsString
  Eigen::DefaultDevice device;
  std::vector<std::string> labels_str = tensoraxis.getLabelsAsStrings(device);
  int iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      BOOST_CHECK_EQUAL(labels_str.at(iter), std::to_string(tensoraxis.getLabels()(i, j)));
    }
    ++iter;
  }

  // Using Char
  Eigen::Tensor<char, 2> labels_char(3, 5);
  labels_char.setConstant('a');
  TensorAxisDefaultDevice<char> tensoraxis_char("1", dimensions, labels_char);

  // Test getLabelsAsString
  std::vector<std::string> labels_char_str = tensoraxis_char.getLabelsAsStrings(device);
  iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      BOOST_CHECK_EQUAL(labels_char_str.at(iter), std::to_string(tensoraxis_char.getLabels()(i, j)));
    }
    ++iter;
  }

  // Use TensorArray8
  Eigen::Tensor<TensorArray8<int>, 2> labels_array(3, 5);
  labels_array.setConstant(TensorArray8<int>({1,2,3,4,5,6,7,8}));
  TensorAxisDefaultDevice<TensorArray8<int>> tensoraxis_array("1", dimensions, labels_array);

  // Test getLabelsAsString
  std::vector<std::string> labels_array_str = tensoraxis_array.getLabelsAsStrings(device);
  iter = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 5; j++) {
      BOOST_CHECK_EQUAL(labels_array_str.at(iter), labels_array(i, j).getTensorArrayAsString());
    }
    ++iter;
  }
}

BOOST_AUTO_TEST_CASE(appendLabelsToAxisFromCsv1DefaultDevice)
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
  Eigen::DefaultDevice device;
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

BOOST_AUTO_TEST_CASE(appendLabelsToAxisFromCsv2DefaultDevice)
{
  // Setup the axis
  int n_dimensions = 2, n_labels = 5;
  Eigen::Tensor<std::string, 1> dimensions(n_dimensions);
  dimensions(0) = "TensorDimension1";
  dimensions(1) = "TensorDimension2";
  TensorAxisDefaultDevice<int> tensoraxis("1", n_dimensions, 0);
  tensoraxis.setDimensions(dimensions);

  // Setup the new labels
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_labels }));
  int iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      labels_values(i, j) = std::to_string(i + j*n_dimensions);
    }
  }

  // Test
  Eigen::DefaultDevice device;
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

BOOST_AUTO_TEST_CASE(makeSelectIndicesFromCsvDefaultDevice)
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
  int n_new_labels = 3;
  Eigen::Tensor<std::string, 2> labels_values(Eigen::array<Eigen::Index, 2>({ n_dimensions, n_new_labels }));
  iter = 0;
  for (int i = 0; i < n_dimensions; ++i) {
    for (int j = 0; j < n_labels; ++j) {
      if (j % 2 == 0) labels_values(i, j/2) = std::to_string(iter);
      ++iter;
    }
  }

  // Test
  Eigen::DefaultDevice device;
  std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>> select_indices;
  tensoraxis.makeSelectIndicesFromCsv(select_indices, labels_values, device);
  for (int i = 0; i < n_labels; ++i) {
    if (i % 2 == 0) BOOST_CHECK_EQUAL(select_indices->getData()(i), 1);
    else BOOST_CHECK_EQUAL(select_indices->getData()(i), 0);
  }
}

BOOST_AUTO_TEST_SUITE_END()