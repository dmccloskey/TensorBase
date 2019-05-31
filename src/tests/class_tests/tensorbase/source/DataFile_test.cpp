/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE DataFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/DataFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(DataFile1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  DataFile* ptr = nullptr;
  DataFile* nullPointer = nullptr;
  ptr = new DataFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  DataFile* ptr = nullptr;
	ptr = new DataFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary) 
{
  DataFile data;

  std::string filename = "DataFileTest.dat";

  Eigen::Tensor<float, 3> random_dat(2,2,2);
  random_dat.setRandom();
	data.storeDataBinary<float, 3>(filename, random_dat);

  Eigen::Tensor<float, 3> test_dat(2,2,2);
	data.loadDataBinary<float, 3>(filename, test_dat);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        BOOST_CHECK_CLOSE(test_dat(i, j, k), random_dat(i, j, k), 1e-6);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()