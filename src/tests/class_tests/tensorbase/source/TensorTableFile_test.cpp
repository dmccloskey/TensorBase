/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorTableFile test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/TensorTableFile.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorTableFile1)

BOOST_AUTO_TEST_CASE(constructor)
{
  TensorTableFile* ptr = nullptr;
  TensorTableFile* nullPointer = nullptr;
  ptr = new TensorTableFile();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorTableFile* ptr = nullptr;
  ptr = new TensorTableFile();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(makeTensorTableShardFilename)
{
  TensorTableFile data;
  BOOST_CHECK_EQUAL(data.makeTensorTableShardFilename("dir/", "table1", 1), "dir/table1_1.tts");
}

BOOST_AUTO_TEST_CASE(storeAndLoadBinary)
{
  TensorTableFile data;
}

BOOST_AUTO_TEST_SUITE_END()