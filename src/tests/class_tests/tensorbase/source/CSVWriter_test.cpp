/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE CSVWriter test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/io/CSVWriter.h>

#include <TensorBase/io/csv.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(CSVWriter1)

BOOST_AUTO_TEST_CASE(constructor) 
{
  CSVWriter* ptr = nullptr;
  CSVWriter* nullPointer = nullptr;
  ptr = new CSVWriter();
  BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructor) 
{
  CSVWriter* ptr = nullptr;
	ptr = new CSVWriter();
  delete ptr;
}

BOOST_AUTO_TEST_CASE(constructor2) 
{
  CSVWriter csvwriter("filename1", ";");

  BOOST_CHECK_EQUAL(csvwriter.getFilename(), "filename1");
  BOOST_CHECK_EQUAL(csvwriter.getDelimeter(), ";");
  BOOST_CHECK_EQUAL(csvwriter.getLineCount(), 0);
}

BOOST_AUTO_TEST_CASE(gettersAndSetters) 
{
  CSVWriter csvwriter;
  csvwriter.setFilename("filename1");
  csvwriter.setDelimeter(";");
  csvwriter.setLineCount(1);

  BOOST_CHECK_EQUAL(csvwriter.getFilename(), "filename1");
  BOOST_CHECK_EQUAL(csvwriter.getDelimeter(), ";");
  BOOST_CHECK_EQUAL(csvwriter.getLineCount(), 1);
}

BOOST_AUTO_TEST_CASE(writeDataInRow) 
{
  std::string filename = "CSVWriterTest.csv";
  std::vector<std::string> headers, line;  
  CSVWriter csvwriter(filename);

  // Write the data to file
  headers = {"Column1", "Column2", "Column3"};
	csvwriter.writeDataInRow(headers.begin(), headers.end());
  line = {"a", "b", "c" };
	csvwriter.writeDataInRow(line.begin(), line.end());
  line = {"1", "2", "3" };
	csvwriter.writeDataInRow(line.begin(), line.end());

  // Read the data back in
  csv::CSVReader test_in(filename);

  int cnt = 0;
  for (csv::CSVRow& row : test_in) {
    if (cnt == 0)
    {
      BOOST_CHECK_EQUAL(row["Column1"].get<>(), "a");
      BOOST_CHECK_EQUAL(row["Column2"].get<>(), "b");
      BOOST_CHECK_EQUAL(row["Column3"].get<>(), "c");
    }
    else if (cnt == 1)
    {
      BOOST_CHECK_EQUAL(row["Column1"].get<>(), "1");
      BOOST_CHECK_EQUAL(row["Column2"].get<>(), "2");
      BOOST_CHECK_EQUAL(row["Column3"].get<>(), "3");
    }
    cnt += 1;
  }
}

BOOST_AUTO_TEST_SUITE_END()