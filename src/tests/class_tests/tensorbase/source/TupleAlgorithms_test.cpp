/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TupleAlgorithms test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/core/TupleAlgorithms.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tupleAlgorithms)


BOOST_AUTO_TEST_CASE(tupleAlgorithmsApply)
{
  auto x = apply(std::make_tuple(1, 2), std::plus<>{});
  BOOST_CHECK_EQUAL(3, x);
}

BOOST_AUTO_TEST_CASE(tupleAlgorithmsTransform)
{
  std::tuple<int, double, long> numeric_tup = std::make_tuple(1, 2.3, 1l);
  auto transform = tuple_transform(numeric_tup,
    [](auto&& t) { return t + 1; });
  BOOST_CHECK_EQUAL(1, std::get<0>(numeric_tup));
  BOOST_CHECK_EQUAL(2, std::get<0>(transform));
  BOOST_CHECK_EQUAL(2.3, std::get<1>(numeric_tup));
  BOOST_CHECK_EQUAL(3.3, std::get<1>(transform));
  BOOST_CHECK_EQUAL(1l, std::get<2>(numeric_tup));
  BOOST_CHECK_EQUAL(2l, std::get<2>(transform));
}

BOOST_AUTO_TEST_CASE(tupleAlgorithmsForEach)
{
  std::tuple<int, double, long> numeric_tup = std::make_tuple(1, 2.3, 1l);

  for_each(numeric_tup,
    [](auto&& t) { t = t + 1; });
  BOOST_CHECK_EQUAL(2, std::get<0>(numeric_tup));
  BOOST_CHECK_EQUAL(3.3, std::get<1>(numeric_tup));
  BOOST_CHECK_EQUAL(2l, std::get<2>(numeric_tup));
}

BOOST_AUTO_TEST_CASE(tupleAlgorithmsApplyIf)
{
  struct ctypes {
    std::string name;
    bool found = false;
  };
  ctypes c1, c2, c3;
  c1.name = "c1"; c2.name = "c2"; c3.name = "c3";
  auto tuple_ctypes = std::make_tuple(c1, c2, c3);
  apply_if(tuple_ctypes,
    [](auto&& t) { return std::forward<decltype(t)>(t).name == "c2"; },
    [](auto&& t) { std::forward<decltype(t)>(t).found = true; });
  BOOST_CHECK(!std::get<0>(tuple_ctypes).found);
  BOOST_CHECK(std::get<1>(tuple_ctypes).found);
  BOOST_CHECK(!std::get<2>(tuple_ctypes).found);
}

BOOST_AUTO_TEST_SUITE_END()