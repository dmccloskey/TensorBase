/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorCollection test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorCollection.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(tensorCollection)

BOOST_AUTO_TEST_CASE(constructor) 
{
  TensorCollection<TensorTable<float,Eigen::DefaultDevice,3>>* ptr = nullptr;
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* nullPointer = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  BOOST_CHECK_NE(ptr, nullPointer);
  delete ptr;
}

BOOST_AUTO_TEST_CASE(destructor)
{
  TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>* ptr = nullptr;
	ptr = new TensorCollection<TensorTable<float, Eigen::DefaultDevice, 3>>();
  delete ptr;
}

// For each iterator in forward order
//https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11/26902803#26902803
template<typename TupleType, typename FunctionType>
void for_each_f(TupleType&&, FunctionType, std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>)
{}

template<std::size_t I, typename TupleType, typename FunctionType, 
  typename = typename std::enable_if<I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
  void for_each_f(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
  f(std::get<I>(t));
  for_each_f(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}
template<typename TupleType, typename FunctionType>
void for_each_f(TupleType&& t, FunctionType f)
{
  for_each_f(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

// For each iterator in reverse order
// https://stackoverflow.com/questions/26959597/is-it-possible-to-apply-a-generic-function-over-tuple-elements
template<int I, class Tuple, typename F> struct for_each_impl_r {
  static void for_each_r(const Tuple& t, F f) {
    for_each_impl_r<I - 1, Tuple, F>::for_each_r(t, f);
    f(get<I>(t));
  }
};
template<class Tuple, typename F> struct for_each_impl_r<0, Tuple, F> {
  static void for_each_r(const Tuple& t, F f) {
    f(get<0>(t));
  }
};
template<class Tuple, typename F>
void for_each_r(const Tuple& t, F f) {
  for_each_impl_r<tuple_size<Tuple>::value - 1, Tuple, F>::for_each_r(t, f);
}

// Implementation 2a
template <class F, size_t... Is>
constexpr auto apply_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is> {}...);
}
template <class Tuple, class F>
constexpr auto apply_t(Tuple t, F f) {
  return apply_impl(f, std::make_index_sequence<std::tuple_size<Tuple>{}> {});
}


/*
@brief For each implementation where a return call back is possible
    but the function must work with a parameter pack
http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
*/
template <class F, size_t... Is>
constexpr auto index_apply_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is> {}...);
}
template <size_t N, class F>
constexpr auto index_apply(F f) {
  return index_apply_impl(f, std::make_index_sequence<N>{});
}

// Implementation 2 examples
template <size_t N, class Tuple>
constexpr auto take_front(Tuple t) {
  return index_apply<N>([&](auto... Is) {
    return std::make_tuple(std::get<Is>(t)...);
  });
}

template <class Tuple, class F>
constexpr auto apply(Tuple t, F f) {
  return apply_t (t, [&](auto... Is) { return f(std::get<Is>(t)...); });
}

template <class Tuple>
constexpr auto reverse(Tuple t) {
  return index_apply < tuple_size<Tuple>{} > (
    [&](auto... Is) {
    return std::make_tuple(
      std::get < std::tuple_size<Tuple>{} - 1 - Is > (t)...);
  });
}

/*
@brief For each implementation without a call back of the tuple
https://www.fluentcpp.com/2019/03/08/stl-algorithms-on-tuples/
*/
template <class Tuple, class F>
constexpr F for_each(Tuple&& t, F&& f)
{
  return for_each_impl(std::forward<Tuple>(t), std::forward<F>(f),
    std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}
template <class Tuple, class F, std::size_t... I>
constexpr F for_each_impl(Tuple&& t, F&& f, std::index_sequence<I...>)
{
  return (void)std::initializer_list<int>{(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))), 0)...}, f;
}

template<typename Tuple, typename Predicate, typename Action>
constexpr void apply_if(Tuple&& tuple, Predicate pred, Action action) {
  for_each(tuple, [pred = std::move(pred), action = std::move(action)](auto&& value) {
    if (pred(value))
      action(std::forward<decltype(value)>(value));
  });
}

template<typename Tuple, typename Action>
void perform(Tuple&& tuple, size_t index, Action action)
{
  size_t currentIndex = 0;
  for_each(tuple, [action = std::move(action), index, &currentIndex](auto&& value)
  {
    if (currentIndex == index)
    {
      action(std::forward<decltype(value)>(value));
    }
    ++currentIndex;
  });
}

/* 
@brief Slice for tuples, arrays, pairs
  Where [I1, I2) is the exclusive range of the subset and the return value is a tuple 
  of either references or values depending on whether the 
  input tuple is an lvalue or an rvalue respectively

References:
  https://stackoverflow.com/questions/8569567/get-part-of-stdtuple

@example
tuple_slice<I1, I2>(t);
*/
template <std::size_t Ofst, class Tuple, std::size_t... I>
constexpr auto slice_impl(Tuple&& t, std::index_sequence<I...>)
{
  return std::forward_as_tuple(
    std::get<I + Ofst>(std::forward<Tuple>(t))...);
}
template <std::size_t I1, std::size_t I2, class Cont>
constexpr auto tuple_slice(Cont&& t)
{
  static_assert(I2 >= I1, "invalid slice");
  static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
    "slice index out of bounds");

  return slice_impl<I1>(std::forward<Cont>(t),
    std::make_index_sequence<I2 - I1>{});
}

template <typename Func, typename Tuple, std::size_t... I>
auto tuple_transform_impl(Tuple&& t, Func&& f, std::index_sequence<I...>)
{
  return std::make_tuple(
    std::forward<Func>(f)(std::get<I>(std::forward<Tuple>(t)))...);
}

template <typename Func, typename Tuple>
auto tuple_transform(Tuple&& t, Func&& f)
{
  return tuple_transform_impl(std::forward<Tuple>(t), std::forward<Func>(f), std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

BOOST_AUTO_TEST_CASE(gettersAndSetters)
{
  TensorCollection<
    TensorTable<float, Eigen::DefaultDevice, 3>,
    TensorTable<int, Eigen::DefaultDevice, 2>,
    TensorTable<char, Eigen::DefaultDevice, 4>
  > tensorCollection;

  // name setters and getters
  tensorCollection.setTableNames({"tfloat", "tint", "tchar"});
  BOOST_CHECK(tensorCollection.getTableNames() == std::vector<std::string>({ "tfloat", "tint", "tchar" }));

  // table getters
  //BOOST_CHECK_EQUAL(tensorCollection.getTableIndex("tfloat"), 0);
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tfloat")->getName(), "tfloat");
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tint")->getName(), "tint");
  //BOOST_CHECK_EQUAL(tensorCollection.getTable("tchar")->getName(), "tchar");
}

BOOST_AUTO_TEST_CASE(Scratch)
{
  auto x = apply(std::make_tuple(1, 2), std::plus<>{});
  BOOST_CHECK_EQUAL(3, x);

  std::tuple<int, double, long> numeric_tup = std::make_tuple(1, 2.3, 1l);
  auto transform = tuple_transform(numeric_tup,
    [](auto&& t) { return t + 1; });
  BOOST_CHECK_EQUAL(1, std::get<0>(numeric_tup));
  BOOST_CHECK_EQUAL(2, std::get<0>(transform));
  BOOST_CHECK_EQUAL(2.3, std::get<1>(numeric_tup));
  BOOST_CHECK_EQUAL(3.3, std::get<1>(transform));
  BOOST_CHECK_EQUAL(1l, std::get<2>(numeric_tup));
  BOOST_CHECK_EQUAL(2l, std::get<2>(transform));

  for_each(numeric_tup,
    [](auto&& t) { t=t+1; });
  BOOST_CHECK_EQUAL(2, std::get<0>(numeric_tup));
  BOOST_CHECK_EQUAL(3.3, std::get<1>(numeric_tup));
  BOOST_CHECK_EQUAL(2l, std::get<2>(numeric_tup));

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