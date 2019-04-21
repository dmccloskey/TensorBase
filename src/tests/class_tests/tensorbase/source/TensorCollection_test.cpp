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

// Implementation 1a
template<typename TupleType, typename FunctionType>
void for_each(TupleType&&, FunctionType, std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>) {}

template<std::size_t I, typename TupleType, typename FunctionType, 
  typename = typename std::enable_if<I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
  void for_each(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
  f(std::get<I>(t));
  for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template<typename TupleType, typename FunctionType>
void for_each(TupleType&& t, FunctionType f)
{
  for_each(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

// Implementation 1b
template<typename TupleType, typename FunctionType>
constexpr auto for_each_get(TupleType&&, FunctionType, std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type >::value>) {}

template<std::size_t I, typename TupleType, typename FunctionType,
  typename = typename std::enable_if<I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type >
constexpr auto for_each_get(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
  f(std::get<I>(t));
  return for_each_get(std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template<typename TupleType, typename FunctionType>
constexpr auto for_each_get(TupleType&& t, FunctionType f)
{
  return for_each_get(std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

// Implementation 2a
template <class F, size_t... Is>
constexpr auto apply_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is> {}...);
}

template <class Tuple, class F>
constexpr auto apply(Tuple t, F f) {
  return apply_impl(f, std::make_index_sequence<std::tuple_size<Tuple>{}> {});
}

// Implementation 2b
template <class F, size_t... Is>
constexpr auto index_apply_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is> {}...);
}

template <size_t N, class F>
constexpr auto index_apply(F f) {
  return index_apply_impl(f, std::make_index_sequence<N>{});
}

template <size_t N, class Tuple>
constexpr auto take_front(Tuple t) {
  return index_apply<N>([&](auto... Is) {
    return std::make_tuple(std::get<Is>(t)...);
  });
}

template <typename T>
struct capture_element{
  void operator()(const T& t) {
    template<typename std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
    if (t == comp) element = t;
  }
  T element;
  T comp;
};

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
  auto some = std::make_tuple("I am good", 255, 2.1);
  for_each(some, [](const auto &x) { std::cout << x << std::endl; });

  auto t = take_front<2>(std::make_tuple("I am good", 255, 2.1));
  for_each(t, [](const auto &x) { std::cout << x << std::endl; });

  auto x = apply(std::make_tuple(1, 2), plus<>{});
  std::cout << x << std::endl;

  //float y=0;
  //for_each_get(std::make_tuple("I am good", 255, 2.1), [&y](const auto& x) {
  //  if (x == 3.3) y=x; 
  //});
  //std::cout << y << std::endl;

  capture_element<float> ce;
  ce.comp = 2.1;
  for_each_get(std::make_tuple("I am good", 255, 2.1), ce);
  std::cout << ce.element << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()