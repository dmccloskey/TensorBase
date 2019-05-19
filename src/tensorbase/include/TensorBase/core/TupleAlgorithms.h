#ifndef TENSORBASE_TUPLEALGORITHMS_H
#define TENSORBASE_TUPLEALGORITHMS_H

#include <tuple>

/*
@brief A collection of tuple iteration algorithms which will most likely
  all be superceded with C++17
*/

namespace TensorBase
{
  /*
  @brief For each implementation where a return call back is possible
    but the function must work with a parameter pack and return
    either single type from a single type tuple
    or a tuple from a multi-type tuple

  References:
    http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
  */
  template <class F, size_t... Is>
  constexpr auto index_apply_impl(F f, std::index_sequence<Is...>) {
    return f(std::integral_constant<size_t, Is> {}...);
  }
  template <class Tuple, class F>
  constexpr auto index_apply(Tuple t, F f) {
    return index_apply_impl(f, std::make_index_sequence < std::tuple_size<Tuple>{} > {});
  }

  // Implementation 2 examples
  template <class Tuple, class F>
  constexpr auto apply(Tuple t, F f) {
    return index_apply(t, [&](auto... Is) { 
      return f(std::get<Is>(t)...); });
  }

  /*
  @brief For each implementation where a transformation
    function is applied to each element of the tuple
    without a call back

  References:
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

  /*
  @brief find_if method call

  TODO: add tests

  @param[in, out] tuple
  @param[in] pred Function implementing the "if" logic
  @returns The first index
  */
  template<typename Tuple, typename Predicate>
  constexpr size_t find_if(Tuple&& tuple, Predicate pred)
  {
    size_t index = std::tuple_size<std::remove_reference_t<Tuple>>::value;
    size_t currentIndex = 0;
    bool found = false;
    for_each(tuple, [&](auto&& value)
    {
      if (!found && pred(value))
      {
        index = currentIndex;
        found = true;
      }
      ++currentIndex;
    });
    return index;
  }

  /*
  @brief find_all method call

  TODO: add tests

  @param[in, out] tuple
  @param[in] pred Function implementing the "if" logic
  @returns The first index
  */
  template<typename Tuple, typename Predicate>
  constexpr std::vector<size_t> find_all(Tuple&& tuple, Predicate pred)
  {
    std::vector<size_t> indices;
    size_t currentIndex = 0;
    for_each(tuple, [&](auto&& value)
    {
      if (pred(value))
      {
        indices.push_back(currentIndex);
      }
      ++currentIndex;
    });
    return indices;
  }

  /*
  @brief apply_if method call

  @param[in, out] tuple
  @param[in] pred Function implementing the "if" logic
  @param[in] action Function implementing the "apply" method
  */
  template<typename Tuple, typename Predicate, typename Action>
  constexpr void apply_if(Tuple&& tuple, Predicate pred, Action action) {
    for_each(tuple, [pred = std::move(pred), action = std::move(action)](auto&& value) {
      if (pred(value))
        action(std::forward<decltype(value)>(value));
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

  /*
  @brief For each implementation where a transformation
    function is applied to each element of the tuple
    and the transformed tuple is returned

  References:
    https://www.fluentcpp.com/2019/03/08/stl-algorithms-on-tuples/
  */
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
}

#endif //TENSORBASE_TUPLEALGORITHMS_H