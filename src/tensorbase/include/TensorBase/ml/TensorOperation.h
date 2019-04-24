/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor operations
  */
  class TensorOperation
  {
    //virtual void undo() = 0;
  };

  /**
    @brief Template class for all Tensor select operations
  */
  class TensorSelect {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void whereClause(T&& t) {};
    template<typename T>
    void havingClause(T&& t) {};
    enum order { ASC, DESC };
    std::vector<std::pair<std::string, std::string>> select_clause; ///< pairs of TensorTable.name and TensorDimension.label
    std::vector<std::pair<std::string, std::string>> group_by_clause; ///< pairs of TensorTable.name and TensorDimension.label
    std::vector<std::tuple<std::string, std::string, order>> order_by_clause; ///< tuple of TensorTable.name, TensorDimension.label, and order
  };

  class TensorSelectUnion;
  class TensorSelectJoin;

  class TensorInsert {
  public:
    template<typename T>
    void operator()(T&& t) {};
    std::vector<std::pair<std::string, std::string>> insert_into_clause; ///< pairs of TensorTable.name and TensorDimension.label
    //Eigen::Tensor<TensorT, TDim> values; ///< values to insert
  };

  class TensorUpdate {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void whereClause(T&& t) {};
    std::vector<std::pair<std::string, std::string>> set_clause; ///< pairs of TensorTable.name and TensorDimension.label
    //Eigen::Tensor<TensorT, TDim> values; ///< values to update
  };

  class TensorDelete {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void whereClause(T&& t) {};
    std::vector<std::pair<std::string, std::string>> delete_clause; ///< pairs of TensorTable.name and TensorDimension.label
  };

  class TensorCreate: public TensorOperation {
  public:
    template<typename... T>
    TensorCollection<T...> operator()(TensorCollection<T...>&& collection, T&&... t) {};
  };

  class TensorCreateTable : TensorCreate {
  public:
    template<
      typename... Types1, template <typename...> class T, 
      typename... Types2, template <typename...> class U,
      typename... Ts>
    void operator()(T<Types1...>&& collection, U<Types2...>&& new_collection, Ts&&... t) {
      new_collection = std::tuple_cat(collection, TensorCollection<Ts...>(t));
    };
  };

  class TensorDrop {
  public:
    template<typename... T>
    void operator()(T&&... collection, T&&... t) {};
  };
};
#endif //TENSORBASE_TENSOROPERATION_H