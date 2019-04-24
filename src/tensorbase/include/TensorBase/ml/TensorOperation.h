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

  class TensorCreateTable {
  public:
    template<typename TC, typename TCr, typename... TTables>
    void operator()(TC& collection, TCr& collection_new, TTables&... t) {
      auto tables_new = std::tuple_cat(collection.tables_, std::make_tuple(t...));
      collection_new.tables_ = tables_new;
    };
  };

  class TensorDrop {
  public:
    template<typename... T>
    void operator()(T&&... collection, T&&... t) {};
  };
};
#endif //TENSORBASE_TENSOROPERATION_H