/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>
#include <TensorBase/core/TupleAlgorithms.h>

namespace TensorBase
{
  /**
    @brief Class for managing heterogenous Tensors

    TODOs:
    - add static checks for the "acceptable" TTables
    - add static checks that the TTables are wrapped in a `std::shared_ptr`
    - implement the method bodies for unimplemented method signatures
  */
  template<class... TTables>
  class TensorCollection
  {
  public:
    TensorCollection() = default;  ///< Default constructor
    TensorCollection(TTables&... tTables) { 
      tables_ = std::make_tuple(tTables...);
    }; 
    ~TensorCollection() = default; ///< Default destructor

    /*
    @brief get all of the table names in the collection

    @note `struct GetTableNamesHelper` is used to accumulate the table names

    @returns a vector of strings with the table names
    */
    std::vector<std::string> getTableNames() const;

    /*
    @brief Add Table

    add a new TensorTable to the tuple of tables

    @param[in] tables Variable number of tables to add

    @returns true if successful, false otherwise
    */
    bool addTables(TTables&... tTables);

    /*
    @brief delete Tables

    remove a TensorTable from the tuple of tables

    @param[in] table_names List of table names to remove

    @returns true if successful, false otherwise
    */
    bool deleteTables(const std::vector<std::string>& table_names);

    /*
    @brief select a specific number of tensor tables, dimensions, and labels

    The functor can describe any number of sort,
    group by, join, or aggregate operations.
    In addition, the functor can also describe any
    transformation operations including matrix multiplication
    and other linear algebra operations supported
    by the Eigen library

    @param[in] func Lambda or functor (?)

    @returns a subset of the TensorCollection
    */
    template<typename Func>
    TensorCollection selectFromTables(Func func);

    /*
    @brief insert values into a particular dimension

    all tables where the current dimension is the only
    dimension composing the axis will be expanded.
    The functor or lambda should describe the values
    to be added and/or the default values to fill
    each affected tensor with.

    @param[in] dimension The name of the dimensions to expand
    @param[in] labels A list of labels to insert into the dimension
    @param[in] func Lambda or functor (?)

    @returns true if successful, false otherwise
    */
    template<typename Func>
    bool insertIntoDimension(std::string& dimension, std::vector<std::string>& labels, Func func);

    /*
    @brief update a specific number of tensor tables, dimensions, and labels

    @param[in] func Lambda or functor (?)

    @returns true if successful, false otherwise
    */
    template<typename Func>
    bool updateTables(Func func);

    /*
    @brief remove labels from a dimension

    all tables where the current dimension is the only
    dimension composing the axis will be reduced.
    The functor or lambda should describe the 
    selection criteria if no specific labels are given.

    @param[in] dimension The name of the dimensions to expand
    @param[in] labels A list of labels to insert into the dimension
    @param[in] func Lambda or functor (?)

    @returns true if successful, false otherwise
    */
    template<typename Func>
    bool removeFromDimension(std::string& dimension, std::vector<std::string>& labels, Func func);

    /*
    @brief write modified table shards to disk

    @returns true if successful, false otherwise
    */
    template<typename Func>
    bool writeShardsToDisk();

    /*
    @brief reads shards from disk and updates
      the tensors in the collection

    @returns true if successful, false otherwise
    */
    template<typename Func>
    bool readShardsFromDisk();

  private:
    std::tuple<TTables...> tables_; ///< tuple of std::shared_ptr TensorTables<TensorT, DeviceT, TDim>
  };

  struct GetTableNamesHelper {
    template<typename T>
    void operator()(T&& t) { names.push_back(std::forward<decltype(t)>(t)->getName()); }
    std::vector<std::string> names;
  };
  template<class ...TTables>
  inline std::vector<std::string> TensorCollection<TTables...>::getTableNames() const
  {
    GetTableNamesHelper getTableNamesHelper;
    for_each(tables_, getTableNamesHelper);
    return getTableNamesHelper.names;
  }
};
#endif //TENSORBASE_TENSORCOLLECTION_H