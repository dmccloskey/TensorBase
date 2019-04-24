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
    @brief write modified table shards to disk

    @returns true if successful, false otherwise
    */
    bool writeShardsToDisk();

    /*
    @brief reads shards from disk and updates
      the tensors in the collection

    @returns true if successful, false otherwise
    */
    bool readShardsFromDisk();

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