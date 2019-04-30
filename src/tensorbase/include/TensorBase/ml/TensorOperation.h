/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>
#include <set>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor Insert/Update/Delete operations
  */
  class TensorOperation
  {
    virtual void redo(TensorCollection& tensor_collection) = 0;
    virtual void undo(TensorCollection& tensor_collection) = 0;
  };

  class TensorInsertIntoAxis {
  public:
    template<typename T>
    void operator()(T&& t) {};
    std::vector<std::pair<std::string, std::string>> insert_into_clause; ///< pairs of TensorTable.name and TensorDimension.label
    //Eigen::Tensor<TensorT, TDim> values; ///< values to insert
  };

  class TensorAddAxis;

  class TensorUpdate {
  public:
    void whereClause(TensorCollection& tensor_collection);
    std::vector<std::pair<std::string, std::string>> set_clause; ///< pairs of TensorTable.name and TensorDimension.label
  };

  class TensorDeleteFromAxis {
  public:
    void deleteFromClause(TensorCollection& tensor_collection);
    void whereClause(TensorCollection& tensor_collection);
    std::vector<std::pair<std::string, std::string>> delete_clause; ///< pairs of TensorTable.name and TensorDimension.label
  };

  class TensorDeleteAxis;

  class TensorAddTables;

  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H