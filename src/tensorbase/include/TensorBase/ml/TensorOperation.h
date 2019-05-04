/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>

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

  /*TODO: Move to seperate files!*/

  class TensorInsertIntoAxis;

  class TensorAddAxis;

  class TensorUpdate;

  class TensorDeleteFromAxis;

  class TensorDeleteAxis;

  class TensorAddTables;

  class TensorDropTables;
};
#endif //TENSORBASE_TENSOROPERATION_H