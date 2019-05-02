/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLECONCEPT_H
#define TENSORBASE_TENSORTABLECONCEPT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>
#include <map>

namespace TensorBase
{
  /// The erasure interface to Tensor Table
  class TensorTableConcept {
  public:
    virtual std::string getName() const = 0;
    virtual std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() = 0;
  };

  /// The erasure wrapper around the Tensor Table interface
  template<typename T>
  class TensorTableWrapper : public TensorTableConcept {
    std::shared_ptr<T> tensor_table_;
  public:
    TensorTableWrapper(const std::shared_ptr<T>& tensor_table) : tensor_table_(tensor_table) {};
    std::string getName() const { return tensor_table_->getName(); };
    std::map<std::string, std::shared_ptr<TensorAxisConcept>>& getAxes() { return tensor_table_->getAxes(); };
  };
};
#endif //TENSORBASE_TENSORTABLECONCEPT_H