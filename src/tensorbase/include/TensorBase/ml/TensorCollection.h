/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTable.h>

namespace TensorBase
{

  /**
    @brief Class for managing heterogenous Tensors
  */
  class TensorCollection
  {
  public:
    TensorCollection() = default;  ///< Default constructor
    ~TensorCollection() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter


  private:
    int id_ = -1;
    std::string name_ = "";
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };
};
#endif //TENSORBASE_TENSORCOLLECTION_H