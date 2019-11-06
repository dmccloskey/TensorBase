/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTIONDEFAULTDEVICE_H
#define TENSORBASE_TENSORCOLLECTIONDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConceptDefaultDevice.h>
#include <TensorBase/ml/TensorCollection.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  class TensorCollectionDefaultDevice : public TensorCollection<Eigen::DefaultDevice>
  {
  public:
    using TensorCollection<Eigen::DefaultDevice>::TensorCollection;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorCollection<Eigen::DefaultDevice>>(this));
    }
  };
};
#endif //TENSORBASE_TENSORCOLLECTIONDEFAULTDEVICE_H