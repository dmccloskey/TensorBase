/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTIONCPU_H
#define TENSORBASE_TENSORCOLLECTIONCPU_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorTableConceptCpu.h>
#include <TensorBase/ml/TensorCollection.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  class TensorCollectionCpu : public TensorCollection<Eigen::ThreadPoolDevice>
  {
  public:
    using TensorCollection<Eigen::ThreadPoolDevice>::TensorCollection;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorCollection<Eigen::ThreadPoolDevice>>(this));
    }
  };
};
#endif //TENSORBASE_TENSORCOLLECTIONCPU_H