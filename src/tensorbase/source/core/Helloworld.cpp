#include <TensorBase/core/Helloworld.h>

namespace TensorBase
{
  Helloworld::Helloworld()
  {        
  }

  Helloworld::~Helloworld()
  {
  }

  double Helloworld::addNumbers(const double& x, const double& y) const
  {
    return x + y;
  }
}