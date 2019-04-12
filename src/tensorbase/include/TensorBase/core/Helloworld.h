#ifndef TENSORBASE_HELLOWORLD_H
#define TENSORBASE_HELLOWORLD_H

namespace TensorBase
{

  class Helloworld
  {
public:
    /// Default constructor
    Helloworld();    
    /// Destructor
    ~Helloworld();

    double addNumbers(const double& x, const double& y) const;

  };
}

#endif //TENSORBASE_HELLOWORLD_H