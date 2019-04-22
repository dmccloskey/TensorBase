/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSION_H
#define TENSORBASE_TENSORDIMENSION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace TensorBase
{
  /**
    @brief Tensor dimension class
  */
  class TensorDimension
  {
  public:
    TensorDimension() = default;  ///< Default constructor
    TensorDimensions(const std::string& name) : name_(name) {};
    TensorDimensions(const std::string& name, const Eigen::Tensor<std::string, 1>& labels) : name_(name) { setLables(labels); };
    ~TensorDimension() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< naem setter
    std::string getName() const { return name_; }; ///< name getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter

    void setLabels(const Eigen::Tensor<std::string, 1>& labels) {
      labels_ = labels;
      setNLabels(labels.size());
    };  ///< labels setter
    Eigen::Tensor<std::string, 1> getLabels() const { return labels_; };  ///< labels getter

  private:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_labels_ = 0;
    Eigen::Tensor<std::string, 1> labels_;

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_labels_, labels_);
    //	}
  };
};
#endif //TENSORBASE_TENSORDIMENSION_H