/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXIS_H
#define TENSORBASE_TENSORAXIS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace TensorBase
{
  /**
    @brief Tensor axis class
  */
  class TensorAxis
  {
  public:
    TensorAxis() = default;  ///< Default constructor
    TensorAxis(const std::string& name,
      const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<std::string, 2>& labels);
    ~TensorAxis() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< naem setter
    std::string getName() const { return name_; }; ///< name getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter
    size_t getNDimensions() const { return n_dimensions_; }; ///< n_labels getter

    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<std::string, 2>& labels) {
      assert(labels.dimension(0) == dimensions.dimension(0));
      tensor_dimension_labels_ = labels;
      tensor_dimension_names_ = dimensions;
      setNDimensions(labels.dimension(0));
      setNLabels(labels.dimension(1));
    };  ///< dimensions and labels setter
    Eigen::Tensor<std::string, 2>& getLabels() { return tensor_dimension_labels_; };  ///< labels getter
    Eigen::Tensor<std::string, 1>& getDimensions() { return tensor_dimension_names_; };  ///< dimensions getter

  private:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter
    void setNDimensions(const size_t& n_dimenions) { n_dimensions_ = n_dimenions; }; ///< n_tensor_dimensions setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_dimensions_ = 0;  ///< the number of "primary keys" or dimensions that compose the axis
    size_t n_labels_ = 0;  ///< The size or length of the axis
    Eigen::Tensor<std::string, 1> tensor_dimension_names_;  ///< array of TensorDimension names
    Eigen::Tensor<std::string, 2> tensor_dimension_labels_;   ///< dim=0: tensor_dimension_name; dim=1 tensor_dimension_labels

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };

  TensorAxis::TensorAxis(const std::string& name,
    const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<std::string, 2>& labels) {
    setName(name);
    setDimensionsAndLabels(dimensions, labels);
  }
};
#endif //TENSORBASE_TENSORAXIS_H