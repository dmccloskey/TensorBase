/**TODO:  Add copyright*/

#ifndef TENSORDIMENSION_TENSORDIMENSION_H
#define TENSORDIMENSION_TENSORDIMENSION_H

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
    //		archive(batch_size_, memory_size_, layer_size_, 
    //		h_data_, h_output_, h_error_, h_derivative_, h_dt_,
    //		d_data_, d_output_, d_error_, d_derivative_, d_dt_,
    //		h_data_updated_, h_output_updated_, h_error_updated_, h_derivative_updated_, h_dt_updated_,
    //		d_data_updated_, d_output_updated_, d_error_updated_, d_derivative_updated_, d_dt_updated_);
    //	}
  };
};
#endif //TENSORDIMENSION_TENSORDIMENSION_H