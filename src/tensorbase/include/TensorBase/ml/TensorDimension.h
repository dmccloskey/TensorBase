/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSION_H
#define TENSORBASE_TENSORDIMENSION_H

#include <TensorBase/ml/TensorData.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

namespace TensorBase
{
  /**
    @brief Tensor dimension class
  */
  template<typename TensorT, typename DeviceT>
  class TensorDimension
  {
  public:
    TensorDimension() = default;  ///< Default constructor
    TensorDimension(const std::string& name) : name_(name) {};
    TensorDimension(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) : name_(name) { setLabels(labels); };
    virtual ~TensorDimension() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter

    virtual void setLabels(const Eigen::Tensor<TensorT, 1>& labels) = 0; ///< labels setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> getLabels() { return labels_->getData(); };  ///< labels getter

  protected:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_labels_ = 0;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_; ///< The actual tensor data

    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_labels_, labels_);
    //	}
  };

  template<typename TensorT>
  class TensorDimensionDefaultDevice : public TensorDimension<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorDimensionDefaultDevice() = default;  ///< Default constructor
    TensorDimensionDefaultDevice(const std::string& name) { setName(name); };
    TensorDimensionDefaultDevice(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) { setName(name); setLabels(labels); };
    ~TensorDimensionDefaultDevice() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataDefaultDevice<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
  };

  template<typename TensorT>
  class TensorDimensionCpu : public TensorDimension<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorDimensionCpu() = default;  ///< Default constructor
    TensorDimensionCpu(const std::string& name) { setName(name); };
    TensorDimensionCpu(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) { setName(name); setLabels(labels); };
    ~TensorDimensionCpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataCpu<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
  };
};
#endif //TENSORBASE_TENSORDIMENSION_H