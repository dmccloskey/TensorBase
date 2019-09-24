/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSION_H
#define TENSORBASE_TENSORDIMENSION_H

#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorArray.h>
#include <TensorBase/io/DataFile.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

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
    TensorDimension(const std::string& name, const std::string& dir) : name_(name), dir_(dir){};
    TensorDimension(const std::string& name, const std::string& dir, const size_t& n_labels) : name_(name), dir_(dir), n_labels_(n_labels) { setLabels(); };
    TensorDimension(const std::string& name, const std::string& dir, const Eigen::Tensor<TensorT, 1>& labels) : name_(name), dir_(dir) { setLabels(labels); };
    virtual ~TensorDimension() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

    void setDir(const std::string& dir) { dir_ = dir; }; ///< dir setter
    std::string getDir() const { return dir_; }; ///< dir getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter

    virtual void setLabels(const Eigen::Tensor<TensorT, 1>& labels) = 0; ///< labels setter
    virtual void setLabels() = 0; ///< labels setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> getLabels() { return labels_->getData(); };  ///< labels getter

    bool syncHAndDData(DeviceT& device) { return labels_->syncHAndDData(device); };  ///< Sync the host and device labels data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { labels_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return labels_->getDataStatus(); };   ///< Get the status of the host and device labels data

    /**
      @brief Load labels from file

      @param[in] filename The name of the data file
      @param[in] device

      @returns Status True on success, False if not
    */
    virtual bool loadLabelsBinary(const std::string& dir, DeviceT& device) = 0;

    /**
      @brief Write labels to file

      @param[in] filename The name of the data file
      @param[in] device

      @returns Status True on success, False if not
    */
    virtual bool storeLabelsBinary(const std::string& dir, DeviceT& device) = 0;

  protected:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter

    int id_ = -1;
    std::string name_ = "";
    std::string dir_ = "";
    size_t n_labels_ = 0;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_; ///< The actual tensor data

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(id_, name_, dir_, n_labels_);
    }
  };

  template<typename TensorT>
  class TensorDimensionDefaultDevice : public TensorDimension<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorDimensionDefaultDevice() = default;  ///< Default constructor
    TensorDimensionDefaultDevice(const std::string& name, const std::string& dir): TensorDimension(name, dir) {};
    TensorDimensionDefaultDevice(const std::string& name, const std::string& dir, const size_t& n_labels) : TensorDimension(name, dir, n_labels) {};
    TensorDimensionDefaultDevice(const std::string& name, const std::string& dir, const Eigen::Tensor<TensorT, 1>& labels): TensorDimension(name, dir, labels) {};
    ~TensorDimensionDefaultDevice() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataDefaultDevice<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_.reset(new TensorDataDefaultDevice<TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& dir, Eigen::DefaultDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->syncHAndDData(device); // H to D
    };
    bool storeLabelsBinary(const std::string& dir, Eigen::DefaultDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->setDataStatus(false, true);
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::DefaultDevice>>(this));
    }
  };

  template<typename TensorT>
  class TensorDimensionCpu : public TensorDimension<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorDimensionCpu() = default;  ///< Default constructor
    TensorDimensionCpu(const std::string& name, const std::string& dir) : TensorDimension(name, dir) {};
    TensorDimensionCpu(const std::string& name, const std::string& dir, const size_t& n_labels) : TensorDimension(name, dir, n_labels) {};
    TensorDimensionCpu(const std::string& name, const std::string& dir, const Eigen::Tensor<TensorT, 1>& labels) : TensorDimension(name, dir, labels) {};
    ~TensorDimensionCpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 1>& labels) override {
      Eigen::array<Eigen::Index, 1> dimensions = labels.dimensions();
      this->labels_.reset(new TensorDataCpu<TensorT, 1>(dimensions));
      this->labels_->setData(labels);
      this->setNLabels(labels.size());
    };
    void setLabels() override {
      Eigen::array<Eigen::Index, 1> dimensions;
      dimensions.at(0) = this->n_labels_;
      this->labels_.reset(new TensorDataCpu<TensorT, 1>(dimensions));
      this->labels_->setData();
    };
    bool loadLabelsBinary(const std::string& dir, Eigen::ThreadPoolDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->syncHAndDData(device); // H to D
    };
    bool storeLabelsBinary(const std::string& dir, Eigen::ThreadPoolDevice& device) override {
      this->syncHAndDData(device); // D to H
      DataFile::loadDataBinary<TensorT, 1>(filename, this->getLabels());
      this->setDataStatus(false, true);
    };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorDimension<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionDefaultDevice<TensorBase::TensorArray8<char>>);

CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDimensionCpu<TensorBase::TensorArray8<char>>);

#endif //TENSORBASE_TENSORDIMENSION_H