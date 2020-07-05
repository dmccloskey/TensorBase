/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDIMENSION_H
#define TENSORBASE_TENSORDIMENSION_H

#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorArray.h>
#include <TensorBase/io/DataFile.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
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
    TensorDimension(const std::string& name) : name_(name){};
    TensorDimension(const std::string& name, const size_t& n_labels) : name_(name), n_labels_(n_labels) { };
    //TensorDimension(const std::string& name, const Eigen::Tensor<TensorT, 1>& labels) : name_(name) { setLabels(labels); };
    virtual ~TensorDimension() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

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
    virtual bool loadLabelsBinary(const std::string& filename, DeviceT& device) = 0;

    /**
      @brief Write labels to file

      @param[in] filename The name of the data file
      @param[in] device

      @returns Status True on success, False if not
    */
    virtual bool storeLabelsBinary(const std::string& filename, DeviceT& device) = 0;

  protected:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_labels_ = 0;
    std::shared_ptr<TensorData<TensorT, DeviceT, 1>> labels_; ///< The actual tensor data

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(id_, name_, n_labels_, labels_);
    }
  };
};

#endif //TENSORBASE_TENSORDIMENSION_H