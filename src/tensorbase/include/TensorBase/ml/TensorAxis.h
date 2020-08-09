/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORAXIS_H
#define TENSORBASE_TENSORAXIS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorArray.h>
#include <TensorBase/io/DataFile.h>
#include <string>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Tensor axis class
  */
  template<typename TensorT, typename DeviceT>
  class TensorAxis
  {
  public:
    TensorAxis() = default;  ///< Default constructor
    TensorAxis(const std::string& name) : name_(name) {};
    TensorAxis(const std::string& name, const size_t& n_dimensions, const size_t& n_labels): name_(name), n_dimensions_(n_dimensions), n_labels_(n_labels) {};
    virtual ~TensorAxis() = default; ///< Default destructor

    template<typename TensorTOther, typename DeviceTOther>
    inline bool operator==(const TensorAxis<TensorTOther, DeviceTOther>& other) const
    {
      return std::tie(
        id_,
        name_, 
        n_labels_,
        n_dimensions_,
        tensor_dimension_names_,
        *tensor_dimension_labels_.get() // compares pointers instead of the underlying classes
      ) == std::tie(
        other.id_,
        other.name_,
        other.n_labels_,
        other.n_dimensions_,
        other.tensor_dimension_names_,
        *other.tensor_dimension_labels_.get()
      );
    }

    inline bool operator!=(const TensorAxis& other) const
    {
      return !(*this == other);
    }

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< naem setter
    std::string getName() const { return name_; }; ///< name getter

    size_t getNLabels() const { return n_labels_; }; ///< n_labels getter
    size_t getNDimensions() const { return n_dimensions_; }; ///< n_labels getter

    void setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels); ///< dimensions and labels setter
    void setDimensions(const Eigen::Tensor<std::string, 1>& dimensions); ///< dimensions setter
    virtual void setLabels(const Eigen::Tensor<TensorT, 2>& labels) = 0; ///< dimensions and labels setter
    virtual void setLabels() = 0; ///< labels setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getLabels() { return tensor_dimension_labels_->getData(); };  ///< labels getter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getLabels() const { return tensor_dimension_labels_->getData(); };  ///< labels getter
    Eigen::TensorMap<Eigen::Tensor<std::string, 1>> getDimensions() { Eigen::TensorMap<Eigen::Tensor<std::string, 1>> data(tensor_dimension_names_.data(), (int)this->n_dimensions_); return data; } ///< dimensions getter

    bool syncDData(DeviceT& device) { return tensor_dimension_labels_->syncDData(device); };  ///< Sync the device labels data
    bool syncHData(DeviceT& device) { return tensor_dimension_labels_->syncHData(device); };  ///< Sync the host labels data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_dimension_labels_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return tensor_dimension_labels_->getDataStatus(); };   ///< Get the status of the host and device labels data

    template<typename T>
    void getLabelsDataPointer(std::shared_ptr<T[]>& data_copy); ///< TensorAxisConcept labels getter

    template<typename T>
    void getLabelsHDataPointer(std::shared_ptr<T[]>& data_copy); ///< TensorAxisConcept labels getter

    virtual std::shared_ptr<TensorAxis<TensorT, DeviceT>> copyToHost(DeviceT& device) = 0;
    virtual std::shared_ptr<TensorAxis<TensorT, DeviceT>> copyToDevice(DeviceT& device) = 0;
    
    /*
    @brief Delete from axis based on a selection index

    @param[in] indices The indices to use for selected deletion (of size nLabels)
    @param[in] device
    */
    void deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);

    /*
    @brief Append labels to the axes

    @param[in] labels The new labels to insert
    @param[in] device
    */
    template<typename T>
    void appendLabelsToAxisConcept(const std::shared_ptr<TensorData<T, DeviceT, 2>>& labels, DeviceT& device);
    virtual void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& labels, DeviceT& device) = 0;

    /*
    @brief Sort the labels of the axis

    @param[in] indices The indices to sort the labels by (of size nLabels)
    @param[in] device
    */
    void sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT& device);
    virtual void makeSortIndices(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<int, DeviceT, 2>>& indices_sort, DeviceT& device) = 0;

    /*
    @brief Select labels from the axis

    @param[in] indices The indices to use for selection (of size nLabels)
    @param[in] labels_select The reduced labels
    @param[in] device
    */
    template<typename T>
    void selectFromAxisConcept(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<T, DeviceT, 2>>& labels_select, DeviceT& device);
    virtual void selectFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& labels_select, DeviceT& device) = 0;

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

    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(DeviceT& device) const;
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(DeviceT& device) const;
    template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(DeviceT& device) const; ///< return a string vector representation of the labels
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const;
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const;
    template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const;

    /*
    @brief Append labels to the axis that are derived from a .csv file
      where the labels are in the form of an array of strings
      and the labels maybe reduntant

    @param[in] labels The new labels to insert
    @param[in] device
    */
    virtual void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, DeviceT& device) = 0;

  protected:
    void setNLabels(const size_t& n_labels) { n_labels_ = n_labels; }; ///< n_labels setter
    void setNDimensions(const size_t& n_dimenions) { n_dimensions_ = n_dimenions; }; ///< n_tensor_dimensions setter

    int id_ = -1;
    std::string name_ = "";
    size_t n_dimensions_ = 0;  ///< the number of "primary keys" or dimensions that compose the axis
    size_t n_labels_ = 0;  ///< The size or length of the axis
    std::vector<std::string> tensor_dimension_names_;  ///< array of TensorDimension names
    std::shared_ptr<TensorData<TensorT, DeviceT, 2>> tensor_dimension_labels_; ///< dim=0: tensor_dimension_name; dim=1 tensor_dimension_labels

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    }
  };

  template<typename TensorT, typename DeviceT>
  inline void TensorAxis<TensorT, DeviceT>::setDimensionsAndLabels(const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels)
  {
    assert(labels.dimension(0) == dimensions.dimension(0));
    setDimensions(dimensions);
    setLabels(labels);
  }

  template<typename TensorT, typename DeviceT>
  inline void TensorAxis<TensorT, DeviceT>::setDimensions(const Eigen::Tensor<std::string, 1>& dimensions)
  {
    setNDimensions(dimensions.size());
    this->tensor_dimension_names_.resize(dimensions.size());
    // copy the tensor
    Eigen::TensorMap<Eigen::Tensor<std::string, 1>> data_copy(this->tensor_dimension_names_.data(), (int)this->n_dimensions_);
    data_copy = dimensions;
  }

  template<typename TensorT, typename DeviceT>
  inline void TensorAxis<TensorT, DeviceT>::deleteFromAxis(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // perform the reduction on the labels and update the axis attributes
    std::shared_ptr<TensorData<TensorT, DeviceT, 2>> labels_select;
    selectFromAxis(indices, labels_select, device);
		if (labels_select->getDimensions().at(1) > 0) {
			tensor_dimension_labels_ = labels_select;
			setNLabels(labels_select->getDimensions().at(1));
		}
		else {
			tensor_dimension_labels_.reset();
			setNLabels(labels_select->getDimensions().at(1));
		}
  }

  template<typename TensorT, typename DeviceT>
  inline void TensorAxis<TensorT, DeviceT>::sortLabels(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, DeviceT & device)
  {
    // Broadcast the sort indices across all of the labels
    std::shared_ptr<TensorData<int, DeviceT, 2>> indices_sort;
    makeSortIndices(indices, indices_sort, device);

    // Apply the sort to the labels
    tensor_dimension_labels_->sort(indices_sort, device);
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  void TensorAxis<TensorT, DeviceT>::getLabelsDataPointer(std::shared_ptr<T[]>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T[]>(tensor_dimension_labels_->getDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  inline void TensorAxis<TensorT, DeviceT>::getLabelsHDataPointer(std::shared_ptr<T[]>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T[]>(tensor_dimension_labels_->getHDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  inline void TensorAxis<TensorT, DeviceT>::appendLabelsToAxisConcept(const std::shared_ptr<TensorData<T, DeviceT, 2>>& labels, DeviceT & device) {
    if (std::is_same<T, TensorT>::value) {
      auto labels_copy = std::reinterpret_pointer_cast<TensorData<TensorT, DeviceT, 2>>(labels);
      appendLabelsToAxis(labels_copy, device);
    }
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  inline void TensorAxis<TensorT, DeviceT>::selectFromAxisConcept(const std::shared_ptr<TensorData<int, DeviceT, 1>>& indices, std::shared_ptr<TensorData<T, DeviceT, 2>>& labels_select, DeviceT & device)
  {
    if (std::is_same<T, TensorT>::value) {
      std::shared_ptr<TensorData<TensorT, DeviceT, 2>> labels_new;
      selectFromAxis(indices, labels_new, device);
      labels_select = std::reinterpret_pointer_cast<TensorData<T, DeviceT, 2>>(labels_new);
    }
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<!std::is_fundamental<T>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(DeviceT& device) const
  {
    // NOTE: the host and device should be syncronized for the primary axis
    //       If this is not true, then this needs to be implemented for each device 
    //       due to the need to synchronize the stream on the GPU
    std::vector<std::string> labels;
    for (int i = 0; i < n_dimensions_; i++) {
      for (int j = 0; j < n_labels_; j++) {
        labels.push_back(getLabels()(i, j).getTensorArrayAsString());
      }
    }
    //setDataStatus(false, true);
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(DeviceT& device) const
  {
    // NOTE: the host and device should be syncronized for the primary axis
    //       If this is not true, then this needs to be implemented for each device 
    //       due to the need to synchronize the stream on the GPU
    std::vector<std::string> labels;
    for (int i = 0; i < n_dimensions_; i++) {
      for (int j = 0; j < n_labels_; j++) {
        labels.push_back(std::to_string(static_cast<char>(getLabels()(i, j))));
      }
    }
    //setDataStatus(false, true);
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(DeviceT& device) const
  {
    // NOTE: the host and device should be syncronized for the primary axis
    //       If this is not true, then this needs to be implemented for each device 
    //       due to the need to synchronize the stream on the GPU
    std::vector<std::string> labels;
    for (int i = 0; i < n_dimensions_; i++) {
      for (int j = 0; j < n_labels_; j++) {
        labels.push_back(std::to_string(getLabels()(i, j)));
      }
    }
    //setDataStatus(false, true);
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<!std::is_fundamental<T>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const
  {
    auto labels_row_t = getLabels().slice(offset, span);
    Eigen::Tensor<std::string, 2> row = labels_row_t.unaryExpr([](TensorT& elem) { return elem.getTensorArrayAsString(); });
    std::vector<std::string> labels(row.data(), row.data() + row.size());
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const
  {
    auto labels_row_t = getLabels().slice(offset, span);
    Eigen::Tensor<std::string, 2> row = labels_row_t.unaryExpr([](const TensorT& elem) { return std::to_string(static_cast<char>(elem)); });
    std::vector<std::string> labels(row.data(), row.data() + row.size());
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value || std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, bool>::value, int>>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const
  {
    auto labels_row_t = getLabels().slice(offset, span);
    Eigen::Tensor<std::string, 2> row = labels_row_t.unaryExpr([](const TensorT& elem) { return std::to_string(elem); });
    std::vector<std::string> labels(row.data(), row.data() + row.size());
    return labels;
  }
};
#endif //TENSORBASE_TENSORAXIS_H