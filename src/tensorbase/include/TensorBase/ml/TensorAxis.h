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

    bool syncHAndDData(DeviceT& device) { return tensor_dimension_labels_->syncHAndDData(device); };  ///< Sync the host and device labels data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { tensor_dimension_labels_->setDataStatus(h_data_updated, d_data_updated); } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return tensor_dimension_labels_->getDataStatus(); };   ///< Get the status of the host and device labels data

    template<typename T>
    void getLabelsDataPointer(std::shared_ptr<T>& data_copy); ///< TensorAxisConcept labels getter

    template<typename T>
    void getLabelsHDataPointer(std::shared_ptr<T>& data_copy); ///< TensorAxisConcept labels getter
    
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

    template<typename T = TensorT, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(DeviceT& device) const;
    template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
    std::vector<std::string> getLabelsAsStrings(DeviceT& device) const; ///< return a string vector representation of the labels
    template<typename T = TensorT, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
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

    /*
    @brief Match the labels that are derived from a .csv file

    @param[out] select_indices A binary vector indicating the indices of the labels that matched
    @param[in] labels The labels to match
    @param[in] device
    */
    virtual void makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, DeviceT, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, DeviceT& device) = 0;

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
    tensor_dimension_labels_ = labels_select;
    setNLabels(labels_select->getDimensions().at(1));
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
  void TensorAxis<TensorT, DeviceT>::getLabelsDataPointer(std::shared_ptr<T>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T>(tensor_dimension_labels_->getDataPointer()); // required for compilation: no conversion should be done
  }

  template<typename TensorT, typename DeviceT>
  template<typename T>
  inline void TensorAxis<TensorT, DeviceT>::getLabelsHDataPointer(std::shared_ptr<T>& data_copy) {
    if (std::is_same<T, TensorT>::value)
      data_copy = std::reinterpret_pointer_cast<T>(tensor_dimension_labels_->getHDataPointer()); // required for compilation: no conversion should be done
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
  template<typename T, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(DeviceT& device) const
  {
    // NOTE: the host and device should be syncronized for the primary axis
    //       If this is not true, then this needs to be implemented for each device 
    //       due to the need to synchronize the stream on the GPU
    //syncHAndDData(device); // D to H
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
  template<typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(DeviceT& device) const
  {
    // NOTE: the host and device should be syncronized for the primary axis
    //       If this is not true, then this needs to be implemented for each device 
    //       due to the need to synchronize the stream on the GPU
    //syncHAndDData(device); // D to H
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
  template<typename T, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const
  {
    auto labels_row_t = getLabels().slice(offset, span);
    Eigen::Tensor<std::string, 2> row = labels_row_t.unaryExpr([](TensorT& elem) { return elem.getTensorArrayAsString(); });
    std::vector<std::string> labels(row.data(), row.data() + row.size());
    return labels;
  }

  template<typename TensorT, typename DeviceT>
  template<typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
  inline std::vector<std::string> TensorAxis<TensorT, DeviceT>::getLabelsAsStrings(const Eigen::array<Eigen::Index, 2>& offset, const Eigen::array<Eigen::Index, 2>& span) const
  {
    auto labels_row_t = getLabels().slice(offset, span);
    Eigen::Tensor<std::string, 2> row = labels_row_t.unaryExpr([](const TensorT& elem) { return std::to_string(elem); });
    std::vector<std::string> labels(row.data(), row.data() + row.size());
    return labels;
  }

  template<typename TensorT>
  class TensorAxisDefaultDevice : public TensorAxis<TensorT, Eigen::DefaultDevice>
  {
  public:
    TensorAxisDefaultDevice() = default;  ///< Default constructor
    TensorAxisDefaultDevice(const std::string& name, const size_t& n_dimensions, const size_t& n_labels) : TensorAxis(name, n_dimensions, n_labels) { this->setLabels(); };
    TensorAxisDefaultDevice(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) : TensorAxis(name) { this->setDimensionsAndLabels(dimensions, labels); };
    ~TensorAxisDefaultDevice() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 2>& labels) override;
    void setLabels() override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& labels, Eigen::DefaultDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& indices_sort, Eigen::DefaultDevice& device) override;
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& labels_select, Eigen::DefaultDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::DefaultDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::DefaultDevice& device) override;
    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::DefaultDevice& device) override;
    void makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::DefaultDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<TensorT, Eigen::DefaultDevice>>(this));
    }
  };
  template<typename TensorT>
  void TensorAxisDefaultDevice<TensorT>::setLabels(const Eigen::Tensor<TensorT, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataDefaultDevice<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_.reset(new TensorDataDefaultDevice<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& labels, Eigen::DefaultDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataDefaultDevice<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat.getDataPointer().get(), labels_concat.getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);

    // Move over the new labels
    this->tensor_dimension_labels_ = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(labels_concat);
  }

  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& indices_sort, Eigen::DefaultDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataDefaultDevice<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({(int)this->getNDimensions(), (int)this->getNLabels()}));
    indices_sort_tmp.setData();

    // create a dummy index along the dimension
    TensorDataDefaultDevice<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_dimension_reshape(indices_dimension.getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // normalize and broadcast the dummy indices across the tensor    
    auto indices_dimension_norm = indices_dimension_reshape - indices_dimension_reshape.constant(1);
    auto indices_dimension_bcast_values = indices_dimension_norm.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_reshape(indices->getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));
    auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(this->getNDimensions());
    auto indices_view_bcast_values = indices_view_norm.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // update the indices_sort_values
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());
    indices_sort_values.device(device) = indices_view_bcast_values + indices_dimension_bcast_values + indices_sort_values.constant(1);

    // move over the results
    indices_sort = std::make_shared<TensorDataDefaultDevice<int, 2>>(indices_sort_tmp);
  }

  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& labels_select, Eigen::DefaultDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataDefaultDevice<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    axis_size_value.device(device) = indices_values.clip(0, 1).sum();

    // allocate memory for the new labels
    TensorDataDefaultDevice<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> new_labels_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(new_labels);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataDefaultDevice<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), indices_select.getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(indices_select);

    // perform the reduction and move over the output
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    labels_select = new_labels_ptr;
  }
  template<typename TensorT>
  inline bool TensorAxisDefaultDevice<TensorT>::loadLabelsBinary(const std::string & filename, Eigen::DefaultDevice & device)
  {
    // Read in the the labels
    this->setDataStatus(true, false);
    Eigen::Tensor<TensorT, 2> labels_data((int)this->n_dimensions_, (int)this->n_labels_);
    DataFile::loadDataBinary<TensorT, 2>(filename + ".ta", labels_data);
    this->getLabels() = labels_data;
    this->syncHAndDData(device); // H to D
    return true;
  }
  template<typename TensorT>
  inline bool TensorAxisDefaultDevice<TensorT>::storeLabelsBinary(const std::string & filename, Eigen::DefaultDevice & device)
  {
    // Store the labels
    this->syncHAndDData(device); // D to H
    DataFile::storeDataBinary<TensorT, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
  }

  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::DefaultDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataDefaultDevice<TensorT, 2> labels_converted(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    labels_converted.setData();
    labels_converted.syncHAndDData(device);
    labels_converted.convertFromStringToTensorT(labels, device);

    // Make the indices select
    TensorDataDefaultDevice<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    indices_select.setData();
    indices_select.syncHAndDData(device);

    // Determine the unique input axis labels
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_unique_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_unique_values_bcast = indices_unique_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v1(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_v1_bcast = labels_new_v1.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v2(labels_converted.getDataPointer().get(), 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1));
    auto labels_new_v2_bcast = labels_new_v2.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    // Select the overlapping labels
    auto labels_new_v1v2_select = (labels_new_v1_bcast == labels_new_v2_bcast).select(indices_unique_values_bcast.constant(1), indices_unique_values_bcast.constant(0));
    std::cout << "labels_new_v1v2_select\n" << labels_new_v1v2_select << std::endl; //DEBUG
    // Reduct along the axis dimensions and then cum sum along the axis labels
    auto labels_new_v1v2_prod = labels_new_v1v2_select.sum(Eigen::array<Eigen::Index, 2>({ 1, 3 })).clip(0, 1);
    std::cout << "labels_new_v1v2_prod\n" << labels_new_v1v2_prod << std::endl; //DEBUG
    auto labels_new_v1v2_cumsum = (labels_new_v1v2_prod.cumsum(1) * labels_new_v1v2_prod).cumsum(2) * labels_new_v1v2_prod;
    std::cout << "labels_new_v1v2_cumsum\n" << labels_new_v1v2_cumsum << std::endl; //DEBUG
    // Select the unique labels marked with a 1
    auto labels_unique_v1v2 = (labels_new_v1v2_cumsum == labels_new_v1v2_cumsum.constant(1)).select(labels_new_v1v2_cumsum.constant(1), labels_new_v1v2_cumsum.constant(0));
    // Collapse back to 1xnlabels by summing along one of the axis labels dimensions
    auto labels_unique = labels_unique_v1v2.sum(Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1);
    std::cout << "labels_unique\n" << labels_unique << std::endl; //DEBUG

    // Determine the new labels to add to the axis 
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_select_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_select_values_bcast = indices_select_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
    // Broadcast along the labels and along the dimensions and for both the labels and the new labels
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_);
    auto labels_values_bcast = labels_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_values(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_values_bcast = labels_new_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
    // Select and sum along the labels and multiple along the dimensions
    auto labels_selected = (labels_values_bcast == labels_new_values_bcast).select(indices_select_values_bcast.constant(1), indices_select_values_bcast.constant(0)).sum(
      Eigen::array<Eigen::Index, 2>({ 3, 4 })).prod(Eigen::array<Eigen::Index, 1>({ 1 })); // not new > 1, new = 0
    // Invert the selection
    auto labels_new = (labels_selected > labels_selected.constant(0)).select(labels_selected.constant(0), labels_selected.constant(1)); // new = 1, not new = 0
    std::cout << "labels_new\n" << labels_new << std::endl; //DEBUG

    // Determine the new and unique labels to add to the axis
    auto labels_new_unique = labels_new * labels_unique;
    // Broadcast back to n_dimensions x labels
    auto labels_new_unique_bcast = labels_new_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));
    std::cout << "labels_new_unique_bcast\n" << labels_new_unique_bcast << std::endl; //DEBUG

    // Store the selection indices
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), (int)labels.dimension(0), (int)labels.dimension(1));
    indices_select_values.device(device) = labels_new_unique_bcast;
    std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>> indices_select_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(indices_select);

    // Determine the number of new labels
    TensorDataDefaultDevice<int, 1> n_labels_new(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_labels_new.setData();
    n_labels_new.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_labels_new_values(n_labels_new.getDataPointer().get());
    n_labels_new_values.device(device) = indices_select_values.sum() / n_labels_new_values.constant((int)labels.dimension(0));
    n_labels_new.syncHAndDData(device); // NOTE: need sync for Gpu
    std::cout << "n_labels_new.getData()\n" << n_labels_new.getData() << std::endl; //DEBUG

    // Allocate memory for the new labels
    TensorDataDefaultDevice<TensorT, 2> labels_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, n_labels_new.getData()(0) }));
    labels_select.setData();
    labels_select.syncHAndDData(device);
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> labels_select_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(labels_select);

    // Select the labels
    labels_converted.select(labels_select_ptr, indices_select_ptr, device);

    // Append the selected labels to the axis
    this->appendLabelsToAxis(labels_select_ptr, device);
  }

  template<typename TensorT>
  inline void TensorAxisDefaultDevice<TensorT>::makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::DefaultDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataDefaultDevice<TensorT, 2> select_labels(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    select_labels.setData();
    select_labels.syncHAndDData(device);
    select_labels.convertFromStringToTensorT(labels, device);

    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_names_selected_reshape(select_labels.getDataPointer().get(), 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1));
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1 }));

    // broadcast the axis labels the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_reshape(this->tensor_dimension_labels_->getDataPointer().get(), (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1);
    auto labels_bcast = labels_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // broadcast the tensor indices the size of the labels queried
    TensorDataDefaultDevice<int, 1> select_indices_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->getNLabels() }));
    select_indices_tmp.setData();
    select_indices_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 4>> indices_reshape(select_indices_tmp.getDataPointer().get(), 1, (int)this->getNLabels(), 1, 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast.constant(1), indices_bcast.constant(0)).sum(
      Eigen::array<Eigen::Index, 2>({ 2, 3 })).prod(Eigen::array<Eigen::Index, 1>({ 0 })).clip(0, 1); // matched = 1, not-matched = 0;

    // assign the select indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> select_indices_values(select_indices_tmp.getDataPointer().get(), (int)this->getNLabels());
    select_indices_values.device(device) = selected;
    select_indices = std::make_shared<TensorDataDefaultDevice<int, 1>>(select_indices_tmp);
  }

  template<typename TensorT>
  class TensorAxisCpu : public TensorAxis<TensorT, Eigen::ThreadPoolDevice>
  {
  public:
    TensorAxisCpu() = default;  ///< Default constructor
    TensorAxisCpu(const std::string& name, const size_t& n_dimensions, const size_t& n_labels) : TensorAxis(name, n_dimensions, n_labels) { this->setLabels(); };
    TensorAxisCpu(const std::string& name, const Eigen::Tensor<std::string, 1>& dimensions, const Eigen::Tensor<TensorT, 2>& labels) : TensorAxis(name) { this->setDimensionsAndLabels(dimensions, labels); };
    ~TensorAxisCpu() = default; ///< Default destructor
    void setLabels(const Eigen::Tensor<TensorT, 2>& labels) override;
    void setLabels() override;
    void appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels, Eigen::ThreadPoolDevice & device) override;
    void makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& indices_sort, Eigen::ThreadPoolDevice& device) override;
    void selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels_select, Eigen::ThreadPoolDevice& device) override;
    bool loadLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override;
    bool storeLabelsBinary(const std::string& filename, Eigen::ThreadPoolDevice& device) override;
    void appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice& device) override;
    void makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice& device) override;
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorAxis<TensorT, Eigen::ThreadPoolDevice>>(this));
    }
  };
  template<typename TensorT>
  void TensorAxisCpu<TensorT>::setLabels(const Eigen::Tensor<TensorT, 2>& labels) {
    Eigen::array<Eigen::Index, 2> labels_dims = labels.dimensions();
    this->tensor_dimension_labels_.reset(new TensorDataCpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData(labels);
    this->setNLabels(labels.dimension(1));
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::setLabels()
  {
    Eigen::array<Eigen::Index, 2> labels_dims;
    labels_dims.at(0) = this->n_dimensions_;
    labels_dims.at(1) = this->n_labels_;
    this->tensor_dimension_labels_.reset(new TensorDataCpu<TensorT, 2>(labels_dims));
    this->tensor_dimension_labels_->setData();
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::appendLabelsToAxis(const std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(labels->getDimensions().at(0) == this->n_dimensions_);

    // update the number of labels
    n_labels_ += labels->getDimensions().at(1);

    // Allocate additional memory for the new labels
    TensorDataCpu<TensorT, 2> labels_concat(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    labels_concat.setData();
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_concat_values(labels_concat.getDataPointer().get(), labels_concat.getDimensions());

    // Concatenate the new labels to the axis
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> new_labels_values(labels->getDataPointer().get(), labels->getDimensions());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), this->tensor_dimension_labels_->getDimensions());
    labels_concat_values.device(device) = labels_values.concatenate(new_labels_values, 1);

    // Move over the new labels
    this->tensor_dimension_labels_ = std::make_shared<TensorDataCpu<TensorT, 2>>(labels_concat);
  }

  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::makeSortIndices(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>>& indices_sort, Eigen::ThreadPoolDevice & device)
  {
    // allocate memory for the indices and set the values to zero
    TensorDataCpu<int, 2> indices_sort_tmp(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), (int)this->getNLabels() }));
    indices_sort_tmp.setData();

    // create a dummy index along the dimension
    TensorDataCpu<int, 1> indices_dimension(Eigen::array<Eigen::Index, 1>({ (int)this->getNDimensions() }));
    indices_dimension.setData();
    for (int i = 0; i < this->getNDimensions(); ++i) {
      indices_dimension.getData()(i) = i + 1;
    }
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_dimension_reshape(indices_dimension.getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // normalize and broadcast the dummy indices across the tensor    
    auto indices_dimension_norm = indices_dimension_reshape - indices_dimension_reshape.constant(1);
    auto indices_dimension_bcast_values = indices_dimension_norm.broadcast(Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));

    // normalize and broadcast the indices across the tensor
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_view_reshape(indices->getDataPointer().get(), Eigen::array<Eigen::Index, 2>({ 1, (int)this->getNLabels() }));
    auto indices_view_norm = (indices_view_reshape - indices_view_reshape.constant(1)) * indices_view_reshape.constant(this->getNDimensions());
    auto indices_view_bcast_values = indices_view_norm.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->getNDimensions(), 1 }));

    // update the indices_sort_values
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_sort_values(indices_sort_tmp.getDataPointer().get(), indices_sort_tmp.getDimensions());
    indices_sort_values.device(device) = indices_view_bcast_values + indices_dimension_bcast_values + indices_sort_values.constant(1);

    // move over the results
    indices_sort = std::make_shared<TensorDataCpu<int, 2>>(indices_sort_tmp);
  }

  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::selectFromAxis(const std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& indices, std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>>& labels_select, Eigen::ThreadPoolDevice & device)
  {
    // temporary memory for calculating the sum of the new axis
    TensorDataCpu<int, 1> axis_size(Eigen::array<Eigen::Index, 1>({ 1 }));
    axis_size.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 0>> axis_size_value(axis_size.getDataPointer().get());

    // calculate the new axis size
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_values(indices->getDataPointer().get(), 1, (int)indices->getTensorSize());
    axis_size_value.device(device) = indices_values.clip(0, 1).sum();

    // allocate memory for the new labels
    TensorDataCpu<TensorT, 2> new_labels(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, axis_size.getData()(0) }));
    new_labels.setData();
    std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>> new_labels_ptr = std::make_shared<TensorDataCpu<TensorT, 2>>(new_labels);

    // broadcast the indices across the dimensions and allocate to memory
    TensorDataCpu<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, (int)this->n_labels_ }));
    indices_select.setData();
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), indices_select.getDimensions());
    indices_select_values.device(device) = indices_values.broadcast(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, 1 }));
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices_select_ptr = std::make_shared<TensorDataCpu<int, 2>>(indices_select);

    // perform the reduction and move over the output
    this->tensor_dimension_labels_->select(new_labels_ptr, indices_select_ptr, device);
    labels_select = new_labels_ptr;
  }
  template<typename TensorT>
  inline bool TensorAxisCpu<TensorT>::loadLabelsBinary(const std::string & filename, Eigen::ThreadPoolDevice & device)
  {
    // Read in the the labels
    this->setDataStatus(true, false);
    Eigen::Tensor<TensorT, 2> labels_data((int)this->n_dimensions_, (int)this->n_labels_);
    DataFile::loadDataBinary<TensorT, 2>(filename + ".ta", labels_data);
    this->getLabels() = labels_data;
    this->syncHAndDData(device); // H to D
    return true;
  }
  template<typename TensorT>
  inline bool TensorAxisCpu<TensorT>::storeLabelsBinary(const std::string & filename, Eigen::ThreadPoolDevice & device)
  {
    // Store the labels
    this->syncHAndDData(device); // D to H
    DataFile::storeDataBinary<TensorT, 2>(filename + ".ta", this->getLabels());
    this->setDataStatus(false, true);
    return true;
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::appendLabelsToAxisFromCsv(const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataCpu<TensorT, 2> labels_converted(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    labels_converted.setData();
    labels_converted.syncHAndDData(device);
    labels_converted.convertFromStringToTensorT(labels, device);

    // Make the indices select
    TensorDataCpu<int, 2> indices_select(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    indices_select.setData();
    indices_select.syncHAndDData(device);

    // Determine the unique input axis labels
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_unique_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_unique_values_bcast = indices_unique_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v1(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_v1_bcast = labels_new_v1.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1) }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_v2(labels_converted.getDataPointer().get(), 1, 1, 1, (int)labels.dimension(0), (int)labels.dimension(1));
    auto labels_new_v2_bcast = labels_new_v2.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    // Select the overlapping labels
    auto labels_new_v1v2_select = (labels_new_v1_bcast == labels_new_v2_bcast).select(indices_unique_values_bcast.constant(1), indices_unique_values_bcast.constant(0));
    // Reduct along the axis dimensions and then cum sum along the axis labels
    auto labels_new_v1v2_prod = labels_new_v1v2_select.sum(Eigen::array<Eigen::Index, 2>({ 1, 3 })).clip(0, 1);
    auto labels_new_v1v2_cumsum = (labels_new_v1v2_prod.cumsum(1) * labels_new_v1v2_prod).cumsum(2) * labels_new_v1v2_prod;
    // Select the unique labels marked with a 1
    auto labels_unique_v1v2 = (labels_new_v1v2_cumsum == labels_new_v1v2_cumsum.constant(1)).select(labels_new_v1v2_cumsum.constant(1), labels_new_v1v2_cumsum.constant(0));
    // Collapse back to 1xnlabels by summing along one of the axis labels dimensions
    auto labels_unique = labels_unique_v1v2.sum(Eigen::array<Eigen::Index, 1>({ 2 })).clip(0, 1);

    // Determine the new labels to add to the axis 
    Eigen::TensorMap<Eigen::Tensor<int, 5>> indices_select_values5(indices_select.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto indices_select_values_bcast = indices_select_values5.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
    // Broadcast along the labels and along the dimensions and for both the labels and the new labels
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_values(this->tensor_dimension_labels_->getDataPointer().get(), 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_);
    auto labels_values_bcast = labels_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1 }));
    Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> labels_new_values(labels_converted.getDataPointer().get(), 1, (int)labels.dimension(0), (int)labels.dimension(1), 1, 1);
    auto labels_new_values_bcast = labels_new_values.broadcast(Eigen::array<Eigen::Index, 5>({ 1, 1, 1, (int)this->n_dimensions_, (int)this->n_labels_ }));
    // Select and sum along the labels and multiple along the dimensions
    auto labels_selected = (labels_values_bcast == labels_new_values_bcast).select(indices_select_values_bcast.constant(1), indices_select_values_bcast.constant(0)).sum(
      Eigen::array<Eigen::Index, 2>({ 3, 4 })).prod(Eigen::array<Eigen::Index, 1>({ 1 })); // not new > 1, new = 0
    // Invert the selection
    auto labels_new = (labels_selected > labels_selected.constant(0)).select(labels_selected.constant(0), labels_selected.constant(1)); // new = 1, not new = 0

    // Determine the new and unique labels to add to the axis
    auto labels_new_unique = labels_new * labels_unique;
    // Broadcast back to n_dimensions x labels
    auto labels_new_unique_bcast = labels_new_unique.broadcast(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), 1 }));

    // Store the selection indices
    Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_select_values(indices_select.getDataPointer().get(), (int)labels.dimension(0), (int)labels.dimension(1));
    indices_select_values.device(device) = labels_new_unique_bcast;
    std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 2>> indices_select_ptr = std::make_shared<TensorDataCpu<int, 2>>(indices_select);

    // Determine the number of new labels
    TensorDataCpu<int, 1> n_labels_new(Eigen::array<Eigen::Index, 1>({ 1 }));
    n_labels_new.setData();
    n_labels_new.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_labels_new_values(n_labels_new.getDataPointer().get());
    n_labels_new_values.device(device) = indices_select_values.sum() / n_labels_new_values.constant((int)labels.dimension(0));
    n_labels_new.syncHAndDData(device); // NOTE: need sync for Gpu

    // Allocate memory for the new labels
    TensorDataCpu<TensorT, 2> labels_select(Eigen::array<Eigen::Index, 2>({ (int)this->n_dimensions_, n_labels_new.getData()(0) }));
    labels_select.setData();
    labels_select.syncHAndDData(device);
    std::shared_ptr<TensorData<TensorT, Eigen::ThreadPoolDevice, 2>> labels_select_ptr = std::make_shared<TensorDataCpu<TensorT, 2>>(labels_select);

    // Select the labels
    labels_converted.select(labels_select_ptr, indices_select_ptr, device);

    // Append the selected labels to the axis
    this->appendLabelsToAxis(labels_select_ptr, device);
  }
  template<typename TensorT>
  inline void TensorAxisCpu<TensorT>::makeSelectIndicesFromCsv(std::shared_ptr<TensorData<int, Eigen::ThreadPoolDevice, 1>>& select_indices, const Eigen::Tensor<std::string, 2>& labels, Eigen::ThreadPoolDevice & device)
  {
    assert(this->n_dimensions_ == (int)labels.dimension(0));

    // Convert to TensorT
    TensorDataCpu<TensorT, 2> select_labels(Eigen::array<Eigen::Index, 2>({ (int)labels.dimension(0), (int)labels.dimension(1) }));
    select_labels.setData();
    select_labels.syncHAndDData(device);
    select_labels.convertFromStringToTensorT(labels, device);

    // reshape to match the axis labels shape and broadcast the length of the labels
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_names_selected_reshape(select_labels.getDataPointer().get(), 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1));
    auto labels_names_selected_bcast = labels_names_selected_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1 }));

    // broadcast the axis labels the size of the labels queried
    Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> labels_reshape(this->tensor_dimension_labels_->getDataPointer().get(), (int)this->getNDimensions(), (int)this->getNLabels(), 1, 1);
    auto labels_bcast = labels_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // broadcast the tensor indices the size of the labels queried
    TensorDataCpu<int, 1> select_indices_tmp(Eigen::array<Eigen::Index, 1>({ (int)this->getNLabels() }));
    select_indices_tmp.setData();
    select_indices_tmp.syncHAndDData(device);
    Eigen::TensorMap<Eigen::Tensor<int, 4>> indices_reshape(select_indices_tmp.getDataPointer().get(), 1, (int)this->getNLabels(), 1, 1);
    auto indices_bcast = indices_reshape.broadcast(Eigen::array<Eigen::Index, 4>({ (int)this->getNDimensions(), 1, select_labels.getDimensions().at(0), select_labels.getDimensions().at(1) }));

    // select the indices and reduce back to a 1D Tensor
    auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast.constant(1), indices_bcast.constant(0)).sum(
      Eigen::array<Eigen::Index, 2>({ 2, 3 })).prod(Eigen::array<Eigen::Index, 1>({ 0 })).clip(0, 1); // matched = 1, not-matched = 0;

    // assign the select indices
    Eigen::TensorMap<Eigen::Tensor<int, 1>> select_indices_values(select_indices_tmp.getDataPointer().get(), (int)this->getNLabels());
    select_indices_values.device(device) = selected;
    select_indices = std::make_shared<TensorDataCpu<int, 1>>(select_indices_tmp);
  }
};

// Cereal registration of TensorTs: float, int, char, double
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisDefaultDevice<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisDefaultDevice<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisDefaultDevice<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisDefaultDevice<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisDefaultDevice<TensorBase::TensorArray8<char>>);

CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<int>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<float>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<double>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<char>);
CEREAL_REGISTER_TYPE(TensorBase::TensorAxisCpu<TensorBase::TensorArray8<char>>);
#endif //TENSORBASE_TENSORAXIS_H