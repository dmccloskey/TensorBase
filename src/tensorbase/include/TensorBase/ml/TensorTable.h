/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORTABLE_H
#define TENSORBASE_TENSORTABLE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorAxis.h>

namespace TensorBase
{
  /**
    @brief Class for managing Tensor data and associated Axes
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorTable
  {
  public:
    TensorTable() = default;  ///< Default constructor
    ~TensorTable() = default; ///< Default destructor

    void setId(const int& id) { id_ = id; }; ///< id setter
    int getId() const { return id_; }; ///< id getter

    void setName(const std::string& name) { name_ = name; }; ///< name setter
    std::string getName() const { return name_; }; ///< name getter

    /**
      @brief Tensor Axes setter

      The method sets the tensor axes and initializes 
      the indices, indices_view, is_modified, in_memory, and
      is_shardable attributes

      @param[in] tensor_axes A vector of TensorAxis
    */
    void setAxes(const std::vector<TensorAxis>& tensor_axes); ///< axes setter

    virtual void initData() = 0;  ///< DeviceT specific initializer

    std::map<std::string, std::shared_ptr<TensorAxis>>& getAxes() { return axes_; }; ///< axes getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndicesView() { return indices_view_; }; ///< indices_view getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsModified() { return is_modified_; }; ///< is_modified getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getInMemory() { return in_memory_; }; ///< in_memory getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsShardable() { return is_shardable_; }; ///< is_shardable getter
    Eigen::array<Eigen::Index, TDim>& getDimensions() { return dimensions_; }  ///< dimensions getter
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& getData() { return data_; }; ///< data getter

    void clear();  ///< clears the axes and all associated data

  private:
    int id_ = -1;
    std::string name_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxis>> axes_; ///< primary axis is dim=0
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> indices_; ///< starting at 1
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> indices_view_; ///< sorted and/or selected indices
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> is_modified_;
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> in_memory_;
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> is_shardable_;

    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> data_; ///< The actual tensor data
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };

  template<typename TensorT, typename DeviceT, int TDim>
  void TensorTable<TensorT, DeviceT, TDim>::setAxes(const std::vector<TensorAxis>& tensor_axes) {
    // Determine the overall dimensions of the tensor
    int axis_cnt = 0;
    for (const TensorAxis& axis : tensor_axes) {
      auto found = axes_.emplace(axis.getName(), std::shared_ptr<TensorAxis>(new TensorAxis(axis)));
      dimensions_.at(axis_cnt) = axis.getNLabels();

      // Build the default indices
      Eigen::Tensor<int, 1> indices(axis.getNLabels());
      for (int i = 0; i < axis.getNLabels(); ++i) {
        indices(i) = i + 1;
      }
      indices_.emplace(axis.getName(), std::shared_ptr<Eigen::Tensor<int, 1>>(new Eigen::Tensor<int, 1>(indices)));

      // Set the defaults
      Eigen::Tensor<int, 1> is_modified(axis.getNLabels());
      is_modified.setZero();
      is_modified_.emplace(axis.getName(), std::shared_ptr<Eigen::Tensor<int, 1>>(new Eigen::Tensor<int, 1>(is_modified)));
      Eigen::Tensor<int, 1> in_memory(axis.getNLabels());
      in_memory.setZero();
      in_memory_.emplace(axis.getName(), std::shared_ptr<Eigen::Tensor<int, 1>>(new Eigen::Tensor<int, 1>(in_memory)));
      Eigen::Tensor<int, 1> is_shardable(axis.getNLabels());
      if (axis_cnt == 0)
        is_shardable.setConstant(1);
      else
        is_shardable.setZero();
      is_shardable_.emplace(axis.getName(), std::shared_ptr<Eigen::Tensor<int, 1>>(new Eigen::Tensor<int, 1>(is_shardable)));

      // Next iteration
      ++axis_cnt;
    }
    indices_view_ = indices_;

    // Allocate memory for the tensor
    initData();
  };

  template<typename TensorT, typename DeviceT, int TDim>
  void TensorTable<TensorT, DeviceT, TDim>::clear() {
    axes_.clear();
    dimensions_ = Eigen::array<Eigen::Index, TDim>();
    indices_.clear();
    indices_view_.clear();
    is_modified_.clear();
    in_memory_.clear();
    is_shardable_.clear();
    data_.reset();
  };

  template<typename TensorT, int TDim>
  class TensorTableDefaultDevice: public TensorTable<TensorT, Eigen::DefaultDevice, TDim>
  {
  public:
    void initData();
  };

  template<typename TensorT, int TDim>
  void TensorTableDefaultDevice<TensorT, TDim>::initData() {
    getData().reset(new TensorDataDefaultDevice<TensorT, TDim>(getDimensions()));
  }
};
#endif //TENSORBASE_TENSORTABLE_H