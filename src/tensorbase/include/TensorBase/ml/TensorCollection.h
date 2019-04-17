/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORCOLLECTION_H
#define TENSORBASE_TENSORCOLLECTION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorAxis.h>

#include <utility> // Tuple magic for C++ 14
#include <cstddef> // Tuple magic for C++ 14

namespace TensorBase
{
  /**
    @brief Structure for mapping tensor data to labels
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorDataToLabels {
  public:
    std::shared_ptr<TensorData<TensorT, DeviceT, TDim>> tensor;
    std::vector<std::string> labels;
  };

  /**
    @brief Class for managing heterogenous Tensors
  */
  template<typename DeviceT, int TDim, typename... TensorTs>
  class TensorCollection
  {
  public:
    TensorCollection() = default;  ///< Default constructor
    ~TensorCollection() = default; ///< Default destructor

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

    std::map<std::string, std::shared_ptr<TensorAxis>>& getAxes() { return axes_; }; ///< axes getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndices() { return indices_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIndicesView() { return indices_view_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsModified() { return is_modified_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getInMemory() { return in_memory_; }; ///< indices getter
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>>& getIsShardable() { return is_shardable_; }; ///< indices getter

    /**
      @brief Tensor setter

      The method allocates memory to the different tensors according to
      the sizes of the labels (which correspond to axis = 0) and
      TensorTypes associated with each of the labels. The method also
      initializes the indices_to_parts attribute.

      @param[in] tensors A parameter pack of TensorTToLabels classes
    */
    void setTensors(const std::vector<std::vector<std::string>>& labels_types);

  private:
    void clearAxes();  ///< clears the axes

    int id_ = -1;
    std::string name_ = "";

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>();
    std::map<std::string, std::shared_ptr<TensorAxis>> axes_; ///< primary axis is dim=0
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> indices_; ///< starting at 1
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> indices_view_; ///< sorted and/or selected indices
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> is_modified_;
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> in_memory_;
    std::map<std::string, std::shared_ptr<Eigen::Tensor<int, 1>>> is_shardable_;

    std::tuple<std::shared_ptr<TensorData<TensorTs, DeviceT, TDim>>...> parts_; ///< The actual tensor data
    std::tuple<TensorTs...> types_;
    Eigen::Tensor<int, TDim> indices_to_parts_;  ///< Map from indices to the tensor data it can be accessed

    // Tuple magic for C++ 14
    // See https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11
    template<class F, class... Ts, std::size_t... Is>
    void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>) {
      using expander = int[];
      (void)expander {
        0, ((void)func(std::get<Is>(tuple)), 0)...
      };
    }
    template<class F, class...Ts>
    void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func) {
      for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
    }
    
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(id_, name_, n_dimensions_, n_labels_, tensor_dimension_names_, tensor_dimension_labels_);
    //	}
  };

  template<typename DeviceT, int TDim, typename... TensorTs>
  void TensorCollection<DeviceT, TDim, TensorTs...>::setAxes(const std::vector<TensorAxis>& tensor_axes) {
    // Determine the overall dimensions of the heterotensor
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
  };

  template<typename DeviceT, int TDim, typename... TensorTs>
  void TensorCollection<DeviceT, TDim, TensorTs...>::clearAxes() {
    axes_.clear();
    dimensions_.clear();
    indices_.clear();
    indices_view_.clear();
    is_modified_.clear();
    in_memory_.clear();
    is_shardable_.clear();
  };

  template<typename DeviceT, int TDim, typename... TensorTs>
  void TensorCollection<DeviceT, TDim, TensorTs...>::setTensors(const std::vector<std::vector<std::string>>& labels_types) {
    static_assert(sizeof...(TensorTs) == Labels_types);
    auto tup = std::make_tuple(labels_types);
    for_each_in_tuple(tup, [](const auto &x) {
    //for (int i = 0; i<labels_types.size(); ++i) {
      Eigen::array<Eigen::Index, TDim> dimensions = dimensions_;
      dimensions.at(0) = x.at(i).size();
      std::get<i>(parts_)->resize(dimensions);
    });
  };

  template<int TDim, typename... TensorTs>
  class TensorCollectionDefaultDevice: public TensorCollection<Eigen::DefaultDevice, TDim, TensorTs...>
  {
  };
};
#endif //TENSORBASE_TENSORCOLLECTION_H