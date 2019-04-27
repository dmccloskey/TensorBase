/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>
#include <set>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor operations
  */
  class TensorOperation
  {
    //virtual void undo() = 0;
  };

  class SelectClause {
  public:
    SelectClause(const Eigen::Tensor<std::string, 1>& table_names,
    const Eigen::Tensor<std::string, 1>& axis_names,
    const Eigen::Tensor<std::string, 1>& dimension_names,
    const Eigen::Tensor<std::string, 1>& label_names):table_names(table_names), axis_names(axis_names), dimension_names(dimension_names), label_names(label_names) {};
    Eigen::Tensor<std::string, 1> table_names;
    Eigen::Tensor<std::string, 1> axis_names;
    Eigen::Tensor<std::string, 1> dimension_names;
    Eigen::Tensor<std::string, 1> label_names;

    /// Get all labels that are on the same table/axis/dimension
    Eigen::Tensor<std::string, 1> getLabels(const std::string& table_name,
      const std::string& axis_name, const std::string& dimension_name) {
      // get the indices of the labels
      Eigen::Tensor<int, 1> indices(table_names.size());
      auto select_table = (table_names == table_names.constant(table_name)).select(indices.constant(1), indices.constant(0));
      auto select_axis = (axis_names == axis_names.constant(axis_name)).select(indices.constant(1), indices.constant(0));
      auto select_dimension = (dimension_names == dimension_names.constant(dimension_name)).select(indices.constant(1), indices.constant(0));
      auto selected = select_table * select_axis * select_dimension;
      
      // get the indices names and sort in ascending order
      Eigen::Tensor<std::string, 1> label_names_selected = (selected == selected.constant(1)).select(label_names, label_names.constant(""));
      std::sort(label_names_selected.data(), label_names_selected.data() + label_names_selected.size(), 
        [](const std::string& lhs, const std::string& rhs) {
        return lhs > rhs;
      });

      // slice out only the selected names
      Eigen::Tensor<int, 0> n_labels = selected.sum();
      Eigen::TensorMap<Eigen::Tensor<std::string, 1>> label_names_sliced(label_names_selected.data(), n_labels(0));
      return label_names_sliced;
    }
  };

  class OrderByClause : public SelectClause {
  public:
    enum order { ASC, DESC };
    std::vector<order> order_by;
  };

  /**
    @brief Template class for all Tensor select operations
  */
  class TensorSelect {
  public:
    void selectClause(TensorCollection& tensor_collection, SelectClause& select_clause) {
      // initialize the tracker for table/axis/dimensions that have been reached
      std::set<std::string> table_axis_dimension_set;
      std::set<std::string> axes_names;
      // iterate through each select combination
      for (int iter = 0; iter < select_clause.table_names.size(); ++iter) {
        // check if the table/axis/dimension has already been selected
        std::string table_axis_dimension = select_clause.table_names(iter) + "/" + select_clause.axis_names(iter) + "/" + select_clause.dimension_names(iter);
        if (table_axis_dimension_set.count(table_axis_dimension) != 0) continue;
        table_axis_dimension_set.insert(table_axis_dimension);
        // iterate throgh each table axis
        for (auto& axis : tensor_collection.tensor_tables_.at(select_clause.table_names(iter))->getAxes()) {
          if (axis.first == select_clause.axis_names(iter)) {
            // zero the view for the axis (only once)
            if (axes_names.count(select_clause.axis_names(iter)) == 0) {
              tensor_collection.tensor_tables_.at(select_clause.table_names(iter))->getIndicesView().at(select_clause.axis_names(iter))->setZero();
            }
            // iterate through each axis dimensions
            for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
              if (axis.second->getDimensions()(d) == select_clause.dimension_names(iter)) {
                // select the labels that are queried
                Eigen::Tensor<std::string, 1> label_names_selected = select_clause.getLabels(
                  select_clause.table_names(iter), select_clause.axis_names(iter), select_clause.dimension_names(iter));
                // reshape to match the axis labels shape
                Eigen::TensorMap<Eigen::Tensor<std::string, 2>> label_names_reshape(label_names_selected.data(), 1, label_names_selected.size());
                // broadcast the length of the labels
                auto labels_names_selected_bcast = label_names_reshape.broadcast(Eigen::array<int, 2>({ (int)axis.second->getNLabels(), 1 }));
                // broadcast the axis labels the size of the labels queried
                Eigen::TensorMap<Eigen::Tensor<std::string, 3>> labels_reshape(axis.second->getLabels().data(), (int)axis.second->getNDimensions(), (int)axis.second->getNLabels(), 1);
                auto labels_bcast = (labels_reshape.chip(d, 0)).broadcast(Eigen::array<int, 2>({ 1, (int)label_names_selected.size() }));
                // broadcast the tensor indices the size of the labels queried
                Eigen::TensorMap<Eigen::Tensor<int, 2>> indices_reshape(tensor_collection.tensor_tables_.at(select_clause.table_names(iter))->getIndices().at(select_clause.axis_names(iter))->data(),
                  (int)axis.second->getNLabels(), 1);
                auto indices_bcast = indices_reshape.broadcast(Eigen::array<int, 2>({ 1, (int)label_names_selected.size() }));
                auto selected = (labels_bcast == labels_names_selected_bcast).select(indices_bcast, indices_bcast.constant(0));
                auto selected_sum = selected.sum(Eigen::array<int, 1>({ 1 }));
                Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_view(tensor_collection.tensor_tables_.at(select_clause.table_names(iter))->getIndicesView().at(select_clause.axis_names(iter))->data(), (int)axis.second->getNLabels());
                indices_view += selected_sum;
              }
            }
          }
        }
      }
    }; ///< set indices of axis to 1 or 0
    void whereClause(TensorCollection& tensor_collection) {}; ///< set other indices of axis to 1 or 0 based on expression
    void groupByClause(TensorCollection& tensor_collectiont) {};
    void havingClause(TensorCollection& tensor_collection) {};
    void orderByClause(TensorCollection& tensor_collection) {};
  private:
    std::set<std::string> selected_tables_;
  };

  template<int TDim>
  class TensorSelectTableSlice {
  public:
    TensorSelectTableSlice(const std::string table_name,
      const Eigen::array<std::string, TDim>& axes_names,
      const Eigen::array<std::string, TDim>& dimension_names,
      const Eigen::array<std::string, TDim>& label_names_start,
      const Eigen::array<std::string, TDim>& label_names_stop) :
      table_name(table_name), axes_names(axes_names), dimension_names(dimension_names),
      label_names_start(label_names_start), label_names_stop(label_names_stop) {
      for (int iter = 0; iter < TDim; ++iter) {
        offsets.at(iter) = -1;
        extents.at(iter) = -1;
      }
    };
    template<typename T>
    void operator()(T&& t){
      if (std::forward<decltype(t)>(t)->getName() == table_name) {
        for (auto& axis : std::forward<decltype(t)>(t)->getAxes()) {
          for (int iter = 0; iter < TDim; ++iter) {
            if (axis.first == axes_names.at(iter)) {
              for (int d = 0; d < axis.second->getDimensions().size(); ++d) {
                if (axis.second->getDimensions()(d) == dimension_names.at(iter)) {
                  for (int l = 0; l < axis.second->getLabels().dimension(1); ++l) {
                    if (axis.second->getLabels()(d, l) == label_names_start.at(iter)) {
                      offsets.at(iter) = l;
                    }
                    else if (axis.second->getLabels()(d, l) == label_names_stop.at(iter)) {
                      extents.at(iter) = l - d + 1;
                    }
                    if (extents.at(iter) != -1 && offsets.at(iter) != -1)
                      break;
                  }
                  break;
                }
              }
              break;
            }
          }
        }
      }
    };
    std::string table_name;
    Eigen::array<std::string, TDim> axes_names;
    Eigen::array<std::string, TDim> dimension_names;
    Eigen::array<std::string, TDim> label_names_start;
    Eigen::array<std::string, TDim> label_names_stop;
    Eigen::array<int, TDim> offsets;
    Eigen::array<int, TDim> extents;
  };

  class TensorSelectUnion;
  class TensorSelectJoin;

  class TensorInsert {
  public:
    template<typename T>
    void operator()(T&& t) {};
    std::vector<std::pair<std::string, std::string>> insert_into_clause; ///< pairs of TensorTable.name and TensorDimension.label
    //Eigen::Tensor<TensorT, TDim> values; ///< values to insert
  };

  class TensorAddAxisToTable;

  class TensorUpdate {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void whereClause(T&& t) {};
    std::vector<std::pair<std::string, std::string>> set_clause; ///< pairs of TensorTable.name and TensorDimension.label
    //Eigen::Tensor<TensorT, TDim> values; ///< values to update
  };

  class TensorDelete {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void whereClause(T&& t) {};
    std::vector<std::pair<std::string, std::string>> delete_clause; ///< pairs of TensorTable.name and TensorDimension.label
  };

  class TensorAddTables {
  public:
    template<typename TC, typename TCr, typename... TTables>
    void operator()(TC& collection, TCr& collection_new, TTables&... t) {
      auto tables_new = std::tuple_cat(collection.tables_, std::make_tuple(t...));
      collection_new.tables_ = tables_new;
    };
  };

  class TensorDropTable {
  public:
    template<typename TC, typename TCr>
    void operator()(TC& collection, TCr& collection_new, const std::string& table_name) {
      // find the indices of the tables
      size_t index = find_if(collection.tables_, [&](auto&& t) {
        if (std::forward<decltype(t)>(t)->getName() == table_name)
          return false;
        else
          return true;
      });
    };
  };
};
#endif //TENSORBASE_TENSOROPERATION_H