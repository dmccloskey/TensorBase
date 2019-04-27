/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSOROPERATION_H
#define TENSORBASE_TENSOROPERATION_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/core/TupleAlgorithms.h>

namespace TensorBase
{
  /**
    @brief Abstract base class for all Tensor operations
  */
  class TensorOperation
  {
    //virtual void undo() = 0;
  };

  /**
    @brief Template class for all Tensor select operations
  */
  class TensorSelect {
  public:
    template<typename T>
    void operator()(T&& t) {};
    template<typename T>
    void selectClause(T&& t) {}; ///< set indices of axis to 1 or 0
    template<typename T>
    void whereClause(T&& t) {}; ///< set other indices of axis to 1 or 0 based on expression
    template<typename T>
    void groupByClause(T&& t) {};
    template<typename T>
    void havingClause(T&& t) {};
    template<typename T>
    void orderByClause(T&& t) {};
    enum order { ASC, DESC };
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> select_clause; ///< tuple of TensorTable.name (optional), TensorAxis.name (optional), TensorDimension.name, TensorDimension.label
    std::vector<std::pair<std::string, std::string>> group_by_clause; ///< pairs of TensorTable.name and TensorDimension.label
    std::vector<std::tuple<std::string, std::string, order>> order_by_clause; ///< tuple of TensorTable.name, TensorDimension.label, and order
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