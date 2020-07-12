/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkDataFrame.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/*
	@brief Specialized `DataFrameManager` for the 0D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class DataFrameManagerTimeStampDefaultDevice : public DataFrameManagerTimeStamp<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using DataFrameManagerTimeStamp::DataFrameManagerTimeStamp;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void DataFrameManagerTimeStampDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void DataFrameManagerTimeStampDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(values_data);
	}

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename LabelsT, typename TensorT>
	class Benchmark1TimePointDefaultDevice : public BenchmarkDataFrame1TimePoint<LabelsT, TensorT, Eigen::DefaultDevice> {
	protected:
		void insert1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		
		void update1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		
		void delete1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
		
	};
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		DataFrameManager0DDefaultDevice<LabelsT, TensorT> dataframe_manager(data_size, false);
		this->insert1TimePoint0D_(dataframe_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		DataFrameManager0DDefaultDevice<LabelsT, TensorT> dataframe_manager(data_size, true);
		this->update1TimePoint0D_(dataframe_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		DataFrameManager0DDefaultDevice<LabelsT, TensorT> dataframe_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			dataframe_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable0D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2> tensorDelete("TTable", "indices", selectClause);
			std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}

	template<typename LabelsT, typename TensorT>
	class DataFrameTensorCollectionGeneratorDefaultDevice : public DataFrameTensorCollectionGenerator<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
    std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::DefaultDevice& device) const;
  };
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> DataFrameTensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::makeTensorCollection(const int& data_size, const double& shard_span_perc, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1), dimensions_3_t(1), dimensions_3_x(1), dimensions_3_y(1);
		dimensions_1.setValues({ "columns" });
		dimensions_2.setValues({ "indices" });
    dimensions_3_t.setValues({ "time" });
    dimensions_3_x.setValues({ "x" });
    dimensions_3_y.setValues({ "y" });
		Eigen::Tensor<TensorArray32<char>, 2> labels_1_1(1, 1), labels_1_2(1, 1), labels_1_3(1, 1), labels_1_4(1, 1);
    labels_1_1.setValues({ { TensorArray32<char>("time_stamp")} });
    labels_1_2.setValues({ { TensorArray32<char>("label")} });
    labels_1_3.setValues({ { TensorArray32<char>("image_2D")} });
    labels_1_4.setValues({ { TensorArray32<char>("is_valid")} });

		// Setup the tables
		// TODO: refactor for the case where LabelsT != TensorT
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("xyztv", dimensions_1, labels_1));
		//auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("xyzt", dimensions_1a, labels_1a));
		//auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("v", dimensions_1b, labels_1b));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("indices", 1, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size , 5}));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
};
#endif //TENSORBASE_BENCHMARKDATAFRAMEDEFAULTDEVICE_H