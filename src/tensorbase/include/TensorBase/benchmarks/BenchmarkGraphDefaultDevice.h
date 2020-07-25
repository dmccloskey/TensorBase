/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkGraph.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountNodePropertyDefaultDevice: public SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectAndCountNodeProperty<LabelsT, TensorT, Eigen::DefaultDevice>::SelectAndCountNodeProperty;
		void setLabelsValuesResults(Eigen::DefaultDevice& device) override;
	};

	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAndCountLinkPropertyDefaultDevice : public SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectAndCountLinkProperty<LabelsT, TensorT, Eigen::DefaultDevice>::SelectAndCountLinkProperty;
		void setLabelsValuesResults(Eigen::DefaultDevice& device) override;
	};

	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectAdjacencyDefaultDevice : public SelectAdjacency<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectAdjacency<LabelsT, TensorT, Eigen::DefaultDevice>::SelectAdjacency;
		void setLabelsValuesResults(Eigen::DefaultDevice& device) override;
	};

	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectBFSDefaultDevice : public SelectBFS<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectBFS<LabelsT, TensorT, Eigen::DefaultDevice>::SelectBFS;
		void setLabelsValuesResults(Eigen::DefaultDevice& device) override;
	};

	/// Specialized class for selecting and counting nodes with particular properties for the DefaultDevice case
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class SelectSSSPDefaultDevice : public SelectSSSP<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using SelectSSSP<LabelsT, TensorT, Eigen::DefaultDevice>::SelectSSSP;
		void setLabelsValuesResults(Eigen::DefaultDevice& device) override;
	};

	/*
	@class Specialized `GraphManagerSparseIndices` for the DefaultDevice case
	*/
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT>
	class GraphManagerSparseIndicesDefaultDevice : public GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using GraphManagerSparseIndices<KGLabelsT, KGTensorT, LabelsT, TensorT, Eigen::DefaultDevice>::GraphManagerSparseIndices;
		void makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) override;
		void makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) override;
	};
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparseIndicesDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeLabelsPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr, Eigen::DefaultDevice& device) {
		TensorDataDefaultDevice<LabelsT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHandDData(device);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(tmp);
	}
	template<typename KGLabelsT, typename KGTensorT, typename LabelsT, typename TensorT, typename DeviceT>
	void GraphManagerSparseIndicesDefaultDevice<KGLabelsT, KGTensorT, LabelsT, TensorT>::makeValuesPtr(const Eigen::array<Eigen::Index, 2>& dimensions, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr, Eigen::DefaultDevice& device) {
		TensorDataDefaultDevice<TensorT, 2> tmp(dimensions);
		tmp.setData();
		tmp.syncHandDData(device);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(tmp);
	}

	/*
	@class Specialized `PixelManager` for the 1D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager1DDefaultDevice : public PixelManager1D<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using PixelManager1D::PixelManager1D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager1DDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager1DDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(values_data);
	}

	/*
	@class Specialized `PixelManager` for the 2D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager2DDefaultDevice : public PixelManager2D<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using PixelManager2D::PixelManager2D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager2DDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager2DDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(values_data);
	}

	/*
	@class Specialized `PixelManager` for the 3D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager3DDefaultDevice : public PixelManager3D<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using PixelManager3D::PixelManager3D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 3>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 3>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager3DDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager3DDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 3>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 3>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 3> values_data(Eigen::array<Eigen::Index, 3>({ values.dimension(0), values.dimension(1), values.dimension(2) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 3>>(values_data);
	}

	/*
	@class Specialized `PixelManager` for the 4D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager4DDefaultDevice : public PixelManager4D<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using PixelManager4D::PixelManager4D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 4>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 4>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager4DDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager4DDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 4>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 4>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 4> values_data(Eigen::array<Eigen::Index, 4>({ values.dimension(0), values.dimension(1), values.dimension(2), values.dimension(3) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 4>>(values_data);
	}

	/*
	@class A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename LabelsT, typename TensorT>
	class Benchmark1LinkDefaultDevice : public BenchmarkPixel1Link<LabelsT, TensorT, Eigen::DefaultDevice> {
	protected:
		void insert1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link0D`
		void insert1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link1D`
		void insert1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link2D`
		void insert1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link3D`
		void insert1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1Link4D`

		void update1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link0D`
		void update1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link1D`
		void update1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link2D`
		void update1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link3D`
		void update1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1Link4D`

		void delete1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link0D`
		void delete1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link1D`
		void delete1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link2D`
		void delete1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link3D`
		void delete1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1Link4D`

    TensorT selectAndSumPixels0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels0D`
    TensorT selectAndSumPixels1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels1D`
    TensorT selectAndSumPixels2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels2D`
    TensorT selectAndSumPixels3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels3D`
    TensorT selectAndSumPixels4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels4D`
	};
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::insert1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1Link0D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::insert1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1Link1D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::insert1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1Link2D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::insert1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1Link3D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::insert1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1Link4D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::update1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1Link0D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::update1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1Link1D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::update1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1Link2D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::update1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1Link3D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::update1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1Link4D_(pixel_manager, transaction_manager, scale, edge_factor, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::delete1Link0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
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
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::delete1Link1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable1D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2> tensorDelete("TTable", "xyzt", selectClause);
			std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::delete1Link2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::delete1Link3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 3>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 3> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 3>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1LinkDefaultDevice<LabelsT, TensorT>::delete1Link4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 4>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::DefaultDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 4> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::DefaultDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisDefaultDevice<LabelsT, TensorT, 4>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1LinkDefaultDevice<LabelsT, TensorT>::selectAndSumPixels0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndSumPixels0DDefaultDevice<int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1LinkDefaultDevice<LabelsT, TensorT>::selectAndSumPixels1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndSumPixels1DDefaultDevice<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1LinkDefaultDevice<LabelsT, TensorT>::selectAndSumPixels2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndSumPixels2DDefaultDevice<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1LinkDefaultDevice<LabelsT, TensorT>::selectAndSumPixels3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndSumPixels3DDefaultDevice<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1LinkDefaultDevice<LabelsT, TensorT>::selectAndSumPixels4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const
  {
    SelectAndSumPixels4DDefaultDevice<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    return select_sum_pixels.result_->getData()(0);
  }

	template<typename LabelsT, typename TensorT>
	class TensorCollectionGeneratorDefaultDevice : public PixelTensorCollectionGenerator<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const override;
	};
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1);
		dimensions_1.setValues({ "xyztv" });
		dimensions_2.setValues({ "indices" });
		Eigen::Tensor<TensorArray8<char>, 2> labels_1(1, 5);
		labels_1.setValues({ { TensorArray8<char>("x"), TensorArray8<char>("y"), TensorArray8<char>("z"), TensorArray8<char>("t"), TensorArray8<char>("v")} });

		// Setup the tables
		// TODO: refactor for the case where LabelsT != TensorT
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
		std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("xyztv", dimensions_1, labels_1));
		//std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("xyzt", dimensions_1a, labels_1a));
		//std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("v", dimensions_1b, labels_1b));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("indices", 1, 0));
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
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(4);
		dimensions_1.setValues({ "values" });
		dimensions_2.setValues({ "x","y","z","t" });
		Eigen::Tensor<TensorArray8<char>, 2> labels_v(1, 1);
		labels_v.setValues({ { TensorArray8<char>("values")} });

		// Setup the tables
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
		std::shared_ptr<TensorAxis<TensorArray8<char>, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("values", dimensions_1, labels_v));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyzt", 4, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ 1, data_size }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		const int dim_span = std::pow(data_size, 0.25);
		Eigen::Tensor<std::string, 1> dimensions_1(3), dimensions_2(1);
		dimensions_1.setValues({ "x","y","z" });
		dimensions_2.setValues({ "t" });
		Eigen::Tensor<LabelsT, 2> labels_1(3, std::pow(dim_span, 3));
		for (int i = 0; i < labels_1.dimension(1); ++i) {
			labels_1(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
			labels_1(1, i) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
			labels_1(2, i) = int(floor(float(i) / float(std::pow(dim_span, 2)))) % dim_span + 1;
		}

		// Setup the tables
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyz", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ dim_span, int(std::pow(dim_span, 3)) })); // NOTE: axes are added in alphabetical order

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		const int dim_span = std::pow(data_size, 0.25);
		Eigen::Tensor<std::string, 1> dimensions_1(2), dimensions_2(1), dimensions_3(1);
		dimensions_1.setValues({ "x","y" });
		dimensions_2.setValues({ "z" });
		dimensions_3.setValues({ "t" });
		Eigen::Tensor<LabelsT, 2> labels_1(2, std::pow(dim_span, 2));
		for (int i = 0; i < labels_1.dimension(1); ++i) {
			labels_1(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
			labels_1(1, i) = int(floor(float(i) / float(std::pow(dim_span, 1)))) % dim_span + 1;
		}
		Eigen::Tensor<LabelsT, 2> labels_2(1, dim_span);
		for (int i = 0; i < labels_2.dimension(1); ++i) {
			labels_2(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
		}

		// Setup the tables
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 3>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 3>>(TensorTableDefaultDevice<TensorT, 3>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xy", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("z", dimensions_2, labels_2));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
		table_1_axis_3_ptr->setDimensions(dimensions_3);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_3_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 3>({ dim_span, int(std::pow(dim_span, 2)), dim_span })); // NOTE: axes are added in alphabetical order

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::DefaultDevice& device) const
	{
		// Setup the axes
		const int dim_span = std::pow(data_size, 0.25);
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1), dimensions_3(1), dimensions_4(1);
		dimensions_1.setValues({ "x" });
		dimensions_2.setValues({ "y" });
		dimensions_3.setValues({ "z" });
		dimensions_4.setValues({ "t" });
		Eigen::Tensor<LabelsT, 2> labels_1(1, dim_span);
		Eigen::Tensor<LabelsT, 2> labels_2(1, dim_span);
		Eigen::Tensor<LabelsT, 2> labels_3(1, dim_span);
		for (int i = 0; i < dim_span; ++i) {
			labels_1(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
			labels_2(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
			labels_3(0, i) = int(floor(float(i) / float(std::pow(dim_span, 0)))) % dim_span + 1;
		}

		// Setup the tables
		std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 4>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 4>>(TensorTableDefaultDevice<TensorT, 4>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("x", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("y", dimensions_2, labels_2));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("z", dimensions_3, labels_3));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::DefaultDevice>> table_1_axis_4_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
		table_1_axis_4_ptr->setDimensions(dimensions_4);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_3_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_4_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 4>({ dim_span, dim_span, dim_span, dim_span }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
};
#endif //TENSORBASE_BENCHMARKGRAPHDEFAULTDEVICE_H