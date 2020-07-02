/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKPIXELSDEFAULTDEVICE_H
#define TENSORBASE_BENCHMARKPIXELSDEFAULTDEVICE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkPixels.h>
#include <TensorBase/ml/TensorCollectionDefaultDevice.h>
#include <TensorBase/ml/TensorOperationDefaultDevice.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/*
	@brief Specialized `PixelManager` for the 0D and DefaultDevice case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager0DDefaultDevice : public PixelManager0D<LabelsT, TensorT, Eigen::DefaultDevice> {
	public:
		using PixelManager0D::PixelManager0D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager0DDefaultDevice<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::DefaultDevice, 2>>& labels_ptr) {
		TensorDataDefaultDevice<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataDefaultDevice<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager0DDefaultDevice<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 2>>& values_ptr) {
		TensorDataDefaultDevice<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataDefaultDevice<TensorT, 2>>(values_data);
	}

	/*
	@brief Specialized `PixelManager` for the 1D and DefaultDevice case
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
	@brief Specialized `PixelManager` for the 2D and DefaultDevice case
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
	@brief Specialized `PixelManager` for the 3D and DefaultDevice case
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
	@brief Specialized `PixelManager` for the 4D and DefaultDevice case
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
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename LabelsT, typename TensorT>
	class Benchmark1TimePointDefaultDevice : public Benchmark1TimePoint<LabelsT, TensorT, Eigen::DefaultDevice> {
	protected:
		void insert1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		void insert1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint1D`
		void insert1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint2D`
		void insert1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint3D`
		void insert1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint4D`

		void update1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		void update1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint1D`
		void update1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint2D`
		void update1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint3D`
		void update1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint4D`

		void delete1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
		void delete1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint1D`
		void delete1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint2D`
		void delete1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint3D`
		void delete1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `delete1TimePoint4D`
	};
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint0D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint1D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint2D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint3D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint4D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint0D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint1D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint2D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint3D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
		PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint4D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
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
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
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
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
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
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
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
	void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::delete1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::DefaultDevice& device) const {
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
	class TensorCollectionGeneratorDefaultDevice : public TensorCollectionGenerator<LabelsT, TensorT, Eigen::DefaultDevice> {
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
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({5, data_size}));

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
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("values", dimensions_1, labels_v));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyzt", 4, 0));
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
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyz", dimensions_1, labels_1));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
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
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xy", dimensions_1, labels_1));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("z", dimensions_2, labels_2));
		auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
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
		auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("x", dimensions_1, labels_1));
		auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("y", dimensions_2, labels_2));
		auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("z", dimensions_3, labels_3));
		auto table_1_axis_4_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
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
#endif //TENSORBASE_BENCHMARKPIXELSDEFAULTDEVICE_H