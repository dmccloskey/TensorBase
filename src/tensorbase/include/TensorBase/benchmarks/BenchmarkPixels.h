/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKPIXELS_H
#define TENSORBASE_BENCHMARKPIXELS_H

#include <ctime> // time format
#include <chrono> // current time
#include <math.h> // std::pow
#include <random> // random number generator

#include <unsupported/Eigen/CXX11/Tensor>

#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorSelect.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
	/// Base class for all select functors
	template<typename LabelsT, typename DeviceT>
	class SelectTable {
	public:
		SelectTable(std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels) : select_labels_(select_labels){};
		~SelectTable() = default;
		virtual void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) = 0;
	protected:
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& select_labels_;
		bool apply_select_ = false;
	};

	/// The select Functor for the 0D case
	template<typename LabelsT, typename DeviceT>
	class SelectTable0D: public SelectTable<LabelsT, DeviceT> {
	public:
		using SelectTable::SelectTable;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override {
			SelectClause<LabelsT, DeviceT> select_clause1("TTable", "indices", this->select_labels_);
			TensorSelect tensorSelect;
			tensorSelect.selectClause(tensor_collection, select_clause1, device);
			if (this->apply_select_) tensorSelect.applySelect(tensor_collection, { "TTable" }, device);
		}
	};

	/// The select Functor for the 1D case
	template<typename LabelsT, typename DeviceT>
	class SelectTable1D : public SelectTable<LabelsT, DeviceT> {
	public:
		using SelectTable::SelectTable;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) override {
			SelectClause<LabelsT, DeviceT> select_clause1("TTable", "xyzt", this->select_labels_);
			TensorSelect tensorSelect;
			tensorSelect.selectClause(tensor_collection, select_clause1, device);
			if (this->apply_select_) tensorSelect.applySelect(tensor_collection, { "TTable" }, device);
		}
	};

	/// The select Functor for the 2D case
	template<typename LabelsT, typename DeviceT>
	class SelectTable2D : public SelectTable<LabelsT, DeviceT> {
	public:
		using SelectTable::SelectTable;
		void operator() (std::shared_ptr<TensorCollection<DeviceT>>& tensor_collection, DeviceT& device) {
			SelectClause<LabelsT, DeviceT> select_clause1("TTable", "t", this->select_labels_);
			TensorSelect tensorSelect;
			tensorSelect.selectClause(tensor_collection, select_clause1, device);
			if (this->apply_select_) tensorSelect.applySelect(tensor_collection, { "TTable" }, device);
		}
	};

	/// The select Functor for the 3D case (Same as 2D)
	template<typename LabelsT, typename DeviceT>
	class SelectTable3D : public SelectTable2D<LabelsT, DeviceT> {
	public:
		using SelectTable2D::SelectTable2D;
	};

	/// The select Functor for the 4D case (Same as 2D)
	template<typename LabelsT, typename DeviceT>
	class SelectTable4D : public SelectTable2D<LabelsT, DeviceT> {
	public:
		using SelectTable2D::SelectTable2D;
	};

	/*
	@brief Class for managing the generation of random pixels in a 4D (3D + time) space
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	class PixelManager {
	public:
		PixelManager(const int& data_size, const bool& use_random_values = false) : data_size_(data_size), use_random_values_(use_random_values){};
		~PixelManager() = default;
		virtual void setDimSizes() = 0;
		virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;
		virtual void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr) = 0;
		virtual void makeValuesPtr(const Eigen::Tensor<TensorT, NDim>& values, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;

		/*
		@brief Generate a random value
		*/
		TensorT getRandomValue();
	protected:
		int data_size_;
		bool use_random_values_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT, int NDim>
	TensorT PixelManager<LabelsT, TensorT, DeviceT, NDim>::getRandomValue() {
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> d{ 0.0f, 10.0f };
		return TensorT(d(gen));
	}

	/*
	@brief Specialized `PixelManager` for the 0D case
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class PixelManager0D : public PixelManager<LabelsT, TensorT, DeviceT, 2> {
	public:
		using PixelManager::PixelManager;
		void setDimSizes();
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
	private:
		int indices_dim_size_;
		int dim_span_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager0D<LabelsT, TensorT, DeviceT>::setDimSizes() {
		indices_dim_size_ = this->data_size_;
		dim_span_ = std::pow(this->data_size_, 0.25);
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager0D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr) {
		setDimSizes();
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 2> values(span, 5);
		for (int i = offset; i < offset + span; ++i) {
			labels(0, i - offset) = LabelsT(i);
			values(i - offset, 0) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
			values(i - offset, 1) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
			values(i - offset, 2) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
			values(i - offset, 3) = int(floor(float(i) / float(std::pow(this->dim_span_, 3)))) % this->dim_span_ + 1;
      if (this->use_random_values_) values(i - offset, 4) = TensorT(-1); // this->getRandomValue();
			else values(i - offset, 4) = TensorT(i);
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}

	/*
	@brief Specialized `PixelManager` for the 1D case
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class PixelManager1D : public PixelManager<LabelsT, TensorT, DeviceT, 2> {
	public:
		using PixelManager::PixelManager;
		void setDimSizes();
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
	private:
		int xyzt_dim_size_;
		int dim_span_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager1D<LabelsT, TensorT, DeviceT>::setDimSizes() {
		xyzt_dim_size_ = this->data_size_;
		dim_span_ = std::pow(this->data_size_, 0.25);
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager1D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr) {
		setDimSizes();
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(4, span);
		Eigen::Tensor<TensorT, 2> values(1, span);
		for (int i = offset; i < offset + span; ++i) {
			labels(0, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
			labels(1, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
			labels(2, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
			labels(3, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 3)))) % this->dim_span_ + 1;
			if (this->use_random_values_) values(0, i - offset) = TensorT(-1); // this->getRandomValue();
			else values(0, i - offset) = TensorT(i);
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}

	/*
	@brief Specialized `PixelManager` for the 2D case
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class PixelManager2D : public PixelManager<LabelsT, TensorT, DeviceT, 2> {
	public:
		using PixelManager::PixelManager;
		void setDimSizes();
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr);
	private:
		int xyz_dim_size_;
		int t_dim_size_;
		int dim_span_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager2D<LabelsT, TensorT, DeviceT>::setDimSizes() {
		dim_span_ = std::pow(this->data_size_, 0.25);
		xyz_dim_size_ = std::pow(dim_span_, 3);
		t_dim_size_ = dim_span_;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager2D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 2>>& values_ptr) {
		setDimSizes();
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 2> values(this->xyz_dim_size_, span);
		for (int i = 0; i < span; ++i) {
			labels(0, i) = int(floor(float(i + offset) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
			Eigen::Tensor<TensorT, 1> new_values(this->xyz_dim_size_);
      if (this->use_random_values_) {
        new_values.setConstant(TensorT(-1)); // new_values.setRandom();
      }
      else {
        new_values.setConstant((i + offset) * this->xyz_dim_size_ + 1);
        new_values = new_values.cumsum(0);
      }
			values.slice(Eigen::array<Eigen::Index, 2>({ 0, i }), Eigen::array<Eigen::Index, 2>({ this->xyz_dim_size_, 1 })) = new_values.reshape(Eigen::array<Eigen::Index, 2>({ this->xyz_dim_size_, 1 }));
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}

	/*
	@brief Specialized `PixelManager` for the 3D case
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class PixelManager3D : public PixelManager<LabelsT, TensorT, DeviceT, 3> {
	public:
		using PixelManager::PixelManager;
		void setDimSizes();
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr);
	private:
		int xy_dim_size_;
		int z_dim_size_;
		int t_dim_size_;
		int dim_span_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager3D<LabelsT, TensorT, DeviceT>::setDimSizes() {
		dim_span_ = std::pow(this->data_size_, 0.25);
		xy_dim_size_ = std::pow(dim_span_, 2);
		z_dim_size_ = dim_span_;
		t_dim_size_ = dim_span_;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager3D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 3>>& values_ptr) {
		setDimSizes();
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 3> values(this->xy_dim_size_, this->z_dim_size_, span);
		for (int i = 0; i < span; ++i) {
			labels(0, i) = int(floor(float(i + offset) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
			Eigen::Tensor<TensorT, 1> new_values(this->xy_dim_size_ * this->z_dim_size_);
      if (this->use_random_values_) {
        new_values.setConstant(TensorT(-1)); // new_values.setRandom();
      }
      else {
        new_values.setConstant((i + offset) * this->xy_dim_size_ * this->z_dim_size_ + 1);
        new_values = new_values.cumsum(0);
      }
			values.slice(Eigen::array<Eigen::Index, 3>({ 0, 0, i }), Eigen::array<Eigen::Index, 3>({ this->xy_dim_size_, this->z_dim_size_, 1 })) = new_values.reshape(Eigen::array<Eigen::Index, 3>({ this->xy_dim_size_, this->z_dim_size_, 1 }));
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}

	/*
	@brief Specialized `PixelManager` for the 4D case
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class PixelManager4D : public PixelManager<LabelsT, TensorT, DeviceT, 4> {
	public:
		using PixelManager::PixelManager;
		void setDimSizes();
		void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 4>>& values_ptr);
	private:
		int x_dim_size_;
		int y_dim_size_;
		int z_dim_size_;
		int t_dim_size_;
		int dim_span_;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager4D<LabelsT, TensorT, DeviceT>::setDimSizes() {
		dim_span_ = std::pow(this->data_size_, 0.25);
		x_dim_size_ = dim_span_;
		y_dim_size_ = dim_span_;
		z_dim_size_ = dim_span_;
		t_dim_size_ = dim_span_;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void PixelManager4D<LabelsT, TensorT, DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, 4>>& values_ptr) {
		setDimSizes();
		// Make the labels and values
		Eigen::Tensor<LabelsT, 2> labels(1, span);
		Eigen::Tensor<TensorT, 4> values(this->x_dim_size_, this->y_dim_size_, this->z_dim_size_, span);
		for (int i = 0; i < span; ++i) {
			labels(0, i) = int(floor(float(i + offset) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
			Eigen::Tensor<TensorT, 1> new_values(this->x_dim_size_ * this->y_dim_size_ * this->z_dim_size_);
      if (this->use_random_values_) {
        new_values.setConstant(TensorT(-1)); // new_values.setRandom();
      }
      else {
        new_values.setConstant((i + offset) * this->x_dim_size_ * this->y_dim_size_ * this->z_dim_size_ + 1);
        new_values = new_values.cumsum(0);
      }
			values.slice(Eigen::array<Eigen::Index, 4>({ 0, 0, 0, i }), Eigen::array<Eigen::Index, 4>({ this->x_dim_size_, this->y_dim_size_, this->z_dim_size_, 1 })) = new_values.reshape(Eigen::array<Eigen::Index, 4>({ this->x_dim_size_, this->y_dim_size_, this->z_dim_size_, 1 }));
		}
		this->makeLabelsPtr(labels, labels_ptr);
		this->makeValuesPtr(values, values_ptr);
	}


	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class Benchmark1TimePoint {
	public:
		Benchmark1TimePoint() = default;
		~Benchmark1TimePoint() = default;
		/*
		@brief insert 1 time-point at a time

		@param[in] n_dims
		@param[in, out] transaction_manager
		@param[in] data_size
		@param[in] device

		@returns A string with the total time of the benchmark in milliseconds
		*/
		std::string insert1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
		std::string update1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
		std::string delete1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const;
	protected:
		virtual void insert1TimePoint0D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint0D`
		virtual void insert1TimePoint1D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint1D`
		virtual void insert1TimePoint2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint2D`
		virtual void insert1TimePoint3D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint3D`
		virtual void insert1TimePoint4D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `insert1TimePoint4D`

		void insert1TimePoint0D_(PixelManager0D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `insert1TimePoint0D`
		void insert1TimePoint1D_(PixelManager1D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `insert1TimePoint1D`
		void insert1TimePoint2D_(PixelManager2D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `insert1TimePoint2D`
		void insert1TimePoint3D_(PixelManager3D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `insert1TimePoint3D`
		void insert1TimePoint4D_(PixelManager4D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `insert1TimePoint4D`

		virtual void update1TimePoint0D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint0D`
		virtual void update1TimePoint1D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint1D`
		virtual void update1TimePoint2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint2D`
		virtual void update1TimePoint3D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint3D`
		virtual void update1TimePoint4D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `update1TimePoint4D`

		void update1TimePoint0D_(PixelManager0D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `update1TimePoint0D`
		void update1TimePoint1D_(PixelManager1D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `update1TimePoint1D`
		void update1TimePoint2D_(PixelManager2D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `update1TimePoint2D`
		void update1TimePoint3D_(PixelManager3D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `update1TimePoint3D`
		void update1TimePoint4D_(PixelManager4D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const; ///< Device agnostic implementation of `update1TimePoint4D`

		virtual void delete1TimePoint0D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint0D`
		virtual void delete1TimePoint1D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint1D`
		virtual void delete1TimePoint2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint2D`
		virtual void delete1TimePoint3D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint3D`
		virtual void delete1TimePoint4D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const = 0; ///< Device specific interface to call `delete1TimePoint4D`
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	std::string Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		if (n_dims == 0) insert1TimePoint0D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 1) insert1TimePoint1D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 2) insert1TimePoint2D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 3) insert1TimePoint3D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 4) insert1TimePoint4D(transaction_manager, data_size, in_memory, device);
		else std::cout << "The given number of dimensions " << n_dims << " is not within the range of 0 to 4." << std::endl;

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	std::string Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		if (n_dims == 0) update1TimePoint0D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 1) update1TimePoint1D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 2) update1TimePoint2D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 3) update1TimePoint3D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 4) update1TimePoint4D(transaction_manager, data_size, in_memory, device);
		else std::cout << "The given number of dimensions " << n_dims << " is not within the range of 0 to 4." << std::endl;

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	std::string Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::delete1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		// Start the timer
		auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

		if (n_dims == 0) delete1TimePoint0D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 1) delete1TimePoint1D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 2) delete1TimePoint2D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 3) delete1TimePoint3D(transaction_manager, data_size, in_memory, device);
		else if (n_dims == 4) delete1TimePoint4D(transaction_manager, data_size, in_memory, device);
		else std::cout << "The given number of dimensions " << n_dims << " is not within the range of 0 to 4." << std::endl;

		// Stop the timer
		auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		std::string milli_time = std::to_string(stop - start);
		return milli_time;
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint0D_(PixelManager0D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);  // BUG: breaks auto max_bcast = indices_view_values.maximum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 1>({ n_labels })); in TensorTableDefaultDevice<TensorT, TDim>::makeAppendIndices
		for (int i = 0; i < data_size; i += span) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2> appendToAxis("TTable", "indices", labels_ptr, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2>>(appendToAxis);
			transaction_manager.executeOperation(appendToAxis_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint1D_(PixelManager1D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
    for (int i = 0; i < data_size; i += span) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2> appendToAxis("TTable", "xyzt", labels_ptr, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2>>(appendToAxis);
			transaction_manager.executeOperation(appendToAxis_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint2D_(PixelManager2D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2> appendToAxis("TTable", "t", labels_ptr, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2>>(appendToAxis);
			transaction_manager.executeOperation(appendToAxis_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint3D_(PixelManager3D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 3>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			TensorAppendToAxis<LabelsT, TensorT, DeviceT, 3> appendToAxis("TTable", "t", labels_ptr, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 3>>(appendToAxis);
			transaction_manager.executeOperation(appendToAxis_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint4D_(PixelManager4D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 4>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			TensorAppendToAxis<LabelsT, TensorT, DeviceT, 4> appendToAxis("TTable", "t", labels_ptr, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 4>>(appendToAxis);
			transaction_manager.executeOperation(appendToAxis_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint0D_(PixelManager0D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size; i += span) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable0D<LabelsT, DeviceT> selectClause(labels_ptr);
			TensorUpdateValues<TensorT, DeviceT, 2> tensorUpdate("TTable", selectClause, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<TensorT, DeviceT, 2>>(tensorUpdate);
			transaction_manager.executeOperation(tensorUpdate_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint1D_(PixelManager1D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size; i += span) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable1D<LabelsT, DeviceT> selectClause(labels_ptr);
			TensorUpdateValues<TensorT, DeviceT, 2> tensorUpdate("TTable", selectClause, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<TensorT, DeviceT, 2>>(tensorUpdate);
			transaction_manager.executeOperation(tensorUpdate_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint2D_(PixelManager2D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, DeviceT> selectClause(labels_ptr);
			TensorUpdateValues<TensorT, DeviceT, 2> tensorUpdate("TTable", selectClause, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<TensorT, DeviceT, 2>>(tensorUpdate);
			transaction_manager.executeOperation(tensorUpdate_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint3D_(PixelManager3D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 3>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, DeviceT> selectClause(labels_ptr);
			TensorUpdateValues<TensorT, DeviceT, 3> tensorUpdate("TTable", selectClause, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<TensorT, DeviceT, 3>>(tensorUpdate);
			transaction_manager.executeOperation(tensorUpdate_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT, typename DeviceT>
	void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::update1TimePoint4D_(PixelManager4D<LabelsT, TensorT, DeviceT>& pixel_manager, TransactionManager<DeviceT>& transaction_manager, const int& data_size, const bool& in_memory, DeviceT& device) const
	{
		std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, DeviceT, 4>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points; ++i) {
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, DeviceT> selectClause(labels_ptr);
			TensorUpdateValues<TensorT, DeviceT, 4> tensorUpdate("TTable", selectClause, values_ptr);
			std::shared_ptr<TensorOperation<DeviceT>> tensorUpdate_ptr = std::make_shared<TensorUpdateValues<TensorT, DeviceT, 4>>(tensorUpdate);
			transaction_manager.executeOperation(tensorUpdate_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}

  /// Helper set of methods for calculating the shard_span, shard_id, and shard_index
  struct TensorCollectionShardHelper {
    /// Round to 1 if below 1
    static int round_1(const int& dim_size, const double& shard_span_perc) {
      int shard_span = dim_size * shard_span_perc;
      if (shard_span <= 0) shard_span = 1;
      return shard_span;
    };
    /// Calculate the shard ID
    static int calc_shard_id(const int& shard_span, const int& i) {
      int shard_id = i / shard_span + 1;
      return shard_id;
    };
    /// Calculate the shard Index
    static int calc_shard_index(const int& shard_span, const int& i) {
      int shard_index = i % shard_span + 1;
      return shard_index;
    };
  };

	/*
	@brief Simulate a typical database table where one axis will be the headers (x, y, z, and t)
		and the other axis will be the index starting from 1
	*/
	template<typename LabelsT, typename TensorT, typename DeviceT>
	class TensorCollectionGenerator {
	public:
		TensorCollectionGenerator() = default;
		~TensorCollectionGenerator() = default;
		std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& n_dims, const int& data_size, const double& shard_span_perc, const bool& is_columnar, DeviceT& device) const;
		virtual std::shared_ptr<TensorCollection<DeviceT>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, DeviceT& device) const = 0;
		virtual std::shared_ptr<TensorCollection<DeviceT>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, DeviceT& device) const = 0;
		virtual std::shared_ptr<TensorCollection<DeviceT>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, DeviceT& device) const = 0;
		virtual std::shared_ptr<TensorCollection<DeviceT>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, DeviceT& device) const = 0;
		virtual std::shared_ptr<TensorCollection<DeviceT>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, DeviceT& device) const = 0;
	};
	template<typename LabelsT, typename TensorT, typename DeviceT>
	std::shared_ptr<TensorCollection<DeviceT>> TensorCollectionGenerator<LabelsT, TensorT, DeviceT>::makeTensorCollection(const int& n_dims, const int& data_size, const double& shard_span_perc, const bool& is_columnar, DeviceT& device) const
	{
		if (n_dims == 0) {
			std::map<std::string, int> shard_span;
			shard_span.emplace("xyztv", 5);
			shard_span.emplace("indices", TensorCollectionShardHelper::round_1(data_size,shard_span_perc));
			return make0DTensorCollection(data_size, shard_span, is_columnar, device);
		}
		else if (n_dims == 1) {
			std::map<std::string, int> shard_span;
			shard_span.emplace("xyzt", TensorCollectionShardHelper::round_1(data_size, shard_span_perc));
			shard_span.emplace("values", 1);
			return make1DTensorCollection(data_size, shard_span, is_columnar, device);
		}
		else if (n_dims == 2) {
			std::map<std::string, int> shard_span;
			shard_span.emplace("xyz", TensorCollectionShardHelper::round_1(std::pow(std::pow(data_size, 0.25), 3), shard_span_perc));
			shard_span.emplace("t", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			return make2DTensorCollection(data_size, shard_span, is_columnar, device);
		}
		else if (n_dims == 3) {
			std::map<std::string, int> shard_span;
			shard_span.emplace("xy", TensorCollectionShardHelper::round_1(std::pow(std::pow(data_size, 0.25), 2), shard_span_perc));
			shard_span.emplace("z", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			shard_span.emplace("t", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			return make3DTensorCollection(data_size, shard_span, is_columnar, device);
		}
		else if (n_dims == 4) {
			std::map<std::string, int> shard_span;
			shard_span.emplace("x", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			shard_span.emplace("y", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			shard_span.emplace("z", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			shard_span.emplace("t", TensorCollectionShardHelper::round_1(std::pow(data_size, 0.25), shard_span_perc));
			return make4DTensorCollection(data_size, shard_span, is_columnar, device);
		}
		else {
			return std::shared_ptr<TensorCollection<DeviceT>>();
		}
	}

	template<typename LabelsT, typename TensorT, typename DeviceT>
	static void runBenchmarkPixels(const std::string& data_dir, const int& n_dims, const int& data_size, const bool& in_memory, const double& shard_span_perc,
		const Benchmark1TimePoint<LabelsT, TensorT, DeviceT>& benchmark_1_tp,
		const TensorCollectionGenerator<LabelsT, TensorT, DeviceT>& tensor_collection_generator, DeviceT& device) {
		std::cout << "Starting insert/delete/update pixel benchmarks for n_dims=" << n_dims << ", data_size=" << data_size << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;

		// Make the nD TensorTables
		std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true, device);

		// Setup the transaction manager
		TransactionManager<DeviceT> transaction_manager;
		transaction_manager.setMaxOperations(data_size + 1);

		// Run the table through the benchmarks
		transaction_manager.setTensorCollection(n_dim_tensor_collection);
		std::cout << n_dims << "D Tensor Table time-point insertion took " << benchmark_1_tp.insert1TimePoint(n_dims, transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << n_dims << "D Tensor Table time-point update took " << benchmark_1_tp.update1TimePoint(n_dims, transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
		std::cout << n_dims << "D Tensor Table time-point deletion took " << benchmark_1_tp.delete1TimePoint(n_dims, transaction_manager, data_size, in_memory, device) << " milliseconds." << std::endl;
	}

	///Parse the command line arguments
	static void parseCmdArgs(const int& argc, char** argv, std::string& data_dir, int& n_dims, int& data_size, bool& in_memory, double& shard_span_perc, int& n_engines, std::string& labels_type, std::string& tensor_type) {
		if (argc >= 2) {
			data_dir = argv[1];
		}
		if (argc >= 3) {
			try {
				n_dims = (std::stoi(argv[2]) >= 0 && std::stoi(argv[2]) <= 4) ? std::stoi(argv[2]) : 4;
			}
			catch (std::exception & e) {
				std::cout << e.what() << std::endl;
			}
		}
		if (argc >= 4) {
			if (argv[3] == std::string("XS")) {
				data_size = 1296;
			}
      else if (argv[3] == std::string("S")) {
        data_size = 104976;
      }
			else if (argv[3] == std::string("M")) {
				data_size = 1048576;
			}
			else if (argv[3] == std::string("L")) {
				data_size = 10556001;
			}
			else if (argv[3] == std::string("XL")) {
				data_size = 1003875856;
			}
      else if (argv[3] == std::string("XXL")) {
        data_size = 1e12;
      }
		}
		if (argc >= 5) {
			in_memory = (argv[4] == std::string("true")) ? true : false;
		}
		if (argc >= 6) {
			try {
				if (std::stoi(argv[5]) == 5) shard_span_perc = 0.05;
				else if (std::stoi(argv[5]) == 20) shard_span_perc = 0.2;
				else if (std::stoi(argv[5]) == 100) shard_span_perc = 1;
			}
			catch (std::exception & e) {
				std::cout << e.what() << std::endl;
			}
		}
    if (argc >= 7) {
      try {
        n_engines = std::stoi(argv[6]);
      }
      catch (std::exception & e) {
        std::cout << e.what() << std::endl;
      }
    }
    if (argc >= 8) {
      if (argv[7] == std::string("int")) {
        labels_type = std::string("int");
      }
      else if (argv[7] == std::string("float")) {
        labels_type = std::string("float");
      }
      else if (argv[7] == std::string("double")) {
        labels_type = std::string("double");
      }
    }
    if (argc >= 9) {
      if (argv[8] == std::string("int")) {
        tensor_type = std::string("int");
      }
      else if (argv[8] == std::string("float")) {
        tensor_type = std::string("float");
      }
      else if (argv[8] == std::string("double")) {
        tensor_type = std::string("double");
      }
    }
	}
};
#endif //TENSORBASE_BENCHMARKPIXELS_H