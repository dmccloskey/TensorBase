/**TODO:  Add copyright*/

#ifndef TENSORBASE_BENCHMARKPIXELSGPU_H
#define TENSORBASE_BENCHMARKPIXELSGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <TensorBase/benchmarks/BenchmarkPixels.h>
#include <TensorBase/ml/TensorCollectionGpu.h>
#include <TensorBase/ml/TensorOperationGpu.h>

using namespace TensorBase;

namespace TensorBaseBenchmarks
{
  /// Specialized class to select a region of pixels and compute the sum for the 0D and Gpu case
  template<typename TensorT>
  class SelectAndSumPixels0DGpu : public SelectAndSumPixels0D<TensorArrayGpu8<char>, TensorT, Eigen::GpuDevice> {
  public:
    using SelectAndSumPixels0D<TensorArrayGpu8<char>, TensorT, Eigen::GpuDevice>::SelectAndSumPixels0D;
    void setLabelsValuesResults(Eigen::GpuDevice& device) override;
  };
  template< typename TensorT>
  inline void SelectAndSumPixels0DGpu<TensorT>::setLabelsValuesResults(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<TensorArrayGpu8<char>, 2> select_labels_xyztv_values(1, 4);
    select_labels_xyztv_values.setValues({ { TensorArrayGpu8<char>("x"), TensorArrayGpu8<char>("y"), TensorArrayGpu8<char>("z"), TensorArrayGpu8<char>("t")} });
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> select_labels_xyztv(select_labels_xyztv_values.dimensions());
    select_labels_xyztv.setData(select_labels_xyztv_values);
    select_labels_xyztv.syncHAndDData(device);
    this->select_labels_xyztv_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(select_labels_xyztv);

    // make the corresponding values and sync to the device
    Eigen::Tensor<TensorT, 1> select_values_xyztv_values(4);
    select_values_xyztv_values.setValues({ this->dim_span_, this->dim_span_, this->dim_span_, this->dim_span_ });
    TensorDataGpuPrimitiveT<TensorT, 1> select_values_xyztv_lt(select_values_xyztv_values.dimensions());
    select_values_xyztv_lt.setData(select_values_xyztv_values);
    select_values_xyztv_lt.syncHAndDData(device);
    this->select_values_xyztv_lt_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(select_values_xyztv_lt);
    select_values_xyztv_values.setValues({ 1, 1, 1, 1 });
    TensorDataGpuPrimitiveT<TensorT, 1> select_values_xyztv_gt(select_values_xyztv_values.dimensions());
    select_values_xyztv_gt.setData(select_values_xyztv_values);
    select_values_xyztv_gt.syncHAndDData(device);
    this->select_values_xyztv_gt_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(select_values_xyztv_gt);

    // make the labels and sync to the device
    Eigen::Tensor<TensorArrayGpu8<char>, 2> select_labels_v_values(1, 1);
    select_labels_v_values.setValues({ { TensorArrayGpu8<char>("v")} });
    TensorDataGpuClassT<TensorArrayGpu8, char, 2> select_labels_v(select_labels_v_values.dimensions());
    select_labels_v.setData(select_labels_v_values);
    select_labels_v.syncHAndDData(device);
    this->select_labels_v_ = std::make_shared<TensorDataGpuClassT<TensorArrayGpu8, char, 2>>(select_labels_v);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<TensorT, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(results);
  }

  /// Specialized class to select a region of pixels and compute the sum for the 1D and Gpu case
  template<typename LabelsT, typename TensorT>
  class SelectAndSumPixels1DGpu : public SelectAndSumPixels1D<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using SelectAndSumPixels1D<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndSumPixels1D;
    void setLabelsValuesResults(Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAndSumPixels1DGpu<LabelsT, TensorT>::setLabelsValuesResults(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    const int data_span_xyzt = std::pow(this->dim_span_, 4);
    Eigen::Tensor<LabelsT, 2> select_labels_xyzt_values(4, data_span_xyzt);
    for (int i = 0; i < data_span_xyzt; ++i) {
      select_labels_xyzt_values(0, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
      select_labels_xyzt_values(1, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
      select_labels_xyzt_values(2, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
      select_labels_xyzt_values(3, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 3)))) % this->dim_span_ + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_xyzt(select_labels_xyzt_values.dimensions());
    select_labels_xyzt.setData(select_labels_xyzt_values);
    select_labels_xyzt.syncHAndDData(device);
    this->select_labels_xyzt_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_xyzt);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<TensorT, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(results);
  }

  /// Specialized class to select a region of pixels and compute the sum for the 2D and Gpu case
  template<typename LabelsT, typename TensorT>
  class SelectAndSumPixels2DGpu : public SelectAndSumPixels2D<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using SelectAndSumPixels2D<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndSumPixels2D;
    void setLabelsValuesResults(Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAndSumPixels2DGpu<LabelsT, TensorT>::setLabelsValuesResults(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    const int data_span_xyz = std::pow(this->dim_span_, 3);
    Eigen::Tensor<LabelsT, 2> select_labels_xyz_values(3, data_span_xyz);
    for (int i = 0; i < data_span_xyz; ++i) {
      select_labels_xyz_values(0, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
      select_labels_xyz_values(1, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
      select_labels_xyz_values(2, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_xyz(select_labels_xyz_values.dimensions());
    select_labels_xyz.setData(select_labels_xyz_values);
    select_labels_xyz.syncHAndDData(device);
    this->select_labels_xyz_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_xyz);

    Eigen::Tensor<LabelsT, 2> select_labels_t_values(1, this->dim_span_);
    for (int i = 0; i < this->dim_span_; ++i) {
      select_labels_t_values(0, i) = i + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_t(select_labels_t_values.dimensions());
    select_labels_t.setData(select_labels_t_values);
    select_labels_t.syncHAndDData(device);
    this->select_labels_t_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_t);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<TensorT, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(results);
  }

  /// Specialized class to select a region of pixels and compute the sum for the 3D and Gpu case
  template<typename LabelsT, typename TensorT>
  class SelectAndSumPixels3DGpu : public SelectAndSumPixels3D<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using SelectAndSumPixels3D<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndSumPixels3D;
    void setLabelsValuesResults(Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAndSumPixels3DGpu<LabelsT, TensorT>::setLabelsValuesResults(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    const int data_span_xy = std::pow(this->dim_span_, 2);
    Eigen::Tensor<LabelsT, 2> select_labels_xy_values(2, data_span_xy);
    for (int i = 0; i < data_span_xy; ++i) {
      select_labels_xy_values(0, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
      select_labels_xy_values(1, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_xy(select_labels_xy_values.dimensions());
    select_labels_xy.setData(select_labels_xy_values);
    select_labels_xy.syncHAndDData(device);
    this->select_labels_xy_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_xy);

    Eigen::Tensor<LabelsT, 2> select_labels_z_values(1, this->dim_span_);
    Eigen::Tensor<LabelsT, 2> select_labels_t_values(1, this->dim_span_);
    for (int i = 0; i < this->dim_span_; ++i) {
      select_labels_z_values(0, i) = i + 1;
      select_labels_t_values(0, i) = i + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_z(select_labels_z_values.dimensions());
    select_labels_z.setData(select_labels_z_values);
    select_labels_z.syncHAndDData(device);
    this->select_labels_z_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_z);
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_t(select_labels_t_values.dimensions());
    select_labels_t.setData(select_labels_t_values);
    select_labels_t.syncHAndDData(device);
    this->select_labels_t_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_t);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<TensorT, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(results);
  }

  /// Specialized class to select a region of pixels and compute the sum for the 4D and Gpu case
  template<typename LabelsT, typename TensorT>
  class SelectAndSumPixels4DGpu : public SelectAndSumPixels4D<LabelsT, TensorT, Eigen::GpuDevice> {
  public:
    using SelectAndSumPixels4D<LabelsT, TensorT, Eigen::GpuDevice>::SelectAndSumPixels4D;
    void setLabelsValuesResults(Eigen::GpuDevice& device) override;
  };
  template<typename LabelsT, typename TensorT>
  inline void SelectAndSumPixels4DGpu<LabelsT, TensorT>::setLabelsValuesResults(Eigen::GpuDevice& device)
  {
    // make the labels and sync to the device
    Eigen::Tensor<LabelsT, 2> select_labels_x_values(1, this->dim_span_);
    Eigen::Tensor<LabelsT, 2> select_labels_y_values(1, this->dim_span_);
    Eigen::Tensor<LabelsT, 2> select_labels_z_values(1, this->dim_span_);
    Eigen::Tensor<LabelsT, 2> select_labels_t_values(1, this->dim_span_);
    for (int i = 0; i < this->dim_span_; ++i) {
      select_labels_x_values(0, i) = i + 1;
      select_labels_y_values(0, i) = i + 1;
      select_labels_z_values(0, i) = i + 1;
      select_labels_t_values(0, i) = i + 1;
    }
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_x(select_labels_x_values.dimensions());
    select_labels_x.setData(select_labels_x_values);
    select_labels_x.syncHAndDData(device);
    this->select_labels_x_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_x);
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_y(select_labels_y_values.dimensions());
    select_labels_y.setData(select_labels_y_values);
    select_labels_y.syncHAndDData(device);
    this->select_labels_y_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_y);
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_z(select_labels_z_values.dimensions());
    select_labels_z.setData(select_labels_z_values);
    select_labels_z.syncHAndDData(device);
    this->select_labels_z_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_z);
    TensorDataGpuPrimitiveT<LabelsT, 2> select_labels_t(select_labels_t_values.dimensions());
    select_labels_t.setData(select_labels_t_values);
    select_labels_t.syncHAndDData(device);
    this->select_labels_t_ = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(select_labels_t);

    // allocate memory for the results
    TensorDataGpuPrimitiveT<TensorT, 1> results(Eigen::array<Eigen::Index, 1>({ 1 }));
    results.setData();
    results.syncHAndDData(device);
    this->result_ = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 1>>(results);
  }

	/*
	@brief Specialized `PixelManager` for the 0D and Gpu case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager0DGpu : public PixelManager0D<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using PixelManager0D<LabelsT, TensorT, Eigen::GpuDevice>::PixelManager0D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager0DGpu<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager0DGpu<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr) {
		TensorDataGpuPrimitiveT<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(values_data);
	}

	/*
	@brief Specialized `PixelManager` for the 1D and Gpu case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager1DGpu : public PixelManager1D<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using PixelManager1D<LabelsT, TensorT, Eigen::GpuDevice>::PixelManager1D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager1DGpu<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager1DGpu<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr) {
		TensorDataGpuPrimitiveT<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(values_data);
	}

	/*
	@brief Specialized `PixelManager` for the 2D and Gpu case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager2DGpu : public PixelManager2D<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using PixelManager2D<LabelsT, TensorT, Eigen::GpuDevice>::PixelManager2D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager2DGpu<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager2DGpu<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 2>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>>& values_ptr) {
		TensorDataGpuPrimitiveT<TensorT, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 2>>(values_data);
	}

	/*
	@brief Specialized `PixelManager` for the 3D and Gpu case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager3DGpu : public PixelManager3D<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using PixelManager3D<LabelsT, TensorT, Eigen::GpuDevice>::PixelManager3D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 3>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 3>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager3DGpu<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager3DGpu<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 3>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 3>>& values_ptr) {
		TensorDataGpuPrimitiveT<TensorT, 3> values_data(Eigen::array<Eigen::Index, 3>({ values.dimension(0), values.dimension(1), values.dimension(2) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 3>>(values_data);
	}

	/*
	@brief Specialized `PixelManager` for the 4D and Gpu case
	*/
	template<typename LabelsT, typename TensorT>
	class PixelManager4DGpu : public PixelManager4D<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		using PixelManager4D<LabelsT, TensorT, Eigen::GpuDevice>::PixelManager4D;
		void makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr);
		void makeValuesPtr(const Eigen::Tensor<TensorT, 4>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 4>>& values_ptr);
	};
	template<typename LabelsT, typename TensorT>
	void PixelManager4DGpu<LabelsT, TensorT>::makeLabelsPtr(const Eigen::Tensor<LabelsT, 2>& labels, std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>>& labels_ptr) {
		TensorDataGpuPrimitiveT<LabelsT, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
		labels_data.setData(labels);
		labels_ptr = std::make_shared<TensorDataGpuPrimitiveT<LabelsT, 2>>(labels_data);
	}
	template<typename LabelsT, typename TensorT>
	void PixelManager4DGpu<LabelsT, TensorT>::makeValuesPtr(const Eigen::Tensor<TensorT, 4>& values, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 4>>& values_ptr) {
		TensorDataGpuPrimitiveT<TensorT, 4> values_data(Eigen::array<Eigen::Index, 4>({ values.dimension(0), values.dimension(1), values.dimension(2), values.dimension(3) }));
		values_data.setData(values);
		values_ptr = std::make_shared<TensorDataGpuPrimitiveT<TensorT, 4>>(values_data);
	}

	/*
	@brief A class for running 1 line insertion, deletion, and update benchmarks
	*/
	template<typename LabelsT, typename TensorT>
	class Benchmark1TimePointGpu : public BenchmarkPixel1TimePoint<LabelsT, TensorT, Eigen::GpuDevice> {
	protected:
		void insert1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
		void insert1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint1D`
		void insert1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint2D`
		void insert1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint3D`
		void insert1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `insert1TimePoint4D`

		void update1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
		void update1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint1D`
		void update1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint2D`
		void update1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint3D`
		void update1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `update1TimePoint4D`

		void delete1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint0D`
		void delete1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint1D`
		void delete1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint2D`
		void delete1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint3D`
		void delete1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `delete1TimePoint4D`

    TensorT selectAndSumPixels0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels0D`
    TensorT selectAndSumPixels1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels1D`
    TensorT selectAndSumPixels2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels2D`
    TensorT selectAndSumPixels3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels3D`
    TensorT selectAndSumPixels4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const override; ///< Device specific interface to call `selectAndSumPixels4D`
  };
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::insert1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager0DGpu<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint0D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::insert1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager1DGpu<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint1D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::insert1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager2DGpu<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint2D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::insert1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager3DGpu<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint3D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::insert1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager4DGpu<LabelsT, TensorT> pixel_manager(data_size, false);
		this->insert1TimePoint4D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::update1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager0DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint0D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::update1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager1DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint1D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::update1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager2DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint2D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::update1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager3DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint3D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::update1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager4DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		this->update1TimePoint4D_(pixel_manager, transaction_manager, data_size, in_memory, device);
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::delete1TimePoint0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager0DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable0D<LabelsT, Eigen::GpuDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2> tensorDelete("TTable", "indices", selectClause);
			std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::delete1TimePoint1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager1DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> values_ptr;
		int span = data_size / std::pow(data_size, 0.25);
		for (int i = 0; i < data_size - 1; i += span) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
			SelectTable1D<LabelsT, Eigen::GpuDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2> tensorDelete("TTable", "xyzt", selectClause);
			std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::delete1TimePoint2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager2DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 2>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::GpuDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 2>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::delete1TimePoint3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager3DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 3>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::GpuDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 3> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 3>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
	template<typename LabelsT, typename TensorT>
	void Benchmark1TimePointGpu<LabelsT, TensorT>::delete1TimePoint4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const {
		PixelManager4DGpu<LabelsT, TensorT> pixel_manager(data_size, true);
		std::shared_ptr<TensorData<LabelsT, Eigen::GpuDevice, 2>> labels_ptr;
		std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 4>> values_ptr;
		int time_points = std::pow(data_size, 0.25);
		for (int i = 0; i < time_points - 1; ++i) { // TOOD: strange run-time error upon deallocation
			labels_ptr.reset();
			values_ptr.reset();
			pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
			SelectTable2D<LabelsT, Eigen::GpuDevice> selectClause(labels_ptr);
			TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 4> tensorDelete("TTable", "t", selectClause);
			std::shared_ptr<TensorOperation<Eigen::GpuDevice>> tensorDelete_ptr = std::make_shared<TensorDeleteFromAxisGpuPrimitiveT<LabelsT, TensorT, 4>>(tensorDelete);
			transaction_manager.executeOperation(tensorDelete_ptr, device);
      if (!in_memory) {
        transaction_manager.commit(device);
        transaction_manager.initTensorCollectionTensorData(device);
      }
		}
	}
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1TimePointGpu<LabelsT, TensorT>::selectAndSumPixels0D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndSumPixels0DGpu<int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    gpuErrchk(cudaStreamSynchronize(device.stream()));
    select_sum_pixels.result_->syncHAndDData(device);
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1TimePointGpu<LabelsT, TensorT>::selectAndSumPixels1D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndSumPixels1DGpu<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    gpuErrchk(cudaStreamSynchronize(device.stream()));
    select_sum_pixels.result_->syncHAndDData(device);
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1TimePointGpu<LabelsT, TensorT>::selectAndSumPixels2D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndSumPixels2DGpu<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    gpuErrchk(cudaStreamSynchronize(device.stream()));
    select_sum_pixels.result_->syncHAndDData(device);
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1TimePointGpu<LabelsT, TensorT>::selectAndSumPixels3D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndSumPixels3DGpu<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    gpuErrchk(cudaStreamSynchronize(device.stream()));
    select_sum_pixels.result_->syncHAndDData(device);
    return select_sum_pixels.result_->getData()(0);
  }
  template<typename LabelsT, typename TensorT>
  inline TensorT Benchmark1TimePointGpu<LabelsT, TensorT>::selectAndSumPixels4D(TransactionManager<Eigen::GpuDevice>& transaction_manager, const int& data_size, const bool& in_memory, Eigen::GpuDevice& device) const
  {
    SelectAndSumPixels4DGpu<int, int> select_sum_pixels(data_size);
    select_sum_pixels(transaction_manager.getTensorCollection(), device);
    if (!in_memory) {
      transaction_manager.initTensorCollectionTensorData(device);
    }
    gpuErrchk(cudaStreamSynchronize(device.stream()));
    select_sum_pixels.result_->syncHAndDData(device);
    return select_sum_pixels.result_->getData()(0);
  }

	template<typename LabelsT, typename TensorT>
	class TensorCollectionGeneratorGpu : public PixelTensorCollectionGenerator<LabelsT, TensorT, Eigen::GpuDevice> {
	public:
		std::shared_ptr<TensorCollection<Eigen::GpuDevice>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::GpuDevice>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::GpuDevice>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::GpuDevice>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const override;
		std::shared_ptr<TensorCollection<Eigen::GpuDevice>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const override;
	};
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> TensorCollectionGeneratorGpu<LabelsT, TensorT>::make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
    Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(1);
    dimensions_1.setValues({ "xyztv" });
		dimensions_2.setValues({ "indices" });
		Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_1(1, 5);
    labels_1.setValues({ { TensorArrayGpu8<char>("x"), TensorArrayGpu8<char>("y"), TensorArrayGpu8<char>("z"), TensorArrayGpu8<char>("t"), TensorArrayGpu8<char>("v")} });

		// Setup the tables
		// TODO: refactor for the case where LabelsT != TensorT
		std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 2>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<TensorT, 2>>(TensorTableGpuPrimitiveT<TensorT, 2>("TTable"));
		std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("xyztv", dimensions_1, labels_1));
		//std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("xyzt", dimensions_1a, labels_1a));
		//std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("v", dimensions_1b, labels_1b));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("indices", 1, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ data_size, 5 }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> TensorCollectionGeneratorGpu<LabelsT, TensorT>::make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
		Eigen::Tensor<std::string, 1> dimensions_1(1), dimensions_2(4);
		dimensions_1.setValues({ "values" });
		dimensions_2.setValues({ "x","y","z","t" });
		Eigen::Tensor<TensorArrayGpu8<char>, 2> labels_v(1, 1);
		labels_v.setValues({ { TensorArrayGpu8<char>("values")} });

		// Setup the tables
		std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 2>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<TensorT, 2>>(TensorTableGpuPrimitiveT<TensorT, 2>("TTable"));
		std::shared_ptr<TensorAxis<TensorArrayGpu8<char>, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuClassT<TensorArrayGpu8, char>>(TensorAxisGpuClassT<TensorArrayGpu8, char>("values", dimensions_1, labels_v));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("xyzt", 4, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ 1, data_size }));

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> TensorCollectionGeneratorGpu<LabelsT, TensorT>::make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
		int dim_span = std::pow(data_size, 0.25);
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
		std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 2>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<TensorT, 2>>(TensorTableGpuPrimitiveT<TensorT, 2>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("xyz", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("t", 1, 0));
		table_1_axis_2_ptr->setDimensions(dimensions_2);
		table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
		table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
		table_1_ptr->setAxes(device);

		// Setup the table data
		table_1_ptr->setData();
		table_1_ptr->setShardSpans(shard_span);
    table_1_ptr->setMaximumDimensions(Eigen::array<Eigen::Index, 2>({ dim_span, int(std::pow(dim_span, 3)) })); // NOTE: axes are added in alphabetical order

		// Setup the collection
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> TensorCollectionGeneratorGpu<LabelsT, TensorT>::make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
		int dim_span = std::pow(data_size, 0.25);
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
		std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 3>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<TensorT, 3>>(TensorTableGpuPrimitiveT<TensorT, 3>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("xy", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("z", dimensions_2, labels_2));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("t", 1, 0));
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
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
	template<typename LabelsT, typename TensorT>
	std::shared_ptr<TensorCollection<Eigen::GpuDevice>> TensorCollectionGeneratorGpu<LabelsT, TensorT>::make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar, Eigen::GpuDevice& device) const
	{
		// Setup the axes
		int dim_span = std::pow(data_size, 0.25);
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
		std::shared_ptr<TensorTable<TensorT, Eigen::GpuDevice, 4>> table_1_ptr = std::make_shared<TensorTableGpuPrimitiveT<TensorT, 4>>(TensorTableGpuPrimitiveT<TensorT, 4>("TTable"));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_1_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("x", dimensions_1, labels_1));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_2_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("y", dimensions_2, labels_2));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_3_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("z", dimensions_3, labels_3));
    std::shared_ptr<TensorAxis<LabelsT, Eigen::GpuDevice>> table_1_axis_4_ptr = std::make_shared<TensorAxisGpuPrimitiveT<LabelsT>>(TensorAxisGpuPrimitiveT<LabelsT>("t", 1, 0));
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
		auto collection_1_ptr = std::make_shared<TensorCollectionGpu>(TensorCollectionGpu());
		collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
		return collection_1_ptr;
	}
};
#endif
#endif //TENSORBASE_BENCHMARKPIXELSGPU_H