/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATAGPU_H
#define TENSORBASE_TENSORDATAGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // CUB sort, select, partition, and runLengthEncode
#include <thrust/remove.h> // THRUST select
#include <thrust/sort.h> // THRUST sort
#include <thrust/device_ptr.h> // THRUST sort, select, partition, and runLengthEncode
#include <thrust/binary_search.h> // THRUST histogram
#include <thrust/device_vector.h> // THRUST histogram
#include <thrust/execution_policy.h> // THRUST sort, select, partition, and runLengthEncode

#include <TensorBase/ml/TensorData.h>
#include <TensorBase/ml/TensorArrayGpu.h>

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
	/*
	Thrust helper functors

	NOTE: These MUST be declared before usage or the CUDA compiler will throw an error
	*/
	template<typename TensorT>
	struct HistogramBinHelper {
		HistogramBinHelper(const TensorT& lower_level, const TensorT& bin_width) : lower_level_(lower_level), bin_width_(bin_width) {}
		__host__ __device__
			TensorT operator()(const TensorT& v) {
			return (v + 1) * bin_width_ + lower_level_;
		}
		TensorT lower_level_ = TensorT(0);
		TensorT bin_width_ = TensorT(1);
	};

	struct isGreaterThanZero {
		__host__ __device__
			bool operator()(const int& x) {
			return x > 0;
		}
	};

  /**
    @brief Tensor data class specialization for Eigen::GpuDevice (single GPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataGpu : public TensorData<TensorT, Eigen::GpuDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::GpuDevice, TDim>::TensorData;
    ~TensorDataGpu() = default;
    // Interface overrides
    void setData(const Eigen::Tensor<TensorT, TDim>& data) override; ///< data setter
    void setData() override;
    bool syncHAndDData(Eigen::GpuDevice& device) override;
    std::shared_ptr<TensorT[]> getHDataPointer() override;
    std::shared_ptr<TensorT[]> getDataPointer() override;
  private:
    	friend class cereal::access;
    	template<class Archive>
    	void serialize(Archive& archive) {
    		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice, TDim>>(this));
    	}
  };
  template<typename TensorT, int TDim>
  void TensorDataGpu<TensorT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data) {
    // allocate cuda and pinned host memory
    TensorT* d_data;
    TensorT* h_data;
    assert(cudaMalloc((void**)(&d_data), getTensorBytes()) == cudaSuccess);
    assert(cudaHostAlloc((void**)(&h_data), getTensorBytes(), cudaHostAllocDefault) == cudaSuccess);
    // copy the tensor
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, getDimensions());
    data_copy = data;
    // define the deleters
    auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
    auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
    this->h_data_.reset(h_data, h_deleter);
    this->d_data_.reset(d_data, d_deleter);
    this->h_data_updated_ = true;
    this->d_data_updated_ = false;
  };
  template<typename TensorT, int TDim>
  void TensorDataGpu<TensorT, TDim>::setData() {
    // allocate cuda and pinned host memory
    TensorT* d_data;
    TensorT* h_data;
    assert(cudaMalloc((void**)(&d_data), getTensorBytes()) == cudaSuccess);
    assert(cudaHostAlloc((void**)(&h_data), getTensorBytes(), cudaHostAllocDefault) == cudaSuccess);
    // define the deleters
    auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
    auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
    this->h_data_.reset(h_data, h_deleter);
    this->d_data_.reset(d_data, d_deleter);
    this->h_data_updated_ = true;
    this->d_data_updated_ = false;
  };
  template<typename TensorT, int TDim>
  bool TensorDataGpu<TensorT, TDim>::syncHAndDData(Eigen::GpuDevice& device) {
    if (this->h_data_updated_ && !this->d_data_updated_) {
      device.memcpyHostToDevice(this->d_data_.get(), this->h_data_.get(), this->getTensorBytes());
      this->d_data_updated_ = true;
      this->h_data_updated_ = false;
      return true;
    }
    else if (!this->h_data_updated_ && this->d_data_updated_) {
      device.memcpyDeviceToHost(this->h_data_.get(), this->d_data_.get(), this->getTensorBytes());
      this->h_data_updated_ = true;
      this->d_data_updated_ = false;
      return true;
    }
    else {
      std::cout << "Both host and device are synchronized." << std::endl;
      return false;
    }
  }
  template<typename TensorT, int TDim>
  inline std::shared_ptr<TensorT[]> TensorDataGpu<TensorT, TDim>::getHDataPointer()
  {
    return h_data_;
  }
  template<typename TensorT, int TDim>
  std::shared_ptr<TensorT[]> TensorDataGpu<TensorT, TDim>::getDataPointer() {
    return d_data_;
  }

  /**
    @brief Tensor data class specialization for Eigen::GpuDevice (single GPU) using primitive template types
  */
  template<typename TensorT, int TDim>
  class TensorDataGpuPrimitiveT : public TensorDataGpu<TensorT, TDim> {
  public:
    using TensorDataGpu<TensorT, TDim>::TensorDataGpu;
    ~TensorDataGpuPrimitiveT() = default;
    // Interface overrides
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> copy(Eigen::GpuDevice& device) override;
    // Algorithm Interface overrides
    void select(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string& sort_order, Eigen::GpuDevice& device) override;
    void sort(const std::string& sort_order, Eigen::GpuDevice& device) override;
    void sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
    void partition(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
		void runLengthEncode(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice& device) override;
		void histogram(const int& n_levels, const TensorT& lower_level, const TensorT& upper_level, std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& histogram, Eigen::GpuDevice& device) override { histogram_(n_levels, lower_level, upper_level, histogram, device); }
		template<typename T = TensorT, std::enable_if_t<std::is_fundamental<T>::value && !std::is_same<char, T>::value, int> = 0>
		void histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::GpuDevice, 1>>& histogram, Eigen::GpuDevice& device);
		template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value || std::is_same<char, T>::value, int> = 0>
		void histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::GpuDevice, 1>>& histogram, Eigen::GpuDevice& device) { /*Do nothing*/ };
    // Other
    void convertFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device) override { convertFromStringToTensorT_(data_new, device); };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, float>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, double>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice, TDim>>(this));
    }
  };
  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> TensorDataGpuPrimitiveT<TensorT, TDim>::copy(Eigen::GpuDevice& device) {
    // initialize the new data
    if (this->d_data_updated_) {
      this->syncHAndDData(device);
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      this->setDataStatus(false, true);
    }
    TensorDataGpuPrimitiveT<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //data_new.setData();
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //this->syncHAndDData(device);
    //data_new_values.device(device) = data_values; // NOTE: .device(device) fails
    return std::make_shared<TensorDataGpuPrimitiveT<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::select(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // Temporary device storage for the size of the selection
    int *d_n_selected;
    assert(cudaMalloc((void**)(&d_n_selected), sizeof(int)) == cudaSuccess);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, this->getDataPointer().get(), indices->getDataPointer().get(), tensor_select->getDataPointer().get(),
      d_n_selected, indices->getTensorSize(), device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run selection
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, this->getDataPointer().get(), indices->getDataPointer().get(), tensor_select->getDataPointer().get(),
      d_n_selected, indices->getTensorSize(), device.stream());

    assert(cudaFree(d_n_selected) == cudaSuccess);
    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string & sort_order, Eigen::GpuDevice & device)
  {
    // Temporary copies for the algorithm 
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> keys_copy = this->copy(device);
    keys_copy->syncHAndDData(device);
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>> values_copy = indices->copy(device);
    values_copy->syncHAndDData(device);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if (sort_order == "ASC")
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        this->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), indices->getDataPointer().get(),
        indices->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());
    else if (sort_order == "DESC")
      cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        this->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), indices->getDataPointer().get(),
        indices->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run sorting operation
    if (sort_order == "ASC")
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        this->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), indices->getDataPointer().get(),
        indices->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());
    else if (sort_order == "DESC")
      cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        this->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), indices->getDataPointer().get(),
        indices->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());

    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::sort(const std::string & sort_order, Eigen::GpuDevice & device)
  {
    // Temporary copies for the algorithm 
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> keys_copy = this->copy(device);
    keys_copy->syncHAndDData(device);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    if (sort_order == "ASC")
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
        keys_copy->getDataPointer().get(), this->getDataPointer().get(),
        this->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());
    else if (sort_order == "DESC")
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
        keys_copy->getDataPointer().get(), this->getDataPointer().get(),
        this->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run sorting operation
    if (sort_order == "ASC")
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
        keys_copy->getDataPointer().get(), this->getDataPointer().get(),
        this->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());
    else if (sort_order == "DESC")
      cub::DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes,
        keys_copy->getDataPointer().get(), this->getDataPointer().get(),
        this->getTensorSize(), 0, sizeof(TensorT) * 8, device.stream());

    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // Temporary copies for the algorithm 
    std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>> keys_copy = indices->copy(device);
    keys_copy->syncHAndDData(device);
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> values_copy = this->copy(device);
    values_copy->syncHAndDData(device);

    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      indices->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), this->getDataPointer().get(),
      indices->getTensorSize(), 0, sizeof(int) * 8, device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      indices->getDataPointer().get(), keys_copy->getDataPointer().get(), values_copy->getDataPointer().get(), this->getDataPointer().get(),
      indices->getTensorSize(), 0, sizeof(int) * 8, device.stream());

    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::partition(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // Temporary copies for the algorithm 
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> values_copy = this->copy(device);
    values_copy->syncHAndDData(device);

    int  *d_num_selected_out;
    assert(cudaMalloc((void**)(&d_num_selected_out), sizeof(int)) == cudaSuccess);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, values_copy->getDataPointer().get(), indices->getDataPointer().get(),
      this->getDataPointer().get(), d_num_selected_out, this->getTensorSize(), device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run selection
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, values_copy->getDataPointer().get(), indices->getDataPointer().get(),
      this->getDataPointer().get(), d_num_selected_out, this->getTensorSize(), device.stream());

    assert(cudaFree(d_temp_storage) == cudaSuccess);
    assert(cudaFree(d_num_selected_out) == cudaSuccess);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::runLengthEncode(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice & device)
  {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, this->getDataPointer().get(),
      unique->getDataPointer().get(), count->getDataPointer().get(), n_runs->getDataPointer().get(), this->getTensorSize(), device.stream());

    // Allocate temporary storage
    assert(cudaMalloc((void**)(&d_temp_storage), temp_storage_bytes) == cudaSuccess);

    // Run encoding
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, this->getDataPointer().get(),
      unique->getDataPointer().get(), count->getDataPointer().get(), n_runs->getDataPointer().get(), this->getTensorSize(), device.stream());

    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }
	template<typename TensorT, int TDim>
	template<typename T, std::enable_if_t<std::is_fundamental<T>::value && !std::is_same<char, T>::value, int>>
	inline void TensorDataGpuPrimitiveT<TensorT, TDim>::histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::GpuDevice, 1>>& histogram, Eigen::GpuDevice& device)
	{
		//// NOTE: Cannot compile due to odd bug in cub::DeviceHistogram::HistogramMultiEven
		////BEGIN CUB HISTOGRAM_____________________________________________________________
		//// Determine temporary device storage requirements
		//void* d_temp_storage = NULL;
		//size_t   temp_storage_bytes = 0;
		//cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
		//	this->getDataPointer().get(), histogram->getDataPointer().get(), n_levels, lower_level, upper_level, this->getTensorSize(), device.stream());

		//// Allocate temporary storage
		//cudaMalloc(&d_temp_storage, temp_storage_bytes);

		//// Compute histograms
		//cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
		//	this->getDataPointer().get(), histogram->getDataPointer().get(), n_levels, lower_level, upper_level, this->getTensorSize(), device.stream());

		//assert(cudaFree(d_temp_storage) == cudaSuccess);
		////END CUB HISTOGRAM_____________________________________________________________

		// Copy the data
		auto data_copy = this->copy(device);
		data_copy->syncHAndDData(device);
		thrust::device_ptr<T> d_data(data_copy->getDataPointer().get());
		thrust::device_ptr<T> d_histogram(histogram->getDataPointer().get());

		// sort data to bring equal elements together
		thrust::sort(thrust::cuda::par.on(device.stream()), d_data, d_data + data_copy->getTensorSize());

		// histogram bins and widths
		const int n_bins = n_levels - 1;
		const T bin_width = (upper_level - lower_level) / (n_levels - T(1));
		thrust::device_vector<T> bin_search(n_bins);
		thrust::sequence(thrust::cuda::par.on(device.stream()), bin_search.begin(), bin_search.end());
		HistogramBinHelper<T> histogramBinHelper(lower_level, bin_width);
		thrust::transform(thrust::cuda::par.on(device.stream()), bin_search.begin(), bin_search.end(), bin_search.begin(), histogramBinHelper);

		// find the end of each bin of values
		thrust::upper_bound(thrust::cuda::par.on(device.stream()), d_data, d_data + data_copy->getTensorSize(),	bin_search.begin(), bin_search.end(),	d_histogram);

		// compute the histogram by taking differences of the cumulative histogram
		thrust::adjacent_difference(thrust::cuda::par.on(device.stream()), d_histogram, d_histogram + histogram->getTensorSize(),	d_histogram);
	}
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value, int>>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    Eigen::DefaultDevice default_device;
    // convert the data from string to TensorT
    this->setDataStatus(true, false);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->h_data_.get(), this->dimensions_);
    data_converted.device(default_device) = data_new.unaryExpr([](const std::string& elem) { return std::stoi(elem); });
    this->syncHAndDData(device);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, float>::value, int>>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    Eigen::DefaultDevice default_device;
    // convert the data from string to TensorT
    this->setDataStatus(true, false);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->h_data_.get(), this->dimensions_);
    data_converted.device(default_device) = data_new.unaryExpr([](const std::string& elem) { return std::stof(elem); });
    this->syncHAndDData(device);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, double>::value, int>>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    Eigen::DefaultDevice default_device;
    // convert the data from string to TensorT
    this->setDataStatus(true, false);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->h_data_.get(), this->dimensions_);
    data_converted.device(default_device) = data_new.unaryExpr([](const std::string& elem) { return std::stod(elem); });
    this->syncHAndDData(device);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    Eigen::DefaultDevice default_device;
    // convert the data from string to TensorT
    this->setDataStatus(true, false);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->h_data_.get(), this->dimensions_);
    data_converted.device(default_device) = data_new.unaryExpr([](const std::string& elem) { return elem.c_str()[0]; });
    this->syncHAndDData(device);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, bool>::value, int>>
  inline void TensorDataGpuPrimitiveT<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    Eigen::DefaultDevice default_device;
    // convert the data from string to TensorT
    this->setDataStatus(true, false);
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->h_data_.get(), this->dimensions_);
    data_converted.device(default_device) = data_new.unaryExpr([](const std::string& elem) { return elem == "1"; });
    this->syncHAndDData(device);
  }

  /**
    @brief Tensor data class specialization for Eigen::GpuDevice (single GPU) using custom class template types
  */
  template<template<class> class ArrayT, class TensorT, int TDim>
  class TensorDataGpuClassT : public TensorDataGpu<ArrayT<TensorT>, TDim> {
  public:
    using TensorDataGpu<ArrayT<TensorT>, TDim>::TensorDataGpu;
    ~TensorDataGpuClassT() = default;
    // Interface overrides
    std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>> copy(Eigen::GpuDevice& device) override;
    // Algorithm Interface overrides
    void select(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string& sort_order, Eigen::GpuDevice& device) override;
    void sort(const std::string& sort_order, Eigen::GpuDevice& device) override;
    void sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
    void partition(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device) override;
    void runLengthEncode(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice& device) override;
		void histogram(const int& n_levels, const ArrayT<TensorT>& lower_level, const ArrayT<TensorT>& upper_level, std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& histogram, Eigen::GpuDevice& device) override {/*Not available*/}
    // Other
    void convertFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device) override { convertFromStringToTensorT_(data_new, device); };
    template<template<class> typename A = ArrayT, typename T = TensorT, std::enable_if_t<std::is_same<A<T>, A<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device);
    template<template<class> typename A = ArrayT, typename T = TensorT, std::enable_if_t<std::is_same<A<T>, A<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device) {/*TODO*/};
    template<template<class> typename A = ArrayT, typename T = TensorT, std::enable_if_t<std::is_same<A<T>, A<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device) {/*TODO*/ };
    template<template<class> typename A = ArrayT, typename T = TensorT, std::enable_if_t<std::is_same<A<T>, A<double>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice& device) {/*TODO*/ };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>(this));
    }
  };
  template<template<class> class ArrayT, class TensorT, int TDim>
  std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>> TensorDataGpuClassT<ArrayT, TensorT, TDim>::copy(Eigen::GpuDevice& device) {
    // initialize the new data
    if (this->d_data_updated_) {
      this->syncHAndDData(device);
      assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
      this->setDataStatus(false, true);
    }
    TensorDataGpuClassT<ArrayT, TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //data_new.setData();
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //this->syncHAndDData(device);
    //data_new_values.device(device) = data_values; // NOTE: .device(device) fails
    return std::make_shared<TensorDataGpuClassT<ArrayT, TensorT, TDim>>(data_new);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::select(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // Create a copy of the data
    auto data_copy = this->copy(device);
    data_copy->syncHAndDData(device);

    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(data_copy->getDataPointer().get());
    thrust::device_ptr<int> d_indices(indices->getDataPointer().get());

    // call remove_if on the flagged entries marked as false (i.e., 0)
    thrust::remove_if(thrust::cuda::par.on(device.stream()), d_data, d_data + data_copy->getTensorSize(), d_indices, thrust::logical_not<bool>());

    // Copy over the selected values
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 1>> tensor_select_values(tensor_select->getDataPointer().get(), (int)tensor_select->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<ArrayT<TensorT>, 1>> data_copy_values(data_copy->getDataPointer().get(), (int)data_copy->getTensorSize());
    Eigen::array<Eigen::Index, 1> offset, span; 
    offset.at(0) = 0;
    span.at(0) = (int)tensor_select->getTensorSize();
    tensor_select_values.slice(offset, span).device(device) = data_copy_values.slice(offset, span);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string & sort_order, Eigen::GpuDevice & device)
  {
    // Create a copy of the data
    auto data_copy = this->copy(device);
    data_copy->syncHAndDData(device);

    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(data_copy->getDataPointer().get());
    thrust::device_ptr<int> d_indices(indices->getDataPointer().get());

    if (sort_order == "ASC") {
      thrust::sort_by_key(thrust::cuda::par.on(device.stream()), d_data, d_data + data_copy->getTensorSize(), d_indices);
    }
    else if (sort_order == "DESC") {
      isGreaterThanGpu8 comp(data_copy->getTensorSize());
      thrust::sort_by_key(thrust::cuda::par.on(device.stream()), d_data, d_data + data_copy->getTensorSize(), d_indices, comp);
    }
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::sort(const std::string & sort_order, Eigen::GpuDevice & device)
  {
    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(this->getDataPointer().get());
    if (sort_order == "ASC") {
      thrust::stable_sort(thrust::cuda::par.on(device.stream()), d_data, d_data + this->getTensorSize());
    }
    else if (sort_order == "DESC") {
      isGreaterThanGpu8 comp(this->getTensorSize());
      thrust::stable_sort(thrust::cuda::par.on(device.stream()), d_data, d_data + this->getTensorSize(), comp);
    }
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(this->getDataPointer().get());
    thrust::device_ptr<int> d_indices(indices->getDataPointer().get());

    thrust::sort_by_key(thrust::cuda::par.on(device.stream()), d_indices, d_indices + this->getTensorSize(), d_data);
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::partition(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(this->getDataPointer().get());
    thrust::device_ptr<int> d_indices(indices->getDataPointer().get());

    // call partition on the flagged entries marked as true (i.e., 1)
    thrust::stable_partition(thrust::cuda::par.on(device.stream()), d_data, d_data + this->getTensorSize(), d_indices, isGreaterThanZero());
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::runLengthEncode(std::shared_ptr<TensorData<ArrayT<TensorT>, Eigen::GpuDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::GpuDevice, 1>>& n_runs, Eigen::GpuDevice & device)
  {
    // make thrust device pointers to the data
    thrust::device_ptr<ArrayT<TensorT>> d_data(this->getDataPointer().get());
    thrust::device_ptr<ArrayT<TensorT>> d_unique(unique->getDataPointer().get());
    thrust::device_ptr<int> d_count(count->getDataPointer().get());

    // compute run lengths
    size_t num_runs = thrust::reduce_by_key(thrust::cuda::par.on(device.stream()),
      d_data, d_data + this->getTensorSize(),          // input key sequence
      thrust::constant_iterator<int>(1),   // input value sequence
      d_unique,                      // output key sequence
      d_count                      // output value sequence
    ).first - d_unique;            // compute the output size

    // update the n_runs
    n_runs->syncHAndDData(device); // D to H
    assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);
    n_runs->getData()(0) = num_runs;
    n_runs->syncHAndDData(device); // H to D
  }
  template<template<class> class ArrayT, class TensorT, int TDim>
  template<template<class> typename A, typename T, std::enable_if_t<std::is_same<A<T>, A<char>>::value, int>>
  inline void TensorDataGpuClassT<ArrayT, TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::GpuDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());

    // convert the data from string to TensorArrayGpu<TensorT>
    this->setDataStatus(true, false);
    std::transform(data_new.data(), data_new.data() + data_new.size(), this->h_data_.get(),
      [](const std::string& elem) -> ArrayT<TensorT> { return ArrayT<TensorT>(elem); });
    this->syncHAndDData(device);

    //// make thrust device pointers to the data
    //thrust::device_ptr<ArrayT<TensorT>> d_data(this->getDataPointer().get());
    //thrust::device_ptr<ArrayT<TensorT>> d_data_new(data_new.data());

    //// convert the data from string to TensorT
    //thrust::transform(thrust::cuda::par.on(device.stream()), d_data_new, d_data_new + data_new.size(), d_data, convertStrToTensorArrayGpu<ArrayT<TensorT>>());
  }
}

// Cereal registration of TensorTs: float, int, char, double and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuPrimitiveT<char, 4>);

CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu8, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu8, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu8, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu8, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu32, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu32, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu32, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu32, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu128, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu128, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu128, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu128, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu512, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu512, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu512, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu512, char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu2048, char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu2048, char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu2048, char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataGpuClassT<TensorBase::TensorArrayGpu2048, char, 4>);
#endif
#endif //TENSORBASE_TENSORDATAGPU_H