/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATAGPU_H
#define TENSORBASE_TENSORDATAGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh> // Sort

#include <TensorBase/ml/TensorData.h>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{
  /**
    @brief Tensor data class specialization for Eigen::GpuDevice (single GPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataGpu : public TensorData<TensorT, Eigen::GpuDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::GpuDevice, TDim>::TensorData;
    ~TensorDataGpu() = default;
    std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> copy(Eigen::GpuDevice& device);
    void select(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device);
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string& sort_order, Eigen::GpuDevice& device);
    void sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice& device);
    std::shared_ptr<TensorT> getDataPointer();
    void setData(const Eigen::Tensor<TensorT, TDim>& data); ///< data setter
    void setData();
    bool syncHAndDData(Eigen::GpuDevice& device);
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice>>(this));
    //	}
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>> TensorDataGpu<TensorT, TDim>::copy(Eigen::GpuDevice& device) {
    // initialize the new data
    TensorDataGpu<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //data_new.setData();
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //data_new_values.device(device) = data_values; // NOTE: .device(device) fails
    return std::make_shared<TensorDataGpu<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataGpu<TensorT, TDim>::select(std::shared_ptr<TensorData<TensorT, Eigen::GpuDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
  {
    // Temporary device storage for the size of the selection
    //int *h_n_selected;  // TODO: comment after debugging
    int *d_n_selected;
    //assert(cudaHostAlloc((void**)(&h_n_selected), sizeof(int), cudaHostAllocDefault) == cudaSuccess);  // TODO: comment after debugging
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

    //device.memcpyDeviceToHost(h_n_selected, d_n_selected, sizeof(int));  // TODO: comment after debugging
    //assert(cudaStreamSynchronize(device.stream()) == cudaSuccess);  // TODO: comment after debugging
    //std::cout << h_n_selected[0] << std::endl;  // TODO: comment after debugging

    //assert(cudaFreeHost(h_n_selected) == cudaSuccess);  // TODO: comment after debugging
    assert(cudaFree(d_n_selected) == cudaSuccess);
    assert(cudaFree(d_temp_storage) == cudaSuccess);
  }

  template<typename TensorT, int TDim>
  inline void TensorDataGpu<TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, const std::string & sort_order, Eigen::GpuDevice & device)
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
  inline void TensorDataGpu<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::GpuDevice, TDim>>& indices, Eigen::GpuDevice & device)
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
  std::shared_ptr<TensorT> TensorDataGpu<TensorT, TDim>::getDataPointer() {
    //if (!this->d_data_updated_) { // NOTE: this will break the interface
    //  this->syncHAndDData(device);
    //}
    return d_data_;
  }
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
}
//CEREAL_REGISTER_TYPE(TensorData::TensorDataGpu<float>);
//// TODO: add double, int, etc.
#endif
#endif //TENSORBASE_TENSORDATAGPU_H