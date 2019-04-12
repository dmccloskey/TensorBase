/**TODO:  Add copyright*/

#ifndef TENSORDATA_TENSORDATA_H
#define TENSORDATA_TENSORDATA_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace TensorData
{
  /**
    @brief Tensor data
  */
	template<typename TensorT, typename DeviceT, int Dim, int... Args>
  class TensorData
  {
public:
    TensorData() = default; ///< Default constructor
    TensorData(const TensorData& other)
		{
			h_data_ = other.h_data_;
			d_data_ = other.d_data_;
			h_data_updated_ = other.h_data_updated_;
			d_data_updated_ = other.d_data_updated_;
		};
    ~TensorData() = default; ///< Default destructor

    inline bool operator==(const TensorData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const TensorData& other) const
    {
      return !(*this == other);
    }

    inline TensorData& operator=(const TensorData& other)
    { 
			h_data_ = other.h_data_;
			d_data_ = other.d_data_;
			h_data_updated_ = other.h_data_updated_;
			d_data_updated_ = other.d_data_updated_;
      dimensions_ = other.dimensions_;
      return *this;
    }

		virtual void setData(const Eigen::Tensor<TensorT, Dim>& data) = 0; ///< data setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, Dim>> getData() { std::shared_ptr<TensorT> h_data = h_data_; Eigen::TensorMap<Eigen::Tensor<TensorT, Dims>> data(h_data.get(), Args); return data; }; ///< data copy getter
		std::shared_ptr<TensorT> getHDataPointer() { return h_data_; }; ///< data pointer getter
		std::shared_ptr<TensorT> getDDataPointer() { return d_data_; }; ///< data pointer getter

		size_t getTensorSize() { return batch_size_ * memory_size_ * layer_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes


		virtual bool syncHAndDData(DeviceT& device) = 0;

		std::pair<bool, bool> getDataStatus() { return std::make_pair(h_data_updated_, d_data_updated_);	};

protected:
		int batch_size_ = 1; ///< Mini batch size
		int memory_size_ = 2; ///< Memory size
		int layer_size_ = 1; ///< Layer size

		std::shared_ptr<TensorT> h_data_ = nullptr;
		std::shared_ptr<TensorT> d_data_ = nullptr;

		bool h_data_updated_ = false;
		bool d_data_updated_ = false;

	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(batch_size_, memory_size_, layer_size_, 
	//		h_data_, h_output_, h_error_, h_derivative_, h_dt_,
	//		d_data_, d_output_, d_error_, d_derivative_, d_dt_,
	//		h_data_updated_, h_output_updated_, h_error_updated_, h_derivative_updated_, h_dt_updated_,
	//		d_data_updated_, d_output_updated_, d_error_updated_, d_derivative_updated_, d_dt_updated_);
	//	}
  };

	template<typename TensorT, int Dim, int... Args>
	class TensorDataCpu : public TensorData<TensorT, Eigen::DefaultDevice, int Dim, int... Args> {
	public:
		void setData(const Eigen::Tensor<TensorT, Dim>& data) {
			TensorT* h_data = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, Dim>> data_copy(h_data, Args);
			data_copy = data;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_data_.reset(h_data, h_deleter);
			this->h_data_.reset(h_data);
			this->h_data_updated_ = true;
			this->d_data_updated_ = true;
		}; ///< data setter
		bool syncHAndDData(Eigen::DefaultDevice& device) { return true; }
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice>>(this));
	//	}
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT, int Dim, int... Args>
	class TensorDataGpu : public TensorData<TensorT, Eigen::GpuDevice, int Dim, int... Args> {
	public:
		void setData(const Eigen::Tensor<TensorT, Dim>& data) {
			// allocate cuda and pinned host memory
			TensorT* d_data;
			TensorT* h_data;
			assert(cudaMalloc((void**)(&d_data), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_data), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, Dim>> data_copy(h_data, Args);
			data_copy = data;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_data_.reset(h_data, h_deleter); 
			this->d_data_.reset(d_data, d_deleter);
			this->h_data_updated_ = true;
			this->d_data_updated_ = false;
		}; ///< data setter
		bool syncHAndDData(Eigen::GpuDevice& device){
			if (this->h_data_updated_ && !this->d_data_updated_) {
				device.memcpyHostToDevice(this->d_data_.get(), this->h_data_.get(), getTensorSize());
				this->d_data_updated_ = true;
				this->h_data_updated_ = false;
				return true;
			}
			else if (!this->h_data_updated_ && this->d_data_updated_) {
				device.memcpyDeviceToHost(this->h_data_.get(), this->d_data_.get(), getTensorSize());
				this->h_data_updated_ = true;
				this->d_data_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<TensorData<TensorT, Eigen::GpuDevice>>(this));
	//	}
	};
#endif
}

//CEREAL_REGISTER_TYPE(TensorData::TensorDataCpu<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(TensorData::TensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //TENSORDATA_TENSORDATA_H