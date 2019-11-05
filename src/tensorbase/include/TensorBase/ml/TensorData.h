/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORDATA_H
#define TENSORBASE_TENSORDATA_H

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/Meta.h>
#include <memory>
#include <array>
#include <TensorBase/ml/TensorArray.h>

#include <cereal/access.hpp>  // serialiation of private members
#include <cereal/types/memory.hpp>
#include <cereal/types/array.hpp>
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{

  /**
    @brief Tensor data base class to handle the underlying memory and resource
      allocation of tensor data

    LIMITATIONS:
    - the memory management assumes a single GPU environment and does not allow for specifying which GPU to use
    - the GPU memory always uses pinned memory, and does not provide an option to use a different type of GPU memory
  */
  template<typename TensorT, typename DeviceT, int TDim>
  class TensorData
  {
  public:
    using tensorT = TensorT;
    TensorData() { device_name_ = typeid(DeviceT).name(); };
    TensorData(const Eigen::array<Eigen::Index, TDim>& dimensions) { 
      setDimensions(dimensions); 
      device_name_ = typeid(DeviceT).name(); };
    TensorData(const TensorData& other)
    {
      h_data_ = other.h_data_;
      d_data_ = other.d_data_;
      h_data_updated_ = other.h_data_updated_;
      d_data_updated_ = other.d_data_updated_;
      dimensions_ = other.dimensions_;
      tensor_size_ = other.tensor_size_;
      device_name_ = other.device_name_;
    };
    virtual ~TensorData() = default; ///< Default destructor

    template<typename TensorTOther, typename DeviceTOther, int TDimOther>
    inline bool operator==(const TensorData<TensorTOther, DeviceTOther, TDimOther>& other) const
    {
      // TODO: CUDA comiler error: use the "typename" keyword to treat nontype "TensorBase::TensorData<TensorT, DeviceT, TDim>::tensorT [with TensorT=TensorTOther, DeviceT=DeviceTOther, TDim=TDimOther]" as a type in a dependent context
      //if (!std::is_same<tensorT, TensorData<TensorTOther, DeviceTOther, TDimOther>::tensorT>::value)
      //  return false;
      return std::tie(
          dimensions_,
          device_name_          
        ) == std::tie(
          other.dimensions_,
          other.device_name_          
        );
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
      tensor_size_ = other.tensor_size_;
      device_name_ = other.device_name_;
      return *this;
    }

    // NOT NEEDED
    //template<typename T, typename D>
    //TensorData(const TensorData<T,D,TDim>& other) {
    //  h_data_ = std::reinterpret_pointer_cast<TensorT>(other.h_data_);
    //  d_data_ = std::reinterpret_pointer_cast<TensorT>(other.d_data_);
    //  h_data_updated_ = other.h_data_updated_;
    //  d_data_updated_ = other.d_data_updated_;
    //  dimensions_ = other.dimensions_;
    //  tensor_size_ = other.tensor_size_;
    //  device_name_ = typeid(DeviceT).name();
    //};

    // NOT NEEDED
    //template<typename T>
    //TensorData(const TensorData<T, DeviceT, TDim>& other) {
    //  h_data_ = std::reinterpret_pointer_cast<TensorT>(other.h_data_);
    //  d_data_ = std::reinterpret_pointer_cast<TensorT>(other.d_data_);
    //  h_data_updated_ = other.h_data_updated_;
    //  d_data_updated_ = other.d_data_updated_;
    //  dimensions_ = other.dimensions_;
    //  tensor_size_ = other.tensor_size_;
    //  device_name_ = typeid(DeviceT).name();
    //};

    virtual std::shared_ptr<TensorData> copy(DeviceT& device) = 0; ///< returns a copy of the TensorData

    virtual void select(std::shared_ptr<TensorData<TensorT, DeviceT, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< return a selection of the TensorData
    virtual void sortIndices(std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, const std::string& sort_order, DeviceT& device) = 0; ///< sort the indices based on the TensorData
    virtual void sort(const std::string& sort_order, DeviceT& device) = 0; ///< sort the TensorData in place
    virtual void sort(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0; ///< sort the TensorData in place

    /*
    @brief Partition the data based on a flag.
      The flagged indices are moved to the front (in order)
      and the non-flagged indices are moved to the back (in reverse order)
    */
    virtual void partition(const std::shared_ptr<TensorData<int, DeviceT, TDim>>& indices, DeviceT& device) = 0;

    /*
    @brief Run Length Encode the data

    @param[out] unique The unique value in the run (same size as the data)
    @param[out] count The counts of the value in the run (same size as the data)
    @param[out] n_runs The number of runs in the data (size of 1);
    */
    virtual void runLengthEncode(std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& unique, std::shared_ptr<TensorData<int, DeviceT, 1>>& count, std::shared_ptr<TensorData<int, DeviceT, 1>>& n_runs, DeviceT& device) = 0;

		/*
		@brief Bin the data for display as a histogram

		NOTE: Only available for primitive types

		@param[out] n_levels The number of bin levels where n_bins = n_levels -1
		@param[out] lower_level The lower sample value boundary of lowest bin
		@param[out] upper_level The upper sample value boundary of upper bin
		@param[out] histogram The bin counts;
		*/
		virtual void histogram(const int& n_levels, const TensorT& lower_level, const TensorT& upper_level, std::shared_ptr<TensorData<TensorT, DeviceT, 1>>& histogram, DeviceT& device) = 0;

    void setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions); ///< Set the tensor dimensions and calculate the tensor size
    Eigen::array<Eigen::Index, TDim> getDimensions() const; ///< Get the tensor dimensions
    size_t getTensorBytes() { return tensor_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    size_t getTensorSize() { return tensor_size_; }; ///< Get the size of the tensor
    int getDims() { return dimensions_.size(); };  ///< TDims getter
    std::string getDeviceName() { return device_name_; }; ///< Device name getter

    virtual void setData(const Eigen::Tensor<TensorT, TDim>& data) = 0; ///< data setter
    virtual void setData() = 0; ///< data setter

    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> getData() { Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data(h_data_.get(), this->getDimensions()); return data; } ///< data copy getter
    virtual std::shared_ptr<TensorT[]> getHDataPointer() = 0; ///< host data pointer getter
    virtual std::shared_ptr<TensorT[]> getDataPointer() = 0; ///< device data pointer getter
    
    virtual bool syncHAndDData(DeviceT& device) = 0;  ///< Sync the host and device data
    void setDataStatus(const bool& h_data_updated, const bool& d_data_updated) { h_data_updated_ = h_data_updated; d_data_updated_ = d_data_updated; } ///< Set the status of the host and device data
    std::pair<bool, bool> getDataStatus() { return std::make_pair(h_data_updated_, d_data_updated_); };   ///< Get the status of the host and device data

    /*
    @brief Intialize the data from a Tensor with type std::string

    @param[out] data_new Eigen::Tensor of type std::string
    */
    virtual void convertFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, DeviceT& device) = 0;

  protected:
    std::shared_ptr<TensorT[]> h_data_ = nullptr;  ///< Shared pointer implementation of the host tensor data
    std::shared_ptr<TensorT[]> d_data_ = nullptr;  ///< Shared pointer implementation of the device (GPU) tensor data

    bool h_data_updated_ = false;  ///< boolean indicator if the host data is up to date
    bool d_data_updated_ = false;  ///< boolean indicator if the device data is up to date
    // MULTI-GPU: more advanced syncronization will need to be implemented when transfering data between different GPUs    

    Eigen::array<Eigen::Index, TDim> dimensions_ = Eigen::array<Eigen::Index, TDim>(); ///< Tensor dimensions (initialized to all zeros)
    size_t tensor_size_ = 0;  ///< Tensor size
    std::string device_name_ = "";

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(dimensions_, tensor_size_, device_name_, h_data_updated_, d_data_updated_);
    }
  };

  template<typename TensorT, typename DeviceT, int TDim>
  inline void TensorData<TensorT, DeviceT, TDim>::setDimensions(const Eigen::array<Eigen::Index, TDim>& dimensions) {
    //dimensions_ = std::array<Eigen::Index, TDim>(); // works on gpu
    dimensions_ = dimensions; // works on cpu but not gpu
    size_t tensor_size = 1;
    for (int i = 0; i < TDim; ++i) {
      //dimensions_.at(i) = dimensions.at(i); // works on gpu
      tensor_size *= dimensions.at(i);
    }
    tensor_size_ = tensor_size;
  }

  template<typename TensorT, typename DeviceT, int TDim>
  inline Eigen::array<Eigen::Index, TDim> TensorData<TensorT, DeviceT, TDim>::getDimensions() const
  { 
    return dimensions_;
  }

  /**
    @brief Tensor data class specialization for Eigen::DefaultDevice (single thread CPU)
  */
  template<typename TensorT, int TDim>
  class TensorDataDefaultDevice : public TensorData<TensorT, Eigen::DefaultDevice, TDim> {
  public:
    using TensorData<TensorT, Eigen::DefaultDevice, TDim>::TensorData;
    ~TensorDataDefaultDevice() = default;
    std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>> copy(Eigen::DefaultDevice& device) override;
    void select(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device) override;
    void sortIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, const std::string& sort_order, Eigen::DefaultDevice& device) override;
    void sort(const std::string& sort_order, Eigen::DefaultDevice& device) override;
    void sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device) override;
    void partition(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device) override;
    void runLengthEncode(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& n_runs, Eigen::DefaultDevice& device) override;
		void histogram(const int& n_levels, const TensorT& lower_level, const TensorT& upper_level, std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& histogram, Eigen::DefaultDevice& device) override { histogram_(n_levels, lower_level, upper_level, histogram, device); }
		template<typename T = TensorT, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
		void histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::DefaultDevice, 1>>& histogram, Eigen::DefaultDevice& device);
		template<typename T = TensorT, std::enable_if_t<!std::is_fundamental<T>::value, int> = 0>
		void histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::DefaultDevice, 1>>& histogram, Eigen::DefaultDevice& device) { /*Do nothing*/ };
    std::shared_ptr<TensorT[]> getHDataPointer() override { return h_data_; }
    std::shared_ptr<TensorT[]> getDataPointer() override { return h_data_; }
    void setData(const Eigen::Tensor<TensorT, TDim>& data) override; ///< data setter
    void setData() override;
    bool syncHAndDData(Eigen::DefaultDevice& device) override { this->d_data_updated_ = true; this->h_data_updated_ = true; return true; }
    void convertFromStringToTensorT(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) override { convertFromStringToTensorT_(data_new, device); };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, int>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, float>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, double>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, char>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, bool>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray8<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray8<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/};
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray8<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray32<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray32<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray32<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray128<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray128<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray128<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray512<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray512<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray512<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray2048<char>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device);
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray2048<int>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
    template<typename T = TensorT, std::enable_if_t<std::is_same<T, TensorArray2048<float>>::value, int> = 0>
    void convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice& device) { /*TODO*/ };
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
    	archive(cereal::base_class<TensorData<TensorT, Eigen::DefaultDevice, TDim>>(this));
    }
  };

  template<typename TensorT, int TDim>
  std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>> TensorDataDefaultDevice<TensorT, TDim>::copy(Eigen::DefaultDevice& device) {
    // initialize the new data
    TensorDataDefaultDevice<TensorT, TDim> data_new(this->getDimensions());
    data_new.setData(this->getData());
    //// copy over the values
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_new_values(data_new.getDataPointer().get(), data_new.getDimensions());
    //const Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_values(this->getDataPointer().get(), this->getDimensions());
    //data_new_values.device(device) = data_values;
    return std::make_shared<TensorDataDefaultDevice<TensorT, TDim>>(data_new);
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::select(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, TDim>>& tensor_select, const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice & device)
  {
		// STL sort helper structure
		std::vector<std::pair<TensorT, int>> data_indices(this->getTensorSize());
		auto make_vec_pair = [](const TensorT& a, const int& b) { return std::make_pair(a, b); };
		std::transform(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), indices->getDataPointer().get(), data_indices.begin(), make_vec_pair);

		// STL remove_if helper methods
		auto isZero = [](const std::pair<TensorT, int>& a) {
			return a.second == 0;
		};

		// call remove_if on the flagged entries marked as false (i.e., 0)
		std::remove_if(data_indices.begin(), data_indices.end(), isZero);

		// extract out the ordered data
		data_indices.resize(tensor_select->getTensorSize());
		std::vector<TensorT> data_copy(tensor_select->getTensorSize());
		auto extract_vec_pair = [](const std::pair<TensorT, int>& a) { return a.first; };
		std::transform(data_indices.begin(), data_indices.end(), data_copy.begin(), extract_vec_pair);

		// Copy over the selected values
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_select_values(tensor_select->getDataPointer().get(), (int)tensor_select->getTensorSize());
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_copy_values(data_copy.data(), (int)tensor_select->getTensorSize());
		tensor_select_values.device(device) = data_copy_values;
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sortIndices(std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, const std::string& sort_order, Eigen::DefaultDevice & device)
  {
		// STL sort helper structure
		std::vector<std::pair<TensorT, int>> data_indices(this->getTensorSize());
		auto make_vec_pair = [](const TensorT& a, const int& b) { return std::make_pair(a, b); };
		std::transform(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), indices->getDataPointer().get(), data_indices.begin(), make_vec_pair);

		// STL sort helper methods
		auto sortAsc = [](const std::pair<TensorT, int>& a, const std::pair<TensorT, int>& b) {
			return a.first < b.first;
		};
		auto sortDesc = [](const std::pair<TensorT, int>& a, const std::pair<TensorT, int>& b) {
			return a.first > b.first;
		};

		// sort by key
		if (sort_order == "ASC") {
			std::sort(data_indices.begin(), data_indices.end(), sortAsc);
		}
		else if (sort_order == "DESC") {
			std::sort(data_indices.begin(), data_indices.end(), sortDesc);
		}

		// extract out the ordered indices
		std::vector<int> sorted_indices(this->getTensorSize());
		auto extract_vec_pair = [](const std::pair<TensorT, int>& a) { return a.second; };
		std::transform(data_indices.begin(), data_indices.end(), sorted_indices.begin(), extract_vec_pair);

		// copy over the values
		Eigen::TensorMap<Eigen::Tensor<int, 1>> indices_values(indices->getDataPointer().get(), this->getTensorSize());
		Eigen::TensorMap<Eigen::Tensor<int, 1>> sorted_values(sorted_indices.data(), this->getTensorSize());
		indices_values.device(device) = sorted_values;
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sort(const std::string & sort_order, Eigen::DefaultDevice & device)
  {
		// STL sort helper methods
		auto sortAsc = [](const TensorT& a, const TensorT& b) {
			return a < b;
		};
		auto sortDesc = [](const TensorT& a, const TensorT& b) {
			return a > b;
		};

		// sort the data in place
		if (sort_order == "ASC") {
			std::stable_sort(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), sortAsc);
		}
		else if (sort_order == "DESC") {
			std::stable_sort(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), sortDesc);
		}
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::sort(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice& device)
  {
		// STL sort helper structure
		std::vector<std::pair<TensorT, int>> data_indices(this->getTensorSize());
		auto make_vec_pair = [](const TensorT& a, const int& b) { return std::make_pair(a, b); };
		std::transform(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), indices->getDataPointer().get(), data_indices.begin(), make_vec_pair);

		// STL sort helper methods
		auto sortByIndex = [](const std::pair<TensorT, int>& a, const std::pair<TensorT, int>& b) {
			return a.second < b.second;
		};

		// sort by key
		std::sort(data_indices.begin(), data_indices.end(), sortByIndex);

		// extract out the ordered data
		std::vector<TensorT> sorted_data(this->getTensorSize());
		auto extract_vec_pair = [](const std::pair<TensorT, int>& a) { return a.first; };
		std::transform(data_indices.begin(), data_indices.end(), sorted_data.begin(), extract_vec_pair);

		// copy over the values
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_values(this->getDataPointer().get(), this->getTensorSize());
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> sorted_values(sorted_data.data(), this->getTensorSize());
		data_values.device(device) = sorted_values;
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::partition(const std::shared_ptr<TensorData<int, Eigen::DefaultDevice, TDim>>& indices, Eigen::DefaultDevice & device)
  {
		// STL sort helper structure
		std::vector<std::pair<TensorT, int>> data_indices(this->getTensorSize());
		auto make_vec_pair = [](const TensorT& a, const int& b) { return std::make_pair(a, b); };
		std::transform(this->getDataPointer().get(), this->getDataPointer().get() + this->getTensorSize(), indices->getDataPointer().get(), data_indices.begin(), make_vec_pair);

		// STL remove_if helper methods
		auto isGreaterThanZero = [](const std::pair<TensorT, int>& a) {
			return a.second > 0;
		};

		// call partition on the flagged entries marked as true (i.e., 1)
		std::stable_partition(data_indices.begin(), data_indices.end(), isGreaterThanZero);

		// extract out the ordered data
		std::vector<TensorT> sorted_data(this->getTensorSize());
		auto extract_vec_pair = [](const std::pair<TensorT, int>& a) { return a.first; };
		std::transform(data_indices.begin(), data_indices.end(), sorted_data.begin(), extract_vec_pair);

		// copy over the values
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> data_values(this->getDataPointer().get(), this->getTensorSize());
		Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> sorted_values(sorted_data.data(), this->getTensorSize());
		data_values.device(device) = sorted_values;
  }
  template<typename TensorT, int TDim>
  inline void TensorDataDefaultDevice<TensorT, TDim>::runLengthEncode(std::shared_ptr<TensorData<TensorT, Eigen::DefaultDevice, 1>>& unique, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& count, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 1>>& n_runs, Eigen::DefaultDevice & device)
  {
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> tensor_values(this->getDataPointer().get(), (int)this->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> unique_values(unique->getDataPointer().get(), (int)unique->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 1>> count_values(count->getDataPointer().get(), (int)count->getTensorSize());
    Eigen::TensorMap<Eigen::Tensor<int, 0>> n_runs_values(n_runs->getDataPointer().get());

    // generate the run length encoding
    int iter = 0;
    int run_index = 0;
    n_runs_values(0) = 0; // initialize the value of the number of runs
    count_values.setZero();
    std::for_each(tensor_values.data(), tensor_values.data() + tensor_values.size() - 1,
      [&tensor_values, &unique_values, &count_values, &n_runs_values, &iter, &run_index, &device](const TensorT& value) {
      if (value == tensor_values(iter + 1) && iter + 1 == tensor_values.size() - 1) { // run, last value
        count_values(run_index) += 2;
        unique_values(run_index) = value;
        n_runs_values(0) += 1;
      }
      else if (value == tensor_values(iter + 1)) { // run
        count_values(run_index) += 1;
      }
      else if (value != tensor_values(iter + 1) && iter + 1 == tensor_values.size() - 1) { // not a run, last value
        count_values(run_index) += 1;
        unique_values(run_index) = value;
        count_values(run_index + 1) += 1;
        unique_values(run_index + 1) = tensor_values(iter + 1);
        n_runs_values(0) += 2;
      }
      else { // not a run
        count_values(run_index) += 1;
        unique_values(run_index) = value;
        run_index += 1;
        n_runs_values(0) += 1;
      }
      ++iter;
    });
  }
	template<typename TensorT, int TDim>
	template<typename T, std::enable_if_t<std::is_fundamental<T>::value, int>>
	inline void TensorDataDefaultDevice<TensorT, TDim>::histogram_(const int& n_levels, const T& lower_level, const T& upper_level, std::shared_ptr<TensorData<T, Eigen::DefaultDevice, 1>>& histogram, Eigen::DefaultDevice& device)
	{
		// Copy the data
		auto data_copy = this->copy(device);
		data_copy->syncHAndDData(device);

		// sort data to bring equal elements together
		std::sort(data_copy->getDataPointer().get(), data_copy->getDataPointer().get() + data_copy->getTensorSize());

		// histogram bins and widths
		const int n_bins = n_levels - 1;
		const T bin_width = (upper_level - lower_level) / (n_levels - T(1));
		std::vector<T> bin_search(n_bins);
		std::generate(bin_search.begin(), bin_search.end(), [n = lower_level, &bin_width]() mutable { return n+=bin_width; });

		// find the end of each bin of values
		auto u_bound = [&data_copy](const T& v) {
			return *std::upper_bound(data_copy->getDataPointer().get(), data_copy->getDataPointer().get() + data_copy->getTensorSize(), v);
		};
		std::transform(bin_search.begin(), bin_search.end(), histogram->getDataPointer().get(), u_bound);

		// compute the histogram by taking differences of the cumulative histogram
		std::adjacent_difference(histogram->getDataPointer().get(), histogram->getDataPointer().get() + histogram->getTensorSize(),
			histogram->getDataPointer().get());
	}
  template<typename TensorT, int TDim>
  void TensorDataDefaultDevice<TensorT, TDim>::setData(const Eigen::Tensor<TensorT, TDim>& data) {
    TensorT* h_data = new TensorT[this->tensor_size_];
    // copy the tensor
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_copy(h_data, this->getDimensions());
    data_copy = data;
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
  };
  template<typename TensorT, int TDim>
  void TensorDataDefaultDevice<TensorT, TDim>::setData() {
    TensorT* h_data = new TensorT[this->tensor_size_];
    this->h_data_.reset(h_data);
    this->h_data_updated_ = true;
    this->d_data_updated_ = true;
  };
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, int>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->getDataPointer().get(), this->dimensions_);
    data_converted.device(device) = data_new.unaryExpr([](const std::string& elem) { return std::stoi(elem); }).reshape(this->dimensions_);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, float>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->getDataPointer().get(), this->dimensions_);
    data_converted.device(device) = data_new.unaryExpr([](const std::string& elem) { return std::stof(elem); }).reshape(this->dimensions_);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, double>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->getDataPointer().get(), this->dimensions_);
    data_converted.device(device) = data_new.unaryExpr([](const std::string& elem) { return std::stod(elem); }).reshape(this->dimensions_);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, char>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->getDataPointer().get(), this->dimensions_);
    data_converted.device(device) = data_new.unaryExpr([](const std::string& elem) { return elem.c_str()[0]; }).reshape(this->dimensions_);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, bool>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(this->getDataPointer().get(), this->dimensions_);
    data_converted.device(device) = data_new.unaryExpr([](const std::string& elem) { return elem == "1"; }).reshape(this->dimensions_);
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, TensorArray8<char>>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    std::transform(data_new.data(), data_new.data() + data_new.size(), getDataPointer().get(), 
      [](const std::string& elem) -> TensorArray8<char> { return TensorArray8<char>(elem); });

    // NOTE: unaryExpr operator does not appear to work with TensorArray8<char>!
    //Eigen::TensorMap<Eigen::Tensor<TensorT, TDim>> data_converted(getDataPointer().get(), dimensions_);
    //data_converted.device(device) = data_converted.unaryExpr([](const std::string& elem) { return TensorArray8<char>(elem); });
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, TensorArray32<char>>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    std::transform(data_new.data(), data_new.data() + data_new.size(), getDataPointer().get(),
      [](const std::string& elem) -> TensorArray32<char> { return TensorArray32<char>(elem); });
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, TensorArray128<char>>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    std::transform(data_new.data(), data_new.data() + data_new.size(), getDataPointer().get(),
      [](const std::string& elem) -> TensorArray128<char> { return TensorArray128<char>(elem); });
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, TensorArray512<char>>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    std::transform(data_new.data(), data_new.data() + data_new.size(), getDataPointer().get(),
      [](const std::string& elem) -> TensorArray512<char> { return TensorArray512<char>(elem); });
  }
  template<typename TensorT, int TDim>
  template<typename T, std::enable_if_t<std::is_same<T, TensorArray2048<char>>::value, int>>
  inline void TensorDataDefaultDevice<TensorT, TDim>::convertFromStringToTensorT_(const Eigen::Tensor<std::string, TDim>& data_new, Eigen::DefaultDevice & device)
  {
    assert(data_new.size() == this->getTensorSize());
    // convert the data from string to TensorT
    std::transform(data_new.data(), data_new.data() + data_new.size(), getDataPointer().get(),
      [](const std::string& elem) -> TensorArray2048<char> { return TensorArray2048<char>(elem); });
  }
}

// Cereal registration of TensorTs: float, int, char, double and TDims: 1, 2, 3, 4
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray8<char>, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray32<char>, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray128<char>, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray512<char>, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray2048<char>, 1>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray8<char>, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray32<char>, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray128<char>, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray512<char>, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray2048<char>, 2>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray8<char>, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray32<char>, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray128<char>, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray512<char>, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray2048<char>, 3>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<int, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<float, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<double, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<char, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray8<char>, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray32<char>, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray128<char>, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray512<char>, 4>);
CEREAL_REGISTER_TYPE(TensorBase::TensorDataDefaultDevice<TensorBase::TensorArray2048<char>, 4>);

#endif //TENSORBASE_TENSORDATA_H