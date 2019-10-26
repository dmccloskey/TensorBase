/**TODO:  Add copyright*/

#include <ctime> // time format
#include <chrono> // current time

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorSelect.h>
#include <math.h>

using namespace TensorBase;

/*
@brief Class for managing the generation of random pixels in a 4D (3D + time) space
*/
template<typename DeviceT, int NDim>
class PixelManager {
public:
	PixelManager(const int& data_size) : data_size_(data_size){};
	~PixelManager() = default;
	virtual void setDimSizes() = 0;
	virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<float, DeviceT, NDim>>& values_ptr) = 0;
	virtual void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_ptr) = 0;
	virtual void makeValuesPtr(const Eigen::Tensor<float, NDim>& values, std::shared_ptr<TensorData<float, DeviceT, NDim>>& values_ptr) = 0;
protected:
	int data_size_;
};

template<typename DeviceT>
class PixelManager1D : public PixelManager<DeviceT, 2> {
public:
	using PixelManager::PixelManager;
	void setDimSizes();
	void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<float, DeviceT, 2>>& values_ptr);
private:
	int xyzt_dim_size_;
	int dim_span_;
};
template<typename DeviceT>
void PixelManager1D<DeviceT>::setDimSizes() {
	xyzt_dim_size_ = this->data_size_;
	dim_span_ = std::pow(this->data_size_, 0.25);
}
template<typename DeviceT>
void PixelManager1D<DeviceT>::getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<int, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<float, DeviceT, 2>>& values_ptr) {
	setDimSizes();
	// Make the labels and values
	Eigen::Tensor<int, 2> labels(4, span);
	Eigen::Tensor<float, 2> values(1, span);
	for (int i=offset; i<offset + span; ++i){
		labels(0, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
		labels(1, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
		labels(2, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
		labels(3, i) = int(floor(float(i) / float(std::pow(this->dim_span_, 3)))) % this->dim_span_ + 1;
		values(0, i) = float(i);
	}
	this->makeLabelsPtr(labels, labels_ptr);
	this->makeValuesPtr(values, values_ptr);
}

class PixelManager1DDefaultDevice : public PixelManager1D<Eigen::DefaultDevice> {
public:
	using PixelManager1D::PixelManager1D;
	void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr);
	void makeValuesPtr(const Eigen::Tensor<float, 2>& values, std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>>& values_ptr);
};
void PixelManager1DDefaultDevice::makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<int, Eigen::DefaultDevice, 2>>& labels_ptr) {
	TensorDataDefaultDevice<int, 2> labels_data(Eigen::array<Eigen::Index, 2>({ labels.dimension(0), labels.dimension(1) }));
	labels_data.setData(labels);
	labels_ptr = std::make_shared<TensorDataDefaultDevice<int, 2>>(labels_data);
}
void PixelManager1DDefaultDevice::makeValuesPtr(const Eigen::Tensor<float, 2>& values, std::shared_ptr<TensorData<float, Eigen::DefaultDevice, 2>>& values_ptr) {
	TensorDataDefaultDevice<float, 2> values_data(Eigen::array<Eigen::Index, 2>({ values.dimension(0), values.dimension(1) }));
	values_data.setData(values);
	values_ptr = std::make_shared<TensorDataDefaultDevice<float, 2>>(values_data);
}

/*
@brief insert 1 pixel at a time

@param[in, out] transaction_manager
@param[in] device

@returns A string giving the total time of the benchmark
*/
template<typename DeviceT>
std::string insert_1_test(TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device) {
	// Start the timer
	auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

	// Insert 1 pixel at a time
	PixelManager1DDefaultDevice pixel_manager(data_size);
	for (int i = 0; i < data_size; ++i) {
		std::shared_ptr<TensorData<int, DeviceT, 2>> labels_ptr;
		std::shared_ptr<TensorData<float, DeviceT, 2>> values_ptr;
		pixel_manager.getInsertData(i, 1, labels_ptr, values_ptr);
		TensorAppendToAxis<int, float, DeviceT, 2> appendToAxis("xyzt", "nDTensorTable", labels_ptr, values_ptr);
		std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<int, float, DeviceT, 2>>(appendToAxis);
		transaction_manager.executeOperation(appendToAxis_ptr, device);
	}

	// Stop the timer
	auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	std::string milli_time = std::to_string(stop - start);
	return milli_time;
};

/*
@brief delete 1 pixel at a time
*/
void delete_1_test() {};

/*
@brief update 1 pixel at a time
*/
void update_1_test() {};

/*
@brief insert 20% of pixels at a time
*/
void insert_20Perc_test() {};

/*
@brief delete 20% of pixels at a time
*/
void delete_20Perc_test() {};

/*
@brief update 20% of pixels at a time
*/
void update_20Perc_test() {};

/*
@brief insert 1 time-point at a time (streaming)
*/
void insert_TP_test() {};

/*
@brief delete 1 time-point at a time (streaming)
*/
void delete_TP_test() {};

/*
@brief update 1 time-point at a time (streaming)
*/
void update_TP_test() {};

/*
@brief Simulate a typical database table where one axis will be the headers (x, y, z, and t)
	and the other axis will be the index starting from 1
*/
template<typename DeviceT>
class TensorCollectionGenerator{
public:
	TensorCollectionGenerator() = default;
	~TensorCollectionGenerator() = default;
	std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& n_dims, const int& data_size, const int& shard_span_perc, const bool& is_columnar);
	virtual std::shared_ptr<TensorCollection<DeviceT>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) = 0;
};
template<typename DeviceT>
std::shared_ptr<TensorCollection<DeviceT>> TensorCollectionGenerator<DeviceT>::makeTensorCollection(const int& n_dims, const int& data_size, const int& shard_span_perc, const bool& is_columnar)
{
	if (n_dims == 0) {
		std::map<std::string, int> shard_span;
		return make0DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 1) {
		std::map<std::string, int> shard_span;
		return make1DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 1) {
		std::map<std::string, int> shard_span;
		return make1DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 2) {
		std::map<std::string, int> shard_span;
		return make2DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 3) {
		std::map<std::string, int> shard_span;
		return make3DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 4) {
		std::map<std::string, int> shard_span;
		return make4DTensorCollection(data_size, shard_span, is_columnar);
	}
	else {
		return std::shared_ptr<TensorCollection<DeviceT>>();
	}
}

class TensorCollectionGeneratorDefaultDevice: public TensorCollectionGenerator<Eigen::DefaultDevice>{
public:
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar);
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar);
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar);
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar);
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar);
};
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice::make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar)
{
	// Setup the axes
	Eigen::Tensor<std::string, 1> dimensions_1(4), dimensions_2(1);
	dimensions_1.setValues({ "x","y","z","t" });
	dimensions_2.setValues({ "values" });
	Eigen::Tensor<TensorArray8<char>, 2> labels_v(1, 1);
	labels_v.setValues({ { TensorArray8<char>("values")} });

	// Setup the tables
	std::shared_ptr<TensorTable<float, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<float, 2>>(TensorTableDefaultDevice<float, 2>("TTable"));
	auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("xyzt", 4, 1));
	table_1_axis_1_ptr->setDimensions(dimensions_1);
	auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<int>>(TensorAxisDefaultDevice<int>("values", dimensions_2, labels_v));
	table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
	table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
	table_1_ptr->setAxes();

	// Setup the table data
	table_1_ptr->setData();
	table_1_ptr->setShardSpans(shard_span);

	// Setup the collection
	auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
	collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
}

void run_pixel_benchmark(const std::string& data_dir, const int& n_dims, const int& data_size, const bool& in_memory, const int& shard_span_perc) {
	std::cout << "Starting insert/delete/update pixel benchmarks for n_dims=" << n_dims << ", data_size=" << data_size << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;
	// Make the control 2D* tables and the nD TensorTables
	TensorCollectionDefaultDevice col_2D_collection, row_2D_collection, n_dim_tensor_collection;
	// TODO...

	// Setup the transaction manager
	TransactionManager<Eigen::DefaultDevice> transaction_manager;
	transaction_manager.setMaxOperations(data_size + 1);

	// Setup the device
	Eigen::DefaultDevice device;

	// Run each table through the pixel by pixel benchmarks
	std::cout << "col_2D insertion pixel by pixel took " << insert_1_test(transaction_manager, data_size, device) << " milliseconds." << std::endl;

	// Run each table through the 20% pixels benchmarks

	// Run each table through the time-point by time-point benchmarks
}

/* Benchmark for toy 4D pixels data where x, y, and z describe the coordinates of the pixel in 3D space (type=int),
	t describes the time of the pixel (type=int), and the value of the pixel (from 0 to 255) describes the intensity of the pixel

Example usage:
	pixels_benchmark [data_dir] [n_dims] [data_size] [in_memory] [shard_size_perc] 

@param[in] n_dims The number of dimensions (i.e., 1-4) with default of 4
	1 dimension: x, y, z, and t on a single axis with a "values" dimensions on the other axis
	2 dimensions: x, y, z on a single axis, and t on another axis
	3 dimensions: y, z on a single axis, x on an axis, and t on an axis
	4 dimensions: x, y, z, and t on seperate axes
@param[in] data_size Options include small, medium, large, and XL (i.e., 1296, 1048576, 1003875856, and 1e12 pixels, respectively) with default of small
	where x, y, z, and t span 1 to 6, 32, 178, and 1000, respectively
@param[in] in_memory Simulate all data loaded into memory (true) or JIT load into memory from disk (false) with default of true
@param[in] shard_size_perc Different shard span configurations.  Options include 5, 20, and 100 with a default of 100
*/
int main(int argc, char** argv)
{
	// Parse the user commands
	std::string data_dir = "C:/Users/dmccloskey/Documents/GitHub/mnist/";
	int n_dims = 4;
	int data_size = 1e3;
	bool in_memory = true;
	int shard_span_perc = 100;
	if (argc >= 2) {
		data_dir = argv[1];
	}
	if (argc >= 3) {
		try {
			n_dims = (std::stoi(argv[2]) > 0 && std::stoi(argv[2]) <= 4) ? std::stoi(argv[2]) : 4;
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	}
	if (argc >= 4) {
		if (argv[3] == std::string("small")) {
			data_size = 1296;
		}
		if (argv[3] == std::string("medium")) {
			data_size = 1048576;
		}
		if (argv[3] == std::string("large")) {
			data_size = 1003875856;
		}
		if (argv[3] == std::string("XL")) {
			data_size = 1e12;
		}
	}
	if (argc >= 5) {
		in_memory = (argv[4] == std::string("true")) ? true : false;
	}
	if (argc >= 6) {
		try {
			if (std::stoi(argv[5]) == 5) shard_span_perc = 5;
			else if (std::stoi(argv[5]) == 20) shard_span_perc = 20;
			else if (std::stoi(argv[5]) == 100) shard_span_perc = 100;
		}
		catch (std::exception & e) {
			std::cout << e.what() << std::endl;
		}
	}

	// run the application
	run_pixel_benchmark(data_dir, n_dims, data_size, in_memory, shard_span_perc);

	return 0;
}
