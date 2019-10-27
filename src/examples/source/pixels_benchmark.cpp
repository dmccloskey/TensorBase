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
template<typename LabelsT, typename TensorT, typename DeviceT, int NDim>
class PixelManager {
public:
	PixelManager(const int& data_size) : data_size_(data_size){};
	~PixelManager() = default;
	virtual void setDimSizes() = 0;
	virtual void getInsertData(const int& offset, const int& span, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;
	virtual void makeLabelsPtr(const Eigen::Tensor<int, 2>& labels, std::shared_ptr<TensorData<LabelsT, DeviceT, 2>>& labels_ptr) = 0;
	virtual void makeValuesPtr(const Eigen::Tensor<float, NDim>& values, std::shared_ptr<TensorData<TensorT, DeviceT, NDim>>& values_ptr) = 0;
protected:
	int data_size_;
};

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
	for (int i=offset; i<offset + span; ++i){
		labels(0, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
		labels(1, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 1)))) % this->dim_span_ + 1;
		labels(2, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 2)))) % this->dim_span_ + 1;
		labels(3, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 3)))) % this->dim_span_ + 1;
		values(0, i - offset) = TensorT(i);
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
	Eigen::Tensor<TensorT, 2> values(1, this->xyz_dim_size_ * span);
	for (int i = offset; i < offset + span; ++i) {
		labels(0, i - offset) = int(floor(float(i) / float(std::pow(this->dim_span_, 0)))) % this->dim_span_ + 1;
		Eigen::Tensor<TensorT, 1> new_values(this->xyz_dim_size_);
		new_values.setConstant(offset);
		new_values = new_values.cumsum(0);
		values.slice(Eigen::array<Eigen::Index, 2>({0, i - offset}), Eigen::array<Eigen::Index, 2>({ 1, this->xyz_dim_size_ })) = new_values.reshape(Eigen::array<Eigen::Index, 2>({ 1, this->xyz_dim_size_ }));
	}
	this->makeLabelsPtr(labels, labels_ptr);
	this->makeValuesPtr(values, values_ptr);
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
@brief A class for running 1 line insertion, deletion, and update benchmarks
*/
template<typename LabelsT, typename TensorT, typename DeviceT>
class Benchmark1TimePoint {
public:
	Benchmark1TimePoint() = delete;
	~Benchmark1TimePoint() = delete;
	/*
	@brief insert 1 time-point at a time

	@param[in] n_dims
	@param[in, out] transaction_manager
	@param[in] data_size
	@param[in] device

	@returns A string with the total time of the benchmark in milliseconds
	*/
	static std::string insert1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device);
private:
	static void insert1TimePoint1D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device);
	static void insert1TimePoint2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device);
};
template<typename LabelsT, typename TensorT, typename DeviceT>
std::string Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint(const int& n_dims, TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device)
{
	// Start the timer
	auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

	if (n_dims == 0) {
		// TODO...
	}
	else if (n_dims == 1) {
		insert1TimePoint1D(transaction_manager, data_size, device);
	}
	else if (n_dims == 2) {
		// TODO...
	}
	else if (n_dims == 3) {
		// TODO...
	}
	else if (n_dims == 4) {
		// TODO...
	}
	else {
		std::cout << "The given number of dimensions "<< n_dims << " is not within the range of 0 to 4." << std::endl;
	}

	// Stop the timer
	auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	std::string milli_time = std::to_string(stop - start);
	return milli_time;
}
template<typename LabelsT, typename TensorT, typename DeviceT>
void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint1D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device)
{
	PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size);
	std::shared_ptr<TensorData<LabelsT, DeviceT, 2>> labels_ptr;
	std::shared_ptr<TensorData<TensorT, DeviceT, 2>> values_ptr;
	//int span = data_size / std::pow(data_size, 0.25);  // BUG: breaks auto max_bcast = indices_view_values.maximum(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 1>({ n_labels })); in TensorTableDefaultDevice<TensorT, TDim>::makeAppendIndices
	int span = 4;
	for (int i = 0; i < data_size; i += span) {
		labels_ptr.reset();
		values_ptr.reset();
		pixel_manager.getInsertData(i, span, labels_ptr, values_ptr);
		TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2> appendToAxis("TTable", "xyzt", labels_ptr, values_ptr);
		std::shared_ptr<TensorOperation<DeviceT>> appendToAxis_ptr = std::make_shared<TensorAppendToAxis<LabelsT, TensorT, DeviceT, 2>>(appendToAxis);
		transaction_manager.executeOperation(appendToAxis_ptr, device);
	}
}
template<typename LabelsT, typename TensorT, typename DeviceT>
void Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint2D(TransactionManager<DeviceT>& transaction_manager, const int& data_size, DeviceT& device)
{
	PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size);
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
	}
}

/*
@brief Simulate a typical database table where one axis will be the headers (x, y, z, and t)
	and the other axis will be the index starting from 1
*/
template<typename LabelsT, typename TensorT, typename DeviceT>
class TensorCollectionGenerator{
public:
	TensorCollectionGenerator() = default;
	~TensorCollectionGenerator() = default;
	std::shared_ptr<TensorCollection<DeviceT>> makeTensorCollection(const int& n_dims, const int& data_size, const int& shard_span_perc, const bool& is_columnar) const;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const = 0;
	virtual std::shared_ptr<TensorCollection<DeviceT>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const = 0;
};
template<typename LabelsT, typename TensorT, typename DeviceT>
std::shared_ptr<TensorCollection<DeviceT>> TensorCollectionGenerator<LabelsT, TensorT, DeviceT>::makeTensorCollection(const int& n_dims, const int& data_size, const int& shard_span_perc, const bool& is_columnar) const
{
	if (n_dims == 0) {
		std::map<std::string, int> shard_span;
		return make0DTensorCollection(data_size, shard_span, is_columnar);
	}
	else if (n_dims == 1) {
		std::map<std::string, int> shard_span;
		if (data_size == 1296) {
			shard_span.emplace("xyzt", 1296);
			shard_span.emplace("values", 1);
		}
		else if (data_size == 1048576) {
			shard_span.emplace("xyzt", 1048576);
			shard_span.emplace("values", 1);
		}
		else if (data_size == 1003875856) {
			shard_span.emplace("xyzt", 1003875856);
			shard_span.emplace("values", 1);
		}
		else if (data_size == 1e12) {
			shard_span.emplace("xyzt", 1e12);
			shard_span.emplace("values", 1);
		}
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

template<typename LabelsT, typename TensorT>
class TensorCollectionGeneratorDefaultDevice: public TensorCollectionGenerator<LabelsT, TensorT, Eigen::DefaultDevice>{
public:
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const;
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const;
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const;
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const;
	std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const;
};
template<typename LabelsT, typename TensorT>
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make0DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
{
	return std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>();
}
template<typename LabelsT, typename TensorT>
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
{
	// Setup the axes
	Eigen::Tensor<std::string, 1> dimensions_1(4), dimensions_2(1);
	dimensions_1.setValues({ "x","y","z","t" });
	dimensions_2.setValues({ "values" });
	Eigen::Tensor<TensorArray8<char>, 2> labels_v(1, 1);
	labels_v.setValues({ { TensorArray8<char>("values")} });

	// Setup the tables
	std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
	auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyzt", 4, 0));
	table_1_axis_1_ptr->setDimensions(dimensions_1);
	auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<TensorArray8<char>>>(TensorAxisDefaultDevice<TensorArray8<char>>("values", dimensions_2, labels_v));
	table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
	table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
	table_1_ptr->setAxes();

	// Setup the table data
	table_1_ptr->setData();
	table_1_ptr->setShardSpans(shard_span);

	// Setup the collection
	auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
	collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
	return collection_1_ptr;
}
template<typename LabelsT, typename TensorT>
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make2DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
{
	return std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>();
}
template<typename LabelsT, typename TensorT>
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
{
	return std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>();
}
template<typename LabelsT, typename TensorT>
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
{
	return std::shared_ptr<TensorCollection<Eigen::DefaultDevice>>();
}

template<typename LabelsT, typename TensorT, typename DeviceT>
static void run_pixel_benchmark(const std::string& data_dir, const int& n_dims, const int& data_size, const bool& in_memory, const int& shard_span_perc, 
	TransactionManager<DeviceT>& transaction_manager,
	const TensorCollectionGenerator<LabelsT, TensorT, DeviceT>& tensor_collection_generator,	DeviceT& device) {
	std::cout << "Starting insert/delete/update pixel benchmarks for n_dims=" << n_dims << ", data_size=" << data_size << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;
	
	// Make the control 2D* tables and the nD TensorTables
	std::shared_ptr<TensorCollection<DeviceT>> col_collection_ptr, row_2D_collection;
	std::shared_ptr<TensorCollection<DeviceT>> n_dim_tensor_collection = tensor_collection_generator.makeTensorCollection(n_dims, data_size, shard_span_perc, true);

	// Run each table through the pixel by pixel benchmarks
	transaction_manager.setTensorCollection(n_dim_tensor_collection);
	std::cout << n_dims << " insertion pixel by pixel took " << Benchmark1TimePoint<LabelsT, TensorT, DeviceT>::insert1TimePoint(n_dims, transaction_manager, data_size, device) << " milliseconds." << std::endl;

	// Run each table through the 20% pixels benchmarks

	// Run each table through the time-point by time-point benchmarks
}

/* Benchmark for toy 4D pixels data where x, y, and z describe the coordinates of the pixel in 3D space (type=int),
	t describes the time of the pixel (type=int), and the value of the pixel (from 0 to 255) describes the intensity of the pixel

Example usage:
	pixels_benchmark [data_dir] [n_dims] [data_size] [in_memory] [shard_size_perc] 
	pixels_benchmark C:/Users/dmccloskey/Documents/GitHub/mnist/ 1 1296 true 1000

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
	int n_dims = 1;
	int data_size = 1296;
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

	// Setup the transaction manager
	TransactionManager<Eigen::DefaultDevice> transaction_manager;
	transaction_manager.setMaxOperations(data_size + 1);

	// Setup the TensorCollectionGenerator
	TensorCollectionGeneratorDefaultDevice<int, float> tensor_collection_generator;

	// Setup the device
	Eigen::DefaultDevice device;

	// run the application
	run_pixel_benchmark(data_dir, n_dims, data_size, in_memory, shard_span_perc, transaction_manager, tensor_collection_generator, device);

	return 0;
}
