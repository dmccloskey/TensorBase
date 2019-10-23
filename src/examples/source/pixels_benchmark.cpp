/**TODO:  Add copyright*/

#include <ctime> // time format
#include <chrono> // current time

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include <TensorBase/ml/TransactionManager.h>
#include <TensorBase/ml/TensorCollection.h>
#include <TensorBase/ml/TensorSelect.h>

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
	for (int i = 0; i < data_size; ++i) {
		//getPixels(data_size, i);

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
TensorCollectionDefaultDevice make_2DColRowTable(const int& data_size, const int& shard_size, const bool& is_columnar) {};
TensorCollectionDefaultDevice make_nDTensorTable(const int& n_dims, const int& data_size, const int& shard_size) {};

void run_pixel_benchmark(const std::string& data_dir, const int& n_dims, const int& data_size, const bool& in_memory, const int& shard_span_perc) {
	std::cout << "Starting insert/delete/update pixel benchmarks for n_dims=" << n_dims << ", data_size=" << data_size << ", in_memory=" << in_memory << ", and shard_span_perc=" << shard_span_perc << std::endl;
	// Make the control 2D* tables and the nD TensorTables
	TensorCollectionDefaultDevice col_2D_collection, row_2D_collection, n_dim_tensor_collection;
	// TODO...

	// Setup the transaction manager
	TransactionManager<Eigen::DefaultDevice> transaction_manager;
	transactionManager.setMaxOperations(data_size + 1);

	// Setup the device
	Eigen::DefaultDevice device;

	// Run each table through the pixel by pixel benchmarks
	std::cout << "col_2D insertion pixel by pixel took " << insert_1_test(transaction_manager, device) << " milliseconds." << std::endl;

	// Run each table through the 20% pixels benchmarks

	// Run each table through the time-point by time-point benchmarks
}

/* Benchmark for toy 4D pixels data

Example usage:
	pixels_benchmark [data_dir] [n_dims] [data_size] [in_memory] [shard_size_perc] 

Command line options
@param[in] n_dims The number of dimensions (i.e., 1-4) with default of 4
@param[in] data_size Options include small, medium, large, and XL (i.e., 1296, 1048576, 1003875856, and 1e12 pixels, respectively) with default of small
@param[in] in_memory Simulate all data loaded into memory (true) or JIT load into memory from disk (false) with default of true
@param[in] shard_size_perc Different shard span configurations.  Options include 1, 5, 20, and 100 with a default of 100
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
		catch (Exception& e) {
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
			if (std::stoi(argv[5]) == 1) shard_span_perc = 1;
			else if (std::stoi(argv[5]) == 5) shard_span_perc = 5;
			else if (std::stoi(argv[5]) == 20) shard_span_perc = 20;
			else if (std::stoi(argv[5]) == 100) shard_span_perc = 100;
		}
		catch (Exception & e) {
			std::cout << e.what() << std::endl;
		}
	}

	// run the application

	return 0;
}