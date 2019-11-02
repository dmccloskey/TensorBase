/**TODO:  Add copyright*/

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include "BenchmarkPixels.h"

using namespace TensorBase;
using namespace TensorBaseBenchmarks;

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
	void insert1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint0D`
	void insert1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint1D`
	void insert1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint2D`
	void insert1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint3D`
	void insert1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `insert1TimePoint4D`

	void update1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint0D`
	void update1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint1D`
	void update1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint2D`
	void update1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint3D`
	void update1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const override; ///< Device specific interface to call `update1TimePoint4D`
};
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
	this->insert1TimePoint0D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
	this->insert1TimePoint1D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
	this->insert1TimePoint2D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
	this->insert1TimePoint3D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::insert1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, false);
	this->insert1TimePoint4D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint0D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager0DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
	this->update1TimePoint0D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint1D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager1DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
	this->update1TimePoint1D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint2D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager2DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
	this->update1TimePoint2D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint3D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager3DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
	this->update1TimePoint3D_(pixel_manager, transaction_manager, data_size, device);
}
template<typename LabelsT, typename TensorT>
void Benchmark1TimePointDefaultDevice<LabelsT, TensorT>::update1TimePoint4D(TransactionManager<Eigen::DefaultDevice>& transaction_manager, const int& data_size, Eigen::DefaultDevice& device) const {
	PixelManager4DDefaultDevice<LabelsT, TensorT> pixel_manager(data_size, true);
	this->update1TimePoint4D_(pixel_manager, transaction_manager, data_size, device);
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
	// Setup the axes
	Eigen::Tensor<std::string, 1> dimensions_1(5), dimensions_2(1);
	dimensions_1.setValues({ "x","y","z","t","v" });
	dimensions_2.setValues({ "indices" });
	Eigen::Tensor<TensorArray8<char>, 2> labels_1(5, 1);
	labels_1.setValues({ { TensorArray8<char>("x")}, { TensorArray8<char>("y")}, { TensorArray8<char>("z")}, { TensorArray8<char>("t")}, { TensorArray8<char>("v")} });

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
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make1DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
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
	std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 2>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 2>>(TensorTableDefaultDevice<TensorT, 2>("TTable"));
	auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xyz", dimensions_1, labels_1));
	auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
	table_1_axis_2_ptr->setDimensions(dimensions_2);
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
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make3DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
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
	std::shared_ptr<TensorTable<TensorT, Eigen::DefaultDevice, 3>> table_1_ptr = std::make_shared<TensorTableDefaultDevice<TensorT, 3>>(TensorTableDefaultDevice<TensorT, 3>("TTable"));
	auto table_1_axis_1_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("xy", dimensions_1, labels_1));
	auto table_1_axis_2_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("z", dimensions_2, labels_2));
	auto table_1_axis_3_ptr = std::make_shared<TensorAxisDefaultDevice<LabelsT>>(TensorAxisDefaultDevice<LabelsT>("t", 1, 0));
	table_1_axis_3_ptr->setDimensions(dimensions_3);
	table_1_ptr->addTensorAxis(table_1_axis_1_ptr);
	table_1_ptr->addTensorAxis(table_1_axis_2_ptr);
	table_1_ptr->addTensorAxis(table_1_axis_3_ptr);
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
std::shared_ptr<TensorCollection<Eigen::DefaultDevice>> TensorCollectionGeneratorDefaultDevice<LabelsT, TensorT>::make4DTensorCollection(const int& data_size, const std::map<std::string, int>& shard_span, const bool& is_columnar) const
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
	table_1_ptr->setAxes();

	// Setup the table data
	table_1_ptr->setData();
	table_1_ptr->setShardSpans(shard_span);

	// Setup the collection
	auto collection_1_ptr = std::make_shared<TensorCollectionDefaultDevice>(TensorCollectionDefaultDevice());
	collection_1_ptr->addTensorTable(table_1_ptr, "TTable");
	return collection_1_ptr;
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
	int n_dims = 2;
	int data_size = 1296;
	bool in_memory = true;
	double shard_span_perc = 1;
	parseCmdArgs(argc, argv, data_dir, n_dims, data_size, in_memory, shard_span_perc);

	// Setup the Benchmarking suite
	Benchmark1TimePointDefaultDevice<int, float> benchmark_1_tp;

	// Setup the TensorCollectionGenerator
	TensorCollectionGeneratorDefaultDevice<int, float> tensor_collection_generator;

	// Setup the device
	Eigen::DefaultDevice device;

	// run the application
	runBenchmarkPixels(data_dir, n_dims, data_size, in_memory, shard_span_perc, benchmark_1_tp, tensor_collection_generator, device);

	return 0;
}