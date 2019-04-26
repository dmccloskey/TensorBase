/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE TensorOperation test suite 
#include <boost/test/included/unit_test.hpp>
#include <TensorBase/ml/TensorOperation.h>

using namespace TensorBase;
using namespace std;

BOOST_AUTO_TEST_SUITE(TensorOperation1)

//BOOST_AUTO_TEST_CASE(constructor) 
//{
//  TensorOperation* ptr = nullptr;
//  TensorOperation* nullPointer = nullptr;
//	ptr = new TensorOperation();
//  BOOST_CHECK_NE(ptr, nullPointer);
//}
//
//BOOST_AUTO_TEST_CASE(destructor) 
//{
//  TensorOperation* ptr = nullptr;
//	ptr = new TensorOperation();
//  delete ptr;
//}

BOOST_AUTO_TEST_CASE(TensorTableCreateAndDrop) {
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { "x-axis-0", "x-axis-1"} });
  labels2.setValues({ { "y-axis-0", "y-axis-1", "y-axis-2" } });
  labels3.setValues({ { "z-axis-0", "z-axis-1", "z-axis-2", "z-axis-3", "z-axis-4" } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorCollection<std::shared_ptr<TensorTableDefaultDevice<float, 3>>> collection_1(tensorTable1_ptr);

  // add a TensorTable to the collection
  TensorTableDefaultDevice<int, 2> tensorTable2("2", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    });
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorCollection<
    std::shared_ptr<TensorTableDefaultDevice<float, 3>>,
    std::shared_ptr<TensorTableDefaultDevice<int, 2>>> collection_add;
  TensorCreateTables()(collection_1, collection_add, tensorTable2_ptr);

  BOOST_CHECK(collection_add.getTableNames() == std::vector<std::string>({ "1", "2" }));
  // TODO: other checks needed here?

  //// remove a tensorTable from the collection
  //TensorCollection<
  //  std::shared_ptr<TensorTableDefaultDevice<float, 3>>
  //> collection_rm;
  //TensorDropTable()(collection_1, collection_rm, "2");
}

BOOST_AUTO_TEST_CASE(TensorTableSelectSlice) 
{
  Eigen::Tensor<std::string, 1> dimensions1(1), dimensions2(1), dimensions3(1);
  dimensions1(0) = "x";
  dimensions2(0) = "y";
  dimensions3(0) = "z";
  int nlabels1 = 2, nlabels2 = 3, nlabels3 = 5;
  Eigen::Tensor<std::string, 2> labels1(1, nlabels1), labels2(1, nlabels2), labels3(1, nlabels3);
  labels1.setValues({ { "x-axis-0", "x-axis-1"} });
  labels2.setValues({ { "y-axis-0", "y-axis-1", "y-axis-2" } });
  labels3.setValues({ { "z-axis-0", "z-axis-1", "z-axis-2", "z-axis-3", "z-axis-4" } });

  TensorTableDefaultDevice<float, 3> tensorTable1("1", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    TensorAxis("3", dimensions3, labels3),
    });
  std::shared_ptr<TensorTableDefaultDevice<float, 3>> tensorTable1_ptr = std::make_shared<TensorTableDefaultDevice<float, 3>>(tensorTable1);

  TensorTableDefaultDevice<int, 2> tensorTable2("2", { TensorAxis("1", dimensions1, labels1),
    TensorAxis("2", dimensions2, labels2),
    });
  std::shared_ptr<TensorTableDefaultDevice<int, 2>> tensorTable2_ptr = std::make_shared<TensorTableDefaultDevice<int, 2>>(tensorTable2);

  TensorCollection<
    std::shared_ptr<TensorTableDefaultDevice<float, 3>>,
    std::shared_ptr<TensorTableDefaultDevice<int, 2>>> collection_1(tensorTable1_ptr, tensorTable2_ptr);

  // select (labels)
  TensorSelectTableSlice<3> tensorSelectTableSlice("1",
    Eigen::array<std::string, 3>({"1", "2", "3"}),
    Eigen::array<std::string, 3>({ "x", "y", "z"}),
    Eigen::array<std::string, 3>({ "x-axis-0", "y-axis-0", "z-axis-0" }),
    Eigen::array<std::string, 3>({ "x-axis-1", "y-axis-2", "z-axis-4" })
    );
  for_each(collection_1.tables_, tensorSelectTableSlice);
  Eigen::array<int, 3> offsets_test = { 0, 0, 0 };
  Eigen::array<int, 3> extents_test = { nlabels1, nlabels2, nlabels3 };
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(tensorSelectTableSlice.offsets.at(i), offsets_test.at(i));
    BOOST_CHECK_EQUAL(tensorSelectTableSlice.extents.at(i), extents_test.at(i));
  }

  // TODO: where to apply the slice operation?
  //Eigen::Tensor<float, 3> test(nlabels1, nlabels2, nlabels3);
  //test.setZero();
  //std::cout << test << std::endl;
  //auto slice = test.slice(tensorSelectTableSlice.offsets, tensorSelectTableSlice.extents);
  //std::cout << slice << std::endl;

  // insert

  // delete

  // update

  //TensorCollection<
  //  std::shared_ptr<TensorTableDefaultDevice<float, 3>>,
  //  std::shared_ptr<TensorTableDefaultDevice<int, 2>>> collection_2(tables_new);

}

BOOST_AUTO_TEST_SUITE_END()