/**TODO:  Add copyright*/

#ifndef TENSORBASE_TENSORARRAY_H
#define TENSORBASE_TENSORARRAY_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream> // `getTensorArrayAsString`
#include <sstream> // `getTensorArrayAsString`

#include <cereal/access.hpp>  // serialiation of private members
#undef min // clashes with std::limit on windows in polymorphic.hpp
#undef max // clashes with std::limit on windows in polymorphic.hpp
#include <cereal/types/polymorphic.hpp>

namespace TensorBase
{ 
  /**
    @brief Base class for all fixed vector types
  */
  template<typename TensorT>
  class TensorArray
  {
  public:
    TensorArray() = default;
    virtual ~TensorArray() = default;

    /// operators are defined on a DeviceT-basis and executed on the specific DeviceT
    bool operator==(const TensorArray& other) const;
    bool operator!=(const TensorArray& other) const;
    bool operator<(const TensorArray& other) const;
    bool operator<=(const TensorArray& other) const;
    bool operator>(const TensorArray& other) const;
    bool operator>=(const TensorArray& other) const;

    size_t getArraySize() const { return array_size_; } ///< array_size getter

    virtual void setTensorArray(const std::string& tensor_array) = 0; ///< tensor_array setter
    virtual void setTensorArray(const std::initializer_list<TensorT>& tensor_array) = 0; ///< tensor_array setter
    virtual void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) = 0; ///< tensor_array setter
    virtual Eigen::Tensor<TensorT, 1> getTensorArray() = 0; ///< tensor_array getter
    virtual std::string getTensorArrayAsString() const = 0; ///< tensor_array getter as a string
    virtual TensorT at(const int& i) const = 0; /// tensor_array accessor

  protected:
    size_t array_size_ = 0;

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(array_size_);
    }
  };

  /* Operators for DefaultDevice and Cpu classes
  */
  namespace TensorArrayComparisonCpu {
    template<typename TensorT>
    int compare(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i)) break;
        if (i == size - 1) return 0;
      }
      return s1.at(i) - s2.at(i);
    }

    template<>
    int compare<char>(const TensorArray<char>& s1, const TensorArray<char>& s2, const int& size)
    {
      int i = 0;
      for (i = 0; i < size; ++i) {
        if (s1.at(i) != s2.at(i) || s1.at(i) == '\0' || s2.at(i) == '\0') break;
        if (i == size - 1) return 0;
      }
      return (const unsigned char)s1.at(i) - (const unsigned char)s2.at(i);
    }
  };

  // Operators for DefaultDevice and Cpu classes
  struct TensorArrayFunctors {
    TensorArrayFunctors(const int& size) : size_(size) {};
    int size_ = 0;
  };
  struct isEqualTo : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) == 0) return true;
      else return false;
    }
  };

  struct isNotEqualTo : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) != 0) return true;
      else return false;
    }
  };

  struct isLessThan : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) < 0) return true;
      else return false;
    }
  };

  struct isGreaterThan : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) > 0) return true;
      else return false;
    }
  };

  struct isLessThanOrEqualTo : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) <= 0) return true;
      else return false;
    }
  };

  struct isGreaterThanOrEqualTo : TensorArrayFunctors {
    using TensorArrayFunctors::TensorArrayFunctors;
    template<typename TensorT>
    bool operator()(const TensorArray<TensorT>& s1, const TensorArray<TensorT>& s2) {
      if (TensorArrayComparisonCpu::compare(s1, s2, this->size_) >= 0) return true;
      else return false;
    }
  };

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator==(const TensorArray<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isEqualTo comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator!=(const TensorArray<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isNotEqualTo comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator<(const TensorArray<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThan comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator<=(const TensorArray<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isLessThanOrEqualTo comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator>(const TensorArray<TensorT> & other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThan comp(this->array_size_);
    return comp(*this, other);
  }

  template<typename TensorT>
  inline bool TensorArray<TensorT>::operator>=(const TensorArray<TensorT>& other) const
  {
    assert(this->array_size_ == other.getArraySize());
    isGreaterThanOrEqualTo comp(this->array_size_);
    return comp(*this, other);
  }

  /**
    @brief Fixed length 8 vector class
  */
  template<typename TensorT>
  class TensorArray8: public TensorArray<TensorT>
  {
  public:
    TensorArray8() = default;
    ~TensorArray8() = default;
    TensorArray8(const std::string& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray8(const std::initializer_list<TensorT>& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray8(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    void setTensorArray(const std::string& tensor_array) override;
    void setTensorArray(const std::initializer_list<TensorT>& tensor_array) override;
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    Eigen::Tensor<TensorT, 1> getTensorArray() override;
    std::string getTensorArrayAsString() const override;
    TensorT at(const int& i) const override;

    /// Inline << operator overload
    friend std::ostream& operator<<(std::ostream& os, const TensorArray8& data) {
      os << data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_;
      return os;
    }
  protected:
    TensorT item_0_ = TensorT(0);
    TensorT item_1_ = TensorT(0);
    TensorT item_2_ = TensorT(0);
    TensorT item_3_ = TensorT(0);
    TensorT item_4_ = TensorT(0);
    TensorT item_5_ = TensorT(0);
    TensorT item_6_ = TensorT(0);
    TensorT item_7_ = TensorT(0);
  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray<TensorT>>(this),
        item_0_, item_1_, item_2_, item_3_, item_4_, item_5_, item_6_, item_7_);
    }
  };

  template<typename TensorT>
  inline TensorT TensorArray8<TensorT>::at(const int & i) const
  {
    if (i == 0) return item_0_;
    else if (i == 1) return item_1_;
    else if (i == 2) return item_2_;
    else if (i == 3) return item_3_;
    else if (i == 4) return item_4_;
    else if (i == 5) return item_5_;
    else if (i == 6) return item_6_;
    else if (i == 7) return item_7_;
    else {
      std::cout << "i " << i << " is greater than 8." << std::endl;
      return TensorT(0);
    }
  }
  template<typename TensorT>
  inline void TensorArray8<TensorT>::setTensorArray(const std::string & tensor_array)
  {
    // check the array size
    assert(8 >= tensor_array.size());
    this->array_size_ = 8;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) {
      this->item_0_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_0_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_1_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_1_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_2_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_2_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_3_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_3_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_4_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_4_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_5_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_5_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_6_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_6_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_7_ = static_cast<TensorT>(*tensor_array_iter);
      ++tensor_array_iter;
    }
    else this->item_7_ = TensorT(0);
  }
  template<typename TensorT>
  inline void TensorArray8<TensorT>::setTensorArray(const std::initializer_list<TensorT>& tensor_array)
  {
    // check the array size
    assert(8 >= tensor_array.size());
    this->array_size_ = 8;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) {
      this->item_0_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_0_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_1_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_1_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_2_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_2_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_3_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_3_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_4_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_4_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_5_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_5_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_6_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_6_ = TensorT(0);
    if (tensor_array_iter != tensor_array.end()) {
      this->item_7_ = *tensor_array_iter;
      ++tensor_array_iter;
    }
    else this->item_7_ = TensorT(0);
  }
  template<typename TensorT>
  inline void TensorArray8<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // check the array size
    assert(8 == tensor_array.dimension(0));
    this->array_size_ = 8;

    // copy the data
    this->item_0_ = tensor_array(0);
    this->item_1_ = tensor_array(1);
    this->item_2_ = tensor_array(2);
    this->item_3_ = tensor_array(3);
    this->item_4_ = tensor_array(4);
    this->item_5_ = tensor_array(5);
    this->item_6_ = tensor_array(6);
    this->item_7_ = tensor_array(7);
  }

  template<typename TensorT>
  inline Eigen::Tensor<TensorT, 1> TensorArray8<TensorT>::getTensorArray()
  {
    Eigen::Tensor<TensorT, 1> tensor_array(this->array_size_); 
    tensor_array(0) = this->item_0_;
    tensor_array(1) = this->item_1_;
    tensor_array(2) = this->item_2_;
    tensor_array(3) = this->item_3_;
    tensor_array(4) = this->item_4_;
    tensor_array(5) = this->item_5_;
    tensor_array(6) = this->item_6_;
    tensor_array(7) = this->item_7_;
    return tensor_array;
  }

  template<typename TensorT>
  inline std::string TensorArray8<TensorT>::getTensorArrayAsString() const
  {
    std::ostringstream os;
    os << *this;
    return std::string(os.str());
  }

  /**
    @brief Fixed length 32 vector class
  */
  template<typename TensorT>
  class TensorArray32 : public TensorArray8<TensorT>
  {
  public:
    TensorArray32() = default;
    ~TensorArray32() = default;
    TensorArray32(const std::string& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray32(const std::initializer_list<TensorT>& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray32(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    void setTensorArray(const std::string& tensor_array) override;
    void setTensorArray(const std::initializer_list<TensorT>& tensor_array) override;
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    Eigen::Tensor<TensorT, 1> getTensorArray() override;
    std::string getTensorArrayAsString() const override;
    TensorT at(const int& i) const override;

    /// Inline << operator overload
    friend std::ostream& operator<<(std::ostream& os, const TensorArray32& data) {
      os << data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_ << data.item_8_ << data.item_9_ << data.item_10_ << data.item_11_ << data.item_12_ << data.item_13_ << data.item_14_ << data.item_15_ << data.item_16_ << data.item_17_ << data.item_18_ << data.item_19_ << data.item_20_ << data.item_21_ << data.item_22_ << data.item_23_ << data.item_24_ << data.item_25_ << data.item_26_ << data.item_27_ << data.item_28_ << data.item_29_ << data.item_30_ << data.item_31_;
      return os;
    }
  protected:
    TensorT item_8_ = TensorT(0); TensorT item_9_ = TensorT(0); TensorT item_10_ = TensorT(0); TensorT item_11_ = TensorT(0); TensorT item_12_ = TensorT(0); TensorT item_13_ = TensorT(0); TensorT item_14_ = TensorT(0); TensorT item_15_ = TensorT(0);
    TensorT item_16_ = TensorT(0); TensorT item_17_ = TensorT(0); TensorT item_18_ = TensorT(0); TensorT item_19_ = TensorT(0); TensorT item_20_ = TensorT(0); TensorT item_21_ = TensorT(0); TensorT item_22_ = TensorT(0); TensorT item_23_ = TensorT(0);
    TensorT item_24_ = TensorT(0); TensorT item_25_ = TensorT(0); TensorT item_26_ = TensorT(0); TensorT item_27_ = TensorT(0); TensorT item_28_ = TensorT(0); TensorT item_29_ = TensorT(0); TensorT item_30_ = TensorT(0); TensorT item_31_ = TensorT(0);

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray8<TensorT>>(this),
        item_8_, item_9_, item_10_, item_11_, item_12_, item_13_, item_14_, item_15_, item_16_, item_17_, item_18_, item_19_, item_20_, item_21_, item_22_, item_23_, item_24_, item_25_, item_26_, item_27_, item_28_, item_29_, item_30_, item_31_);
    }
  };

  template<typename TensorT>
  inline TensorT TensorArray32<TensorT>::at(const int & i) const
  {
    if (i == 0) return item_0_;
    else if (i == 1) return item_1_;
    else if (i == 2) return item_2_;
    else if (i == 3) return item_3_;
    else if (i == 4) return item_4_;
    else if (i == 5) return item_5_;
    else if (i == 6) return item_6_;
    else if (i == 7) return item_7_;
    else if (i == 8) return item_8_;
    else if (i == 9) return item_9_;
    else if (i == 10) return item_10_;
    else if (i == 11) return item_11_;
    else if (i == 12) return item_12_;
    else if (i == 13) return item_13_;
    else if (i == 14) return item_14_;
    else if (i == 15) return item_15_;
    else if (i == 16) return item_16_;
    else if (i == 17) return item_17_;
    else if (i == 18) return item_18_;
    else if (i == 19) return item_19_;
    else if (i == 20) return item_20_;
    else if (i == 21) return item_21_;
    else if (i == 22) return item_22_;
    else if (i == 23) return item_23_;
    else if (i == 24) return item_24_;
    else if (i == 25) return item_25_;
    else if (i == 26) return item_26_;
    else if (i == 27) return item_27_;
    else if (i == 28) return item_28_;
    else if (i == 29) return item_29_;
    else if (i == 30) return item_30_;
    else if (i == 31) return item_31_;
    else {
      std::cout << "i " << i << " is greater than 32." << std::endl;
      return TensorT(0);
    }
  }
  template<typename TensorT>
  inline void TensorArray32<TensorT>::setTensorArray(const std::string & tensor_array)
  {
    // check the array size
    assert(32 >= tensor_array.size());
    this->array_size_ = 32;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) { this->item_0_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_0_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_1_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_1_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_2_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_2_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_3_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_3_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_4_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_4_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_5_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_5_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_6_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_6_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_7_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_7_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_8_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_8_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_9_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_9_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_10_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_10_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_11_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_11_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_12_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_12_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_13_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_13_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_14_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_14_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_15_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_15_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_16_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_16_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_17_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_17_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_18_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_18_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_19_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_19_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_20_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_20_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_21_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_21_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_22_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_22_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_23_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_23_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_24_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_24_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_25_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_25_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_26_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_26_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_27_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_27_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_28_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_28_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_29_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_29_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_30_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_30_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_31_ = static_cast<TensorT>(*tensor_array_iter); ++tensor_array_iter; }
    else { this->item_31_ = TensorT(0); }


  }
  template<typename TensorT>
  inline void TensorArray32<TensorT>::setTensorArray(const std::initializer_list<TensorT>& tensor_array)
  {
    // check the array size
    assert(32 >= tensor_array.size());
    this->array_size_ = 32;

    // copy the data
    auto tensor_array_iter = tensor_array.begin();
    if (tensor_array_iter != tensor_array.end()) { this->item_0_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_0_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_1_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_1_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_2_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_2_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_3_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_3_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_4_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_4_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_5_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_5_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_6_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_6_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_7_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_7_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_8_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_8_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_9_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_9_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_10_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_10_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_11_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_11_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_12_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_12_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_13_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_13_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_14_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_14_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_15_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_15_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_16_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_16_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_17_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_17_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_18_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_18_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_19_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_19_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_20_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_20_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_21_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_21_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_22_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_22_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_23_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_23_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_24_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_24_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_25_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_25_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_26_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_26_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_27_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_27_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_28_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_28_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_29_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_29_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_30_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_30_ = TensorT(0); }
    if (tensor_array_iter != tensor_array.end()) { this->item_31_ = *tensor_array_iter; ++tensor_array_iter; }
    else { this->item_31_ = TensorT(0); }

  }
  template<typename TensorT>
  inline void TensorArray32<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // check the array size
    assert(32 == tensor_array.dimension(0));
    this->array_size_ = 32;

    // copy the data
    this->item_0_ = tensor_array(0);
    this->item_1_ = tensor_array(1);
    this->item_2_ = tensor_array(2);
    this->item_3_ = tensor_array(3);
    this->item_4_ = tensor_array(4);
    this->item_5_ = tensor_array(5);
    this->item_6_ = tensor_array(6);
    this->item_7_ = tensor_array(7);
    this->item_8_ = tensor_array(8);
    this->item_9_ = tensor_array(9);
    this->item_10_ = tensor_array(10);
    this->item_11_ = tensor_array(11);
    this->item_12_ = tensor_array(12);
    this->item_13_ = tensor_array(13);
    this->item_14_ = tensor_array(14);
    this->item_15_ = tensor_array(15);
    this->item_16_ = tensor_array(16);
    this->item_17_ = tensor_array(17);
    this->item_18_ = tensor_array(18);
    this->item_19_ = tensor_array(19);
    this->item_20_ = tensor_array(20);
    this->item_21_ = tensor_array(21);
    this->item_22_ = tensor_array(22);
    this->item_23_ = tensor_array(23);
    this->item_24_ = tensor_array(24);
    this->item_25_ = tensor_array(25);
    this->item_26_ = tensor_array(26);
    this->item_27_ = tensor_array(27);
    this->item_28_ = tensor_array(28);
    this->item_29_ = tensor_array(29);
    this->item_30_ = tensor_array(30);
    this->item_31_ = tensor_array(31);
  }

  template<typename TensorT>
  inline Eigen::Tensor<TensorT, 1> TensorArray32<TensorT>::getTensorArray()
  {
    Eigen::Tensor<TensorT, 1> tensor_array(this->array_size_);
    tensor_array(0) = this->item_0_;
    tensor_array(1) = this->item_1_;
    tensor_array(2) = this->item_2_;
    tensor_array(3) = this->item_3_;
    tensor_array(4) = this->item_4_;
    tensor_array(5) = this->item_5_;
    tensor_array(6) = this->item_6_;
    tensor_array(7) = this->item_7_;
    tensor_array(8) = this->item_8_;
    tensor_array(9) = this->item_9_;
    tensor_array(10) = this->item_10_;
    tensor_array(11) = this->item_11_;
    tensor_array(12) = this->item_12_;
    tensor_array(13) = this->item_13_;
    tensor_array(14) = this->item_14_;
    tensor_array(15) = this->item_15_;
    tensor_array(16) = this->item_16_;
    tensor_array(17) = this->item_17_;
    tensor_array(18) = this->item_18_;
    tensor_array(19) = this->item_19_;
    tensor_array(20) = this->item_20_;
    tensor_array(21) = this->item_21_;
    tensor_array(22) = this->item_22_;
    tensor_array(23) = this->item_23_;
    tensor_array(24) = this->item_24_;
    tensor_array(25) = this->item_25_;
    tensor_array(26) = this->item_26_;
    tensor_array(27) = this->item_27_;
    tensor_array(28) = this->item_28_;
    tensor_array(29) = this->item_29_;
    tensor_array(30) = this->item_30_;
    tensor_array(31) = this->item_31_;
    return tensor_array;
  }

  template<typename TensorT>
  inline std::string TensorArray32<TensorT>::getTensorArrayAsString() const
  {
    std::ostringstream os;
    os << *this;
    return std::string(os.str());
  }
};
#endif //TENSORBASE_TENSORARRAY_H