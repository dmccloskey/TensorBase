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
    @brief Fixed length 64 vector class
  */
  template<typename TensorT>
  class TensorArray64 : public TensorArray8<TensorT>
  {
  public:
    TensorArray64() = default;
    ~TensorArray64() = default;
    TensorArray64(const std::string& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray64(const std::initializer_list<TensorT>& tensor_array) { this->setTensorArray(tensor_array); }
    TensorArray64(const Eigen::Tensor<TensorT, 1>& tensor_array) { this->setTensorArray(tensor_array); }
    void setTensorArray(const std::string& tensor_array) override;
    void setTensorArray(const std::initializer_list<TensorT>& tensor_array) override;
    void setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array) override;
    Eigen::Tensor<TensorT, 1> getTensorArray() override;
    std::string getTensorArrayAsString() const override;
    TensorT at(const int& i) const override;

    /// Inline << operator overload
    friend std::ostream& operator<<(std::ostream& os, const TensorArray64& data) {
      os << data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_
        data.item_0_ << data.item_1_ << data.item_2_ << data.item_3_ << data.item_4_ << data.item_5_ << data.item_6_ << data.item_7_
      return os;
    }
  protected:
    TensorT item_8_ = TensorT(0); TensorT item_9_ = TensorT(0); TensorT item_10_ = TensorT(0); TensorT item_11_ = TensorT(0); TensorT item_12_ = TensorT(0); TensorT item_13_ = TensorT(0); TensorT item_14_ = TensorT(0); TensorT item_15_ = TensorT(0);
    TensorT item_16_ = TensorT(0); TensorT item_17_ = TensorT(0); TensorT item_18_ = TensorT(0); TensorT item_19_ = TensorT(0); TensorT item_20_ = TensorT(0); TensorT item_21_ = TensorT(0); TensorT item_22_ = TensorT(0); TensorT item_23_ = TensorT(0);

  private:
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive& archive) {
      archive(cereal::base_class<TensorArray8<TensorT>>(this),
        item_0_, item_1_, item_2_, item_3_, item_4_, item_5_, item_6_, item_7_);
    }
  };

  template<typename TensorT>
  inline TensorT TensorArray64<TensorT>::at(const int & i) const
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
      std::cout << "i " << i << " is greater than 64." << std::endl;
      return TensorT(0);
    }
  }
  template<typename TensorT>
  inline void TensorArray64<TensorT>::setTensorArray(const std::string & tensor_array)
  {
    // check the array size
    assert(64 >= tensor_array.size());
    this->array_size_ = 64;

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
  inline void TensorArray64<TensorT>::setTensorArray(const std::initializer_list<TensorT>& tensor_array)
  {
    // check the array size
    assert(64 >= tensor_array.size());
    this->array_size_ = 64;

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
  inline void TensorArray64<TensorT>::setTensorArray(const Eigen::Tensor<TensorT, 1>& tensor_array)
  {
    // check the array size
    assert(64 == tensor_array.dimension(0));
    this->array_size_ = 64;

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
  inline Eigen::Tensor<TensorT, 1> TensorArray64<TensorT>::getTensorArray()
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
  inline std::string TensorArray64<TensorT>::getTensorArrayAsString() const
  {
    std::ostringstream os;
    os << *this;
    return std::string(os.str());
  }
};
#endif //TENSORBASE_TENSORARRAY_H