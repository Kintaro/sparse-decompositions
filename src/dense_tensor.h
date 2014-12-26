// template <typename T, size_t P>
// class Tensor {
//  public:
//   std::size_t modes() const;
//   std::size_t size(const std::size_t mode) const;

//   inline Tensor(const Tensor&);
//   inline Tensor(Tensor&&);
//   inline Tensor& operator=(const Tensor&);
//   inline Tensor& operator=(Tensor&&);

//   inline const T& Get(const std::array<std::size_t, P>& index) const;

//   // inline const Scalar& coeff(const std::array<Index, NumIndices>& indices)
//   // const
//   // {
//   //   eigen_internal_assert(checkIndexRange(indices));
//   //   return m_storage.data()[linearizedIndex(indices)];
//   // }

//   // inline const Scalar& coeff(Index index) const
//   // {
//   //   eigen_internal_assert(index >= 0 && index < size());
//   //   return m_storage.data()[index];
//   // }

//   // template<typename... IndexTypes>
//   // inline Scalar& coeffRef(Index firstIndex, Index secondIndex,
// IndexTypes...
//   // otherIndices)
//   // {
//   //   static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of
//   // indices used to access a tensor coefficient must be equal to the rank of
//   // the tensor.");
//   //   return coeffRef(std::array<Index, NumIndices>{{firstIndex,
// secondIndex,
//   // otherIndices...}});
//   // }

//   // inline Scalar& coeffRef(const std::array<Index, NumIndices>& indices)
//   // {
//   //   eigen_internal_assert(checkIndexRange(indices));
//   //   return m_storage.data()[linearizedIndex(indices)];
//   // }

//   // inline Scalar& coeffRef(Index index)
//   // {
//   //   eigen_internal_assert(index >= 0 && index < size());
//   //   return m_storage.data()[index];
//   // }

//   // template<typename... IndexTypes>
//   // inline const Scalar& operator()(Index firstIndex, Index secondIndex,
//   // IndexTypes... otherIndices) const
//   // {
//   //   static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of
//   // indices used to access a tensor coefficient must be equal to the rank of
//   // the tensor.");
//   //   return this->operator()(std::array<Index, NumIndices>{{firstIndex,
//   // secondIndex, otherIndices...}});
//   // }

//   // inline const Scalar& operator()(const std::array<Index, NumIndices>&
//   // indices) const
//   // {
//   //   eigen_assert(checkIndexRange(indices));
//   //   return coeff(indices);
//   // }

//   // inline const Scalar& operator()(Index index) const
//   // {
//   //   eigen_internal_assert(index >= 0 && index < size());
//   //   return coeff(index);
//   // }

//   // inline const Scalar& operator[](Index index) const
//   // {
//   //   static_assert(NumIndices == 1, "The bracket operator is only for
// vectors,
//   // use the parenthesis operator instead.");
//   //   return coeff(index);
//   // }

//   // template<typename... IndexTypes>
//   // inline Scalar& operator()(Index firstIndex, Index secondIndex,
//   // IndexTypes... otherIndices)
//   // {
//   //   static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of
//   // indices used to access a tensor coefficient must be equal to the rank of
//   // the tensor.");
//   //   return operator()(std::array<Index, NumIndices>{{firstIndex,
// secondIndex,
//   // otherIndices...}});
//   // }

//   // inline Scalar& operator()(const std::array<Index, NumIndices>& indices)
//   // {
//   //   eigen_assert(checkIndexRange(indices));
//   //   return coeffRef(indices);
//   // }

//   // inline Scalar& operator()(Index index)
//   // {
//   //   eigen_assert(index >= 0 && index < size());
//   //   return coeffRef(index);
//   // }

//   // inline Scalar& operator[](Index index)
//   // {
//   //   static_assert(NumIndices == 1, "The bracket operator is only for
// vectors,
//   // use the parenthesis operator instead.");
//   //   return coeffRef(index);
//   // }

//   // inline Tensor()
//   //   : m_storage()
//   // {
//   // }

//   // inline Tensor(const Self& other)
//   //   : m_storage(other.m_storage)
//   // {
//   // }

//   // inline Tensor(Self&& other)
//   //   : m_storage(other.m_storage)
//   // {
//   // }

//   // template<typename... IndexTypes>
//   // inline Tensor(Index firstDimension, IndexTypes... otherDimensions)
//   //   : m_storage()
//   // {
//   //   static_assert(sizeof...(otherDimensions) + 1 == NumIndices, "Number of
//   // dimensions used to construct a tensor must be equal to the rank of the
//   // tensor.");
//   //   resize(std::array<Index, NumIndices>{{firstDimension,
//   // otherDimensions...}});
//   // }

//   // inline Tensor(std::array<Index, NumIndices> dimensions)
//   //   : m_storage(internal::array_prod(dimensions), dimensions)
//   // {
//   //   EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
//   // }

//   // template<typename... IndexTypes>
//   // void resize(Index firstDimension, IndexTypes... otherDimensions)
//   // {
//   //   static_assert(sizeof...(otherDimensions) + 1 == NumIndices, "Number of
//   // dimensions used to resize a tensor must be equal to the rank of the
//   // tensor.");
//   //   resize(std::array<Index, NumIndices>{{firstDimension,
//   // otherDimensions...}});
//   // }

//   // void resize(const std::array<Index, NumIndices>& dimensions)
//   // {
//   //   std::size_t i;
//   //   Index size = Index(1);
//   //   for (i = 0; i < NumIndices; i++) {
//   //     internal::check_rows_cols_for_overflow<Dynamic>::run(size,
//   // dimensions[i]);
//   //     size *= dimensions[i];
//   //   }
//   //   #ifdef EIGEN_INITIALIZE_COEFFS
//   //     bool size_changed = size != this->size();
//   //     m_storage.resize(size, dimensions);
//   //     if(size_changed) EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
//   //   #else
//   //     m_storage.resize(size, dimensions);
//   //   #endif
//   // }

//  private:
//   std::unique_ptr<Storage<T, P>> storage_ = nullptr;
// };

// template <typename T, size_t P>
// std::size_t Tensor<T, P>::modes() const {
//   return storage_->modes();
// }

// template <typename T, size_t P>
// std::size_t Tensor<T, P>::size(const std::size_t mode) const {
//   return storage_->size()[mode];
// }

// template <typename T, size_t P>
// Tensor<T, P>::Tensor(const Tensor<T, P>& other)
//     : storage_(other.storage_) {}

// template <typename T, size_t P>
// Tensor<T, P>::Tensor(Tensor<T, P>&& other)
//     : storage_(std::move(other.storage_)) {}

// template <typename T, size_t P>
// Tensor<T, P>& Tensor<T, P>::operator=(const Tensor<T, P>& other) {
//   if (this != &other) {
//     storage_.reset(other.storage_->Copy());
//   }
// }

// template <typename T, size_t P>
// Tensor<T, P>& Tensor<T, P>::operator=(Tensor<T, P>&& other)
//     : storage_(std::move(other.storage_)) {}

// template <typename T, size_t P>
// const T& Tensor<T, P>::Get(const std::array<std::size_t, P>& index) const {
//   return storage_->Get(index);
// }
