#if !defined MATRIX_STORAGE_2018_08_24_12_32_44
#define MATRIX_STORAGE_2018_08_24_12_32_44

#include <initializer_list>

namespace std::experimental::la {
    template<class Scalar, size_t RowCount, size_t ColCount>
    struct fixed_size_matrix {
        using scalar_t = Scalar;
        using matrix_t = fixed_size_matrix<Scalar, RowCount, ColCount>;
        template<class Other>
        using multiply_t = fixed_size_matrix<Scalar, RowCount, Other::col>;
        using transpose_t = fixed_size_matrix<Scalar, ColCount, RowCount>;
        using submatrix_t = fixed_size_matrix<Scalar, RowCount - 1, ColCount - 1>;
        
        constexpr static size_t row = RowCount;
        constexpr static size_t col = ColCount;
        
        constexpr fixed_size_matrix() = default;
        fixed_size_matrix(std::initializer_list<Scalar>) noexcept;            // Pass by value or rref?
        constexpr Scalar operator()(size_t, size_t) const;
        constexpr Scalar& operator()(size_t, size_t);
        
        constexpr Scalar* begin() noexcept;
        constexpr const Scalar* cbegin() const noexcept;
        constexpr Scalar* end() noexcept;
        constexpr const Scalar* cend() const noexcept;
        
        Scalar _Data[RowCount * ColCount];
    };
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::operator()(size_t i, size_t j) const
{
    return _Data[i * RowCount + j];
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar& std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::operator()(size_t i, size_t j)
{
    return _Data[i * RowCount + j];
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar* std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::begin() noexcept
{
    return _Data;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr const Scalar* std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::cbegin() const noexcept
{
    return _Data;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar* std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::end() noexcept
{
    return _Data + RowCount * ColCount;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr const Scalar* std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::cend() const noexcept
{
    return _Data + RowCount * ColCount;
}

#endif //MATRIX_STORAGE_2018_08_24_12_32_44
