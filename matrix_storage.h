#if !defined MATRIX_STORAGE_2018_08_24_12_32_44
#define MATRIX_STORAGE_2018_08_24_12_32_44

#include <initializer_list>

namespace std::experimental::la {
    struct fixed_size_matrix_t{};
    struct dynamic_size_matrix_t{};
    template<class T>
    using is_fixed_size = typename enable_if<std::is_base_of<fixed_size_matrix_t, T>::value>::type;
    template<class T>
    using is_dynamic_size = typename enable_if<std::is_base_of<dynamic_size_matrix_t, T>::value>::type;
    
    template<class Scalar, size_t RowCount, size_t ColCount>
    struct fixed_size_matrix : public fixed_size_matrix_t
    {
        using scalar_t = Scalar;
        using matrix_t = fixed_size_matrix<Scalar, RowCount, ColCount>;
        template<class Other>
        using multiply_t = fixed_size_matrix<Scalar, RowCount, Other::col>;
        using transpose_t = fixed_size_matrix<Scalar, ColCount, RowCount>;
        using submatrix_t = fixed_size_matrix<Scalar, RowCount - 1, ColCount - 1>;
        
        constexpr static size_t row = RowCount;
        constexpr static size_t col = ColCount;
        
        constexpr fixed_size_matrix() = default;
        constexpr fixed_size_matrix(std::initializer_list<Scalar>) noexcept;            // Pass by value or rref?
        constexpr Scalar operator()(size_t, size_t) const;
        constexpr Scalar& operator()(size_t, size_t);
        
        constexpr Scalar* begin() noexcept;
        constexpr const Scalar* cbegin() const noexcept;
        constexpr Scalar* end() noexcept;
        constexpr const Scalar* cend() const noexcept;
        
        Scalar _Data[RowCount * ColCount];
    };
    
    template<class Scalar, class Alloc = std::allocator<Scalar>>
    struct dynamic_size_matrix : public dynamic_size_matrix_t
    {
        using scalar_t = Scalar;
        using matrix_t = dynamic_size_matrix<Scalar, Alloc>;
        template<class Other>
        using multiply_t = dynamic_size_matrix<Scalar, Alloc>;
        using transpose_t = dynamic_size_matrix<Scalar, Alloc>;
        using submatrix_t = dynamic_size_matrix<Scalar, Alloc>;
        
        constexpr dynamic_size_matrix() = default;
        constexpr dynamic_size_matrix(dynamic_size_matrix const&);
        constexpr dynamic_size_matrix(std::pair<size_t, size_t>);
        constexpr Scalar operator()(size_t, size_t) const;
        constexpr Scalar& operator()(size_t, size_t);
        
        constexpr Scalar* begin() noexcept;
        constexpr const Scalar* cbegin() const noexcept;
        constexpr Scalar* end() noexcept;
        constexpr const Scalar* cend() const noexcept;
        
        size_t _RowCount = 0;
        size_t _ColCount = 0;
        std::unique_ptr<Scalar[]> _Data = nullptr;
    };
}

////////////////////////////////////////////////////////
// fixed_size_matrix implementation
////////////////////////////////////////////////////////
template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr std::experimental::la::fixed_size_matrix<Scalar, RowCount, ColCount>::fixed_size_matrix(std::initializer_list<Scalar> il) noexcept
{
    std::copy(il.begin(), il.end(), _Data);
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

////////////////////////////////////////////////////////
// dynamic_size_matrix implementation
////////////////////////////////////////////////////////
template<class Scalar, class Alloc>
inline constexpr std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::dynamic_size_matrix(dynamic_size_matrix const& rhs)
    : _RowCount(rhs._RowCount)
    , _ColCount(rhs._ColCount)
    , _Data(new Scalar[_RowCount * _ColCount])
{
    if (rhs._Data)
    {
        std::copy(rhs._Data.get(), rhs._Data.get() + _RowCount * _ColCount, _Data.get());
    }
}

template<class Scalar, class Alloc>
inline constexpr std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::dynamic_size_matrix(std::pair<size_t, size_t> size)
    : _RowCount(size.first)
    , _ColCount(size.first)
    , _Data(new Scalar[_RowCount * _ColCount])
{
}

template<class Scalar, class Alloc>
inline constexpr Scalar std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::operator()(size_t i, size_t j) const
{
    return _Data.get()[i * _RowCount + j];
}

template<class Scalar, class Alloc>
inline constexpr Scalar& std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::operator()(size_t i, size_t j)
{
    return _Data.get()[i * _RowCount + j];
}

template<class Scalar, class Alloc>
inline constexpr Scalar* std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::begin() noexcept
{
    return _Data.get();
}

template<class Scalar, class Alloc>
inline constexpr const Scalar* std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::cbegin() const noexcept
{
    return _Data.get();
}

template<class Scalar, class Alloc>
inline constexpr Scalar* std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::end() noexcept
{
    return _Data.get() + _RowCount * _ColCount;
}

template<class Scalar, class Alloc>
inline constexpr const Scalar* std::experimental::la::dynamic_size_matrix<Scalar, Alloc>::cend() const noexcept
{
    return _Data.get() + _RowCount * _ColCount;
}

#endif //MATRIX_STORAGE_2018_08_24_12_32_44
