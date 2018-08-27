#if !defined LINEAR_ALGEBRA_18_07_29_15_04_10
#define LINEAR_ALGEBRA_18_07_29_15_04_10

#include <initializer_list>

namespace std::experimental::la {
    ////////////////////////////////////////////////////////
    // matrix
    ////////////////////////////////////////////////////////
    template <class Rep>
    struct matrix
    {
        using scalar_t = typename Rep::scalar_t;
        using matrix_t = typename Rep::matrix_t;
        // Constructors
        constexpr matrix() = default;
        constexpr explicit matrix(const matrix_t&) noexcept;
        constexpr matrix(std::initializer_list<scalar_t>) noexcept;
        // Accessors
        constexpr matrix_t const& data() const noexcept;
        constexpr matrix_t& data() noexcept;
        constexpr scalar_t operator()(size_t, size_t) const;                    // If the parameters are too high, what do we do? Throw, UB or error code?
        constexpr scalar_t& operator()(size_t, size_t);                    // If the parameters are too high, what do we do? Throw, UB or error code?
        // Equality operators
        constexpr bool operator==(matrix<Rep> const& rhs) const noexcept;
        constexpr bool operator!=(matrix<Rep> const& rhs) const noexcept;
        // Scalar binary operators
        constexpr matrix<Rep>& operator*=(scalar_t const& rhs) noexcept;
        constexpr matrix<Rep>& operator/=(scalar_t const& rhs) noexcept;
        // Matrix binary operators
        constexpr matrix<Rep>& operator+=(matrix<Rep> const& rhs) noexcept;
        constexpr matrix<Rep>& operator-=(matrix<Rep> const& rhs) noexcept;
    private:
        matrix_t _Data;
    };
    
    // Scalar binary operators
    template<class Rep>
    constexpr matrix<Rep> operator*(matrix<Rep> const& lhs, typename matrix<Rep>::scalar_t const& rhs) noexcept;
    
    template<class Rep>
    constexpr matrix<Rep> operator*(typename matrix<Rep>::scalar_t const& rhs, matrix<Rep> const& lhs) noexcept;
    
    template<class Rep>
    constexpr matrix<Rep> operator/(matrix<Rep> const& lhs, typename matrix<Rep>::scalar_t const& rhs) noexcept;
    
    // Matrix binary operators
    template<class Rep>
    constexpr matrix<Rep> operator+(matrix<Rep> const& lhs, matrix<Rep> const& rhs) noexcept;
    
    template<class Rep>
    constexpr matrix<Rep> operator-(matrix<Rep> const& lhs, matrix<Rep> const& rhs) noexcept;
    
    template<class Rep1, class Rep2>
    constexpr auto operator*(matrix<Rep1> const& lhs, matrix<Rep2> const& rhs) noexcept;
    
    // Matrix Functions
    template<class Rep>
    constexpr auto transpose(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr auto submatrix(matrix<Rep> const&, size_t p, size_t q) noexcept;
    
    // Vector Functions
    template<class Rep>
    constexpr typename Rep::scalar_t inner_product(matrix<Rep> const& lhs, matrix<Rep> const& rhs) noexcept;
    
    template<class Rep>
    constexpr typename Rep::scalar_t modulus(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr typename Rep::scalar_t modulus_squared(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr matrix<Rep> unit(matrix<Rep> const&) noexcept;
    
    // SquareMatrix predicates
    template<class Rep>
    constexpr bool is_identity(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr bool is_invertible(matrix<Rep> const&) noexcept;
    
    // SquareMatrix functions
    template<class Rep>
    constexpr matrix<Rep> identity() noexcept;
    
    template<class Rep>
    constexpr typename Rep::scalar_t determinant(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr matrix<typename Rep::transpose_t> classical_adjoint(matrix<Rep> const&) noexcept;
    
    template<class Rep>
    constexpr matrix<Rep> inverse(matrix<Rep> const&);
}

////////////////////////////////////////////////////////
// matrix implementation
////////////////////////////////////////////////////////
// Constructors
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>::matrix(const matrix_t& dat) noexcept
: _Data(dat)
{}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>::matrix(std::initializer_list<scalar_t> il) noexcept
: _Data(il)
{}

// Accessors
template<class Rep>
inline constexpr typename std::experimental::la::matrix<Rep>::matrix_t const& std::experimental::la::matrix<Rep>::data() const noexcept
{
    return _Data;
}

template<class Rep>
inline constexpr typename std::experimental::la::matrix<Rep>::matrix_t& std::experimental::la::matrix<Rep>::data() noexcept
{
    return _Data;
}

template<class Rep>
inline constexpr typename std::experimental::la::matrix<Rep>::scalar_t std::experimental::la::matrix<Rep>::operator()(size_t i, size_t j) const
{
    return _Data(i, j);
}

template<class Rep>
inline constexpr typename std::experimental::la::matrix<Rep>::scalar_t& std::experimental::la::matrix<Rep>::operator()(size_t i, size_t j)
{
    return _Data(i, j);
}

// Equality operators
template<class Rep>
inline constexpr bool std::experimental::la::matrix<Rep>::operator==(matrix<Rep> const& rhs) const noexcept
{
    return Rep::equal(data(), rhs.data());
}

template<class Rep>
inline constexpr bool std::experimental::la::matrix<Rep>::operator!=(matrix<Rep> const& rhs) const noexcept
{
    return Rep::not_equal(data(), rhs.data());
}

// Scalar member binary operators
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>& std::experimental::la::matrix<Rep>::operator*=(typename matrix<Rep>::scalar_t const& rhs) noexcept
{
    Rep::scalar_multiply(_Data, rhs);
    return *this;
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>& std::experimental::la::matrix<Rep>::operator/=(typename matrix<Rep>::scalar_t const& rhs) noexcept
{
    Rep::divide(_Data, rhs);
    return *this;
}

// Matrix member binary operators
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>& std::experimental::la::matrix<Rep>::operator+=(matrix<Rep> const& rhs) noexcept
{
    Rep::add(_Data, rhs.data());
    return *this;
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep>& std::experimental::la::matrix<Rep>::operator-=(matrix<Rep> const& rhs) noexcept
{
    Rep::subtract(_Data, rhs.data());
    return *this;
}

// Scalar non-member binary operators
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::operator*(std::experimental::la::matrix<Rep> const& lhs, typename std::experimental::la::matrix<Rep>::scalar_t const& rhs) noexcept
{
    auto res(lhs);
    return res *= rhs;
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::operator*(typename std::experimental::la::matrix<Rep>::scalar_t const& lhs, std::experimental::la::matrix<Rep> const& rhs) noexcept
{
    auto res(rhs);
    Rep::scalar_multiply(res.data(), lhs);
    return res;
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::operator/(std::experimental::la::matrix<Rep> const& lhs, typename std::experimental::la::matrix<Rep>::scalar_t const& rhs) noexcept
{
    auto res(lhs);
    return res /= rhs;
}

// Matrix non-member binary operators
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::operator+(std::experimental::la::matrix<Rep> const& lhs, std::experimental::la::matrix<Rep> const& rhs) noexcept
{
    auto res(lhs);
    return res += rhs;
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::operator-(std::experimental::la::matrix<Rep> const& lhs, std::experimental::la::matrix<Rep> const& rhs) noexcept
{
    auto res(lhs);
    return res -= rhs;
}

template<class Rep1, class Rep2>
inline constexpr auto std::experimental::la::operator*(std::experimental::la::matrix<Rep1> const& lhs, std::experimental::la::matrix<Rep2> const& rhs) noexcept
{
    return matrix<typename Rep1::template multiply_t<Rep2>>(Rep1::template matrix_multiply<Rep2>(lhs.data(), rhs.data()));
}

// Matrix functions
template<class Rep>
inline constexpr auto std::experimental::la::transpose(matrix<Rep> const& mat) noexcept
{
    return matrix<typename Rep::transpose_t>(Rep::transpose(mat.data()));
}

template<class Rep>
inline constexpr auto std::experimental::la::submatrix(matrix<Rep> const& mat, size_t p, size_t q) noexcept
{
    return matrix<typename Rep::submatrix_t>(Rep::submatrix(mat.data(), p, q));
}

// Vector functions
template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::la::inner_product(matrix<Rep> const& lhs, matrix<Rep> const& rhs) noexcept
{
    return Rep::inner_product(lhs.data(), rhs.data());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::la::modulus(matrix<Rep> const& vec) noexcept
{
    return Rep::modulus(vec.data());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::la::modulus_squared(matrix<Rep> const& vec) noexcept
{
    return Rep::modulus_squared(vec.data());
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::unit(matrix<Rep> const& vec) noexcept
{
    return matrix<Rep>(Rep::unit(vec.data()));
}

// Square matrix predicates
template<class Rep>
inline constexpr bool std::experimental::la::is_identity(matrix<Rep> const& mat) noexcept
{
    return Rep::is_identity(mat.data());
}

template<class Rep>
inline constexpr bool std::experimental::la::is_invertible(matrix<Rep> const& mat) noexcept
{
    return Rep::is_invertible(mat.data());
}

// SquareMatrix functions
template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::identity() noexcept
{
    return matrix<Rep>(Rep::identity());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::la::determinant(matrix<Rep> const& mat) noexcept
{
    return Rep::determinant(mat.data());
}

template<class Rep>
inline constexpr std::experimental::la::matrix<typename Rep::transpose_t> std::experimental::la::classical_adjoint(matrix<Rep> const& mat) noexcept
{
    return matrix<typename Rep::transpose_t>(Rep::classical_adjoint(mat.data()));
}

template<class Rep>
inline constexpr std::experimental::la::matrix<Rep> std::experimental::la::inverse(matrix<Rep> const& mat)
{
    return matrix<Rep>(Rep::inverse(mat.data()));
}

#endif
