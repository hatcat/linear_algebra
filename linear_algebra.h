#if !defined LINEAR_ALGEBRA_18_07_29_15_04_10
#define LINEAR_ALGEBRA_18_07_29_15_04_10

#include <initializer_list>

namespace std {
	namespace experimental {
		////////////////////////////////////////////////////////
		// matrix
		////////////////////////////////////////////////////////
		template <class Rep>
		struct matrix
		{
			using scalar_t = typename Rep::scalar_t;
			using matrix_t = typename Rep::matrix_t;
			// Constructors
			constexpr matrix() noexcept = default;
			constexpr explicit matrix(const matrix_t&) noexcept;
			constexpr matrix(std::initializer_list<scalar_t>) noexcept;
			constexpr matrix(scalar_t const(&src)[Rep::row * Rep::col]) noexcept;
			// Accessors
			constexpr matrix_t const& data() const noexcept;
			constexpr matrix_t& data() noexcept;
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
}

////////////////////////////////////////////////////////
// matrix implementation
////////////////////////////////////////////////////////
// Constructors
template<class Rep>
inline constexpr std::experimental::matrix<Rep>::matrix(const matrix_t& dat) noexcept
	: _Data(dat)
{}

template<class Rep>
inline constexpr std::experimental::matrix<Rep>::matrix(std::initializer_list<scalar_t> il) noexcept
	: _Data(il)
{}

template<class Rep>
inline constexpr std::experimental::matrix<Rep>::matrix(scalar_t const(&src)[Rep::row * Rep::col]) noexcept
	: _Data(src)
{}

// Accessors
template<class Rep>
inline constexpr typename std::experimental::matrix<Rep>::matrix_t const& std::experimental::matrix<Rep>::data() const noexcept
{
	return _Data;
}

template<class Rep>
inline constexpr typename std::experimental::matrix<Rep>::matrix_t& std::experimental::matrix<Rep>::data() noexcept
{
	return _Data;
}

// Equality operators
template<class Rep>
inline constexpr bool std::experimental::matrix<Rep>::operator==(matrix<Rep> const& rhs) const noexcept
{
	return Rep::equal(data(), rhs.data());
}

template<class Rep>
inline constexpr bool std::experimental::matrix<Rep>::operator!=(matrix<Rep> const& rhs) const noexcept
{
	return Rep::not_equal(data(), rhs.data());
}

// Scalar member binary operators
template<class Rep>
inline constexpr std::experimental::matrix<Rep>& std::experimental::matrix<Rep>::operator*=(typename matrix<Rep>::scalar_t const& rhs) noexcept
{
	Rep::multiply(_Data, rhs);
	return *this;
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep>& std::experimental::matrix<Rep>::operator/=(typename matrix<Rep>::scalar_t const& rhs) noexcept
{
	Rep::divide(_Data, rhs);
	return *this;
}

// Matrix member binary operators
template<class Rep>
inline constexpr std::experimental::matrix<Rep>& std::experimental::matrix<Rep>::operator+=(matrix<Rep> const& rhs) noexcept
{
	Rep::add(_Data, rhs.data());
	return *this;
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep>& std::experimental::matrix<Rep>::operator-=(matrix<Rep> const& rhs) noexcept
{
	Rep::subtract(_Data, rhs.data());
	return *this;
}

// Scalar non-member binary operators
template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::operator*(std::experimental::matrix<Rep> const& lhs, typename std::experimental::matrix<Rep>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	return res *= rhs;
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::operator*(typename std::experimental::matrix<Rep>::scalar_t const& lhs, std::experimental::matrix<Rep> const& rhs) noexcept
{
	auto res(rhs);
	Rep::multiply(res.data(), lhs);
	return res;
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::operator/(std::experimental::matrix<Rep> const& lhs, typename std::experimental::matrix<Rep>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	return res /= rhs;
}

// Matrix non-member binary operators
template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::operator+(std::experimental::matrix<Rep> const& lhs, std::experimental::matrix<Rep> const& rhs) noexcept
{
	auto res(lhs);
	return res += rhs;
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::operator-(std::experimental::matrix<Rep> const& lhs, std::experimental::matrix<Rep> const& rhs) noexcept
{
	auto res(lhs);
	return res -= rhs;
}

template<class Rep1, class Rep2>
inline constexpr auto std::experimental::operator*(std::experimental::matrix<Rep1> const& lhs, std::experimental::matrix<Rep2> const& rhs) noexcept
{
	return std::experimental::matrix<typename Rep1::template other<Rep1::row, Rep2::col>>(Rep1::template multiply<Rep2::col>(lhs.data(), rhs.data()));
}

// Matrix functions
template<class Rep>
inline constexpr auto std::experimental::transpose(matrix<Rep> const& mat) noexcept
{
	return matrix<typename Rep::transpose_t>(Rep::transpose(mat.data()));
}

template<class Rep>
inline constexpr auto std::experimental::submatrix(matrix<Rep> const& mat, size_t p, size_t q) noexcept
{
	return matrix<typename Rep::submatrix_t>(Rep::submatrix(mat.data(), p, q));
}

// Vector functions
template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::inner_product(matrix<Rep> const& lhs, matrix<Rep> const& rhs) noexcept
{
	return Rep::inner_product(lhs.data(), rhs.data());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::modulus(matrix<Rep> const& vec) noexcept
{
	return Rep::modulus(vec.data());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::modulus_squared(matrix<Rep> const& vec) noexcept
{
	return Rep::modulus_squared(vec.data());
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::unit(matrix<Rep> const& vec) noexcept
{
	return matrix<Rep>(Rep::unit(vec.data()));
}

// Square matrix predicates
template<class Rep>
inline constexpr bool std::experimental::is_identity(matrix<Rep> const& mat) noexcept
{
	return Rep::is_identity(mat.data());
}

template<class Rep>
inline constexpr bool std::experimental::is_invertible(matrix<Rep> const& mat) noexcept
{
	return Rep::is_invertible(mat);
}

// SquareMatrix functions
template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::identity() noexcept
{
	return matrix<Rep>(Rep::identity());
}

template<class Rep>
inline constexpr typename Rep::scalar_t std::experimental::determinant(matrix<Rep> const& mat) noexcept
{
	return Rep::determinant(mat.data());
}

template<class Rep>
inline constexpr std::experimental::matrix<typename Rep::transpose_t> std::experimental::classical_adjoint(matrix<Rep> const& mat) noexcept
{
	return matrix<typename Rep::transpose_t>(Rep::classical_adjoint(mat.data()));
}

template<class Rep>
inline constexpr std::experimental::matrix<Rep> std::experimental::inverse(matrix<Rep> const& mat)
{
	return matrix<Rep>(Rep::inverse(mat.data()));
}

#endif