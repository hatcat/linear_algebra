#pragma once

#include <initializer_list>
#include <algorithm>

namespace std {
	namespace experimental {
		// Define Vector concept (where row = 1)
		// Define SquareMatrix concept (where row = column)

		// matrix
		template<class rep>
		struct matrix
		{
			using scalar_t = typename rep::scalar_t;
			using matrix_t = typename rep::matrix_t;

			// Constructors
			constexpr matrix() noexcept;
			constexpr explicit matrix(const matrix_t&) noexcept;
			constexpr matrix(std::initializer_list<scalar_t>) noexcept;				// Pass by value or rref?
			constexpr matrix(scalar_t const(&src)[rep::row * rep::col]) noexcept;	// Really, this should be a span or a range

			// Accessors
			constexpr matrix_t const& data() const noexcept;
			constexpr matrix_t& data() noexcept;

			// Equality operators
			constexpr bool operator==(matrix<rep> const& rhs) const noexcept;
			constexpr bool operator!=(matrix<rep> const& rhs) const noexcept;

			// Scalar binary operators
			constexpr matrix<rep>& operator*=(scalar_t const& rhs) noexcept;
			constexpr matrix<rep>& operator/=(scalar_t const& rhs) noexcept;

			// Matrix binary operators
			constexpr matrix<rep>& operator+=(matrix<rep> const& rhs) noexcept;
			constexpr matrix<rep>& operator-=(matrix<rep> const& rhs) noexcept;

		private:
			matrix_t _Data;
		};

		// Unary operators
		template<class rep>
		constexpr matrix<rep> operator+(matrix<rep> const& mat) noexcept;

		template<class rep>
		constexpr matrix<rep> operator-(matrix<rep> const& mat) noexcept;

		// Scalar binary operators
		template<class rep>
		constexpr matrix<rep> operator*(matrix<rep> const& lhs, typename matrix<rep>::scalar_t const& rhs) noexcept;

		template<class rep>
		constexpr matrix<rep> operator*(typename matrix<rep>::scalar_t const& rhs, matrix<rep> const& lhs) noexcept;

		template<class rep>
		constexpr matrix<rep> operator/(matrix<rep> const& lhs, typename matrix<rep>::scalar_t const& rhs) noexcept;

		// Matrix binary operators
		template<class rep>
		constexpr matrix<rep> operator+(matrix<rep> const& lhs, matrix<rep> const& rhs) noexcept;

		template<class rep>
		constexpr matrix<rep> operator-(matrix<rep> const& lhs, matrix<rep> const& rhs) noexcept;

		template<class rep1, class rep2>
		constexpr auto operator*(matrix<rep1> const& lhs, matrix<rep2> const& rhs) noexcept;

		// Matrix Functions
		template<class rep>
		constexpr auto transpose(matrix<rep> const&) noexcept;

		template<class rep>
		constexpr auto submatrix(matrix<rep> const&, size_t p, size_t q) noexcept;

		// Vector Functions
		template<class rep>
		constexpr typename rep::scalar_t inner_product(matrix<rep> const& lhs, matrix<rep> const& rhs) noexcept;

		template<class rep>
		constexpr typename rep::scalar_t modulus(matrix<rep> const&) noexcept;

		template<class rep>
		constexpr typename rep::scalar_t modulus_squared(matrix<rep> const&) noexcept;

		template<class rep>
		constexpr matrix<rep> unit(matrix<rep> const&) noexcept;

		// SquareMatrix predicates
		template<class rep>
		constexpr bool is_identity(matrix<rep> const&) noexcept;

		template<class rep>
		constexpr bool is_invertible(matrix<rep> const&) noexcept;

		// SquareMatrix functions
		template<class rep>
		constexpr matrix<rep> identity() noexcept;

		template<class rep>
		constexpr typename rep::scalar_t determinant(matrix<rep> const&) noexcept;

		template<class rep1>
		constexpr auto classical_adjoint(matrix<rep1> const&) noexcept;

		template<class rep>
		constexpr matrix<rep> inverse(matrix<rep> const&);
	}
}

// Constructors
template<class rep>
inline constexpr std::experimental::matrix<rep>::matrix() noexcept
{
}

template<class rep>
inline constexpr std::experimental::matrix<rep>::matrix(const matrix_t& dat) noexcept
	: _Data(dat)
{
}

template<class rep>
inline constexpr std::experimental::matrix<rep>::matrix(std::initializer_list<scalar_t> il) noexcept
	: _Data(il)
{
}

template<class rep>
inline constexpr std::experimental::matrix<rep>::matrix(scalar_t const(&src)[rep::row * rep::col]) noexcept
	: _Data(src)
{
}

// Accessors
template<class rep>
inline constexpr typename std::experimental::matrix<rep>::matrix_t const& std::experimental::matrix<rep>::data() const noexcept
{
	return _Data;
}

template<class rep>
inline constexpr typename std::experimental::matrix<rep>::matrix_t& std::experimental::matrix<rep>::data() noexcept
{
	return _Data;
}

// Equality operators
template<class rep>
inline constexpr bool std::experimental::matrix<rep>::operator==(std::experimental::matrix<rep> const& rhs) const noexcept
{
	return rep::equal(data(), rhs.data());
}

template<class rep>
inline constexpr bool std::experimental::matrix<rep>::operator!=(std::experimental::matrix<rep> const& rhs) const noexcept
{
	return rep::not_equal(data(), rhs.data());
}

// Scalar binary operators
template<class rep>
inline constexpr std::experimental::matrix<rep>& std::experimental::matrix<rep>::operator*=(typename matrix<rep>::scalar_t const& rhs) noexcept
{
	_Data = rep::matrix_multiply_scalar(_Data, rhs);
	return *this;
}

template<class rep>
inline constexpr std::experimental::matrix<rep>& std::experimental::matrix<rep>::operator/=(typename matrix<rep>::scalar_t const& rhs) noexcept
{
	_Data = rep::divide(_Data, rhs);
	return *this;
}

// Matrix binary operators
template<class rep>
inline constexpr std::experimental::matrix<rep>& std::experimental::matrix<rep>::operator+=(matrix<rep> const& rhs) noexcept
{
	_Data = rep::add(_Data, rhs.data());
	return *this;
}

template<class rep>
inline constexpr std::experimental::matrix<rep>& std::experimental::matrix<rep>::operator-=(matrix<rep> const& rhs) noexcept
{
	_Data = rep::subtract(_Data, rhs.data());
	return *this;
}

// Unary operators
template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator+(matrix<rep> const& mat) noexcept
{
	return rep::positive(mat);
}

template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator-(matrix<rep> const& mat) noexcept
{
	return rep::negate(mat);
}

// Scalar binary operators
template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator*(std::experimental::matrix<rep> const& lhs, typename matrix<rep>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	return res *= rhs;
}

template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator*(typename matrix<rep>::scalar_t const& lhs, std::experimental::matrix<rep> const& rhs) noexcept
{
	return std::experimental::matrix<rep>(rep::scalar_multiply_matrix(lhs, rhs.data()));
}

template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator/(std::experimental::matrix<rep> const& lhs, typename matrix<rep>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	return res /= rhs;
}

// Matrix binary operators
template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator+(std::experimental::matrix<rep> const& lhs, std::experimental::matrix<rep> const& rhs) noexcept
{
	auto res(lhs);
	return res += rhs;
}

template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::operator-(std::experimental::matrix<rep> const& lhs, std::experimental::matrix<rep> const& rhs) noexcept
{
	auto res(lhs);
	return res -= rhs;
}

template<class rep1, class rep2>
inline constexpr auto std::experimental::operator*(std::experimental::matrix<rep1> const& lhs, std::experimental::matrix<rep2> const& rhs) noexcept
{
	return matrix<rep1::other<rep1::row, rep2::col>>(rep1::matrix_multiply_matrix<rep2::col>(lhs.data(), rhs.data()));
}

// Matrix functions
template<class rep>
inline constexpr auto std::experimental::transpose(std::experimental::matrix<rep> const& mat) noexcept
{
	auto res = transpose<rep>(mat.data());
	return matrix<rep::other<rep::col, rep::row>>(res);
}

template<class rep>
inline constexpr auto std::experimental::submatrix(std::experimental::matrix<rep> const& mat, size_t p, size_t q) noexcept
{
	auto res = rep::submatrix(mat.data(), p, q);
	return matrix<rep::other<rep::row - 1, rep::col - 1>>(res);
}

// Vector functions
template<class rep> // Requires Vector
inline constexpr typename rep::scalar_t std::experimental::inner_product(std::experimental::matrix<rep> const& lhs, std::experimental::matrix<rep> const& rhs) noexcept
{
	return rep::inner_product(lhs.data(), rhs.data());
}

template<class rep> // Requires Vector
inline constexpr typename rep::scalar_t std::experimental::modulus(std::experimental::matrix<rep> const& vec) noexcept
{
	return rep::modulus(vec.data());
}

template<class rep> // Requires Vector
inline constexpr typename rep::scalar_t std::experimental::modulus_squared(std::experimental::matrix<rep> const& vec) noexcept
{
	return rep::modulus_squared(vec.data());
}

template<class rep> // Requires Vector
inline constexpr std::experimental::matrix<rep> std::experimental::unit(std::experimental::matrix<rep> const& vec) noexcept
{
	return matrix<rep>(rep::unit(vec.data()));
}

// Square matrix predicates
template<class rep>
inline constexpr bool std::experimental::is_identity(std::experimental::matrix<rep> const& mat) noexcept
{
	return rep::is_identity(mat.data());
}

template<class rep>
inline constexpr bool std::experimental::is_invertible(std::experimental::matrix<rep> const& mat) noexcept
{
	return rep::is_invertible(mat);
}

// SquareMatrix functions
template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::identity() noexcept
{
	return matrix<rep>(rep::identity());
}

template<class rep>
inline constexpr typename rep::scalar_t std::experimental::determinant(std::experimental::matrix<rep> const& mat) noexcept
{
	return rep::determinant(mat.data());
}

template<class rep>
inline constexpr auto std::experimental::classical_adjoint(std::experimental::matrix<rep> const& mat) noexcept
{
	auto res = rep::classical_adjoint(mat.data());
	return matrix<rep::other<rep::col, rep::row>>(res);
}

template<class rep>
inline constexpr std::experimental::matrix<rep> std::experimental::inverse(std::experimental::matrix<rep> const& mat)
{
	return std::experimental::matrix<rep>(rep::inverse(mat.data()));
}
