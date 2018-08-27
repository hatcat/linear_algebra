#if !defined MATRIX_TRAITS_18_07_29_15_03_45
#define MATRIX_TRAITS_18_07_29_15_03_45

#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>

/*
TO DO:
--1. Eliminate dynamic_matrix_traits
--2. Migrate matrix_traits::matrix_type to additional header
3. Create two trait types, fixed_size and dynamic_size, and derive from those
4. Update la::matrix API to reflect fixed vs dynamic size
5. Create resizable matrix type
6. Reimplement matrix_traits to accommodate dynamic_size as well as fixed_size
*/

namespace std::experimental::la {
	////////////////////////////////////////////////////////
	// matrix_traits
	////////////////////////////////////////////////////////
	template<class Storage>
	struct matrix_traits
	{
		using scalar_t = typename Storage::scalar_t;
		using matrix_t = typename Storage::matrix_t;
		using transpose_t = typename Storage::transpose_t;
		using submatrix_t = typename Storage::submatrix_t;
		template<class Traits2>
		using multiply_t = typename Storage::template multiply_t<typename Traits2::matrix_t>;

		static constexpr bool equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		static constexpr bool not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		static constexpr void scalar_multiply(matrix_t& lhs, scalar_t const& rhs) noexcept;
		template <class Traits2> static constexpr typename multiply_t<Traits2>::matrix_t matrix_multiply(matrix_t const& lhs, typename Traits2::matrix_t const& rhs) noexcept;
		static constexpr void divide(matrix_t& lhs, scalar_t const& rhs) noexcept;
		static constexpr void add(matrix_t& lhs, matrix_t const& rhs) noexcept;
		static constexpr void subtract(matrix_t& lhs, matrix_t const& rhs) noexcept;
		static constexpr auto submatrix(matrix_t const& mat, size_t m, size_t n) noexcept;
		static constexpr typename transpose_t::matrix_t transpose(matrix_t const& mat) noexcept;
		static constexpr scalar_t inner_product(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		static constexpr scalar_t modulus(matrix_t const& mat) noexcept;
		static constexpr scalar_t modulus_squared(matrix_t const& mat) noexcept;
		static constexpr matrix_t unit(matrix_t const& mat) noexcept;
        static constexpr bool is_identity(matrix_t const& mat) noexcept;
		static constexpr matrix_t identity() noexcept;
		static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
		static constexpr typename transpose_t::matrix_t classical_adjoint(matrix_t const& mat) noexcept;
		static constexpr matrix_t inverse(matrix_t const& mat);
	};
}

////////////////////////////////////////////////////////
// matrix_impl implementation
////////////////////////////////////////////////////////
template <class Scalar>
template <class InputIterator>
constexpr bool std::experimental::la::matrix_impl<Scalar>::equal(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept
{
	return std::equal(lhs, end, rhs);
}

template <class Scalar>
template <class InputIterator>
constexpr bool std::experimental::la::matrix_impl<Scalar>::not_equal(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept
{
	return !equal(lhs, end, rhs);
}

template<class Scalar>
template <class Iterator>
inline constexpr void std::experimental::la::matrix_impl<Scalar>::scalar_multiply(Iterator lhs, Iterator end, Scalar rhs, Iterator res) noexcept
{
	std::transform(lhs, end, res, [&](const auto& el) {return el * rhs; });
}

template<class Scalar>
template <class InputIterator, class OutputIterator>
inline constexpr InputIterator std::experimental::la::matrix_impl<Scalar>::matrix_multiply(InputIterator lhs, InputIterator rhs, OutputIterator res, size_t m, size_t n, size_t q)
{
	for (auto i = 0U; i < m; ++i)
	{
		for (auto j = 0U; j < q; ++j)
		{
			auto dp = Scalar(0);
			for (auto k = 0U; k < n; ++k)
			{
				dp += *lhs++ * *rhs;
				rhs += q;
			}
			*res++ = dp;
			lhs -= n;
			rhs -= q * n;
			++rhs;
		}
		lhs += n;
		rhs -= q;
	}
	return res;
}

template<class Scalar>
template <class Iterator>
inline constexpr void std::experimental::la::matrix_impl<Scalar>::divide(Iterator lhs, Iterator end, Scalar rhs, Iterator res)
{
	std::transform(lhs, end, res, [&](const auto& el) {return el / rhs; });
}

template<class Scalar>
template <class Iterator, class OperandIterator>
constexpr void std::experimental::la::matrix_impl<Scalar>::add(Iterator lhs, Iterator end, OperandIterator rhs, Iterator res) noexcept
{
	std::transform(lhs, end, rhs, res, [&](const auto& lel, const auto& rel) {return lel + rel; });
}

template<class Scalar>
template <class Iterator, class OperandIterator>
constexpr void std::experimental::la::matrix_impl<Scalar>::subtract(Iterator lhs, Iterator end, OperandIterator rhs, Iterator res) noexcept
{
	std::transform(lhs, end, rhs, res, [&](const auto& lel, const auto& rel) {return lel - rel; });
}

template<class Scalar>
template <class InputIterator>
constexpr bool std::experimental::la::matrix_impl<Scalar>::is_identity(InputIterator lhs, size_t m) noexcept
{
	auto x = m + 1;
	for (auto y = 0; y != m * m; ++y, ++lhs)
	{
		if (x == (m + 1) && *lhs != Scalar(1))
		{
			return false;
		}
		else if (x != m + 1)
		{
			if (*lhs != Scalar(0)) return false;
		}
		if (--x == 0) x = m + 1;
	}
	return true;
}

template<class Scalar>
template <class OutputIterator>
constexpr OutputIterator std::experimental::la::matrix_impl<Scalar>::identity(OutputIterator res, size_t m) noexcept
{
	auto r = res;
	auto x = m + 1;
	for (auto y = 0; y != m * m; ++y, ++res)
	{
		if (x == (m + 1))
		{
			*res = Scalar(1);
		}
		else if (x != m + 1)
		{
			*res = Scalar(0);
		}
		if (--x == 0) x = m + 1;
	}
	return r;
}


template<class Scalar>
template <class InputIterator>
constexpr Scalar std::experimental::la::matrix_impl<Scalar>::inner_product(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept
{
	return Scalar(std::inner_product(lhs, end, rhs, Scalar(0)));
}

template<class Scalar>
template <class InputIterator>
constexpr Scalar std::experimental::la::matrix_impl<Scalar>::modulus(InputIterator lhs, InputIterator end) noexcept
{
	return Scalar(std::sqrt(modulus_squared(lhs, end)));
}

template<class Scalar>
template <class InputIterator>
constexpr Scalar std::experimental::la::matrix_impl<Scalar>::modulus_squared(InputIterator lhs, InputIterator end) noexcept
{
	return std::accumulate(lhs, end, Scalar(0), [&](Scalar tot, const auto& el) {return tot + (el * el); });
}

template<class Scalar>
template <class InputIterator, class OutputIterator>
constexpr OutputIterator std::experimental::la::matrix_impl<Scalar>::unit(InputIterator lhs, InputIterator end, OutputIterator res)
{
	auto mod = modulus(lhs, end);
	return std::transform(lhs, end, res, [&](const auto& el) { return el / mod; });
}

template<class Scalar>
template <class InputIterator, class OutputIterator>
inline constexpr OutputIterator std::experimental::la::matrix_impl<Scalar>::submatrix(InputIterator lhs, OutputIterator res, size_t m, size_t n, size_t i, size_t j)
{
	for (auto r = 0U; r < m; ++r)
	{
		for (auto c = 0U; c < n; ++c)
		{
			if (r != i && c != j)
			{
				*res++ = *lhs;
			}
			++lhs;
		}
	}
	return res;
}

////////////////////////////////////////////////////////
// matrix_traits implementation
////////////////////////////////////////////////////////
template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return matrix_impl<Storage::scalar_t>::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin());
}

template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return matrix_impl<Storage::scalar_t>::not_equal(lhs.cbegin(), lhs.cend(), rhs.cbegin());
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::scalar_multiply(matrix_t& lhs, scalar_t const& rhs) noexcept
{
	matrix_impl<Storage::scalar_t>::scalar_multiply(lhs.begin(), lhs.end(), rhs, lhs.begin());
}

template<class Storage>
template<class Traits2>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::multiply_t<Traits2>::matrix_t std::experimental::la::matrix_traits<Storage>::matrix_multiply(matrix_t const& lhs, typename Traits2::matrix_t const& rhs) noexcept
{
	auto res = typename matrix_traits<Storage>::multiply_t<Traits2>::matrix_t{};
	matrix_impl<Storage::scalar_t>::matrix_multiply(lhs.cbegin(), rhs.cbegin(), res.begin(), Storage::row, Traits2::matrix_t::col, Traits2::matrix_t::col);
	return res;
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::divide(matrix_t& lhs, scalar_t const& rhs) noexcept
{
	matrix_impl<Storage::scalar_t>::divide(lhs.begin(), lhs.end(), rhs, lhs.begin());
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::add(matrix_t& lhs, matrix_t const& rhs) noexcept
{
	matrix_impl<Storage::scalar_t>::add(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin());
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::subtract(matrix_t& lhs, matrix_t const& rhs) noexcept
{
	matrix_impl<Storage::scalar_t>::subtract(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin());
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::inner_product(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	static_assert(Storage::row == 1 || Storage::col == 1);
	static_assert(Storage::row != Storage::col);
	return matrix_impl<Storage::scalar_t>::inner_product(lhs.begin(), lhs.end(), rhs.begin());
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::modulus(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == 1 || Storage::col == 1);
	static_assert(Storage::row != Storage::col);
	return matrix_impl<Storage::scalar_t>::modulus(mat.begin(), mat.end());
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::modulus_squared(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == 1 || Storage::col == 1);
	static_assert(Storage::row != Storage::col);
	return matrix_impl<Storage::scalar_t>::modulus_squared(mat.begin(), mat.end());
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::matrix_t std::experimental::la::matrix_traits<Storage>::unit(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == 1 || Storage::col == 1);
	static_assert(Storage::row != Storage::col);
	auto res(mat);
	matrix_impl<Storage::scalar_t>::unit(mat.begin(), mat.end(), res.begin());
	return res;
}

template<class Storage>
inline constexpr auto std::experimental::la::matrix_traits<Storage>::submatrix(matrix_t const& mat, size_t i, size_t j) noexcept
{
	static_assert(Storage::row > 1 && Storage::col > 1);
	auto res = typename submatrix_t::matrix_t{};
	matrix_impl<Storage::scalar_t>::submatrix(mat.cbegin(), res.begin(), Storage::row, Storage::col, i, j);
	return res;
}

template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::is_identity(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == Storage::col);
	return matrix_impl<Storage::scalar_t>::is_identity(mat.cbegin(), Storage::row);
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::matrix_t std::experimental::la::matrix_traits<Storage>::identity() noexcept
{
	static_assert(Storage::row == Storage::col);
	auto res = matrix_t{};
	matrix_impl<Storage::scalar_t>::identity(res.begin(), Storage::row);
	return res;
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::determinant(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == Storage::col);
	if constexpr (Storage::row == 1) return mat._Data[0];
	else if constexpr (Storage::row == 2) return (mat._Data[0] * mat._Data[3]) - (mat._Data[1] * mat._Data[2]);
	else if constexpr (Storage::row > 2)
	{
		auto det = scalar_t(0);
		auto sign = scalar_t(1);
		for (auto f = 0; f < Storage::row; ++f)
		{
			auto sub = submatrix(mat, 0, f);
			auto cofactor = sign * mat._Data[f] * submatrix_t::determinant(sub);
			det += cofactor;
			sign = -sign;
		}
		return det;
	}
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::transpose_t::matrix_t std::experimental::la::matrix_traits<Storage>::classical_adjoint(matrix_t const& mat) noexcept
{
	static_assert(Storage::row == Storage::col);
	auto res = matrix_t{};
	for (auto i = 0; i < Storage::row; ++i)
	{
		auto sign = i % 2 == 0 ? scalar_t(1) : scalar_t(-1);
		for (auto j = 0; j < Storage::row; ++j)
		{
			auto sub = submatrix(mat, i, j);
			auto det = submatrix_t::determinant(sub);
			res._Data[i * Storage::row + j] = sign * det;
			sign = -sign;
		}
	}
	return transpose(res);
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::matrix_t std::experimental::la::matrix_traits<Storage>::inverse(matrix_t const& mat)
{
	auto adj = classical_adjoint(mat);
	auto det = determinant(mat);
	std::transform(adj._Data, adj._Data + (Storage::row * Storage::row), adj._Data, [&](const auto& el) { return el / det; });
	return adj;
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::transpose_t::matrix_t std::experimental::la::matrix_traits<Storage>::transpose(typename std::experimental::la::matrix_traits<Storage>::matrix_t const& mat) noexcept
{
	auto res = transpose_t::matrix_t{};
	for (auto i = 0; i < Storage::row; ++i)
	{
		for (auto j = 0; j < Storage::col; ++j)
		{
			res._Data[i + j * Storage::row] = mat._Data[i * Storage::col + j];
		}
	}
	return res;
}

#endif

/*
From Simon Brand:

template<class rep, size_t dim, class scalar_type>
inline constexpr std::experimental::la::matrix<rep> std::experimental::la::operator*(std::experimental::la::matrix<rep> const& lhs, scalar_type const& rhs) noexcept
{
auto filler = [&]<size_t... Idx>(std::index_sequence<Idx...>){
auto res(lhs);
(res.i[Idx] *= rhs, ...);
return res; };
return filler (std::make_index_sequence<dim>{});
}

*/
