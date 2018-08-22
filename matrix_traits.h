#if !defined MATRIX_TRAITS_18_07_29_15_03_45
#define MATRIX_TRAITS_18_07_29_15_03_45

#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>

/*
TO DO:
1. Eliminate dynamic_matrix_traits
2. Migrate matrix_traits::matrix_type to additional header
3. Create resizable matrix type
4. Create two trait types, fixed_size and dynamic_size, and derive from those
5. Reimplement matrix_traits to accommodate dynamic_size as well as fixed_size
*/

namespace std::experimental::la {
	template <class Scalar>
	struct matrix_impl
	{
		template <class InputIterator>
		static constexpr bool equal(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept;

		template <class InputIterator>
		static constexpr bool not_equal(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept;

		template <class Iterator>
		static constexpr void multiply(Iterator lhs, Iterator end, Scalar rhs, Iterator res) noexcept;

		template <class InputIterator, class OutputIterator>
		static constexpr InputIterator multiply(InputIterator lhs, InputIterator rhs, OutputIterator res, size_t m, size_t n, size_t q);

		template <class Iterator>
		static constexpr void divide(Iterator lhs, Iterator end, Scalar rhs, Iterator res);

		template <class Iterator, class OperandIterator>
		static constexpr void add(Iterator lhs, Iterator end, OperandIterator rhs, Iterator res) noexcept;

		template <class Iterator, class OperandIterator>
		static constexpr void subtract(Iterator lhs, Iterator end, OperandIterator rhs, Iterator res) noexcept;

		template <class InputIterator>
		static constexpr bool is_identity(InputIterator lhs, size_t m) noexcept;

		template <class OutputIterator>
		static constexpr OutputIterator identity(OutputIterator res, size_t m) noexcept;

		template <class InputIterator>
		static constexpr Scalar determinant(InputIterator const& lhs, size_t m);

		template <class InputIterator, class OutputIterator>
		static constexpr InputIterator inverse(InputIterator lhs, OutputIterator res, size_t m, size_t n);

		template <class InputIterator>
		static constexpr Scalar inner_product(InputIterator lhs, InputIterator end, InputIterator rhs) noexcept;

		template <class InputIterator>
		static constexpr Scalar modulus(InputIterator lhs, InputIterator end) noexcept;

		template <class InputIterator>
		static constexpr Scalar modulus_squared(InputIterator lhs, InputIterator end) noexcept;

		template <class InputIterator, class OutputIterator>
		static constexpr OutputIterator unit(InputIterator lhs, InputIterator end, OutputIterator res);

		template <class InputIterator, class OutputIterator>
		static constexpr OutputIterator submatrix(InputIterator lhs, OutputIterator res, size_t m, size_t n, size_t i, size_t j);
	};

	////////////////////////////////////////////////////////
	// matrix_traits
	////////////////////////////////////////////////////////
	template<class Scalar, size_t RowCount, size_t ColCount>
	struct matrix_traits
	{
		static const size_t row = RowCount;
		static const size_t col = ColCount;
		struct matrix_type {
			constexpr matrix_type() = default;
			matrix_type(std::initializer_list<Scalar>) noexcept;			// Pass by value or rref?

			constexpr Scalar operator()(size_t, size_t) const;

			Scalar _Data[RowCount * ColCount];
		};

		template<size_t OtherRow, size_t OtherCol>
		using other = matrix_traits<Scalar, OtherRow, OtherCol>;
		using matrix_t = matrix_type;
		using scalar_t = Scalar;
		using transpose_t = matrix_traits<Scalar, ColCount, RowCount>;
		using submatrix_t = matrix_traits<Scalar, RowCount - 1, ColCount - 1>;

		static constexpr bool equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		static constexpr bool not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		static constexpr void multiply(matrix_t& lhs, scalar_t const& rhs) noexcept;
		template <size_t ColCount2> static constexpr typename matrix_traits<Scalar, RowCount, ColCount2>::matrix_t multiply(matrix_t const& lhs, typename matrix_traits<Scalar, ColCount, ColCount2>::matrix_t const& rhs) noexcept;
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
inline constexpr void std::experimental::la::matrix_impl<Scalar>::multiply(Iterator lhs, Iterator end, Scalar rhs, Iterator res) noexcept
{
	std::transform(lhs, end, res, [&](const auto& el) {return el * rhs; });
}

template<class Scalar>
template <class InputIterator, class OutputIterator>
inline constexpr InputIterator std::experimental::la::matrix_impl<Scalar>::multiply(InputIterator lhs, InputIterator rhs, OutputIterator res, size_t m, size_t n, size_t q)
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
template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_type::matrix_type(std::initializer_list<Scalar> il) noexcept
{
	assert(il.size() <= row * col);
	std::copy(std::begin(il), std::end(il), _Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_type::operator()(size_t i, size_t j) const
{
	return _Data[i * ColCount + j];
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return matrix_impl<Scalar>::equal(lhs._Data, lhs._Data + (row * col), rhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return matrix_impl<Scalar>::not_equal(lhs._Data, lhs._Data + (row * col), rhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr void std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::multiply(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t& lhs, typename matrix_traits<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	matrix_impl<Scalar>::multiply(lhs._Data, lhs._Data + (row * col), rhs, lhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount1>
template<size_t ColCount2>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount2>::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount1>::multiply(typename matrix_traits<Scalar, RowCount, ColCount1>::matrix_t const& lhs, typename matrix_traits<Scalar, ColCount1, ColCount2>::matrix_t const& rhs) noexcept
{
	auto res = typename matrix_traits<Scalar, RowCount, ColCount2>::matrix_t{};
	matrix_impl<Scalar>::multiply(lhs._Data, rhs._Data, res._Data, row, col, ColCount2);
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr void std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::divide(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t& lhs, typename matrix_traits<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	matrix_impl<Scalar>::divide(lhs._Data, lhs._Data + (row * col), rhs, lhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr void std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::add(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t& lhs, typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	matrix_impl<Scalar>::add(lhs._Data, lhs._Data + (row * col), rhs._Data, lhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr void std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::subtract(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t& lhs, typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	matrix_impl<Scalar>::subtract(lhs._Data, lhs._Data + (row * col), rhs._Data, lhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::inner_product(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	static_assert(RowCount == 1 || ColCount == 1);
	static_assert(RowCount != ColCount);
	return matrix_impl<Scalar>::inner_product(lhs._Data, lhs._Data + (row * col), rhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::modulus(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == 1 || ColCount == 1);
	static_assert(RowCount != ColCount);
	return matrix_impl<Scalar>::modulus(mat._Data, mat._Data + (row * col));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::modulus_squared(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == 1 || ColCount == 1);
	static_assert(RowCount != ColCount);
	return matrix_impl<Scalar>::modulus_squared(mat._Data, mat._Data + (row * col));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::unit(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == 1 || ColCount == 1);
	static_assert(RowCount != ColCount);
	auto res(mat);
	matrix_impl<Scalar>::unit(mat._Data, mat._Data + (row * col), res._Data);
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr auto std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::submatrix(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat, size_t i, size_t j) noexcept
{
	static_assert(RowCount > 1 && ColCount > 1);
	auto res = typename submatrix_t::matrix_t{};
	matrix_impl<Scalar>::submatrix(mat._Data, res._Data, row, col, i, j);
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::is_identity(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == ColCount);
	return matrix_impl<Scalar>::is_identity(mat._Data, row);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::identity() noexcept
{
	static_assert(RowCount == ColCount);
	auto res = matrix_t{};
	matrix_impl<Scalar>::identity(res._Data, row);
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::determinant(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == ColCount);
	if constexpr (RowCount == 1) return mat._Data[0];
	else if constexpr (RowCount == 2) return (mat._Data[0] * mat._Data[3]) - (mat._Data[1] * mat._Data[2]);
	else if constexpr (RowCount > 2)
	{
		auto det = scalar_t(0);
		auto sign = scalar_t(1);
		for (auto f = 0; f < RowCount; ++f)
		{
			auto sub = matrix_traits<Scalar, RowCount, RowCount>::submatrix(mat, 0, f);
			auto cofactor = sign * mat._Data[f] * matrix_traits<Scalar, RowCount - 1, RowCount - 1>::determinant(sub);
			det += cofactor;
			sign = -sign;
		}
		return det;
	}
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::transpose_t::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::classical_adjoint(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	static_assert(RowCount == ColCount);
	auto res = matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		auto sign = i % 2 == 0 ? scalar_t(1) : scalar_t(-1);
		for (auto j = 0; j < RowCount; ++j)
		{
			auto sub = matrix_traits<Scalar, RowCount, RowCount>::submatrix(mat, i, j);
			auto det = matrix_traits<Scalar, RowCount - 1, RowCount - 1>::determinant(sub);
			res._Data[i * RowCount + j] = sign * det;
			sign = -sign;
		}
	}
	return transpose(res);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::inverse(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat)
{
	auto adj = classical_adjoint(mat);
	auto det = determinant(mat);
	std::transform(adj._Data, adj._Data + (RowCount * RowCount), adj._Data, [&](const auto& el) { return el / det; });
	return adj;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::transpose_t::matrix_t std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::transpose(typename std::experimental::la::matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res = typename matrix_traits<Scalar, RowCount, ColCount>::transpose_t::matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < ColCount; ++j)
		{
			res._Data[i + j * RowCount] = mat._Data[i * ColCount + j];
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
