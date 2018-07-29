#if !defined MATRIX_TRAITS_18_07_29_15_03_45
#define MATRIX_TRAITS_18_07_29_15_03_45

#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>

namespace std {
	namespace experimental {
		////////////////////////////////////////////////////////
		// matrix_traits
		////////////////////////////////////////////////////////
		template<class Scalar, size_t RowCount, size_t ColCount> struct matrix_traits;

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct matrix_traits_base
		{
			static const size_t row = RowCount;
			static const size_t col = ColCount;
			struct matrix_type {
				constexpr matrix_type() noexcept = default;
				matrix_type(std::initializer_list<Scalar>) noexcept;			// Pass by value or rref?
				matrix_type(Scalar const(&)[RowCount * ColCount]) noexcept;		// Really, this should be a span or a range
				Scalar _Data[RowCount * ColCount];
			};

			template<size_t OtherRow, size_t OtherCol>
			using other = matrix_traits<Scalar, OtherRow, OtherCol>;
			using matrix_t = matrix_type;
			using scalar_t = Scalar;

			static constexpr bool equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr bool not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t matrix_multiply_scalar(matrix_t const& lhs, Scalar const& rhs) noexcept;
			static constexpr matrix_t scalar_multiply_matrix(Scalar const& lhs, matrix_t const& rhs) noexcept;
			template <size_t ColCount2> static constexpr typename matrix_traits<Scalar, RowCount, ColCount2>::matrix_t matrix_multiply_matrix(matrix_t const& lhs, typename matrix_traits<Scalar, ColCount, ColCount2>::matrix_t const& rhs) noexcept;
			static constexpr matrix_t divide(matrix_t const& lhs, Scalar const& rhs) noexcept;
			static constexpr matrix_t add(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t subtract(matrix_t const& lhs, matrix_t const& rhs) noexcept;
		};

		template<class Scalar, size_t RowCount>
		struct unity_matrix_traits : public matrix_traits_base<Scalar, RowCount, RowCount>
		{
			template<typename... Args> unity_matrix_traits(Args&&... args) noexcept;

			using matrix_t = typename matrix_traits_base<Scalar, RowCount, RowCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr matrix_t identity() noexcept;
			static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
		};

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct vector_traits : public matrix_traits_base<Scalar, RowCount, ColCount>
		{
			template<typename... Args> vector_traits(Args&&... args) noexcept;

			using matrix_t = typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr scalar_t inner_product(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr scalar_t modulus(matrix_t const& mat) noexcept;
			static constexpr scalar_t modulus_squared(matrix_t const& mat) noexcept;
			static constexpr matrix_t unit(matrix_t const& mat) noexcept;
		};

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct non_vector_traits : public matrix_traits_base<Scalar, RowCount, ColCount>
		{
			template<typename... Args> non_vector_traits(Args&&... args) noexcept;

			using matrix_t = typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr typename matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t submatrix(matrix_t const& mat, size_t m, size_t n) noexcept;
		};

		template<class Scalar, size_t RowCount>
		struct square_matrix_traits : public non_vector_traits<Scalar, RowCount, RowCount>
		{
			template<typename... Args> square_matrix_traits(Args&&... args) noexcept;

			using matrix_t = typename non_vector_traits<Scalar, RowCount, RowCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr matrix_t identity() noexcept;
			static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
			static constexpr typename square_matrix_traits<Scalar, RowCount>::matrix_t classical_adjoint(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
		};

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct matrix_traits : public std::conditional_t < RowCount == 1 && ColCount == 1, unity_matrix_traits<Scalar, RowCount>,
			std::conditional_t<RowCount == 1 || ColCount == 1, vector_traits<Scalar, RowCount, ColCount>,
			std::conditional_t<RowCount == ColCount, square_matrix_traits<Scalar, RowCount>,
			non_vector_traits<Scalar, RowCount, ColCount>>>>
		{
			template<typename... Args> matrix_traits(Args&&... args) noexcept;

			using parent = std::conditional_t < RowCount == 1 && ColCount == 1, unity_matrix_traits<Scalar, RowCount>,
				std::conditional_t<RowCount == 1 || ColCount == 1, vector_traits<Scalar, RowCount, ColCount>,
				std::conditional_t<RowCount == ColCount, square_matrix_traits<Scalar, RowCount>,
				non_vector_traits<Scalar, RowCount, ColCount>>>>;
			using matrix_t = typename parent::matrix_t;
			using scalar_t = Scalar;
			static const size_t row = RowCount;
			static const size_t col = ColCount;
		};

		template<class Scalar, class Allocator = std::allocator<Scalar>>
		struct dynamic_matrix_traits
		{};

		template<class Scalar, size_t RowCount, size_t ColCount>
		static constexpr typename matrix_traits<Scalar, ColCount, RowCount>::matrix_t transpose(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept;
	}
}

////////////////////////////////////////////////////////
// matrix_traits implementation
////////////////////////////////////////////////////////
// matrix_traits_base
template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_type::matrix_type(std::initializer_list<Scalar> il) noexcept
{
	assert(il.size() <= RowCount * ColCount);
	std::copy(std::begin(il), std::end(il), _Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_type::matrix_type(Scalar const(&src)[RowCount * ColCount]) noexcept
{
	std::copy(src, src + (RowCount * ColCount), _Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return std::equal(lhs._Data, lhs._Data + (RowCount * ColCount), rhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return !std::equal(lhs._Data, lhs._Data + (RowCount * ColCount), rhs._Data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_multiply_scalar(typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs._Data, lhs._Data + (RowCount * ColCount), res._Data, [&](const auto& el) {return el * rhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::scalar_multiply_matrix(typename matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& lhs, typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(rhs);
	std::transform(rhs._Data, rhs._Data + (RowCount * ColCount), res._Data, [&](const auto& el) {return el * lhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount1>
template<size_t ColCount2>
inline constexpr typename std::experimental::matrix_traits<Scalar, RowCount, ColCount2>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount1>::matrix_multiply_matrix(typename matrix_traits_base<Scalar, RowCount, ColCount1>::matrix_t const& lhs, typename matrix_traits<Scalar, ColCount1, ColCount2>::matrix_t const& rhs) noexcept
{
	auto res = typename matrix_traits<Scalar, RowCount, ColCount2>::matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < ColCount2; ++j)
		{
			auto dp = Scalar(0);
			for (auto k = 0; k < ColCount1; ++k)
			{
				dp += lhs._Data[i * ColCount1 + k] * rhs._Data[j + ColCount2 * k];
			}
			res._Data[j + i * RowCount] = dp;
		}
	}
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::divide(typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs._Data, lhs._Data + (RowCount * ColCount), res._Data, [&](const auto& el) {return el / rhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::add(typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs._Data, lhs._Data + (RowCount * ColCount), rhs._Data, res._Data, [&](const auto& lel, const auto& rel) {return lel + rel; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::subtract(typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs._Data, lhs._Data + (RowCount * ColCount), rhs._Data, res._Data, [&](const auto& lel, const auto& rel) {return lel - rel; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t std::experimental::non_vector_traits<Scalar, RowCount, ColCount>::submatrix(typename matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& mat, size_t p, size_t q) noexcept
{
	auto res = typename matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t{};
	auto i = 0, j = 0;
	for (int r = 0; r < RowCount; ++r)
	{
		for (int c = 0; c < ColCount; ++c)
		{
			if (r != p && c != q)
			{
				auto res_offset = i * (ColCount - 1) + j;
				assert(res_offset < ((RowCount - 1) * (ColCount - 1)));
				res._Data[i * (ColCount - 1) + j] = mat._Data[r * ColCount + c];
				if (++j == ColCount - 1)
				{
					j = 0;
					++i;
				}
			}
		}
	}
	return res;
}

// unity_matrix_traits

template<class Scalar, size_t RowCount>
template<typename... Args>
inline std::experimental::unity_matrix_traits<Scalar, RowCount>::unity_matrix_traits(Args&&... args) noexcept
	: matrix_traits_base<Scalar, RowCount, RowCount>(std::forward(args...))
{}

template<class Scalar, size_t RowCount>
inline constexpr Scalar std::experimental::unity_matrix_traits<Scalar, RowCount>::determinant(typename unity_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	return mat._Data[0];
}

// vector_traits

template<class Scalar, size_t RowCount, size_t ColCount>
template<typename... Args>
inline std::experimental::vector_traits<Scalar, RowCount, ColCount>::vector_traits(Args&&... args) noexcept
	: matrix_traits_base<Scalar, RowCount, ColCount>(std::forward(args...))
{}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::inner_product(typename vector_traits<Scalar, RowCount, ColCount>::matrix_t const& lhs, typename vector_traits<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	return std::inner_product(lhs._Data, lhs._Data + ColCount, rhs._Data, Scalar(0));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::modulus(typename vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	return scalar_t(std::sqrt(modulus_squared(mat)));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::modulus_squared(typename vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	return std::accumulate(mat._Data, mat._Data + (RowCount * ColCount), Scalar(0), [&](Scalar tot, const auto& el) {return tot + (el * el); });
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::vector_traits<Scalar, RowCount, ColCount>::matrix_t std::experimental::vector_traits<Scalar, RowCount, ColCount>::unit(typename vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res(mat);
	auto mod = modulus(mat);
	std::transform(mat._Data, mat._Data + (RowCount * ColCount), res._Data, [&](const auto& el) { return el / mod; });
	return res;
}

// square_matrix_traits
template<class Scalar, size_t RowCount>
template<typename... Args>
inline std::experimental::square_matrix_traits<Scalar, RowCount>::square_matrix_traits(Args&&... args) noexcept
	: non_vector_traits<Scalar, RowCount, RowCount>(std::forward(args...))
{}

template<class Scalar, size_t RowCount>
inline constexpr bool std::experimental::square_matrix_traits<Scalar, RowCount>::is_identity(typename square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < RowCount; ++j)
		{
			if (i == j && mat._Data[i * RowCount + j] == scalar_t(1)) continue;
			else if (i != j && mat._Data[i * RowCount + j] == scalar_t(0)) continue;
			return false;
		}
	}
	return true;
}

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::identity() noexcept
{
	auto res = matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		res._Data[i + i * RowCount] = Scalar(1);
	}
	return res;
}

template<class Scalar, size_t RowCount>
inline constexpr Scalar std::experimental::square_matrix_traits<Scalar, RowCount>::determinant(typename square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
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

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::classical_adjoint(typename square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
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
	return transpose<Scalar, RowCount, RowCount>(res);
}

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::inverse(typename square_matrix_traits<Scalar, RowCount>::matrix_t const& mat)
{
	auto adj = classical_adjoint(mat);
	auto det = determinant(mat);
	std::transform(adj._Data, adj._Data + (RowCount * RowCount), adj._Data, [&](const auto& el) { return el / det; });
	return adj;
}

// Free functions
template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits<Scalar, ColCount, RowCount>::matrix_t std::experimental::transpose(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res = typename matrix_traits<Scalar, RowCount, ColCount>::template other<ColCount, RowCount>::matrix_t{};
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
inline constexpr std::experimental::matrix<rep> std::experimental::operator*(std::experimental::matrix<rep> const& lhs, scalar_type const& rhs) noexcept
{
auto filler = [&]<size_t... Idx>(std::index_sequence<Idx...>){
auto res(lhs);
(res.i[Idx] *= rhs, ...);
return res; };
return filler (std::make_index_sequence<dim>{});
}

*/
