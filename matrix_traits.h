#pragma once

#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>

namespace std {
	namespace experimental {
		template<class Scalar, size_t RowCount, size_t ColCount> struct matrix_traits;

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct matrix_traits_base
		{
			static const size_t row = RowCount;
			static const size_t col = ColCount;
			struct matrix_type {
				constexpr matrix_type() noexcept = default;
				matrix_type(std::initializer_list<Scalar>) noexcept;
				matrix_type(Scalar const(&)[RowCount * ColCount]) noexcept;

				Scalar m_data[RowCount * ColCount];
			};

			template<size_t OtherRow, size_t OtherCol>
			using other = matrix_traits<Scalar, OtherRow, OtherCol>;
			using matrix_t = matrix_type;
			using scalar_t = Scalar;

			constexpr matrix_traits_base() noexcept = default;
			matrix_traits_base(std::initializer_list<Scalar>) noexcept;
			matrix_traits_base(Scalar const(&)[RowCount * ColCount]) noexcept;
			static constexpr bool equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr bool not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t matrix_multiply_scalar(matrix_t const& lhs, Scalar const& rhs) noexcept;
			static constexpr matrix_t divide(matrix_t const& lhs, Scalar const& rhs) noexcept;
			static constexpr matrix_t add(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t subtract(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t positive(matrix_t const& mat) noexcept;
			static constexpr matrix_t negate(matrix_t const& mat) noexcept;
			static constexpr matrix_t scalar_multiply_matrix(Scalar const& lhs, matrix_t const& rhs) noexcept;
			template <size_t ColCount2>
			static constexpr typename matrix_traits<Scalar, RowCount, ColCount2>::matrix_t matrix_multiply_matrix(matrix_t const& lhs, typename matrix_traits<Scalar, ColCount, ColCount2>::matrix_t const& rhs) noexcept;

		protected:
			matrix_t m_data;
		};

		template<class Scalar, size_t RowCount>
		struct unity_matrix_traits : public matrix_traits_base<Scalar, RowCount, RowCount>
		{
			template<typename... Args> unity_matrix_traits(Args&&... args);

			using matrix_t = typename matrix_traits_base<Scalar, RowCount, RowCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr bool is_invertible(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
			static constexpr matrix_t identity() noexcept;
			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
		};

		template<class Scalar, size_t RowCount, size_t ColCount>
		struct vector_traits : public matrix_traits_base<Scalar, RowCount, ColCount>
		{
			template<typename... Args> vector_traits(Args&&... args) noexcept;

			using matrix_t = matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t;
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

			using matrix_t = matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr typename matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t submatrix(matrix_t const& mat, size_t m, size_t n) noexcept;
		};

		template<class Scalar, size_t RowCount>
		struct square_matrix_traits : public non_vector_traits<Scalar, RowCount, RowCount>
		{
			template<typename... Args> square_matrix_traits(Args&&... args) noexcept;

			using matrix_t = typename non_vector_traits<Scalar, RowCount, RowCount>::matrix_t;
			using scalar_t = Scalar;

			static constexpr bool is_invertible(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
			static constexpr matrix_t identity() noexcept;
			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr Scalar determinant(matrix_t const& mat) noexcept;
			static constexpr typename square_matrix_traits<Scalar, RowCount>::matrix_t classical_adjoint(matrix_t const& mat) noexcept;
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
		};

		template<class Scalar, class Allocator = std::allocator<Scalar>>
		struct dynamic_matrix_traits
		{};

		template<class Scalar, size_t RowCount, size_t ColCount>
		static constexpr typename matrix_traits<Scalar, ColCount, RowCount>::matrix_t transpose(typename matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept;
	}
}

// matrix_traits_base

template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_type::matrix_type(std::initializer_list<Scalar> il) noexcept
{
	assert(il.size() <= RowCount * ColCount);
	std::copy(std::begin(il), std::end(il), m_data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_type::matrix_type(Scalar const(&src)[RowCount * ColCount]) noexcept
{
	std::copy(src, src + (RowCount * ColCount), m_data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_traits_base(std::initializer_list<Scalar> il) noexcept
	: m_data(il)
{
	assert(il.size() <= RowCount * ColCount);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_traits_base(Scalar const(&src)[RowCount * ColCount]) noexcept
	: m_data(src)
{
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return std::equal(lhs.m_data, lhs.m_data + (RowCount * ColCount), rhs.m_data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr bool std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return !std::equal(lhs.m_data, lhs.m_data + (RowCount * ColCount), rhs.m_data);
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_multiply_scalar(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (RowCount * ColCount), res.m_data, [&](const auto& el) {return el * rhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::divide(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (RowCount * ColCount), res.m_data, [&](const auto& el) {return el / rhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::add(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (RowCount * ColCount), rhs.m_data, res.m_data, [&](const auto& lel, const auto& rel) {return lel + rel; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::subtract(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& lhs, matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (RowCount * ColCount), rhs.m_data, res.m_data, [&](const auto& lel, const auto& rel) {return lel - rel; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::positive(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	return mat;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::negate(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res(mat);
	std::transform(mat.m_data, mat.m_data + (RowCount * ColCount), [&](auto& el) { el = -el; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount>::scalar_multiply_matrix(matrix_traits_base<Scalar, RowCount, ColCount>::scalar_t const& lhs, matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	auto res(rhs);
	std::transform(rhs.m_data, rhs.m_data + (RowCount * ColCount), res.m_data, [&](const auto& el) {return el * lhs; });
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount1>
template<size_t ColCount2>
inline constexpr typename std::experimental::matrix_traits<Scalar, RowCount, ColCount2>::matrix_t std::experimental::matrix_traits_base<Scalar, RowCount, ColCount1>::matrix_multiply_matrix(matrix_traits_base<Scalar, RowCount, ColCount1>::matrix_t const& lhs, typename matrix_traits<Scalar, ColCount1, ColCount2>::matrix_t const& rhs) noexcept
{
	auto res = matrix_traits<Scalar, RowCount, ColCount2>::matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < ColCount2; ++j)
		{
			auto dp = Scalar(0);
			for (auto k = 0; k < ColCount1; ++k)
			{
				dp += lhs.m_data[i * ColCount1 + k] * rhs.m_data[j + ColCount2 * k];
			}
			res.m_data[i + j * RowCount] = dp;
		}
	}
	return res;
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits<Scalar, ColCount, RowCount>::matrix_t std::experimental::transpose(typename std::experimental::matrix_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res = matrix_traits<Scalar, RowCount, ColCount>::other<ColCount, RowCount>::matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < ColCount; ++j)
		{
			res.m_data[i + j * RowCount] = mat.m_data[i * ColCount + j];
		}
	}
	return res;
}

// unity_matrix_traits

template<class Scalar, size_t RowCount>
inline constexpr Scalar std::experimental::unity_matrix_traits<Scalar, RowCount>::determinant(unity_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	return mat.m_data[0];
}

// vector_traits

template<class Scalar, size_t RowCount, size_t ColCount>
template<typename... Args>
inline std::experimental::vector_traits<Scalar, RowCount, ColCount>::vector_traits(Args&&... args) noexcept
	: matrix_traits_base<Scalar, RowCount, ColCount>(std::forward(args))
{}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::inner_product(vector_traits<Scalar, RowCount, ColCount>::matrix_t const& lhs, vector_traits<Scalar, RowCount, ColCount>::matrix_t const& rhs) noexcept
{
	return std::inner_product(lhs.m_data, lhs.m_data + ColCount, rhs.m_data, Scalar(0));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::modulus(vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	return sqrt(modulus_squared(mat));
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr Scalar std::experimental::vector_traits<Scalar, RowCount, ColCount>::modulus_squared(vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	return std::accumulate(mat.m_data, mat.m_data + (RowCount * ColCount), Scalar(0), [&](Scalar tot, const auto& el) {return tot + (el * el); });
}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::vector_traits<Scalar, RowCount, ColCount>::matrix_t std::experimental::vector_traits<Scalar, RowCount, ColCount>::unit(vector_traits<Scalar, RowCount, ColCount>::matrix_t const& mat) noexcept
{
	auto res(mat);
	auto mod = modulus(mat);
	std::transform(mat.m_data, mat.m_data + (RowCount * ColCount), res.m_data, [&](const auto& el) { return el / mod; });
	return res;
}

// non_vector_traits

template<class Scalar, size_t RowCount, size_t ColCount>
template<typename... Args>
inline std::experimental::non_vector_traits<Scalar, RowCount, ColCount>::non_vector_traits(Args&&... args) noexcept
	: matrix_traits_base<Scalar, RowCount, ColCount>(std::forward(args))
{}

template<class Scalar, size_t RowCount, size_t ColCount>
inline constexpr typename std::experimental::matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t std::experimental::non_vector_traits<Scalar, RowCount, ColCount>::submatrix(matrix_traits_base<Scalar, RowCount, ColCount>::matrix_t const& mat, size_t p, size_t q) noexcept
{
	auto res = matrix_traits<Scalar, RowCount - 1, ColCount - 1>::matrix_t{};
	auto i = 0, j = 0;
	for (int r = 0; r < RowCount; ++r)
	{
		for (int c = 0; c < ColCount; ++c)
		{
			if (r != p && c != q)
			{
				auto res_offset = i * (ColCount - 1) + j;
				assert(res_offset < ((RowCount - 1) * (ColCount - 1)));
				res.m_data[i * (ColCount - 1) + j] = mat.m_data[r * ColCount + c];
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

// square_matrix_traits

template<class Scalar, size_t RowCount>
template<typename... Args>
inline std::experimental::square_matrix_traits<Scalar, RowCount>::square_matrix_traits(Args&&... args) noexcept
	: non_vector_traits<Scalar, RowCount, RowCount>(std::forward(args))
{}

template<class Scalar, size_t RowCount>
inline constexpr bool std::experimental::square_matrix_traits<Scalar, RowCount>::is_identity(square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < RowCount; ++j)
		{
			if (i == j && mat.m_data[i * RowCount + j] == Scalar(1)) continue;
			else if (mat.m_data[i * RowCount + j] == Scalar(0)) continue;
			return false;
		}
	}
	return true;
}

template<class Scalar, size_t RowCount>
inline constexpr bool std::experimental::square_matrix_traits<Scalar, RowCount>::is_invertible(square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	return determinant(mat) != 0;
}

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::identity() noexcept
{
	auto res = matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		res.m_data[i + i * RowCount] = Scalar(1);
	}
	return res;
}

template<class Scalar, size_t RowCount>
inline constexpr Scalar std::experimental::square_matrix_traits<Scalar, RowCount>::determinant(square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	if constexpr (RowCount == 1) return mat.m_data[0];
	else if constexpr (RowCount == 2) return (mat.m_data[0] * mat.m_data[3]) - (mat.m_data[2] * mat.m_data[2]);
	else if constexpr (RowCount > 2)
	{
		Scalar det = 0;
		Scalar sign = 1;
		for (auto f = 0; f < RowCount; ++f)
		{
			auto sub = matrix_traits<Scalar, RowCount, RowCount>::submatrix(mat, 0, f);
			auto cofactor = sign * mat.m_data[f] * matrix_traits<Scalar, RowCount - 1, RowCount - 1>::determinant(sub);
			det += cofactor;
			sign = -sign;
		}
		return det;
	}
}

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::classical_adjoint(square_matrix_traits<Scalar, RowCount>::matrix_t const& mat) noexcept
{
	auto res = matrix_t{};
	for (auto i = 0; i < RowCount; ++i)
	{
		for (auto j = 0; j < RowCount; ++j)
		{
			auto sub = matrix_traits<Scalar, RowCount, RowCount>::submatrix(mat, i, j);
			auto det = matrix_traits<Scalar, RowCount - 1, RowCount - 1>::determinant(sub);
			res.m_data[i * RowCount + j] = det;
		}
	}
	return transpose<Scalar, RowCount, RowCount>(res);
}

template<class Scalar, size_t RowCount>
inline constexpr typename std::experimental::square_matrix_traits<Scalar, RowCount>::matrix_t std::experimental::square_matrix_traits<Scalar, RowCount>::inverse(square_matrix_traits<Scalar, RowCount>::matrix_t const& mat)
{
	auto adj = classical_adjoint(mat);
	auto det = determinant(mat);
	std::transform(adj.m_data, adj.m_data + (RowCount * RowCount), adj.m_data, [&](const auto& el) { return el / det; });
	return adj;
}

/*
From Simon Brand:

template<class rep, size_t dim, class Scalar>
inline constexpr std::experimental::matrix<rep> std::experimental::operator*(std::experimental::matrix<rep> const& lhs, Scalar const& rhs) noexcept
{
	auto filler = [&]<size_t... Idx>(std::index_sequence<Idx...>){
		auto res(lhs);
		(res.i[Idx] *= rhs, ...);
		return res; };
	return filler (std::make_index_sequence<dim>{});
}

*/
