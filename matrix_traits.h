#pragma once

#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>

namespace std {
	namespace experimental {
		template<class scalar_type, size_t row_count, size_t col_count> struct matrix_traits;

		template<class scalar_type, size_t row_count, size_t col_count>
		struct matrix_traits_base
		{
			static const size_t row = row_count;
			static const size_t col = col_count;
			struct matrix_type {
				constexpr matrix_type() noexcept = default;
				matrix_type(std::initializer_list<scalar_type>) noexcept;
				matrix_type(scalar_type const(&)[row_count * col_count]) noexcept;

				scalar_type m_data[row_count * col_count];
			};
			template<size_t other_row, size_t other_column>
			using other = matrix_traits<scalar_type, other_row, other_column>;
			using matrix_t = matrix_type;
			using scalar_t = scalar_type;
			constexpr matrix_traits_base() noexcept = default;
			matrix_traits_base(std::initializer_list<scalar_type>) noexcept;
			matrix_traits_base(scalar_type const(&)[row_count * col_count]) noexcept;
			static constexpr bool equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr bool not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t matrix_multiply_scalar(matrix_t const& lhs, scalar_type const& rhs) noexcept;
			static constexpr matrix_t divide(matrix_t const& lhs, scalar_type const& rhs) noexcept;
			static constexpr matrix_t add(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t subtract(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr matrix_t positive(matrix_t const& mat) noexcept;
			static constexpr matrix_t negate(matrix_t const& mat) noexcept;
			static constexpr matrix_t scalar_multiply_matrix(scalar_type const& lhs, matrix_t const& rhs) noexcept;
			template <size_t col_count2>
			static constexpr typename matrix_traits<scalar_type, row_count, col_count2>::matrix_t matrix_multiply_matrix(matrix_t const& lhs, typename matrix_traits<scalar_type, col_count, col_count2>::matrix_t const& rhs) noexcept;

		protected:
			matrix_t m_data;
		};

		template<class scalar_type, size_t row_count>
		struct unity_matrix_traits : public matrix_traits_base<scalar_type, row_count, row_count>
		{
			template<typename... Args> unity_matrix_traits(Args&&... args);

			using matrix_t = typename matrix_traits_base<scalar_type, row_count, row_count>::matrix_t;
			using scalar_t = scalar_type;

			static constexpr bool is_invertible(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
			static constexpr matrix_t identity() noexcept;
			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
		};

		template<class scalar_type, size_t row_count, size_t col_count>
		struct vector_traits : public matrix_traits_base<scalar_type, row_count, col_count>
		{
			template<typename... Args> vector_traits(Args&&... args) noexcept;

			using matrix_t = matrix_traits_base<scalar_type, row_count, col_count>::matrix_t;
			using scalar_t = scalar_type;

			static constexpr scalar_t inner_product(matrix_t const& lhs, matrix_t const& rhs) noexcept;
			static constexpr scalar_t modulus(matrix_t const& mat) noexcept;
			static constexpr scalar_t modulus_squared(matrix_t const& mat) noexcept;
			static constexpr matrix_t unit(matrix_t const& mat) noexcept;
		};

		template<class scalar_type, size_t row_count, size_t col_count>
		struct non_vector_traits : public matrix_traits_base<scalar_type, row_count, col_count>
		{
			template<typename... Args> non_vector_traits(Args&&... args) noexcept;

			using matrix_t = matrix_traits_base<scalar_type, row_count, col_count>::matrix_t;
			using scalar_t = scalar_type;

			static constexpr typename matrix_traits<scalar_type, row_count - 1, col_count - 1>::matrix_t submatrix(matrix_t const& mat, size_t m, size_t n) noexcept;
		};

		template<class scalar_type, size_t row_count>
		struct square_matrix_traits : public non_vector_traits<scalar_type, row_count, row_count>
		{
			template<typename... Args> square_matrix_traits(Args&&... args) noexcept;

			using matrix_t = typename non_vector_traits<scalar_type, row_count, row_count>::matrix_t;
			using scalar_t = scalar_type;

			static constexpr bool is_invertible(matrix_t const& mat) noexcept;
			static constexpr matrix_t inverse(matrix_t const& mat);
			static constexpr matrix_t identity() noexcept;
			static constexpr bool is_identity(matrix_t const& mat) noexcept;
			static constexpr scalar_type determinant(matrix_t const& mat) noexcept;
			static constexpr typename square_matrix_traits<scalar_type, row_count>::matrix_t classical_adjoint(matrix_t const& mat) noexcept;
		};

		template<class scalar_type, size_t row_count, size_t col_count>
		struct matrix_traits : public std::conditional_t < row_count == 1 && col_count == 1, unity_matrix_traits<scalar_type, row_count>,
										std::conditional_t<row_count == 1 || col_count == 1, vector_traits<scalar_type, row_count, col_count>,
										  std::conditional_t<row_count == col_count, square_matrix_traits<scalar_type, row_count>,
											non_vector_traits<scalar_type, row_count, col_count>>>>
		{
			template<typename... Args> matrix_traits(Args&&... args) noexcept;

			using parent = std::conditional_t < row_count == 1 && col_count == 1, unity_matrix_traits<scalar_type, row_count>,
							std::conditional_t<row_count == 1 || col_count == 1, vector_traits<scalar_type, row_count, col_count>,
							  std::conditional_t<row_count == col_count, square_matrix_traits<scalar_type, row_count>,
								non_vector_traits<scalar_type, row_count, col_count>>>>;
			using matrix_t = typename parent::matrix_t;
			using scalar_t = scalar_type;
		};

		template<class scalar_type, size_t row_count, size_t col_count>
		static constexpr typename matrix_traits<scalar_type, col_count, row_count>::matrix_t transpose(typename matrix_traits<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept;
	}
}

// matrix_traits_base

template<class scalar_type, size_t row_count, size_t col_count>
inline std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_type::matrix_type(std::initializer_list<scalar_type> il) noexcept
{
	assert(il.size() <= row_count * col_count);
	std::copy(std::begin(il), std::end(il), m_data);
}

template<class scalar_type, size_t row_count, size_t col_count>
inline std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_type::matrix_type(scalar_type const(&src)[row_count * col_count]) noexcept
{
	std::copy(src, src + (row_count * col_count), m_data);
}

template<class scalar_type, size_t row_count, size_t col_count>
inline std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_traits_base(std::initializer_list<scalar_type> il) noexcept
	: m_data(il)
{
	assert(il.size() <= row_count * col_count);
}

template<class scalar_type, size_t row_count, size_t col_count>
inline std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_traits_base(scalar_type const(&src)[row_count * col_count]) noexcept
	: m_data(src)
{
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr bool std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return std::equal(lhs.m_data, lhs.m_data + (row_count * col_count), rhs.m_data);
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr bool std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
	return !std::equal(lhs.m_data, lhs.m_data + (row_count * col_count), rhs.m_data);
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_multiply_scalar(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& lhs, matrix_traits_base<scalar_type, row_count, col_count>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (row_count * col_count), res.m_data, [&](const auto& el) {return el * rhs; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::divide(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& lhs, matrix_traits_base<scalar_type, row_count, col_count>::scalar_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (row_count * col_count), res.m_data, [&](const auto& el) {return el / rhs; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::add(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& lhs, matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (row_count * col_count), rhs.m_data, res.m_data, [&](const auto& lel, const auto& rel) {return lel + rel; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::subtract(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& lhs, matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& rhs) noexcept
{
	auto res(lhs);
	std::transform(lhs.m_data, lhs.m_data + (row_count * col_count), rhs.m_data, res.m_data, [&](const auto& lel, const auto& rel) {return lel - rel; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::positive(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	return mat;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::negate(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	auto res(mat);
	std::transform(mat.m_data, mat.m_data + (row_count * col_count), [&](auto& el) { el = -el; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count>::scalar_multiply_matrix(matrix_traits_base<scalar_type, row_count, col_count>::scalar_t const& lhs, matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& rhs) noexcept
{
	auto res(rhs);
	std::transform(rhs.m_data, rhs.m_data + (row_count * col_count), res.m_data, [&](const auto& el) {return el * lhs; });
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count1>
template<size_t col_count2>
inline constexpr typename std::experimental::matrix_traits<scalar_type, row_count, col_count2>::matrix_t std::experimental::matrix_traits_base<scalar_type, row_count, col_count1>::matrix_multiply_matrix(matrix_traits_base<scalar_type, row_count, col_count1>::matrix_t const& lhs, typename matrix_traits<scalar_type, col_count1, col_count2>::matrix_t const& rhs) noexcept
{
	auto res = matrix_traits<scalar_type, row_count, col_count2>::matrix_t{};
	for (auto i = 0; i < row_count; ++i)
	{
		for (auto j = 0; j < col_count2; ++j)
		{
			auto dp = scalar_type(0);
			for (auto k = 0; k < col_count1; ++k)
			{
				dp += lhs.m_data[i * col_count1 + k] * rhs.m_data[j + col_count2 * k];
			}
			res.m_data[i + j * row_count] = dp;
		}
	}
	return res;
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits<scalar_type, col_count, row_count>::matrix_t std::experimental::transpose(typename std::experimental::matrix_traits<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	auto res = matrix_traits<scalar_type, row_count, col_count>::other<col_count, row_count>::matrix_t{};
	for (auto i = 0; i < row_count; ++i)
	{
		for (auto j = 0; j < col_count; ++j)
		{
			res.m_data[i + j * row_count] = mat.m_data[i * col_count + j];
		}
	}
	return res;
}

// unity_matrix_traits

template<class scalar_type, size_t row_count>
inline constexpr scalar_type std::experimental::unity_matrix_traits<scalar_type, row_count>::determinant(unity_matrix_traits<scalar_type, row_count>::matrix_t const& mat) noexcept
{
	return mat.m_data[0];
}

// vector_traits

template<class scalar_type, size_t row_count, size_t col_count>
template<typename... Args>
inline std::experimental::vector_traits<scalar_type, row_count, col_count>::vector_traits(Args&&... args) noexcept
	: matrix_traits_base<scalar_type, row_count, col_count>(std::forward(args))
{}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr scalar_type std::experimental::vector_traits<scalar_type, row_count, col_count>::inner_product(vector_traits<scalar_type, row_count, col_count>::matrix_t const& lhs, vector_traits<scalar_type, row_count, col_count>::matrix_t const& rhs) noexcept
{
	return std::inner_product(lhs.m_data, lhs.m_data + col_count, rhs.m_data, scalar_type(0));
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr scalar_type std::experimental::vector_traits<scalar_type, row_count, col_count>::modulus(vector_traits<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	return sqrt(modulus_squared(mat));
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr scalar_type std::experimental::vector_traits<scalar_type, row_count, col_count>::modulus_squared(vector_traits<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	return std::accumulate(mat.m_data, mat.m_data + (row_count * col_count), scalar_type(0), [&](scalar_type tot, const auto& el) {return tot + (el * el); });
}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::vector_traits<scalar_type, row_count, col_count>::matrix_t std::experimental::vector_traits<scalar_type, row_count, col_count>::unit(vector_traits<scalar_type, row_count, col_count>::matrix_t const& mat) noexcept
{
	auto res(mat);
	auto mod = modulus(mat);
	std::transform(mat.m_data, mat.m_data + (row_count * col_count), res.m_data, [&](const auto& el) { return el / mod; });
	return res;
}

// non_vector_traits

template<class scalar_type, size_t row_count, size_t col_count>
template<typename... Args>
inline std::experimental::non_vector_traits<scalar_type, row_count, col_count>::non_vector_traits(Args&&... args) noexcept
	: matrix_traits_base<scalar_type, row_count, col_count>(std::forward(args))
{}

template<class scalar_type, size_t row_count, size_t col_count>
inline constexpr typename std::experimental::matrix_traits<scalar_type, row_count - 1, col_count - 1>::matrix_t std::experimental::non_vector_traits<scalar_type, row_count, col_count>::submatrix(matrix_traits_base<scalar_type, row_count, col_count>::matrix_t const& mat, size_t p, size_t q) noexcept
{
	auto res = matrix_traits<scalar_type, row_count - 1, col_count - 1>::matrix_t{};
	auto i = 0, j = 0;
	for (int r = 0; r < row_count; ++r)
	{
		for (int c = 0; c < col_count; ++c)
		{
			if (r != p && c != q)
			{
				auto res_offset = i * (col_count - 1) + j;
				assert(res_offset < ((row_count - 1) * (col_count - 1)));
				res.m_data[i * (col_count - 1) + j] = mat.m_data[r * col_count + c];
				if (++j == col_count - 1)
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

template<class scalar_type, size_t row_count>
template<typename... Args>
inline std::experimental::square_matrix_traits<scalar_type, row_count>::square_matrix_traits(Args&&... args) noexcept
	: non_vector_traits<scalar_type, row_count, row_count>(std::forward(args))
{}

template<class scalar_type, size_t row_count>
inline constexpr bool std::experimental::square_matrix_traits<scalar_type, row_count>::is_identity(square_matrix_traits<scalar_type, row_count>::matrix_t const& mat) noexcept
{
	for (auto i = 0; i < row_count; ++i)
	{
		for (auto j = 0; j < row_count; ++j)
		{
			if (i == j && mat.m_data[i * row_count + j] == scalar_type(1)) continue;
			else if (mat.m_data[i * row_count + j] == scalar_type(0)) continue;
			return false;
		}
	}
	return true;
}

template<class scalar_type, size_t row_count>
inline constexpr bool std::experimental::square_matrix_traits<scalar_type, row_count>::is_invertible(square_matrix_traits<scalar_type, row_count>::matrix_t const& mat) noexcept
{
	return determinant(mat) != 0;
}

template<class scalar_type, size_t row_count>
inline constexpr typename std::experimental::square_matrix_traits<scalar_type, row_count>::matrix_t std::experimental::square_matrix_traits<scalar_type, row_count>::identity() noexcept
{
	auto res = matrix_t{};
	for (auto i = 0; i < row_count; ++i)
	{
		res.m_data[i + i * row_count] = scalar_type(1);
	}
	return res;
}

template<class scalar_type, size_t row_count>
inline constexpr scalar_type std::experimental::square_matrix_traits<scalar_type, row_count>::determinant(square_matrix_traits<scalar_type, row_count>::matrix_t const& mat) noexcept
{
	if constexpr (row_count == 1) return mat.m_data[0];
	else if constexpr (row_count == 2) return (mat.m_data[0] * mat.m_data[3]) - (mat.m_data[2] * mat.m_data[2]);
	else if constexpr (row_count > 2)
	{
		scalar_type det = 0;
		scalar_type sign = 1;
		for (auto f = 0; f < row_count; ++f)
		{
			auto sub = matrix_traits<scalar_type, row_count, row_count>::submatrix(mat, 0, f);
			auto cofactor = sign * mat.m_data[f] * matrix_traits<scalar_type, row_count - 1, row_count - 1>::determinant(sub);
			det += cofactor;
			sign = -sign;
		}
		return det;
	}
}

template<class scalar_type, size_t row_count>
inline constexpr typename std::experimental::square_matrix_traits<scalar_type, row_count>::matrix_t std::experimental::square_matrix_traits<scalar_type, row_count>::classical_adjoint(square_matrix_traits<scalar_type, row_count>::matrix_t const& mat) noexcept
{
	auto res = matrix_t{};
	for (auto i = 0; i < row_count; ++i)
	{
		for (auto j = 0; j < row_count; ++j)
		{
			auto sub = matrix_traits<scalar_type, row_count, row_count>::submatrix(mat, i, j);
			auto det = matrix_traits<scalar_type, row_count - 1, row_count - 1>::determinant(sub);
			res.m_data[i * row_count + j] = det;
		}
	}
	return transpose<scalar_type, row_count, row_count>(res);
}

template<class scalar_type, size_t row_count>
inline constexpr typename std::experimental::square_matrix_traits<scalar_type, row_count>::matrix_t std::experimental::square_matrix_traits<scalar_type, row_count>::inverse(square_matrix_traits<scalar_type, row_count>::matrix_t const& mat)
{
	auto adj = classical_adjoint(mat);
	auto det = determinant(mat);
	std::transform(adj.m_data, adj.m_data + (row_count * row_count), adj.m_data, [&](const auto& el) { return el / det; });
	return adj;
}

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
