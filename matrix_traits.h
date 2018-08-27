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
--3. Create two trait types, fixed_size and dynamic_size, and derive from those
--4. Update la::matrix API to reflect fixed vs dynamic size
--5. Create resizable matrix type
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
        using transpose_t = matrix_traits<typename Storage::transpose_t>;
        using submatrix_t = matrix_traits<typename Storage::submatrix_t>;
        template<class Traits2>
        using multiply_t = matrix_traits<typename Storage::template multiply_t<typename Traits2::matrix_t>>;
        
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
        static constexpr bool is_invertible(matrix_t const& mat) noexcept;
        static constexpr matrix_t identity() noexcept;
        static constexpr scalar_t determinant(matrix_t const& mat) noexcept;
        static constexpr typename transpose_t::matrix_t classical_adjoint(matrix_t const& mat) noexcept;
        static constexpr matrix_t inverse(matrix_t const& mat);
    };
}

////////////////////////////////////////////////////////
// matrix_traits implementation
////////////////////////////////////////////////////////
template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
    return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin());
}

template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::not_equal(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
    return !std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin());
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::scalar_multiply(matrix_t& lhs, scalar_t const& rhs) noexcept
{
    std::transform(lhs.begin(), lhs.end(), lhs.begin(), [&](const auto& el) {return el * rhs; });
}

template<class Storage>
template<class Traits2>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::template multiply_t<Traits2>::matrix_t std::experimental::la::matrix_traits<Storage>::matrix_multiply(matrix_t const& lhs, typename Traits2::matrix_t const& rhs) noexcept
{
    auto res = typename matrix_traits<Storage>::template multiply_t<Traits2>::matrix_t{};
    auto out = res.begin();
    auto l_in = lhs.cbegin();
    auto r_in = rhs.cbegin();
    for (auto i = 0U; i < Storage::row; ++i)
    {
        for (auto j = 0U; j < Traits2::matrix_t::col; ++j)
        {
            auto dp = typename Storage::scalar_t(0);
            for (auto k = 0U; k < Traits2::matrix_t::col; ++k)
            {
                dp += *l_in++ * *r_in;
                r_in += Traits2::matrix_t::col;
            }
            *out++ = dp;
            l_in -= Traits2::matrix_t::col;
            r_in -= Traits2::matrix_t::col * Traits2::matrix_t::col;
            ++r_in;
        }
        l_in += Traits2::matrix_t::col;
        r_in -= Traits2::matrix_t::col;
    }
    return res;
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::divide(matrix_t& lhs, scalar_t const& rhs) noexcept
{
    std::transform(lhs.begin(), lhs.end(), lhs.begin(), [&](const auto& el) {return el / rhs; });
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::add(matrix_t& lhs, matrix_t const& rhs) noexcept
{
    std::transform(lhs.begin(), lhs.end(), rhs.cbegin(), lhs.begin(), [&](const auto& lel, const auto& rel) {return lel + rel; });
}

template<class Storage>
inline constexpr void std::experimental::la::matrix_traits<Storage>::subtract(matrix_t& lhs, matrix_t const& rhs) noexcept
{
    std::transform(lhs.begin(), lhs.end(), rhs.cbegin(), lhs.begin(), [&](const auto& lel, const auto& rel) {return lel - rel; });
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::inner_product(matrix_t const& lhs, matrix_t const& rhs) noexcept
{
    static_assert(Storage::row == 1 || Storage::col == 1);
    static_assert(Storage::row != Storage::col);
    return typename Storage::scalar_t(std::inner_product(lhs.cbegin(), lhs.cend(), rhs.cbegin(), typename Storage::scalar_t(0)));
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::modulus(matrix_t const& mat) noexcept
{
    static_assert(Storage::row == 1 || Storage::col == 1);
    static_assert(Storage::row != Storage::col);
    return typename Storage::scalar_t(std::sqrt(modulus_squared(mat)));
}

template<class Storage>
inline constexpr typename Storage::scalar_t std::experimental::la::matrix_traits<Storage>::modulus_squared(matrix_t const& mat) noexcept
{
    static_assert(Storage::row == 1 || Storage::col == 1);
    static_assert(Storage::row != Storage::col);
    return std::accumulate(mat.cbegin(), mat.cend(), typename Storage::scalar_t(0), [&](typename Storage::scalar_t tot, const auto& el) {return tot + (el * el); });
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::matrix_t std::experimental::la::matrix_traits<Storage>::unit(matrix_t const& mat) noexcept
{
    static_assert(Storage::row == 1 || Storage::col == 1);
    static_assert(Storage::row != Storage::col);
    auto res(mat);
    auto mod = modulus(mat);
    std::transform(mat.cbegin(), mat.cend(), res.begin(), [&](const auto& el) { return el / mod; });
    return res;
}

template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::is_identity(matrix_t const& mat) noexcept
{
    static_assert(Storage::row == Storage::col);
    auto l_in = mat.cbegin();
    auto x = Storage::row + 1;
    for (auto y = 0; y != Storage::row * Storage::row; ++y, ++l_in)
    {
        if (x == (Storage::row + 1) && *l_in != typename Storage::scalar_t(1))
        {
            return false;
        }
        else if (x != Storage::row + 1)
        {
            if (*l_in != typename Storage::scalar_t(0)) return false;
        }
        if (--x == 0) x = Storage::row + 1;
            }
    return true;
}

template<class Storage>
inline constexpr bool std::experimental::la::matrix_traits<Storage>::is_invertible(matrix_t const& mat) noexcept
{
    // TODO
    return true;
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::matrix_t std::experimental::la::matrix_traits<Storage>::identity() noexcept
{
    static_assert(Storage::row == Storage::col);
    auto res = matrix_t{};
    auto out = res.begin();
    auto x = Storage::row + 1;
    for (auto y = 0; y != Storage::row * Storage::row; ++y, ++out)
    {
        if (x == (Storage::row + 1))
        {
            *out = typename Storage::scalar_t(1);
        }
        else if (x != Storage::row + 1)
        {
            *out = typename Storage::scalar_t(0);
        }
        if (--x == 0) x = Storage::row + 1;
    }
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
inline constexpr auto std::experimental::la::matrix_traits<Storage>::submatrix(matrix_t const& mat, size_t i, size_t j) noexcept
{
    static_assert(Storage::row > 1 && Storage::col > 1);
    auto l_in = mat.cbegin();
    auto res = typename submatrix_t::matrix_t{};
    auto r_out = res.begin();
    for (auto r = 0U; r < Storage::row; ++r)
    {
        for (auto c = 0U; c < Storage::col; ++c)
        {
            if (r != i && c != j)
            {
                *r_out++ = *l_in;
            }
            ++l_in;
        }
    }
    return res;
}

template<class Storage>
inline constexpr typename std::experimental::la::matrix_traits<Storage>::transpose_t::matrix_t std::experimental::la::matrix_traits<Storage>::transpose(typename std::experimental::la::matrix_traits<Storage>::matrix_t const& mat) noexcept
{
    auto res = typename transpose_t::matrix_t{};
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
