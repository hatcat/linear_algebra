#include "matrix_storage.h"
#include "matrix_traits.h"
#include "linear_algebra.h"

void fixed_size_float_test()
{
    using namespace std::experimental::la;
    auto v1 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{};
    auto v2 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 0.0f };
    auto v3 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 1.0f, 2.0f };
    auto v4 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 1.0f, 2.0f };
    auto m1 = matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{};
    auto m2 = matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{
        0.0f, 0.2f,
        0.4f, 0.6f };
    auto m3 = matrix<matrix_traits<fixed_size_matrix<float, 2, 3>>>{
        0.0f, 0.1f, 0.2f,
        0.3f, 0.4f, 0.5f };
    auto m4 = matrix<matrix_traits<fixed_size_matrix<float, 3, 2>>>{
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f };
    auto m5 = matrix<matrix_traits<fixed_size_matrix<float, 3, 3>>>{
        1.0f, 2.0f, 3.0f,
        2.0f, 3.0f, 4.0f,
        1.0f, 5.0f, 7.0f };
    
    // test accessors
    assert(m3(1, 2) == 0.5f);
    
    // test equality operators
    assert(v2 != v3);
    assert(v3 == v4);
    
    // test scalar binary operators
    auto sbo1 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 2.0f, 2.0f } *3.0f;
    auto sbo2 = 3.0f * matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 2.0f, 2.0f };
    auto sbo3 = matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 2.0f, 2.0f } / 4.0f;
    
    assert(sbo1 == (matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 6.0f, 6.0f }));
    assert(sbo2 == (matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 6.0f, 6.0f }));
    assert(sbo3 == (matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 0.5f, 0.5f }));
    
    // test matrix binary operators
    auto mbo1 = v3 + v4;
    auto mbo2 = v3 - v4;
    auto mbo3 = m3 * m4;
    assert(mbo1 == (matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 2.0f, 4.0f }));
    assert(mbo2 == (matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 0.0f, 0.0f }));
    assert(mbo3 == (matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{ 1.3f, 1.6f,
        4.0f, 5.2f }));
    
    // test matrix functions
    auto mf1 = transpose(m3);
    auto mf2 = submatrix(m3, 0, 1);
    assert(mf1 == (decltype(mf1){ 0.0f, 0.3f,
        0.1f, 0.4f,
        0.2f, 0.5f }));
    assert(mf2 == (decltype(mf2){ 0.3f, 0.5f }));
    
    // test vector functions
    auto vf1 = inner_product(v3, v4);
    auto vf2 = modulus(matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 3.0f, 4.0f });
    auto vf3 = modulus_squared(matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 3.0f, 4.0f });
    auto vf4 = modulus(unit(matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 1.0f, 1.0f }));
    auto vf4a = modulus(matrix<matrix_traits<fixed_size_matrix<float, 1, 2>>>{ 0.707106769f, 0.707106769f });
    assert(vf1 == 5.0f);
    assert(vf2 == 5.0f);
    assert(vf3 == 25.0f);
    assert(vf4 - vf4a < std::numeric_limits<float>::min());
    
    // test SquareMatrix predicates
    assert(!(is_identity(matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{
        1.0f, 0.0f,
        1.0f, 1.0f })));
    assert((is_identity(matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{
        1.0f, 0.0f,
        0.0f, 1.0f })));
    assert((is_invertible(matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>(m2))));
    
    // test SquareMatrix functions
    auto smf1 = identity<matrix_traits<fixed_size_matrix<float, 2, 2>>>();
    auto smf2 = determinant<matrix_traits<fixed_size_matrix<float, 3, 3>>>(m5);
    auto smf3 = classical_adjoint<matrix_traits<fixed_size_matrix<float, 3, 3>>>(m5);
    auto smf4 = inverse(m2);
    auto smf5 = m2 * smf4;
    auto smf6 = smf4 * m2;
    assert(smf1 == (matrix<matrix_traits<fixed_size_matrix<float, 2, 2>>>{ 1.0f, 0.0f,
        0.0f, 1.0f }));
    assert(smf2 == 2.0f);
    assert(smf3 == (matrix<matrix_traits<fixed_size_matrix<float, 3, 3>>>{ 1.0f, 1.0f, -1.0f,
        -10.0f, 4.0f, 2.0f,
        7.0f, -3.0f, -1.0f }));
    assert(is_identity(smf5));    // Rounding error
    assert(is_identity(smf6));    // Rounding error
}

void dynamic_size_float_test()
{
    using namespace std::experimental::la;
    auto v1 = matrix<matrix_traits<dynamic_size_matrix<float>>>{};
    auto v2 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 1U) };
    auto v3 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) };
    auto v4 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) };
    auto m1 = matrix<matrix_traits<dynamic_size_matrix<float>>>{};
    auto m2 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(2U, 2U) };
    auto m3 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(2U, 3U) };
    auto m4 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(3U, 2U) };
    auto m5 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(3U, 3U) };
    
    // test accessors
    assert(m3(1, 2) == 0.5f);
    
    // test equality operators
    assert(v2 != v3);
    assert(v3 == v4);
    
    // test scalar binary operators
    auto sbo1 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) } *3.0f;
    auto sbo2 = 3.0f * matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) };
    auto sbo3 = matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) } / 4.0f;
    
    assert(sbo1 == (matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) }));
    assert(sbo2 == (matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) }));
    assert(sbo3 == (matrix<matrix_traits<dynamic_size_matrix<float>>>{ std::pair(1U, 2U) }));
}

int main()
{
    fixed_size_float_test();
    dynamic_size_float_test();
}
