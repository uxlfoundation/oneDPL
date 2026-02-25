// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
#include <initializer_list>

struct
{
    template <std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange,
              typename V, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange&& r_out, const V& value, Proj proj = {})
    {
        using ret_type = std::ranges::remove_copy_result<std::ranges::borrowed_iterator_t<InRange>,
                                                         std::ranges::borrowed_iterator_t<OutRange>>;
        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (!std::ranges::equal_to{}(std::invoke(proj, in[i]), value))
             {
                 if (j < std::ranges::size(r_out))
                     out[j++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out + j};
    }
    
    void test_self()
    {
#if TEST_CPP20_SPAN_PRESENT
        int input[10] = {0,0, 1, 2,2, 8, 1,1,1, 8};
        int output[9] = {-9, -8, -7, -6, -5, -4, -3, -2, -1};

        // Define test cases with expected outputs and expected end positions
        struct TestCase {
            int in_size;
            int out_size;
            std::initializer_list<int> expected_output;
            int expected_in_end;
            int expected_out_end;
        } test_cases[] = {
        // insz, outsz,  expected,  instop, outstop
            {0,  0, {},                 0,  0}, // Empty ranges
            {10, 0, {},                 0,  0}, // Empty output range
            {1,  1, {0},                1,  1}, // One element ranges
            {10, 1, {0},                1,  1}, // One element output range
            {10, 5, {0, 0, 2, 2, 8},    9,  5}, // Output range is not big enough
            {10, 6, {0, 0, 2, 2, 8, 8}, 10, 6}, // Output range is just enough
            {10, 7, {0, 0, 2, 2, 8, 8}, 10, 6}, // Output range is bigger than needed
        };

        auto& self = *this;
        for (const TestCase& test_case : test_cases) {
            constexpr int shift = 1;
            std::span<int> in_span(input, test_case.in_size);
            std::span<int> out_span(output + shift, test_case.out_size);

            auto result = self(in_span, out_span, 1);

            // Verify the returned iterators point to the correct end positions
            EXPECT_EQ(in_span.begin() + test_case.expected_in_end, result.in, "Checker problem: wrong input stop");
            EXPECT_EQ(out_span.begin() + test_case.expected_out_end, result.out, "Checker problem: wrong output stop");

            // Verify the output matches the expected result and nothing is overwritten
            for (int i = 0; i < 9; ++i)
            {
                if (i < shift || i >= shift + test_case.expected_out_end)
                {
                    EXPECT_EQ(i - 9, output[i], "Checker problem: out of range modification");
                }
                else
                {
                    EXPECT_EQ(test_case.expected_output.begin()[i - shift], output[i], "Checker problem: wrong output");
                    output[i] = i - 9; // Restore the original output data
                }
            }
        }
#endif // TEST_CPP20_SPAN_PRESENT
    }
} remove_copy_checker;
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto almost_always_two = [](auto i) {
        if (i%7 > 0 && (i - 1)%3 == 0)
            return i;
        return 2;
    };
    using many_twos = decltype(almost_always_two);

    remove_copy_checker.test_self();

    test_range_algo<0, int, data_in_out_lim>{179}(dpl_ranges::remove_copy, remove_copy_checker, 0);
    test_range_algo<1, int, data_in_out_lim, many_twos>{1127}(dpl_ranges::remove_copy, remove_copy_checker, 2);
    test_range_algo<2, int, data_in_out_lim>{}(dpl_ranges::remove_copy, remove_copy_checker, 1, proj);
    test_range_algo<3, P2, data_in_out_lim, many_twos>{}(dpl_ranges::remove_copy, remove_copy_checker, 2, &P2::x);
    test_range_algo<4, P2, data_in_out_lim>{}(dpl_ranges::remove_copy, remove_copy_checker, 0, &P2::proj);
    test_range_algo<5, int, data_in_out_lim>{big_sz}(dpl_ranges::remove_copy, remove_copy_checker, 1);
    test_range_algo<6, int, data_in_out_lim, many_twos>{big_sz}(dpl_ranges::remove_copy, remove_copy_checker, 2);
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
