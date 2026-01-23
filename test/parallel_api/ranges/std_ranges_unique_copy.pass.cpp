// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
#include <span>
#include <initializer_list>

struct
{
    template <std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange,
              typename Comp, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange&& r_out, Comp comp, Proj proj = {})
    {
        using ret_type = std::ranges::unique_copy_result<std::ranges::borrowed_iterator_t<InRange>,
                                                         std::ranges::borrowed_iterator_t<OutRange>>;
        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (i == 0 || !bool(std::invoke(comp, std::invoke(proj, in[i - 1]), std::invoke(proj, in[i]))))
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
            {10, 1, {0},                2,  1}, // One element output range
            {10, 5, {0, 1, 2, 8, 1},    9,  5}, // Output range is not big enough
            {10, 6, {0, 1, 2, 8, 1, 8}, 10, 6}, // Output range is just enough
            {10, 7, {0, 1, 2, 8, 1, 8}, 10, 6}, // Output range is bigger
        };

        auto& self = *this;
        for (const TestCase& test_case : test_cases) {
            constexpr int shift = 1;
            std::span<int> in_span(input, test_case.in_size);
            std::span<int> out_span(output + shift, test_case.out_size);

            auto result = self(in_span, out_span, std::equal_to<int>{});

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
    }
} unique_copy_checker;
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    unique_copy_checker.test_self();

    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // input generator with a fair chance of repeating the previous value
    auto repeat_sometimes = [](auto i) {
        static decltype(i) last = 0;
        if (i==0)
            last = 0; // reset
        else if (i%7 > 0 && (last + i - 1)%3 == 0)
            last = i;
        return last;
    };
    using repeating_gen = decltype(repeat_sometimes);
    
    auto equal_tens = [](auto i, auto j) { return i/10 == j/10; };

    test_range_algo<0, int, data_in_out_lim>{163}(dpl_ranges::unique_copy, unique_copy_checker, std::ranges::equal_to{}, proj);
    test_range_algo<1, int, data_in_out_lim, repeating_gen>{837}(dpl_ranges::unique_copy, unique_copy_checker, equal_tens);
    test_range_algo<2, int, data_in_out_lim>{}(dpl_ranges::unique_copy, unique_copy_checker, std::ranges::not_equal_to{}, proj);
    test_range_algo<3, int, data_in_out_lim, repeating_gen>{}(dpl_ranges::unique_copy, unique_copy_checker, std::ranges::equal_to{}, proj);
    test_range_algo<4, P2, data_in_out_lim>{}(dpl_ranges::unique_copy, unique_copy_checker, equal_tens, &P2::x);
    test_range_algo<5, P2, data_in_out_lim>{}(dpl_ranges::unique_copy, unique_copy_checker, std::ranges::equal_to{}, &P2::proj);
    test_range_algo<6, int, data_in_out_lim, repeating_gen>{big_sz}(dpl_ranges::unique_copy, unique_copy_checker, std::ranges::equal_to{});
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
