// -*- C++ -*-
//===-- histogram.pass.cpp ------------------------------------------------===//
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

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/histogram_serial_impl.h"

using namespace TestUtils;


struct test_histogram_even_bins
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<TestUtils::is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 expected_bin_first,
               Iterator2 expected_bin_last, Iterator3 bin_first, Iterator3 bin_last, Size n, T bin_min, T bin_max,
               Size trash)
    {
        const Size bin_size = bin_last - bin_first;
        histogram_sequential(in_first, in_last, bin_size, bin_min, bin_max, expected_bin_first);
        auto orr = ::oneapi::dpl::histogram(exec, in_first, in_last, bin_size, bin_min, bin_max, bin_first);
        EXPECT_TRUE(bin_last == orr, "histogram returned wrong iterator");
        EXPECT_EQ_N(expected_bin_first, bin_first, bin_size, "wrong result from histogram");
        ::std::fill_n(bin_first, bin_size, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<!TestUtils::is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 expected_bin_first,
               Iterator2 expected_bin_last, Iterator3 bin_first, Iterator3 bin_last, Size n, T bin_min, T bin_max,
               Size trash)
    {
    }
};

struct test_histogram_range_bins
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    std::enable_if_t<TestUtils::is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 boundary_first, Iterator2 boundary_last,
               Iterator3 expected_bin_first, Iterator3 /* expected_bin_last */, Iterator4 bin_first, Iterator4 bin_last,
               Size trash)
    {
        const Size bin_size = boundary_last - boundary_first - 1;
        histogram_sequential(in_first, in_last, boundary_first, boundary_last, expected_bin_first);
        auto orr = ::oneapi::dpl::histogram(exec, in_first, in_last, boundary_first, boundary_last, bin_first);
        EXPECT_TRUE(bin_last == orr, "histogram returned wrong iterator");
        EXPECT_EQ_N(expected_bin_first, bin_first, bin_size, "wrong result from histogram");
        ::std::fill_n(bin_first, bin_size, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    std::enable_if_t<!TestUtils::is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 boundary_first, Iterator2 boundary_last,
               Iterator3 expected_bin_first, Iterator3 /* expected_bin_last */, Iterator4 bin_first, Iterator4 bin_last,
               Size trash)
    {
    }
};

template <::std::size_t CallNumber, typename Size, typename T>
void
test_range_and_even_histogram(Size n, T min_boundary, T max_boundary, T overflow, Size jitter, Size num_bins,
                              Size trash)
{
    //possibly spill over by overflow/2 on each side of range
    Sequence<T> in(n, [&](size_t k) {
        return T(std::rand() % Size(max_boundary - min_boundary + overflow)) + min_boundary - overflow / T(2);
    });

    Sequence<Size> expected(num_bins, [](size_t k) { return 0; });
    Sequence<Size> out(num_bins, [&](size_t k) { return trash; });

    invoke_on_all_policies<CallNumber * 4>()(test_histogram_even_bins(), in.begin(), in.end(), expected.begin(),
                                                    expected.end(), out.begin(), out.end(), Size(in.size()),
                                                    min_boundary, max_boundary, trash);
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<CallNumber * 4 + 1>()(test_histogram_even_bins(), in.cbegin(), in.cend(),
                                                        expected.begin(), expected.end(), out.begin(), out.end(),
                                                        Size(in.size()), min_boundary, max_boundary, trash);
#    endif // !ONEDPL_FPGA_DEVICE

    T offset = (max_boundary - min_boundary) / T(num_bins);
    Sequence<T> boundaries(num_bins + 1, [&](size_t k) { return k * offset + (std::rand() % jitter) + min_boundary; });

    invoke_on_all_policies<CallNumber * 4 + 2>()(test_histogram_range_bins(), in.begin(), in.end(),
                                                        boundaries.begin(), boundaries.end(), expected.begin(),
                                                        expected.end(), out.begin(), out.end(), trash);
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<CallNumber * 4 + 3>()(test_histogram_range_bins(), in.cbegin(), in.cend(),
                                                        boundaries.cbegin(), boundaries.cend(), expected.begin(),
                                                        expected.end(), out.begin(), out.end(), trash);
#    endif // !ONEDPL_FPGA_DEVICE

}

template <::std::size_t CallNumber, typename T, typename Size>
void
test_histogram(T min_boundary, T max_boundary, T overflow, Size jitter, Size trash)
{
    for (Size bin_size = 4; bin_size <= 20000; bin_size = Size(3.1415 * bin_size))
    {
        for (Size n = 0; n <= 100000; n = n <= 16 ? n + 1 : Size(3.1415 * n))
        {
            test_range_and_even_histogram<CallNumber>(n, min_boundary, max_boundary, overflow, jitter, bin_size, trash);
        }
    }
}

int
main()
{
    test_histogram<0, float, uint32_t>(10000.0f, 110000.0f, 300.0f, uint32_t(50), uint32_t(99999));

#if !ONEDPL_FPGA_DEVICE
    test_histogram<1, std::int32_t, uint64_t>(-50000, 50000, 10000, uint64_t(5), uint64_t(99999));
#endif //!ONEDPL_FPGA_DEVICE

    return done();
}
