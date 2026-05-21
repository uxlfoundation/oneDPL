// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

#include <cstdint>

using namespace TestUtils;

template <typename In, typename Init, typename Out>
struct test_inclusive_scan_with_plus
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1> || std::is_same_v<Iterator1, Iterator2>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, [[maybe_unused]] T init, T trash)
    {
        using namespace std;
        Iterator3 orr;
        // If the types are different, apply the init
        constexpr bool use_init = !std::is_same_v<Iterator1, Iterator2>;
        if constexpr (use_init)
        {
            inclusive_scan_serial(in_first, in_last, expected_first, std::plus<>{}, init);
            orr = inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, std::plus<>{}, init);
        }
        else
        {
            inclusive_scan_serial(in_first, in_last, expected_first);
            orr = inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first);
        }
        EXPECT_TRUE(out_last == orr, "inclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan");
        std::fill_n(out_first, n, trash);
    }
    // inclusive_scan with reverse_iterator between different iterator types results in a compilation error even if
    // the call should be valid. Please see: https://github.com/uxlfoundation/oneDPL/issues/2296
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1> && !std::is_same_v<Iterator1, Iterator2>>
    operator()(Policy&& /*exec*/, Iterator1 /*in_first*/, Iterator1 /*in_last*/, Iterator2 /*out_first*/,
               Iterator2 /*out_last*/, Iterator3 /*expected_first*/, Iterator3 /*expected_last*/, Size /*n*/,
               T /*init*/, T /*trash*/)
    {
    }
};

template <typename In, typename Init, typename Out, typename Convert>
void
test_with_plus(Init init, Out trash, Convert convert)
{
    for (size_t n = 0; n <= TestUtils::get_scan_test_max_n(); n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, convert);
        Sequence<Out> expected(n);
        Sequence<Out> out(n, [&](std::int32_t) { return trash; });

        invoke_on_all_policies<0>()(test_inclusive_scan_with_plus<In, Init, Out>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, trash);
        invoke_on_all_policies<1>()(test_inclusive_scan_with_plus<In, Init, Out>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, trash);
    }

#if TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
    // testing of large number of items may take too much time in debug mode
    unsigned long n = TestUtils::test_queue_is_cpu() ? TestUtils::get_scan_test_max_n() :
#    if PSTL_USE_DEBUG
                                                     1000000;
#    else
                                                     100000000;
#    endif

    Sequence<In> in(n, convert);
    Sequence<Out> expected(n);
    Sequence<Out> out(n, [&](std::int32_t) { return trash; });
    invoke_on_all_hetero_policies<4>()(test_inclusive_scan_with_plus<In, Init, Out>(), in.begin(), in.end(),
                                       out.begin(), out.end(), expected.begin(), expected.end(), in.size(), init,
                                       trash);
#endif // TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
}

int
main()
{
    // Since the implicit "+" forms of the scan delegate to the generic forms,
    // there's little point in using a highly restricted type, so just use double.
    test_with_plus<float64_t, float64_t, float64_t>(
        0.0, -666.0, [](std::uint32_t k) { return float64_t((k % 991 + 1) ^ (k % 997 + 2)); });
    test_with_plus<std::int32_t, std::int32_t, std::int32_t>(
        0.0, -666.0, [](std::uint32_t k) { return std::int32_t((k % 991 + 1) ^ (k % 997 + 2)); });

    // When testing from bool to uint32_t, we must give a uint32_t init type to scan over integers
    test_with_plus<bool, std::uint32_t, std::uint32_t>(0, 123456,
                                                       [](std::uint32_t k) { return std::uint32_t{k % 2 == 0}; });

    return done();
}
