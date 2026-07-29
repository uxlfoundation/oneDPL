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
    std::size_t n = TestUtils::test_queue_is_cpu() ? TestUtils::get_scan_test_max_n() :
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

#if TEST_DPCPP_BACKEND_PRESENT && defined(SYCL_IMPLEMENTATION_INTEL)
// Immitate segmented scan to avoid too large precision errors
void test_with_bfloat16(std::size_t n)
{
    using T = sycl::ext::oneapi::bfloat16;
    // Truncate n to be a multiple of num_segments for simplicity
    const std::size_t num_segments = n / 100;
    const std::size_t row_n = n / num_segments;
    const std::size_t total_n = row_n * num_segments;

    auto q = TestUtils::get_test_queue();

    std::vector<T> expected(n);
    T* in = sycl::malloc_shared<T>(n, q);
    T* out = sycl::malloc_shared<T>(n, q);

    // Initialize and compute expected results
    for (std::size_t seg = 0; seg < num_segments; ++seg)
    {
        T prefix = 0;
        for (std::size_t i = 0; i < row_n; ++i)
        {
            T value = static_cast<T>(i % 3 + 1); // 1, 2, 3, 1, 2, 3, ...
            in[seg * row_n + i] = value;
            prefix += value;
            expected[seg * row_n + i] = prefix;
        }
    }

    auto policy = oneapi::dpl::execution::make_device_policy(q);
    for (std::size_t seg = 0; seg < num_segments; ++seg)
    {
        std::size_t offset = seg * row_n;
        oneapi::dpl::inclusive_scan(policy, in + offset, in + offset + row_n, out + offset, std::plus<T>());
    }

    // Validation
    // EXPECT_EQ* utilities cannot be used because their precision requirements are too strict for bfloat16
    auto approx_equal = [](float act, float exp) {return std::fabs(act - exp) <= 0.01 * std::fabs(exp); }; // 1% tolerance
    for (std::size_t i = 0; i < total_n; ++i)
    {
        if (!approx_equal(static_cast<float>(out[i]), static_cast<float>(expected[i])))
        {
            std::string message = "inclusive_scan failed for bfloat16 at index " + std::to_string(i) +
                                  ": expected " + std::to_string(static_cast<float>(expected[i])) +
                                  ", got " + std::to_string(static_cast<float>(out[i]));
            EXPECT_TRUE(false, message.c_str());
        }
    }
    sycl::free(in, q);
    sycl::free(out, q);
}
#endif

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

#if TEST_DPCPP_BACKEND_PRESENT && defined(SYCL_IMPLEMENTATION_INTEL)
    test_with_bfloat16(1000);
    test_with_bfloat16(35000);
#endif

    return done();
}
