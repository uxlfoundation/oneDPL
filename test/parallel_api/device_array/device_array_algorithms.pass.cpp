// -*- C++ -*-
//===-- device_array_algorithms.pass.cpp ---------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Integration of oneapi::dpl::experimental::device_array with oneDPL algorithms, with raw SYCL
// kernels, and with the range algorithms.

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/experimental/device_array>
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)
#endif

#include "support/utils.h"

// device_array is only available where oneapi::dpl::span is, which under C++17 requires the SYCL
// implementation to provide sycl::span.
#if TEST_DPCPP_BACKEND_PRESENT && TEST_SPAN_PRESENT
#    define TEST_DEVICE_ARRAY_PRESENT 1
#else
#    define TEST_DEVICE_ARRAY_PRESENT 0
#endif

#if TEST_DEVICE_ARRAY_PRESENT
#include "support/utils_sycl.h"
#include "support/utils_invoke.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace
{

template <typename _Tp>
using device_array = oneapi::dpl::experimental::device_array<_Tp>;

// Kernel names for the raw parallel_for submissions, so the test does not depend on unnamed lambda
// support.
class span_capture_kernel;
class pointer_capture_kernel;

std::vector<int>
shuffled_host(std::size_t __n)
{
    std::vector<int> __v(__n);
    // deterministic unsorted sequence
    for (std::size_t __i = 0; __i < __n; ++__i)
        __v[__i] = int((__i * 7919 + 13) % __n);
    return __v;
}

void
test_sort(sycl::queue __q)
{
    const std::size_t __n = 4096;
    const std::vector<int> __host = shuffled_host(__n);

    device_array<int> __d(__host, __q);

    auto __policy = oneapi::dpl::execution::make_device_policy(__q);
    oneapi::dpl::sort(CLONE_TEST_POLICY_IDX(__policy, 0), oneapi::dpl::begin(__d), oneapi::dpl::end(__d));

    std::vector<int> __expected = __host;
    std::sort(__expected.begin(), __expected.end());
    EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "sort over oneapi::dpl::begin/end of a device_array");
}

// transform writing into a second container, and reduce as a read-only path.
void
test_transform_and_reduce(sycl::queue __q)
{
    const std::size_t __n = 2048;
    std::vector<int> __host(__n);
    std::iota(__host.begin(), __host.end(), 1);

    device_array<int> __in(__host, __q);
    device_array<int> __out(__n, 0, __q);

    auto __policy = oneapi::dpl::execution::make_device_policy(__q);
    oneapi::dpl::transform(CLONE_TEST_POLICY_IDX(__policy, 1), oneapi::dpl::begin(__in), oneapi::dpl::end(__in),
                           oneapi::dpl::begin(__out), [](int __x) { return __x * 3; });

    std::vector<int> __expected(__n);
    std::transform(__host.begin(), __host.end(), __expected.begin(), [](int __x) { return __x * 3; });
    EXPECT_EQ_RANGES(__expected, __out.to_vector(__q), "transform into a device_array");

    const std::int64_t __expected_sum =
        std::accumulate(__expected.begin(), __expected.end(), std::int64_t(0), std::plus<std::int64_t>{});
    const std::int64_t __sum = oneapi::dpl::reduce(CLONE_TEST_POLICY_IDX(__policy, 2), oneapi::dpl::begin(__out),
                                                   oneapi::dpl::end(__out), std::int64_t(0), std::plus<std::int64_t>{});
    EXPECT_EQ(__expected_sum, __sum, "reduce over a device_array");

    // The const overloads of begin/end are the read-only path.
    const device_array<int>& __const_out = __out;
    const std::int64_t __const_sum =
        oneapi::dpl::reduce(CLONE_TEST_POLICY_IDX(__policy, 3), oneapi::dpl::begin(__const_out),
                            oneapi::dpl::end(__const_out), std::int64_t(0), std::plus<std::int64_t>{});
    EXPECT_EQ(__expected_sum, __const_sum, "reduce over a const device_array");
}

// Raw SYCL kernels, capturing the span by value and then the bare pointer.
void
test_raw_kernels(sycl::queue __q)
{
    const std::size_t __n = 1024;
    std::vector<int> __host(__n);
    std::iota(__host.begin(), __host.end(), 0);

    device_array<int> __d(__host, __q);

    __q.parallel_for<span_capture_kernel>(sycl::range<1>(__n), [__s = __d.span()](sycl::id<1> __i) { __s[__i] *= 2; })
        .wait_and_throw();

    std::vector<int> __expected(__n);
    std::transform(__host.begin(), __host.end(), __expected.begin(), [](int __x) { return __x * 2; });
    EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "raw kernel capturing d.span() by value");

    __q.parallel_for<pointer_capture_kernel>(sycl::range<1>(__n),
                                             [__p = __d.span().data()](sycl::id<1> __i) { __p[__i.get(0)] += 1; })
        .wait_and_throw();

    for (int& __v : __expected)
        __v += 1;
    EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "raw kernel capturing d.span().data()");
}

// The range algorithms take the span directly. Both gates are needed: the range algorithms are
// C++20 only, and the C++17 sycl::span fallback is not a std::ranges view or borrowed range.
#    if defined(ONEDPL_HAS_RANGE_ALGORITHMS) && TEST_CPP20_SPAN_PRESENT
void
test_range_algorithms(sycl::queue __q)
{
    const std::size_t __n = 4096;
    const std::vector<int> __host = shuffled_host(__n);

    auto __policy = oneapi::dpl::execution::make_device_policy(__q);

    // sort and for_each over the whole span.
    {
        device_array<int> __d(__host, __q);
        oneapi::dpl::ranges::sort(CLONE_TEST_POLICY_IDX(__policy, 4), __d.span());

        std::vector<int> __expected = __host;
        std::sort(__expected.begin(), __expected.end());
        EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "ranges::sort over d.span()");

        oneapi::dpl::ranges::for_each(CLONE_TEST_POLICY_IDX(__policy, 5), __d.span(), [](int& __x) { __x += 10; });
        for (int& __v : __expected)
            __v += 10;
        EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "ranges::for_each over d.span()");
    }

    // sort over a subrange only; the tail beyond k must be untouched.
    {
        device_array<int> __d(__host, __q);
        const std::size_t __k = 1000;
        const std::size_t __offset = 7;
        oneapi::dpl::ranges::sort(CLONE_TEST_POLICY_IDX(__policy, 6), __d.span().subspan(__offset, __k));

        std::vector<int> __expected = __host;
        std::sort(__expected.begin() + __offset, __expected.begin() + __offset + __k);
        EXPECT_EQ_RANGES(__expected, __d.to_vector(__q), "ranges::sort over a subspan touched the tail");
    }
}
#    endif // ONEDPL_HAS_RANGE_ALGORITHMS && TEST_CPP20_SPAN_PRESENT

} // namespace
#endif // TEST_DEVICE_ARRAY_PRESENT

int
main()
{
#if TEST_DEVICE_ARRAY_PRESENT
    sycl::queue q = TestUtils::get_test_queue();

    test_sort(q);
    test_transform_and_reduce(q);
    test_raw_kernels(q);
#    if defined(ONEDPL_HAS_RANGE_ALGORITHMS) && TEST_CPP20_SPAN_PRESENT
    test_range_algorithms(q);
#    endif
#endif // TEST_DEVICE_ARRAY_PRESENT

    return TestUtils::done(TEST_DEVICE_ARRAY_PRESENT);
}
