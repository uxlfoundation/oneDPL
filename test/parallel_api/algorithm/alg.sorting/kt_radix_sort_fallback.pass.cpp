// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that ineligible cases correctly fall back to the legacy radix sort or
// merge sort when ONEDPL_ENABLE_KT_RADIX_SORT_IN_SORT=1. These tests verify
// that the eligibility traits correctly reject cases that KT cannot handle.

#include "support/test_config.h"

#if !TEST_DPCPP_BACKEND_PRESENT
int main() { return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT); }
#else

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

#include <vector>
#include <algorithm>
#include <cstdint>

// Custom non-monotonic comparator (should force fallback to merge sort)
struct CustomCompare
{
    bool operator()(int a, int b) const
    {
        // Sort by absolute value
        return std::abs(a) < std::abs(b);
    }
};

// Non-arithmetic type (should force fallback to merge sort)
struct NonArithmetic
{
    int value;
    int id;  // Use int instead of std::string to be device-copyable

    bool operator<(const NonArithmetic& other) const
    {
        return value < other.value;
    }

    bool operator==(const NonArithmetic& other) const
    {
        return value == other.value && id == other.id;
    }
};

// Test with a custom comparator (not std::less/std::greater)
template <typename Policy>
void test_custom_comparator(Policy&& policy)
{
    constexpr std::size_t n = 300000; // Above __min_size
    std::vector<int> data(n);

    // Generate data with positive and negative values
    for (std::size_t i = 0; i < n; ++i)
        data[i] = static_cast<int>(i % 100) - 50;

    // Reference sort
    std::vector<int> expected = data;
    std::stable_sort(expected.begin(), expected.end(), CustomCompare{});

    // Test with USM
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> dt(policy, data.begin(), data.end());

    oneapi::dpl::stable_sort(policy, dt.get_data(), dt.get_data() + n, CustomCompare{});
    dt.retrieve_data(data.begin());

    EXPECT_EQ_N(expected.begin(), data.begin(), n, "custom comparator fallback");
}

// Test with non-arithmetic type (should use merge sort)
template <typename Policy>
void test_non_arithmetic_type(Policy&& policy)
{
    constexpr std::size_t n = 300000; // Above __min_size
    std::vector<NonArithmetic> data(n);

    for (std::size_t i = 0; i < n; ++i)
        data[i] = NonArithmetic{static_cast<int>(n - i), static_cast<int>(i)};

    // Reference sort
    std::vector<NonArithmetic> expected = data;
    std::stable_sort(expected.begin(), expected.end());

    // Test with buffer
    {
        sycl::buffer<NonArithmetic> buf(data.data(), sycl::range<1>(n));
        oneapi::dpl::stable_sort(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf));
    }
    EXPECT_EQ_N(expected.begin(), data.begin(), n, "non-arithmetic type fallback");
}

// Test with permutation_iterator (should use legacy radix sort)
template <typename Policy>
void test_permutation_iterator(Policy&& policy)
{
    constexpr std::size_t n = 300000; // Above __min_size
    std::vector<std::uint32_t> indices(n);
    std::vector<std::uint32_t> values(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        indices[i] = static_cast<std::uint32_t>(n - i - 1); // Reverse order
        values[i] = static_cast<std::uint32_t>(i);
    }

    // Create reference using direct access
    std::vector<std::uint32_t> expected(n);
    for (std::size_t i = 0; i < n; ++i)
        expected[i] = values[indices[i]];
    std::sort(expected.begin(), expected.end());

    // Test with buffer
    std::vector<std::uint32_t> result(n);
    {
        sycl::buffer<std::uint32_t> idx_buf(indices.data(), sycl::range<1>(n));
        sycl::buffer<std::uint32_t> val_buf(values.data(), sycl::range<1>(n));

        auto perm_iter = oneapi::dpl::make_permutation_iterator(oneapi::dpl::begin(val_buf), oneapi::dpl::begin(idx_buf));

        // Note: transform_iterator over read-only projections is known to fail in the legacy
        // radix sort (19 errors, pre-existing). permutation_iterator should at least compile.
        // We test the compile path here; runtime correctness depends on legacy path behavior.

        // This will attempt to sort through the permutation_iterator
        // It should compile and delegate to legacy radix sort
        oneapi::dpl::stable_sort(policy, perm_iter, perm_iter + n);
    }
    // Read back results
    for (std::size_t i = 0; i < n; ++i)
        result[i] = values[indices[i]];

    EXPECT_EQ_N(expected.begin(), result.begin(), n, "permutation_iterator fallback");
}

// Test with size below __min_size (should use legacy radix sort)
template <typename Policy>
void test_below_min_size(Policy&& policy)
{
    constexpr std::size_t n = 10000; // Well below __min_size = 262144
    std::vector<std::uint32_t> data(n);

    for (std::size_t i = 0; i < n; ++i)
        data[i] = static_cast<std::uint32_t>(n - i);

    // Reference sort
    std::vector<std::uint32_t> expected = data;
    std::stable_sort(expected.begin(), expected.end());

    // Test with USM
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, std::uint32_t> dt(policy, data.begin(), data.end());

    oneapi::dpl::stable_sort(policy, dt.get_data(), dt.get_data() + n);
    dt.retrieve_data(data.begin());

    EXPECT_EQ_N(expected.begin(), data.begin(), n, "below min_size fallback");
}

int main()
{
    auto policy = TestUtils::get_dpcpp_test_policy();

    test_custom_comparator(policy);
    test_non_arithmetic_type(policy);
    test_permutation_iterator(policy);
    test_below_min_size(policy);

    return TestUtils::done();
}

#endif // TEST_DPCPP_BACKEND_PRESENT
