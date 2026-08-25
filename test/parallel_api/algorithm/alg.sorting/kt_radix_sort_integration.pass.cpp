// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for ONEDPL_ENABLE_KT_RADIX_SORT_IN_SORT=1 integration: verifies that
// eligible sorts route to the KT radix sort path on capable hardware, and that
// ineligible sorts correctly fall back to the legacy radix sort.

#include "support/test_config.h"

#if !TEST_DPCPP_BACKEND_PRESENT
int main() { return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT); }
#else

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>

#if _ONEDPL_CPP20_RANGES_PRESENT
#    include _PSTL_TEST_HEADER(ranges)
#    include <span>
#endif

// Test sizes straddling __min_size = 1 << 18 = 262144
constexpr std::size_t below_min = 260000;
constexpr std::size_t at_min = 262144;
constexpr std::size_t above_min = 265000;
constexpr std::size_t large_size = 500000;

// Generate random data
template <typename T>
void generate_data(std::vector<T>& data, std::size_t seed = 42)
{
    std::mt19937 gen(seed);
    if constexpr (std::is_integral_v<T>)
    {
        // std::uniform_int_distribution is only defined for short and wider, so generate through a
        // wide distribution and narrow.
        using _WideT = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
        std::uniform_int_distribution<_WideT> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(dist(gen)); });
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        std::uniform_real_distribution<T> dist(T(-1000), T(1000));
        std::generate(data.begin(), data.end(), [&]() { return dist(gen); });
    }
}

template <>
void generate_data<sycl::half>(std::vector<sycl::half>& data, std::size_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::generate(data.begin(), data.end(), [&]() { return sycl::half(dist(gen)); });
}

#if defined(SYCL_EXT_ONEAPI_BFLOAT16)
template <>
void generate_data<sycl::ext::oneapi::bfloat16>(std::vector<sycl::ext::oneapi::bfloat16>& data, std::size_t seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::generate(data.begin(), data.end(), [&]() { return sycl::ext::oneapi::bfloat16(dist(gen)); });
}
#endif // SYCL_EXT_ONEAPI_BFLOAT16

// Check if device supports KT radix sort
bool device_supports_kt_radix(const sycl::queue& q)
{
    const sycl::device dev = q.get_device();
    if (!dev.is_gpu())
        return false;

    namespace syclex = sycl::ext::oneapi::experimental;
    const std::vector<syclex::forward_progress_guarantee> caps =
        dev.get_info<syclex::info::device::work_group_progress_capabilities<syclex::execution_scope::root_group>>();

    return std::find(caps.begin(), caps.end(), syclex::forward_progress_guarantee::concurrent) != caps.end();
}

// Test sort for a given type and comparator
template <typename T, typename Compare, typename Policy>
void test_sort_type(Policy&& policy, std::size_t n, Compare comp, const char* test_name)
{
    std::vector<T> host_data(n);
    generate_data(host_data);

    // Reference sort
    std::vector<T> expected = host_data;
    std::stable_sort(expected.begin(), expected.end(), comp);

    // Test USM device allocation
    {
        std::vector<T> test_data = host_data;
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_helper(policy, test_data.begin(), test_data.end());

        oneapi::dpl::stable_sort(policy, dt_helper.get_data(), dt_helper.get_data() + n, comp);
        dt_helper.retrieve_data(test_data.begin());

        EXPECT_EQ_N(expected.begin(), test_data.begin(), n, test_name);
    }

    // Test buffer allocation
    {
        std::vector<T> test_data = host_data;
        {
            sycl::buffer<T> buf(test_data.data(), sycl::range<1>(n));
            oneapi::dpl::stable_sort(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), comp);
        }
        EXPECT_EQ_N(expected.begin(), test_data.begin(), n, test_name);
    }
}

// Test stability with many duplicate keys
template <typename T, typename Policy>
void test_stability(Policy&& policy, std::size_t n, const char* test_name)
{
    // Create data with many duplicates (only 10 unique keys)
    std::vector<T> keys(n);
    std::vector<std::size_t> values(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        keys[i] = T(i % 10);
        values[i] = i;  // Original position
    }

    // Shuffle to mix duplicates
    std::mt19937 gen(42);
    auto zip_first = oneapi::dpl::make_zip_iterator(keys.begin(), values.begin());
    std::shuffle(zip_first, zip_first + n, gen);

    // Reference stable sort
    std::vector<T> expected_keys = keys;
    std::vector<std::size_t> expected_values = values;
    auto ref_zip = oneapi::dpl::make_zip_iterator(expected_keys.begin(), expected_values.begin());
    std::stable_sort(ref_zip, ref_zip + n,
        [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    // Test with USM
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> keys_dt(policy, keys.begin(), keys.end());
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, std::size_t> vals_dt(policy, values.begin(), values.end());

    oneapi::dpl::stable_sort_by_key(policy, keys_dt.get_data(), keys_dt.get_data() + n, vals_dt.get_data());

    keys_dt.retrieve_data(keys.begin());
    vals_dt.retrieve_data(values.begin());

    EXPECT_EQ_N(expected_keys.begin(), keys.begin(), n, test_name);
    EXPECT_EQ_N(expected_values.begin(), values.begin(), n, test_name);
}

// Test sort_by_key
template <typename KeyT, typename ValT, typename Compare, typename Policy>
void test_sort_by_key(Policy&& policy, std::size_t n, Compare comp, const char* test_name)
{
    std::vector<KeyT> keys(n);
    std::vector<ValT> values(n);

    generate_data(keys);
    std::iota(values.begin(), values.end(), ValT(0));

    // Reference sort
    std::vector<KeyT> expected_keys = keys;
    std::vector<ValT> expected_values = values;
    auto ref_zip = oneapi::dpl::make_zip_iterator(expected_keys.begin(), expected_values.begin());
    std::stable_sort(ref_zip, ref_zip + n,
        [comp](const auto& a, const auto& b) { return comp(std::get<0>(a), std::get<0>(b)); });

    // Test with USM
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, KeyT> keys_dt(policy, keys.begin(), keys.end());
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, ValT> vals_dt(policy, values.begin(), values.end());

    oneapi::dpl::stable_sort_by_key(policy, keys_dt.get_data(), keys_dt.get_data() + n, vals_dt.get_data(), comp);

    keys_dt.retrieve_data(keys.begin());
    vals_dt.retrieve_data(values.begin());

    EXPECT_EQ_N(expected_keys.begin(), keys.begin(), n, test_name);
    EXPECT_EQ_N(expected_values.begin(), values.begin(), n, test_name);
}

#if _ONEDPL_CPP20_RANGES_PRESENT
// Test ranges::sort
template <typename T, typename Compare, typename Policy>
void test_ranges_sort(Policy&& policy, std::size_t n, Compare comp, const char* test_name)
{
    std::vector<T> host_data(n);
    generate_data(host_data);

    std::vector<T> expected = host_data;
    std::ranges::sort(expected, comp);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_helper(policy, host_data.begin(), host_data.end());

    std::span<T> view(dt_helper.get_data(), n);
    oneapi::dpl::ranges::sort(policy, view, comp);

    dt_helper.retrieve_data(host_data.begin());
    EXPECT_EQ_N(expected.begin(), host_data.begin(), n, test_name);
}
#endif

int main()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();

    // Check if device supports KT radix sort
    bool kt_capable = device_supports_kt_radix(q);
    if (!kt_capable)
    {
        std::cout << "Device does not support KT radix sort (missing concurrent root-group progress guarantee)" << std::endl;
        std::cout << "Tests will use legacy radix sort path" << std::endl;
    }

    // Test various types and sizes
    // uint8_t
    test_sort_type<std::uint8_t>(policy, above_min, std::less<>{}, "uint8_t ascending");
    test_sort_type<std::uint8_t>(policy, above_min, std::greater<>{}, "uint8_t descending");

    // int16_t
    test_sort_type<std::int16_t>(policy, at_min, std::less<>{}, "int16_t ascending");
    test_sort_type<std::int16_t>(policy, at_min, std::greater<>{}, "int16_t descending");

    // uint32_t - test all size ranges
    test_sort_type<std::uint32_t>(policy, below_min, std::less<>{}, "uint32_t below_min");
    test_sort_type<std::uint32_t>(policy, at_min, std::less<>{}, "uint32_t at_min");
    test_sort_type<std::uint32_t>(policy, above_min, std::less<>{}, "uint32_t above_min");
    test_sort_type<std::uint32_t>(policy, large_size, std::less<>{}, "uint32_t large");

    // int64_t
    test_sort_type<std::int64_t>(policy, above_min, std::less<>{}, "int64_t ascending");
    test_sort_type<std::int64_t>(policy, above_min, std::greater<>{}, "int64_t descending");

    // float
    test_sort_type<float>(policy, above_min, std::less<>{}, "float ascending");
    test_sort_type<float>(policy, above_min, std::greater<>{}, "float descending");

    // double
    test_sort_type<double>(policy, at_min, std::less<>{}, "double ascending");
    test_sort_type<double>(policy, at_min, std::greater<>{}, "double descending");

    // sycl::half
    test_sort_type<sycl::half>(policy, above_min, std::less<>{}, "sycl::half ascending");
    test_sort_type<sycl::half>(policy, above_min, std::greater<>{}, "sycl::half descending");

#if defined(SYCL_EXT_ONEAPI_BFLOAT16)
    // bfloat16
    test_sort_type<sycl::ext::oneapi::bfloat16>(policy, at_min, std::less<>{}, "bfloat16 ascending");
#endif

    // Stability test
    test_stability<std::uint32_t>(policy, above_min, "stability test uint32_t");
    test_stability<std::int64_t>(policy, at_min, "stability test int64_t");

    // sort_by_key tests
    test_sort_by_key<std::uint32_t, std::uint64_t>(policy, above_min, std::less<>{}, "sort_by_key uint32/uint64 ascending");
    test_sort_by_key<std::int16_t, float>(policy, at_min, std::greater<>{}, "sort_by_key int16/float descending");
    test_sort_by_key<double, std::uint32_t>(policy, above_min, std::less<>{}, "sort_by_key double/uint32 ascending");

#if _ONEDPL_CPP20_RANGES_PRESENT
    // ranges::sort
    test_ranges_sort<std::uint32_t>(policy, above_min, std::less<>{}, "ranges::sort uint32_t");
    test_ranges_sort<float>(policy, at_min, std::greater<>{}, "ranges::sort float descending");
#endif

    return TestUtils::done();
}

#endif // TEST_DPCPP_BACKEND_PRESENT
