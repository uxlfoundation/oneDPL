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

#ifndef _SORT_BY_KEY_COMMON_H
#define _SORT_BY_KEY_COMMON_H

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"
#if TEST_DPCPP_BACKEND_PRESENT
#   include "support/sycl_alloc_utils.h"
#endif

#include <tuple> // std::tie
#include <random> // std::default_random_engine, uniform_real_distribution
#include <vector> // std::vector
#include <iterator> // std::distance
#include <algorithm> // std::generate, std::remove_if, std::stable_sort, std::is_sorted, std::transform, std::shuffle
#include <functional> // std::less, std::greater
#include <type_traits> // std::is_same
#include <unordered_map> // std::unordered_map

struct StableSortTag{};
struct UnstableSortTag{};

struct Particle
{
    float mass = 0;
    float velocity = 0;
    float coordinates[3] = {0, 0, 0};
    using energy_type = float;
    energy_type energy() const
    {
        // kinetic energy of the particle
        return 0.5 * mass * velocity * velocity;
    }
    bool operator==(const Particle& other) const
    {
        return std::tie(coordinates[0], coordinates[1], coordinates[2]) ==
               std::tie(other.coordinates[0], other.coordinates[1], other.coordinates[2]);
    }
};

// Arbitrary non-power-of-2 element counts for better coverage:
// small size often uses single-work-group specialization for device policy
// large size often uses multi-work-group specialization
constexpr std::size_t large_size = 81207;
constexpr std::size_t small_size = 4134;

template<typename KeyIt, typename ValIt, typename Size>
Size
remove_duplicates_by_key(const KeyIt& keys_begin, const ValIt& vals_begin, Size n)
{
    using KeyT = typename std::iterator_traits<KeyIt>::value_type;
    std::unordered_map<KeyT, Size> histogram;
    std::for_each(keys_begin, keys_begin + n, [&histogram](const auto& key) { ++histogram[key]; });
    auto first = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);
    auto has_duplicates = [&histogram](const auto& pair) {return histogram[std::get<0>(pair)] > 1; };
    auto new_last = std::remove_if(first, first + n, has_duplicates);
    return std::distance(first, new_last);
}

template<typename KeyIt, typename ValIt, typename Size>
void
generate_data(KeyIt keys_begin, ValIt vals_begin, Size keys_n, Size vals_n, std::uint32_t seed)
{
    using KeyT = typename std::iterator_traits<KeyIt>::value_type;
    using ValT = typename std::iterator_traits<ValIt>::value_type;
    if constexpr (std::is_same_v<ValT, Particle>)
    {
        static_assert(std::is_same_v<KeyT, Particle::energy_type>);
        std::default_random_engine gen{seed};
        std::uniform_real_distribution<float> mass_dist(0.0, 1000.0);
        std::uniform_real_distribution<float> velocity_dist(0.0, 1.0);
        std::uniform_real_distribution<float> coord_dist(0.0, 1.0);
        std::generate(vals_begin, vals_begin + vals_n, [&gen, &mass_dist, &velocity_dist, &coord_dist]() {
            return Particle{mass_dist(gen), velocity_dist(gen), {coord_dist(gen), coord_dist(gen), coord_dist(gen)}};
        });
        std::transform(vals_begin, vals_begin + keys_n, keys_begin, [](const auto& particle) { return particle.energy(); });
    }
    else
    {
        TestUtils::generate_arithmetic_data(keys_begin, keys_n, seed);
        TestUtils::generate_arithmetic_data(vals_begin, vals_n, seed + 1);
        // avoid having value duplicates at the same locations as key duplicates for a more robust stability check
        std::shuffle(vals_begin, vals_begin + keys_n, std::default_random_engine(seed));
    }
}

template<typename Policy, typename KeyIt, typename ValIt, typename Size, typename... Compare>
void
call_sort(Policy&& policy, KeyIt keys_begin, ValIt vals_begin, Size n, StableSortTag, Compare... compare)
{
    oneapi::dpl::stable_sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, compare...);
}

template<typename Policy, typename KeyIt, typename ValIt, typename Size, typename... Compare>
void
call_sort(Policy&& policy, KeyIt keys_begin, ValIt vals_begin, Size n, UnstableSortTag, Compare... compare)
{
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_begin + n, vals_begin, compare...);
}

template<typename KeyIt, typename ValIt, typename Size, typename Compare = std::less<>>
void
call_reference_sort(KeyIt ref_keys_begin, ValIt ref_vals_begin, Size n, Compare compare = {})
{
    auto first = oneapi::dpl::make_zip_iterator(ref_keys_begin, ref_vals_begin);
    std::stable_sort(first, first + n, [compare](const auto& lhs, const auto& rhs) {
        return compare(std::get<0>(lhs), std::get<0>(rhs));
    });
}

template<typename KeyIt, typename ValIt, typename KeysOrigIt, typename ValsOrigIt, typename Size, typename... Compare>
void
check_sort(const KeyIt& keys_begin, const ValIt& vals_begin,
           const KeysOrigIt& keys_orig_begin, const ValsOrigIt& vals_orig_begin,
           Size keys_n, Size vals_n, StableSortTag, Compare... compare)
{
    using KeyT = typename std::iterator_traits<KeyIt>::value_type;
    using ValT = typename std::iterator_traits<ValIt>::value_type;
    std::vector<KeyT> keys_expected(keys_orig_begin, keys_orig_begin + keys_n);
    std::vector<ValT> vals_expected(vals_orig_begin, vals_orig_begin + vals_n);
    call_reference_sort(keys_expected.begin(), vals_expected.begin(), keys_n, compare...);
    EXPECT_EQ_N(keys_expected.begin(), keys_begin, keys_n, "wrong result stable sort: keys");
    // TODO: investigate how to make sure that the values are reordered together with their keys
    // currently, the check does not guarantee it,
    // but the probability of missing it very low due to having random values and further shuffling
    EXPECT_EQ_N(vals_expected.begin(), vals_begin, keys_n, "wrong result stable sort: values");
    EXPECT_EQ_N(vals_expected.begin() + keys_n, vals_begin + keys_n, vals_n - keys_n,
                "wrong result stable sort: remaining values should not be touched");
}

template<typename KeyIt, typename ValIt, typename KeysOrigIt, typename ValsOrigIt, typename Size, typename... Compare>
void
check_sort(const KeyIt& keys_begin, const ValIt& vals_begin,
           const KeysOrigIt& keys_orig_begin, const ValsOrigIt& vals_orig_begin,
           Size keys_n, Size vals_n, UnstableSortTag, Compare... compare)
{
    using KeyT = typename std::iterator_traits<KeyIt>::value_type;
    using ValT = typename std::iterator_traits<ValIt>::value_type;
    std::vector<KeyT> keys_expected(keys_orig_begin, keys_orig_begin + keys_n);
    std::vector<ValT> vals_expected(vals_orig_begin, vals_orig_begin + vals_n);
    call_reference_sort(keys_expected.begin(), vals_expected.begin(), keys_n, compare...);
    EXPECT_EQ_N(keys_expected.begin(), keys_begin, keys_n, "wrong result non-stable sort: keys");
    EXPECT_EQ_N(vals_expected.begin() + keys_n, vals_begin + keys_n, vals_n - keys_n,
                "wrong result non-stable sort: remaining values should not be touched");

    // Remove key-value pairs with duplicate keys
    // The resulting value sequence is deterministic even for non-stable sort
    auto expected_unique_n = remove_duplicates_by_key(keys_expected.begin(), vals_expected.begin(), keys_n);
    std::vector<KeyT> keys(keys_begin, keys_begin + keys_n);
    std::vector<ValT> vals(vals_begin, vals_begin + vals_n);
    auto unique_n = remove_duplicates_by_key(keys.begin(), vals.begin(), keys_n);
    EXPECT_EQ(expected_unique_n, unique_n, "the number of unique keys does not much the expected value");
    EXPECT_EQ_N(vals_expected.begin(), vals.begin(), expected_unique_n, "wrong result non-stable sort: values");
}

template<typename KeyT, typename ValT, typename Size, typename Policy, typename StabilityTag, typename... Compare>
void
test_with_std_policy(Policy&& policy, Size n, StabilityTag stability_tag, Compare... compare)
{
    Size keys_n = n;
    Size vals_n = n + 5; // to test that the remaining values are not touched
    std::vector<KeyT> origin_keys(keys_n);
    std::vector<ValT> origin_vals(vals_n);
    generate_data(origin_keys.data(), origin_vals.data(), keys_n, vals_n, 42);
    std::vector<KeyT> keys(origin_keys);
    std::vector<ValT> vals(origin_vals);

    call_sort(policy, keys.begin(), vals.begin(), keys_n, stability_tag, compare...);
    check_sort(keys.begin(), vals.begin(), origin_keys.begin(), origin_vals.begin(), keys_n, vals_n, stability_tag, compare...);
}

#if TEST_DPCPP_BACKEND_PRESENT
template <typename KeyT, typename ValT, sycl::usm::alloc alloc_type, std::uint32_t KernelNameID,
          typename Size, typename StabilityTag, typename ...Compare>
void
test_with_usm(sycl::queue& q, Size n, StabilityTag stability_tag, Compare... compare)
{
    Size keys_n = n;
    Size vals_n = n + 5; // to test that the remaining values are not touched

    std::vector<KeyT> origin_keys(keys_n);
    std::vector<ValT> origin_vals(vals_n);
    generate_data(origin_keys.data(), origin_vals.data(), keys_n, vals_n, 42);
    std::vector<KeyT> keys(origin_keys);
    std::vector<ValT> vals(origin_vals);
    TestUtils::usm_data_transfer<alloc_type, KeyT> keys_device(q, keys.begin(), keys.end());
    TestUtils::usm_data_transfer<alloc_type, ValT> vals_device(q, vals.begin(), vals.end());

    // calling sort
    auto policy = TestUtils::make_device_policy<TestUtils::unique_kernel_name<class USM, KernelNameID>>(q);
    call_sort(policy, keys_device.get_data(), vals_device.get_data(), keys_n, stability_tag, compare...);

    // checking results
    keys_device.retrieve_data(keys.begin());
    vals_device.retrieve_data(vals.begin());
   // sort_by_key with device policy guarantees stability, hence StableSortTag{} is passed
    check_sort(keys.begin(), vals.begin(), origin_keys.begin(), origin_vals.begin(), keys_n, vals_n, StableSortTag{}, compare...);
}

template <typename KeyT, typename ValT, std::uint32_t KernelNameID,
          typename Size, typename StabilityTag, typename... Compare>
void
test_with_buffers(sycl::queue& q, Size n, StabilityTag stability_tag, Compare... compare)
{
    std::vector<KeyT> origin_keys(n);
    std::vector<ValT> origin_vals(n);
    generate_data(origin_keys.data(), origin_vals.data(), n, n, 42);
    std::vector<KeyT> keys(origin_keys);
    std::vector<ValT> vals(origin_vals);
    {
        sycl::buffer<KeyT> keys_device(keys.data(), n);
        sycl::buffer<ValT> vals_device(vals.data(), n);
        auto policy = TestUtils::make_device_policy<TestUtils::unique_kernel_name<class Buffer, KernelNameID>>(q);
        call_sort(policy, oneapi::dpl::begin(keys_device), oneapi::dpl::begin(vals_device), n, stability_tag, compare...);
    }
   // sort_by_key with device policy guarantees stability, hence StableSortTag{} is passed
    check_sort(keys.begin(), vals.begin(), origin_keys.begin(), origin_vals.begin(), n, n, StableSortTag{}, compare...);
}

template <typename StabilityTag>
void
test_device_policy(StabilityTag stability_tag)
{
    sycl::queue q = TestUtils::get_test_queue();
    auto custom_greater = [](const auto& lhs, const auto& rhs) { return lhs > rhs; }; // Cover merge-sort from device backend

    test_with_usm<std::int16_t, float, sycl::usm::alloc::shared, 1>(q, large_size, stability_tag, std::greater{});
    test_with_usm<std::uint32_t, std::uint32_t, sycl::usm::alloc::device, 2>(q, large_size, stability_tag);
    test_with_buffers<float, float, 3>(q, small_size, stability_tag, custom_greater);
    test_with_buffers<Particle::energy_type, Particle, 4>(q, large_size, stability_tag, custom_greater);
    test_with_buffers<Particle::energy_type, Particle, 5>(q, small_size, stability_tag);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename KeyT, typename ValT, typename Size, typename StabilityTag, typename... Compare>
void
test_std_polcies(Size n, StabilityTag stability_tag, Compare... compare)
{
    test_with_std_policy<KeyT, ValT>(oneapi::dpl::execution::seq, n, stability_tag, compare...);
    test_with_std_policy<KeyT, ValT>(oneapi::dpl::execution::unseq, n, stability_tag, compare...);
    test_with_std_policy<KeyT, ValT>(oneapi::dpl::execution::par, n, stability_tag, compare...);
    test_with_std_policy<KeyT, ValT>(oneapi::dpl::execution::par_unseq, n, stability_tag, compare...);
}

template <typename StabilityTag>
void
test_all_policies(StabilityTag stability_tag)
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_device_policy(stability_tag);
#endif // TEST_DPCPP_BACKEND_PRESENT
    test_std_polcies<int, int>(large_size, stability_tag);
    test_std_polcies<std::size_t, float>(large_size, stability_tag, std::greater{});
    test_std_polcies<Particle::energy_type, Particle>(small_size, stability_tag);
}

#endif // _SORT_BY_KEY_COMMON_H
