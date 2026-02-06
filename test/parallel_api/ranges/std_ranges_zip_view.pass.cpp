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
#include <oneapi/dpl/ranges>

#if _ENABLE_STD_RANGES_TESTING
#include <vector>

void test_zip_view_base_op()
{
    namespace dpl_ranges = oneapi::dpl::ranges;

    constexpr int max_n = 100;
    std::vector<int> vec1(max_n);
    std::vector<int> vec2(max_n/2);

    auto zip_view = dpl_ranges::views::zip(vec1, vec2);

    static_assert(std::is_trivially_copyable_v<decltype(zip_view)>);

    static_assert(std::random_access_iterator<decltype(zip_view.begin())>);
    static_assert(std::sentinel_for<decltype(zip_view.end()), decltype(zip_view.begin())>);

    EXPECT_TRUE(zip_view.end() - zip_view.begin() == max_n/2,
        "Difference operation between an iterator and a sentinel (zip_view) returns a wrong result.");

    EXPECT_TRUE(zip_view[2] == *(zip_view.begin() + 2), 
        "Subscription or dereferencing operation for zip_view returns a wrong result.");

    EXPECT_TRUE(std::ranges::size(zip_view) == max_n/2, "zip_view::size method returns a wrong result.");
    EXPECT_TRUE((bool)zip_view, "zip_view::operator bool() method returns a wrong result.");

    EXPECT_TRUE(zip_view[0] == zip_view.front(), "zip_view::front method returns a wrong result.");
    EXPECT_TRUE(zip_view[zip_view.size() - 1] == zip_view.back(), "zip_view::back method returns a wrong result.");
    EXPECT_TRUE(!zip_view.empty(), "zip_view::empty() method returns a wrong result.");

    using zip_view_t = dpl_ranges::zip_view<std::ranges::iota_view<int>>;
    static_assert(std::is_trivially_copyable_v<zip_view_t>);

    auto zip_view_0 = zip_view_t();
    EXPECT_TRUE(!zip_view_0.empty(), "zip_view::empty() method returns a wrong result.");
}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    test_zip_view_base_op();

    namespace dpl_ranges = oneapi::dpl::ranges;

// Suppress warnings about array bounds in GCC, due to static analysis limitations;
// A false positive in case of std::sort call:
// https://github.com/gcc-mirror/gcc/blob/releases/gcc-13/libstdc++-v3/include/bits/stl_algo.h#L1859
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Warray-bounds"
#elif defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 6385) // array bounds (MSVC)
#endif

    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto zip_view = dpl_ranges::views::zip(data, std::views::iota(0, max_n)) | std::views::take(5);
    assert(zip_view.size() == 5);
    assert(zip_view.begin() + 5 == zip_view.end());
    std::ranges::for_each(zip_view, test_std_ranges::f_mutuable, [](auto&& val) ->decltype(auto) { return std::get<0>(val); });
    for(int i = 0; i < zip_view.size(); ++i)
        EXPECT_TRUE(std::get<0>(zip_view[i]) == i*i && std::get<1>(zip_view[i]) == i, "Wrong effect for std::ranges::for_each with zip_view.");

    test_std_ranges::call_with_host_policies(dpl_ranges::for_each, zip_view, test_std_ranges::f_mutuable,
        [](const auto& val) { return std::get<1>(val); });

    {
    int data2[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto zip_view_sort = dpl_ranges::views::zip(data2, data2);

    [[maybe_unused]] oneapi::dpl::zip_iterator<int*, int*> zip_it = zip_view_sort.begin(); //check conversion to oneapi::dpl::zip_iterator

    [[maybe_unused]] auto it1 = zip_view_sort.begin();
    [[maybe_unused]] auto it2 = zip_view_sort.end();
    assert(it2 - it1 == max_n);    
    std::sort(zip_view_sort.begin(), zip_view_sort.begin() + max_n,
        [](const auto& val1, const auto& val2) { return std::get<0>(val1) > std::get<0>(val2); });

    for(int i = 0; i < max_n; ++i)
        EXPECT_TRUE(std::get<0>(zip_view_sort[i]) == max_n - 1 - i && std::get<1>(zip_view_sort[i]) == max_n - 1 - i,
            "Wrong effect for std::sort with zip_view.");

    std::ranges::sort(zip_view_sort, std::less{}, [](auto&& val) { return std::get<0>(val); });
    for(int i = 0; i < max_n; ++i)
        EXPECT_TRUE(std::get<0>(zip_view_sort[i]) == i && std::get<1>(zip_view_sort[i]) == i,
            "Wrong effect for std::ranges::sort with zip_view.");

    static_assert(std::ranges::random_access_range<decltype(zip_view_sort)>);
    static_assert(std::random_access_iterator<decltype(zip_view_sort.begin())>);

    test_std_ranges::call_with_host_policies(dpl_ranges::sort, zip_view_sort, std::greater{},
        [](const auto& val) { return std::get<0>(val); });

    for(int i = 0; i < max_n; ++i)
        EXPECT_TRUE(std::get<0>(zip_view_sort[i]) == max_n - 1 - i && std::get<1>(zip_view_sort[i]) == max_n - 1 - i,
            "Wrong effect for oneapi::dpl::ranges::sort with zip_view.");
    }

#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop //Warray-bounds
#elif defined(_MSC_VER)
  #pragma warning(pop)
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    {
    const char* err_msg = "Wrong effect for oneapi::dpl::ranges::sort with zip_view and a device policy.";

    const int n = test_std_ranges::medium_size;
    std::vector<int> vals(n), keys(n);

    //test with random number and projection usage
    std::default_random_engine gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0, 100.0);

    std::generate(vals.begin(), vals.end(), [&] { return dist(gen); });
    std::generate(keys.begin(), keys.end(), [&] { return dist(gen); });

    std::vector<int> vals_exp(vals);
    std::vector<int> keys_exp(keys);

    auto exec = TestUtils::get_dpcpp_test_policy();
    {
        using namespace test_std_ranges;
        usm_subrange<int> cont_vals(exec, vals.data(), n);
        usm_subrange<int> cont_keys(exec, keys.data(), n);
        auto view_vals = cont_vals();
        auto view_keys = cont_keys();
        auto view_s = dpl_ranges::views::zip(view_vals, view_keys);

        //call Range based sort with a device policy
        dpl_ranges::stable_sort(exec, view_s, std::ranges::greater{}, [](const auto& a) { return std::get<1>(a);});

        //call a reference sort function
        auto first = oneapi::dpl::make_zip_iterator(vals_exp.begin(), keys_exp.begin());
        std::stable_sort(first, first + n, [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b);});
    }

    //result check
    EXPECT_EQ_N(vals_exp.begin(), vals.begin(), n, err_msg);
    EXPECT_EQ_N(keys_exp.begin(), keys.begin(), n, err_msg);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
