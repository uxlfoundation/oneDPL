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
#include <cstdint>
#include "support/test_config.h"

#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if TEST_DPCPP_BACKEND_PRESENT
template <typename... _Args>
auto
std_set(oneapi::dpl::unseq_backend::_IntersectionTag<std::true_type>, _Args&&... args)
{
    return std::set_intersection(std::forward<_Args>(args)...);
}
template <typename... _Args>
auto
std_set(oneapi::dpl::unseq_backend::_DifferenceTag<std::true_type>, _Args&&... args)
{
    return std::set_difference(std::forward<_Args>(args)...);
}
template <typename... _Args>
auto
std_set(oneapi::dpl::unseq_backend::_SymmetricDifferenceTag<std::true_type>, _Args&&... args)
{
    return std::set_symmetric_difference(std::forward<_Args>(args)...);
}
template <typename... _Args>
auto
std_set(oneapi::dpl::unseq_backend::_UnionTag<std::true_type>, _Args&&... args)
{
    return std::set_union(std::forward<_Args>(args)...);
}

template <typename SetTag>
bool
test_serial_set_op_count(SetTag set_tag)
{
    // Test for set operation with serial policy
    std::cout << "Test for set operation count only" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {3, 4, 5, 6, 7};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__noop_temp_data __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }
    return true;
}

template <typename SetTag>
bool
test_serial_set_op_count_and_write(SetTag set_tag)
{
    // Test for set operation with serial policy
    std::cout << "Test for set operation with count and write" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {3, 4, 5, 6, 7};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__temp_data_array<10, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i].__v != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}
template <typename SetTag>
bool
test_serial_set_op_count_and_write2(SetTag set_tag)
{
    // Test for set operation with serial policy
    std::cout << "Test for set operation with count and write" << std::endl;
    std::vector<int> v1 = {1};
    std::vector<int> v2 = {1, 1};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__temp_data_array<10, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i].__v != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename SetTag>
bool
test_serial_set_op_count_and_write_limited(SetTag set_tag)
{
    std::cout << "Test for set operation with count and write limited" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v2 = {3, 4, 4, 4, 5, 6, 7, 11, 12, 13, 14, 15};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__temp_data_array<11, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 4, 2, 10, __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin() + 4, v1.begin() + 4 + 5, v2.begin() + 2, v2.begin() + 2 + 5, v3.begin(),
                       std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i].__v != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename SetTag>
bool
test_serial_set_op_count_and_write2_large_setA(SetTag set_tag)
{
    // Test for set operation with serial policy
    std::cout << "Test for set operation with count and write" << std::endl;
    std::vector<int> v1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2};
    std::vector<int> v2 = {1};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__temp_data_array<15, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i].__v != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename SetTag>
bool
test_serial_set_op_count_and_write2_large_setB(SetTag set_tag)
{
    // Test for set operation with serial policy
    std::cout << "Test for set operation with count and write" << std::endl;
    std::vector<int> v1 = {1};
    std::vector<int> v2 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__temp_data_array<15, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__get_set_operation<SetTag> __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std_set(set_tag, v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch, expected " << res - v3.begin() << " got " << count << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i].__v != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename Rng>
bool
test_right_biased_lower_bound_impl(Rng __rng, std::size_t __location, std::less<typename Rng::value_type> __comp)
{
    auto expected_res = std::lower_bound(__rng.begin(), __rng.begin() + __location, __rng[__location], __comp);
    auto res = oneapi::dpl::__internal::__biased_lower_bound</*last_biased=*/true>(
        __rng.begin(), std::size_t{0}, __location, __rng[__location], __comp);

    if (res != expected_res - __rng.begin())
    {
        std::cout << "Failed: lower_bound mismatch" << std::endl;
        return false;
    }
    return true;
}

bool
test_right_biased_lower_bound()
{
    std::cout << "Test for right biased lower bound" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10};
    bool ret = true;
    for (std::size_t i = 0; i < v1.size(); ++i)
        ret &= test_right_biased_lower_bound_impl(v1, i, std::less<int>());

    std::vector<int> v2 = {3, 4, 4, 4, 5, 6, 7, 11, 12, 13, 14, 15};
    for (std::size_t i = 0; i < v2.size(); ++i)
        ret &= test_right_biased_lower_bound_impl(v2, i, std::less<int>());

    return ret;
}

// Find the merge path intersection for a diagonal as ground truth by walking the path
template <typename _Rng1, typename _Rng2, typename _Comp>
auto
find_merge_path_intersection(_Rng1 __rng1, _Rng2 __rng2, std::size_t __diag_idx, _Comp __comp)
{
    std::size_t idx1 = 0;
    std::size_t idx2 = 0;

    //take __diag_idx steps
    for (std::size_t i = 0; i < __diag_idx; ++i)
    {
        if (idx1 < __rng1.size() && idx2 < __rng2.size())
        {
            if (__comp(__rng2[idx2], __rng1[idx1]))
                ++idx2;
            else
                ++idx1;
        }
        else if (idx1 < __rng1.size())
        {
            ++idx1;
        }
        else
        {
            ++idx2;
        }
    }
    return std::make_tuple(idx1, idx2);
}

// Find the balanced path intersection for a diagonal as ground truth by walking the path
template <typename _Rng1, typename _Rng2, typename _Comp>
auto
find_balanced_path_intersection(_Rng1 __rng1, _Rng2 __rng2, std::size_t __diag_idx, _Comp __comp)
{
    std::size_t idx1 = 0;
    std::size_t idx2 = 0;
    bool star = false;

    bool next_matched_ele_from_rng1 = true;
    //take __diag_idx steps
    for (std::size_t i = 0; i < __diag_idx; ++i)
    {
        if (idx1 < __rng1.size() && idx2 < __rng2.size())
        {
            if (__comp(__rng2[idx2], __rng1[idx1]))
            {
                next_matched_ele_from_rng1 = true;
                ++idx2;
            }
            else if (__comp(__rng1[idx1], __rng2[idx2]))
            {
                next_matched_ele_from_rng1 = true;
                ++idx1;
            }
            else // they match
            {
                if (next_matched_ele_from_rng1)
                {
                    ++idx1;
                }
                else
                {
                    ++idx2;
                }
                next_matched_ele_from_rng1 = !next_matched_ele_from_rng1;
            }
        }
        else if (idx1 < __rng1.size())
        {
            next_matched_ele_from_rng1 = true;
            ++idx1;
        }
        else
        {
            next_matched_ele_from_rng1 = true;
            ++idx2;
        }
    }
    if (!next_matched_ele_from_rng1)
    {
        idx2++;
        star = true;
    }
    return std::make_tuple(idx1, idx2, star);
}

template <typename _Rng1, typename _Rng2, typename _Comp>
bool
test_find_balanced_path_impl(_Rng1 __rng1, _Rng2 __rng2, _Comp __comp)
{
    for (std::size_t diag_idx = 0; diag_idx < __rng1.size() + __rng2.size(); ++diag_idx)
    {
        auto [merge_path_idx1, merge_path_idx2] = find_merge_path_intersection(__rng1, __rng2, diag_idx, __comp);
        auto [expected_balanced_path_idx1, expected_balanced_path_idx2, expected_star] =
            find_balanced_path_intersection(__rng1, __rng2, diag_idx, __comp);
        auto [balanced_path_idx1, balanced_path_idx2, star] =
            oneapi::dpl::__par_backend_hetero::__find_balanced_path_start_point(__rng1, __rng2, merge_path_idx1,
                                                                                merge_path_idx2, __comp);
        if (balanced_path_idx1 != expected_balanced_path_idx1 || balanced_path_idx2 != expected_balanced_path_idx2 ||
            star != expected_star)
        {
            std::cout << "rng1[" << __rng1.size() << "]: ";
            for (auto i : __rng1)
                std::cout << i << " ";
            std::cout << std::endl;
            std::cout << "rng2[" << __rng2.size() << "]: ";
            for (auto i : __rng2)
                std::cout << i << " ";
            std::cout << std::endl;

            std::cout << "Failed: balanced path mismatch on diagonal " << diag_idx << " of "
                      << __rng1.size() + __rng2.size() << std::endl;
            std::cout << " Merge Path: " << merge_path_idx1 << " " << merge_path_idx2 << std::endl;
            std::cout << "Expected: " << expected_balanced_path_idx1 << " " << expected_balanced_path_idx2 << " "
                      << expected_star << std::endl;
            std::cout << "Actual: " << balanced_path_idx1 << " " << balanced_path_idx2 << " " << star << std::endl;
            return false;
        }
    }
    return true;
}

bool
test_find_balanced_path()
{
    std::cout << "Test for find balanced path" << std::endl;
    //                     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    std::vector<int> v1 = {1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9};
    //                     0  1  2  3  4  5  6  7  8
    std::vector<int> v2 = {3, 4, 4, 4, 5, 5, 5, 6, 7};
    //                     0  1  2  3  4  5  6  7  8  9 10 11
    std::vector<int> v3 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4};
    //                     0  1  2  3  4  5  6  7  8
    std::vector<int> v4 = {5, 7, 7, 8, 9, 9, 9, 9, 9};
    bool ret = test_find_balanced_path_impl(v1, v2, std::less<int>());
    ret &= test_find_balanced_path_impl(v2, v1, std::less<int>());
    ret &= test_find_balanced_path_impl(v3, v4, std::less<int>());
    ret &= test_find_balanced_path_impl(v4, v3, std::less<int>());
    ret &= test_find_balanced_path_impl(v1, v4, std::less<int>());
    ret &= test_find_balanced_path_impl(v4, v1, std::less<int>());
    ret &= test_find_balanced_path_impl(v1, v3, std::less<int>());
    ret &= test_find_balanced_path_impl(v3, v1, std::less<int>());
    ret &= test_find_balanced_path_impl(v2, v4, std::less<int>());
    ret &= test_find_balanced_path_impl(v4, v2, std::less<int>());
    ret &= test_find_balanced_path_impl(v2, v3, std::less<int>());
    ret &= test_find_balanced_path_impl(v3, v2, std::less<int>());

    return ret;
}

template <typename SetTag>
void
test_variety_of_combinations_of_setops(SetTag set_tag)
{
    EXPECT_TRUE(test_serial_set_op_count(set_tag), "test for serial set_intersection operation returning count only");
    EXPECT_TRUE(test_serial_set_op_count_and_write(set_tag), "test for serial set_intersection operation");
    EXPECT_TRUE(test_serial_set_op_count_and_write2(set_tag), "test for serial set_intersection operation2");

// Test for MS STL, serial set algorithms are returning wrong count for certain inputs
    EXPECT_TRUE(test_serial_set_op_count_and_write2_large_setA(set_tag),
                "test for serial set_intersection operation2 large SetA");
    EXPECT_TRUE(test_serial_set_op_count_and_write2_large_setB(set_tag),
                "test for serial set_intersection operation2 large SetB");
    EXPECT_TRUE(test_serial_set_op_count_and_write_limited(set_tag),
                "test for serial set_intersection operation limited");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    std::cout << "Test intersection" << std::endl;
    test_variety_of_combinations_of_setops(oneapi::dpl::unseq_backend::_IntersectionTag<std::true_type>{});
    std::cout << "Test difference" << std::endl;
    test_variety_of_combinations_of_setops(oneapi::dpl::unseq_backend::_DifferenceTag<std::true_type>{});
    std::cout << "Test union" << std::endl;
    test_variety_of_combinations_of_setops(oneapi::dpl::unseq_backend::_UnionTag<std::true_type>{});
    std::cout << "Test symmetric diff" << std::endl;
    test_variety_of_combinations_of_setops(oneapi::dpl::unseq_backend::_SymmetricDifferenceTag<std::true_type>{});
    EXPECT_TRUE(test_right_biased_lower_bound(), "test for right biased lower bound");
    EXPECT_TRUE(test_find_balanced_path(), "test for find balanced path");
#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
