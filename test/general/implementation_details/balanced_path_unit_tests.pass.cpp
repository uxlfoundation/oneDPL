// -*- C++ -*-
//===-- balanced_path_unit_tests.pass.cpp -----------------------------------------------===//
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



bool test_serial_set_intersection_op_count()
{
    // Test for set_intersection operation with serial policy
    std::cout << "Test for set_intersection operation count only" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {3, 4, 5, 6, 7};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__noop_temp_data __temp_data{};
    oneapi::dpl::__par_backend_hetero::__set_intersection __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch" << std::endl;
        return false;
    }
    return true;
}

bool test_serial_set_intersection_op_count_and_write()
{
    // Test for set_intersection operation with serial policy
    std::cout << "Test for set_intersection operation with count and write" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {3, 4, 5, 6, 7};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__set_temp_data<10, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__set_intersection __set_op;
    std::uint16_t count = __set_op(v1, v2, 0, 0, v1.size() + v2.size(), __temp_data, std::less<int>());

    auto res = std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch" << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i] != v3[i])
        {
            std::cout << "Failed: data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}


bool test_serial_set_intersection_op_count_and_write_limited()
{
    // Test for set_intersection operation with serial policy
    std::cout << "Test for set_intersection operation with count and write limited" << std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v2 = {3, 4, 4, 4, 5, 6, 7, 11, 12, 13, 14, 15};
    std::vector<int> v3(v1.size() + v2.size());

    oneapi::dpl::__par_backend_hetero::__set_temp_data<10, int> __temp_data{};
    oneapi::dpl::__par_backend_hetero::__set_intersection __set_op;
    std::uint16_t count = __set_op(v1, v2, 4, 2, 10, __temp_data, std::less<int>());

    auto res = std::set_intersection(v1.begin()+4, v1.begin() + 4 + 5, v2.begin() + 2, v2.begin() + 2 + 5, v3.begin(), std::less<int>());

    if (count != res - v3.begin())
    {
        std::cout << "Failed: count mismatch" << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        if (__temp_data.__data[i] != v3[i])
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
    auto res = oneapi::dpl::__par_backend_hetero::__right_biased_lower_bound(__rng.begin(), std::size_t{0}, __location, __rng[__location], __comp);
    
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
    std::cout<<"Test for right biased lower bound"<<std::endl;
    std::vector<int> v1 = {1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10};
    bool ret = true;
    for (std::size_t i = 0; i < v1.size(); ++i)
        ret &= test_right_biased_lower_bound_impl(v1, i, std::less<int>());
    
    std::vector<int> v2 = {3, 4, 4, 4, 5, 6, 7, 11, 12, 13, 14, 15};
    for (std::size_t i = 0; i < v2.size(); ++i)
        ret &= test_right_biased_lower_bound_impl(v2, i, std::less<int>());

    return ret;
}


// template <typename _InRng1, typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare>
// std::uint16_t
// operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
//            _SizeType __num_eles_min, _TempOutput& __temp_out, _Compare __comp) const



int
main()
{

    EXPECT_TRUE(test_serial_set_intersection_op_count(), "test for serial set_intersection operation returning count only");
    EXPECT_TRUE(test_serial_set_intersection_op_count_and_write(), "test for serial set_intersection operation");
    EXPECT_TRUE(test_serial_set_intersection_op_count_and_write_limited(), "test for serial set_intersection operation limited");
    EXPECT_TRUE(test_right_biased_lower_bound(), "test for right biased lower bound");
    return TestUtils::done();
}
