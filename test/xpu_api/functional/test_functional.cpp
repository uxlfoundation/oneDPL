// -*- C++ -*-
//===-- test_functional.cpp -----------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include <vector>

#include "support/utils.h"

int main()
{

#ifdef ONEDPL_STANDARD_POLICIES_ONLY
    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 1; data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 1;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        auto t = std::copy_if( oneapi::dpl::execution::par, data.begin(), data.end(), result.begin(), oneapi::dpl::identity() );

        EXPECT_EQ(std::distance(result.begin(), t), 4, "Wrong distance");
    }

    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = 3; data[1] = -1; data[2] = -4; data[3] = 1; data[4] = -5; data[5] = -9; data[6] = 2; data[7] = 6;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        std::exclusive_scan( oneapi::dpl::execution::par, data.begin(), data.begin() + 8, result.begin(), T(0) , oneapi::dpl::minimum<T>() );

        EXPECT_EQ(result[0], 0, "#1");
        EXPECT_EQ(result[1], 0, "#1");
        EXPECT_EQ(result[2], -1, "#1");
        EXPECT_EQ(result[3], -4, "#1");
        EXPECT_EQ(result[4], -4, "#1");
        EXPECT_EQ(result[5], -5, "#1");
        EXPECT_EQ(result[6], -9, "#1");
        EXPECT_EQ(result[7], -9, "#1");
    }

    {
        using T = int;
        std::vector<T> data(8);
        std::vector<T> result(8);

        data[0] = -3; data[1] = 1; data[2] = 4; data[3] = -1; data[4] = 5; data[5] = 9; data[6] = -2; data[7] = 6;
        std::fill( oneapi::dpl::execution::par, result.begin(), result.end(), T(0) );

        std::exclusive_scan( oneapi::dpl::execution::par, data.begin(), data.begin() + 8, result.begin(), T(0) , oneapi::dpl::maximum<T>() );

        EXPECT_EQ(result[0], 0, "#1");
        EXPECT_EQ(result[1], 0, "#1");
        EXPECT_EQ(result[2], 1, "#1");
        EXPECT_EQ(result[3], 4, "#1");
        EXPECT_EQ(result[4], 4, "#1");
        EXPECT_EQ(result[5], 5, "#1");
        EXPECT_EQ(result[6], 9, "#1");
        EXPECT_EQ(result[7], 9, "#1");

    }
#else
    /* IDENTITY TEST: */
    {
        using T = int;

        // create buffer
        sycl::buffer<T, 1> src_buf{sycl::range<1>(8)};
        sycl::buffer<T, 1> dst_buf{sycl::range<1>(8)};

        {
            auto data = src_buf.template get_access<sycl::access::mode::write>();
            data[0] = 1; data[1] = 0; data[2] = 0; data[3] = 1;
            data[4] = 0; data[5] = 1; data[6] = 0; data[7] = 1;
        }

        auto data_it = oneapi::dpl::begin(src_buf);
        auto data_end_it = oneapi::dpl::end(src_buf);

        auto result_it = oneapi::dpl::begin(dst_buf);

        // create named policy from existing one
        auto new_policy = TestUtils::make_device_policy<class IdentX>(oneapi::dpl::execution::dpcpp_default);

        auto t = std::copy_if( new_policy, data_it, data_end_it, result_it, oneapi::dpl::identity() );

        auto count = std::distance(result_it,t);

        EXPECT_EQ(count, 4, "#1");
    }

    /* MAXIMUM TEST: */
    {
        using T = int;

        // create buffer
        sycl::buffer<T, 1> src_buf{sycl::range<1>(8)};
        sycl::buffer<T, 1> dst_buf{sycl::range<1>(8)};

        {   
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            auto dst = dst_buf.template get_access<sycl::access::mode::write>();
            src[0] = -3; src[1] = 1; src[2] = 4; src[3] = -1; src[4] = 5; src[5] = 9; src[6] = -2; src[7] = 6;
            dst[0] = 0;  dst[1] = 0; dst[2] = 0;  dst[3] = 0; dst[4] = 0;  dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);
        
        // create named policy from existing one
        auto new_policy = TestUtils::make_device_policy<class MaxX>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::exclusive_scan(new_policy, src_it, src_end_it, dst_it, T(0), oneapi::dpl::maximum<T>());
        
        auto dst = dst_buf.template get_host_access(sycl::read_only);
        EXPECT_EQ(dst[0], 0, "#1");
        EXPECT_EQ(dst[1], 0, "#1");
        EXPECT_EQ(dst[2], 1, "#1");
        EXPECT_EQ(dst[3], 4, "#1");
        EXPECT_EQ(dst[4], 4, "#1");
        EXPECT_EQ(dst[5], 5, "#1");
        EXPECT_EQ(dst[6], 9, "#1");
        EXPECT_EQ(dst[7], 9, "#1");
    }

    /* MINIMUM TEST: */
    {
        using T = int;

        // create buffer
        sycl::buffer<T, 1> src_buf{sycl::range<1>(8)};
        sycl::buffer<T, 1> dst_buf{sycl::range<1>(8)};
    
        {   
            auto src = src_buf.template get_access<sycl::access::mode::write>();
            auto dst = dst_buf.template get_access<sycl::access::mode::write>();

            src[0] = 3; src[1] = -1; src[2] = -4; src[3] = 1; src[4] = -5; src[5] = -9; src[6] = 2; src[7] = 6;
            dst[0] = 0;  dst[1] = 0; dst[2] = 0;  dst[3] = 0; dst[4] = 0;  dst[5] = 0; dst[6] = 0; dst[7] = 0;
        }

        auto dst_it = oneapi::dpl::begin(dst_buf);
        auto src_it = oneapi::dpl::begin(src_buf);
        auto src_end_it = oneapi::dpl::end(src_buf);

        // create named policy from existing one
        auto new_policy = TestUtils::make_device_policy<class MinX>(oneapi::dpl::execution::dpcpp_default);
        // call algorithm:
        std::exclusive_scan(new_policy, src_it, src_end_it, dst_it, T(0), oneapi::dpl::minimum<T>());
    
        auto dst = dst_buf.template get_host_access(sycl::read_only);
        EXPECT_EQ(dst[0], 0, "#1");
        EXPECT_EQ(dst[1], 0, "#1");
        EXPECT_EQ(dst[2], -1, "#1");
        EXPECT_EQ(dst[3], -4, "#1");
        EXPECT_EQ(dst[4], -4, "#1");
        EXPECT_EQ(dst[5], -5, "#1");
        EXPECT_EQ(dst[6], -9, "#1");
        EXPECT_EQ(dst[7], -9, "#1");
    }
#endif

    return TestUtils::done();
}

