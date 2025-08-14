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

#if _ENABLE_STD_RANGES_TESTING

#include <oneapi/dpl/memory>
#include <oneapi/dpl/algorithm>

#include "std_ranges_test.h"

namespace test_std_ranges
{

template<typename>
constexpr int test_mode_id = 0;

template<typename Elem, int no_init_val>
struct test_memory_algo
{
    void run_host(auto algo, auto checker, auto&&... args)
    {
        std::allocator<Elem> alloc;
        run_one_policy(alloc, oneapi::dpl::execution::seq, algo, checker, args...);
        run_one_policy(alloc, oneapi::dpl::execution::unseq, algo, checker, args...);
        run_one_policy(alloc, oneapi::dpl::execution::par, algo, checker, args...);
        run_one_policy(alloc, oneapi::dpl::execution::par_unseq, algo, checker, std::forward<decltype(args)>(args)...);
    }
#if TEST_DPCPP_BACKEND_PRESENT
    void run_device(auto algo, auto checker, auto&&... args)
    {
        //sycl::usm::alloc _alloc_type
        auto policy = TestUtils::get_dpcpp_test_policy();
        sycl::usm_allocator<Elem, sycl::usm::alloc::shared> q_alloc{policy.queue()};

        run_one_policy(q_alloc, policy, algo, checker, std::forward<decltype(args)>(args)...);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT

    void run(auto algo, auto checker, auto&&... args)
    {
        run_host(algo, checker, args...);
#if TEST_DPCPP_BACKEND_PRESENT
        run_device(algo, checker, std::forward<decltype(args)>(args)...);
#endif //TEST_DPCPP_BACKEND_PRESENT
    }

private:
    // Tests both subrange and span
    void run_one_policy(auto& alloc, auto&& policy, auto algo, auto checker, auto&&... args)
    {
        const std::size_t n_in = medium_size;
        Elem* data_in1 = alloc.allocate(n_in);
        Elem* data_in2 = alloc.allocate(n_in);
        std::memset(reinterpret_cast<void*>(data_in1), no_init_val, n_in*sizeof(Elem));
        std::memset(reinterpret_cast<void*>(data_in2), no_init_val, n_in*sizeof(Elem));
        std::ranges::subrange subrange_in(data_in1, data_in1 + n_in);
        std::span span_in(data_in2, n_in);

        // Two ranges: uninitialized_copy, uninitialized_move
        if constexpr (test_mode_id<std::remove_cvref_t<decltype(algo)>> == 1)
        {
            const std::size_t n_out = n_in / 2; // to check minimal size logic
            Elem* data_out1 = alloc.allocate(n_out);
            Elem* data_out2 = alloc.allocate(n_out);
            std::ranges::subrange subrange_out(data_out1, data_out1 + n_out);
            std::span span_out(data_out2, n_out);
            std::memset(reinterpret_cast<void*>(data_out1), no_init_val, n_out*sizeof(Elem));
            std::memset(reinterpret_cast<void*>(data_out2), no_init_val, n_out*sizeof(Elem));

            std::uninitialized_fill(data_in1, data_in1 + n_in, 5);
            std::uninitialized_fill(data_in2, data_in2 + n_in, 5);

            run_impl(policy, algo, checker, std::move(subrange_in), std::move(subrange_out), args...);
#if TEST_CPP20_SPAN_PRESENT
            run_impl(std::forward<decltype(policy)>(policy), algo, checker,
                     std::move(span_in), std::move(span_out), std::forward<decltype(args)>(args)...);
#endif
            alloc.deallocate(data_out1, n_out);
            alloc.deallocate(data_out2, n_out);
        }
        // One range: destroy, uninitialized_fill, uninitialized_default_construct, uninitialized_value_construct
        else
        {
            run_impl(policy, algo, checker, std::move(subrange_in), args...);
#if TEST_CPP20_SPAN_PRESENT
            run_impl(std::forward<decltype(policy)>(policy), algo, checker, std::move(span_in),
                     std::forward<decltype(args)>(args)...);
#endif
        }
        alloc.deallocate(data_in1, n_in);
        alloc.deallocate(data_in2, n_in);
    }

    void run_impl(auto&& policy, auto algo, auto checker, auto&&... args)
    {
        auto res = algo(std::forward<decltype(policy)>(policy), std::forward<decltype(args)>(args)...);
        auto [bres1, bres2] = checker(res, std::forward<decltype(args)>(args)...);

        std::string wrong_return = std::string("wrong return value from memory algo with ranges: ") +
                                    typeid(algo).name() + typeid(policy).name();
        std::string wrong_effect = std::string("wrong effect from memory algo with ranges: ") +
                                    typeid(algo).name() + typeid(policy).name();
        EXPECT_TRUE(bres1, wrong_return.c_str());
        EXPECT_TRUE(bres2, wrong_effect.c_str());
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
