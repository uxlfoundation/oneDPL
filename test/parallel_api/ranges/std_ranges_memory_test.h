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
        run_one_policy(alloc, oneapi::dpl::execution::seq, algo, checker, std::forward<decltype(args)>(args)...);
        run_one_policy(alloc, oneapi::dpl::execution::unseq, algo, checker, std::forward<decltype(args)>(args)...);
        run_one_policy(alloc, oneapi::dpl::execution::par, algo, checker,  std::forward<decltype(args)>(args)...);
        run_one_policy(alloc, oneapi::dpl::execution::par_unseq, algo, checker, std::forward<decltype(args)>(args)...);
    }
#if TEST_DPCPP_BACKEND_PRESENT
    void run_device(auto algo, auto checker, auto&&... args)
    {
        //sycl::usm::alloc _alloc_type
        auto policy = TestUtils::get_dpcpp_test_policy();
        sycl::queue q = policy.queue();
        sycl::usm_allocator<Elem, sycl::usm::alloc::shared> q_alloc{policy.queue()};

        run_one_policy(q_alloc, policy, algo, checker, std::forward<decltype(args)>(args)...);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT

private:
    void run_one_policy(auto& alloc, auto&& policy, auto algo, auto checker, auto&&... args)
    {
        const std::size_t n = medium_size;
        Elem* pData = alloc.allocate(n);

        std::memset(reinterpret_cast<void*>(pData), no_init_val, n*sizeof(Elem));
        std::ranges::subrange r(pData, pData + n);

        if constexpr (test_mode_id<std::remove_cvref_t<decltype(algo)>> == 1)
        {//two ranges, constructor calls
            const std::size_t n1 = n1/2;
            Elem* pData1 = alloc.allocate(n1);
            std::memset(reinterpret_cast<void*>(pData1), no_init_val, n1*sizeof(Elem));
            std::ranges::subrange r1(pData1, pData1 + n1);
            std::uninitialized_fill(pData1, pData1 + n1, 5);

            run(std::forward<decltype(policy)>(policy), algo, checker, std::move(r1), std::move(r), std::forward<decltype(args)>(args)...);

            alloc.deallocate(pData1, n1);
        }
        else if constexpr (test_mode_id<std::remove_cvref_t<decltype(algo)>> == 2)
        { //one range, destructor calls
            std::uninitialized_fill(pData, pData + n, 5);
            run(std::forward<decltype(policy)>(policy), algo, checker, std::move(r), std::forward<decltype(args)>(args)...);
        }
        else //one range, constructor calls
            run(std::forward<decltype(policy)>(policy), algo, checker, std::move(r), std::forward<decltype(args)>(args)...);

        alloc.deallocate(pData, n);
    }

    void run(auto&& policy, auto algo, auto checker, auto&&... args)
    {
        auto res = algo(std::forward<decltype(policy)>(policy), std::forward<decltype(args)>(args)...);
        auto [bres1, bres2] = checker(res, std::forward<decltype(args)>(args)...);
        EXPECT_TRUE(bres1, (std::string("wrong return value from memory algo with ranges: ") + typeid(algo).name()
            + typeid(policy).name()).c_str());
        EXPECT_TRUE(bres2, (std::string("wrong effect from memory algo with ranges: ") + typeid(algo).name()
            + typeid(policy).name()).c_str());
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
