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

template<typename Elem, int no_init_val>
struct test_memory_algo
{

    void run_host(auto algo, auto checker, auto&&... args)
    {
        run_host_one_policy(oneapi::dpl::execution::seq, algo, checker, std::forward<decltype(args)>(args)...);
        run_host_one_policy(oneapi::dpl::execution::unseq, algo, checker, std::forward<decltype(args)>(args)...);
        run_host_one_policy(oneapi::dpl::execution::par, algo, checker,  std::forward<decltype(args)>(args)...);
        run_host_one_policy(oneapi::dpl::execution::par_unseq, algo, checker, std::forward<decltype(args)>(args)...);
    }
#if TEST_DPCPP_BACKEND_PRESENT
    template<sycl::usm::alloc _alloc_type>
    void run_device(auto algo, auto checker, auto&&... args)
    {

        sycl::queue __q;
        //usm_data_transfer(sycl::queue __q, _Size __sz)
        TestUtils::usm_data_transfer<_alloc_type, T> __mem(q, medium_size);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT

private:
//    template<typename Allocator>
    void run_host_one_policy(auto&& policy, auto algo, auto checker, auto&&... args)
    {
        const std::size_t n = medium_size;
        std::allocator<Elem> alloc;
        Elem* pData = alloc.allocate(n);

        std::memset(pData, no_init_val, n*sizeof(Elem));
        std::ranges::subrange r(pData, pData + n);

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
