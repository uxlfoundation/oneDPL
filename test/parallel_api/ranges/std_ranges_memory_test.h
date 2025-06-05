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

#include <oneapi/dpl/memory>

#include "support/std_ranges_test.h"


namespace test_std_ranges
{

template<sycl::usm::alloc _alloc_type>
struct test_memory_algo
{

    void operator()(auto __algo)
    {

        sycl::queue __q;
        //usm_data_transfer(sycl::queue __q, _Size __sz)
        TestUtils::usm_data_transfer<_alloc_type, T> __mem(q, medium_size);
    }

    void operator()(auto&& __policy, auto __algo, auto __checker, auto&&... args)
    {
        __algo(std::forward<decltype(__policy)>(__policy), std::forward<decltype(args)>(args)...);
        const bool __bres = __checker(std::forward<decltype(args)>(args)...);
        EXPECT_TRUE(__bres, (std::string("wrong return value from memory algo with ranges: ") + typeid(__algo).name()
            + typeid(__policy).name()).c_str());
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
