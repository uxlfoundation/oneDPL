// -*- C++ -*-
//===-- zip_iterator_find.pass.cpp ---------------------------------------------===//
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

#include "zip_iterator_funcs.h"
#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#   include "support/utils_sycl.h"
#endif

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (find_if) not properly copyable");
        }

        auto f_for_last = [n](T1 x) { return x == n - 1; };
        auto f_for_none = [](T1 x) { return x == -1; };
        auto f_for_first = [](T1 x) { return x % 2 == 0; };

        auto tuple_res1 = std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_last), 0>{f_for_last});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE((tuple_res1 - tuple_first1) == n - 1, "wrong effect from find_if_1 (tuple)");
        auto tuple_res2 = std::find_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_none), 0>{f_for_none});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res2 == tuple_last1, "wrong effect from find_if_2 (tuple)");
        auto tuple_res3 = std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), tuple_first1, tuple_last1,
                                       TuplePredicate<decltype(f_for_first), 0>{f_for_first});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res3 == tuple_first1, "wrong effect from find_if_3 (tuple)");

        // current test doesn't work with zip iterators
        auto tuple_res4 = std::find(make_new_policy<new_kernel_name<Policy, 3>>(exec), tuple_first1, tuple_last1,
                                    std::make_tuple(T1{-1}, T1{-1}));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(tuple_res4 == tuple_last1, "wrong effect from find (tuple)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_find_if<ValueType>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
    //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

    // Run tests for USM shared memory
    test_usm_and_buffer<sycl::usm::alloc::shared>();
    // Run tests for USM device memory
    test_usm_and_buffer<sycl::usm::alloc::device>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return done(TEST_DPCPP_BACKEND_PRESENT);
}

