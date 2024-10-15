// -*- C++ -*-
//===-- zip_iterator_for.pass.cpp ---------------------------------------------===//
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

DEFINE_TEST(test_for_each)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(6);
        auto f = [](T1& val) { ++val; };
        std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(std::make_tuple(first1, first1));
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (for_each) not properly copyable");
        }

        std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                      TuplePredicate<decltype(f), 0>{f});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1), "wrong effect from for_each(tuple)");
    }
};

DEFINE_TEST(test_for_each_structured_binding)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each_structured_binding, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(6);
        auto f = [](T1& val) { ++val; };
        std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, 
                        "zip_iterator (structured_binding) not properly copyable");
        }

        std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                      [f](auto value)
                      {
                          auto [x, y] = value;
                          f(x);
                          f(y);
                      });
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 2), "wrong effect from for_each(tuple)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_for_each<ValueType>>();
    PRINT_DEBUG("test_for_each_structured_binding");
    test1buffer<alloc_type, test_for_each_structured_binding<ValueType>>();
}
#endif

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
#endif

    return done(TEST_DPCPP_BACKEND_PRESENT);
}

