// -*- C++ -*-
//===-- zip_iterator_sort.pass.cpp ---------------------------------------------===//
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

DEFINE_TEST(test_stable_sort)
{
    DEFINE_TEST_CONSTRUCTOR(test_stable_sort, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename std::iterator_traits<Iterator1>::value_type;

        auto value = T(333);
        std::iota(host_keys.get(), host_keys.get() + n, value);
        std::copy_n(host_keys.get(), n, host_vals.get());
        update_data(host_keys, host_vals);

        auto tuple_first = oneapi::dpl::make_zip_iterator(first1, first2);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last1, last2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first)>, "zip_iterator (stable_sort) not properly copyable");
        }

        std::stable_sort(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                         TuplePredicate<std::greater<T>, 0>{std::greater<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        retrieve_data(host_keys, host_vals);
        EXPECT_TRUE(std::is_sorted(host_keys.get(), host_keys.get() + n, std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
        EXPECT_TRUE(std::is_sorted(host_vals.get(), host_vals.get() + n, std::greater<T>()),
                    "wrong effect from stable_sort (tuple)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_stable_sort");
    test2buffers<alloc_type, test_stable_sort<ValueType>>();
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

