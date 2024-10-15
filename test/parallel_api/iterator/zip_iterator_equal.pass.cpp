// -*- C++ -*-
//===-- zip_iterator_equal.pass.cpp ---------------------------------------------===//
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

DEFINE_TEST(test_equal)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename std::iterator_traits<Iterator1>::value_type;

        auto value = T(42);
        std::iota(host_keys.get(), host_keys.get() + n, value);
        std::iota(host_vals.get(), host_vals.get() + n, value);
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (equal1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>, "zip_iterator (equal2) not properly copyable");
        }

        bool is_equal = std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                   TuplePredicate<std::equal_to<T>, 0>{std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(is_equal, "wrong effect from equal(tuple) 1");

        host_vals.retrieve_data();
        *(host_vals.get() + n - 1) = T{0};
        host_vals.update_data();

        is_equal = std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1, tuple_first2,
                              TuplePredicate<std::equal_to<T>, 0>{std::equal_to<T>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(!is_equal, "wrong effect from equal(tuple) 2");
    }
};

DEFINE_TEST(test_equal_structured_binding)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal_structured_binding, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename std::iterator_traits<Iterator1>::value_type;

        auto value = T(42);
        std::iota(host_keys.get(), host_keys.get() + n, value);
        std::iota(host_vals.get(), host_vals.get() + n, value);
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>,
                        "zip_iterator (equal_structured_binding1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>,
                        "zip_iterator (equal_structured_binding2) not properly copyable");
        }

        auto compare = [](auto tuple_first1, auto tuple_first2)
        {
            const auto& [a, b] = tuple_first1;
            const auto& [c, d] = tuple_first2;

            static_assert(std::is_reference_v<decltype(a)>, "tuple element type is not a reference");
            static_assert(std::is_reference_v<decltype(b)>, "tuple element type is not a reference");
            static_assert(std::is_reference_v<decltype(c)>, "tuple element type is not a reference");
            static_assert(std::is_reference_v<decltype(d)>, "tuple element type is not a reference");

            return (a == c) && (b == d);
        };

        bool is_equal = std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                   compare);
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(is_equal, "wrong effect from equal(tuple with use of structured binding) 1");

        host_vals.retrieve_data();
        *(host_vals.get() + n - 1) = T{0};
        host_vals.update_data();

        is_equal = std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), tuple_first1, tuple_last1, tuple_first2,
                              compare);
        EXPECT_TRUE(!is_equal, "wrong effect from equal(tuple with use of structured binding) 2");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_equal");
    test2buffers<alloc_type, test_equal<ValueType>>();
    PRINT_DEBUG("test_equal_structured_binding");
    test2buffers<alloc_type, test_equal_structured_binding<ValueType>>();
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

