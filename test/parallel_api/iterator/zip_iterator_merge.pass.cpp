// -*- C++ -*-
//===-- zip_iterator_merge.pass.cpp ---------------------------------------------===//
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

DEFINE_TEST(test_merge)
{
    DEFINE_TEST_CONSTRUCTOR(test_merge, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;
        typedef typename std::iterator_traits<Iterator2>::value_type T2;
        typedef typename std::iterator_traits<Iterator3>::value_type T3;

        T1 odd = T1{1};
        T2 even = T2{0};
        size_t size1 = n >= 2 ? n / 2 : n;
        size_t size2 = n >= 3 ? n / 3 : n;
        std::for_each(host_keys.get(), host_keys.get() + size1,
                      [&odd](T1& value)
                      {
                          value = odd;
                          odd += 2;
                      });
        std::for_each(host_vals.get(), host_vals.get() + size2,
                      [&even](T2& value)
                      {
                          value = even;
                          even += 2;
                      });
        std::fill(host_res.get(), host_res.get() + n, T3{ -1 });
        update_data(host_keys, host_vals, host_res);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(first1 + size1, first1 + size1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(first2 + size2, first2 + size2);
        auto tuple_first3 = oneapi::dpl::make_zip_iterator(first3, first3);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (merge1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>, "zip_iterator (merge2) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first3)>, "zip_iterator (merge3) not properly copyable");
        }

        auto tuple_last3 = std::merge(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
                                      tuple_last2, tuple_first3, TuplePredicate<std::less<T2>, 0>{std::less<T2>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        size_t res_size = tuple_last3 - tuple_first3;
        TestDataTransfer<UDTKind::eRes, Size> host_res_merge(*this, res_size);

        size_t exp_size = size1 + size2;
        bool is_correct = res_size == exp_size;
        EXPECT_TRUE(is_correct, "wrong result from merge (tuple)");

        retrieve_data(host_keys, host_vals, host_res_merge);

        auto host_first1 = host_keys.get();
        auto host_first3 = host_res_merge.get();

        for (size_t i = 0; i < std::min(res_size, exp_size) && is_correct; ++i)
            if ((i < size2 * 2 && *(host_first3 + i) != i) ||
                (i >= size2 * 2 && *(host_first3 + i) != *(host_first1 + i - size2)))
                is_correct = false;
        EXPECT_TRUE(is_correct, "wrong effect from merge (tuple)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_merge");
    test3buffers<alloc_type, test_merge<ValueType>>(2);
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

