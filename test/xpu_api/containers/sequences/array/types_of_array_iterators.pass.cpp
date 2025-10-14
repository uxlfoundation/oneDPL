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

#include "support/test_config.h"

#include <oneapi/dpl/array>
#include <oneapi/dpl/cstddef>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/type_traits>

#include "support/utils.h"

template <typename T>
class KernelTest1;
template <typename T>
class KernelTest2;

template <class C>
bool
test_iterators()
{
    bool ret = true;
    {
        sycl::queue deviceQueue = TestUtils::get_test_queue();
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buf1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest1<C>>([=]() {
                typedef dpl::iterator_traits<typename C::iterator> ItT;
                typedef dpl::iterator_traits<typename C::const_iterator> CItT;

                ret_acc[0] &=
                    (dpl::is_same<typename ItT::iterator_category, dpl::random_access_iterator_tag>::value == true);

                ret_acc[0] &= (dpl::is_same<typename ItT::value_type, typename C::value_type>::value == true);

                ret_acc[0] &= (dpl::is_same<typename ItT::reference, typename C::reference>::value == true);

                ret_acc[0] &= (dpl::is_same<typename ItT::pointer, typename C::pointer>::value == true);
                ret_acc[0] &= (dpl::is_same<typename ItT::difference_type, typename C::difference_type>::value == true);

                ret_acc[0] &=
                    (dpl::is_same<typename CItT::iterator_category, typename dpl::random_access_iterator_tag>::value ==
                     true);
                ret_acc[0] &= (dpl::is_same<typename CItT::value_type, typename C::value_type>::value == true);
                ret_acc[0] &= (dpl::is_same<typename CItT::reference, typename C::const_reference>::value == true);

                ret_acc[0] &= (dpl::is_same<typename CItT::pointer, typename C::const_pointer>::value == true);

                ret_acc[0] &=
                    (dpl::is_same<typename CItT::difference_type, typename C::difference_type>::value == true);
            });
        });
    }
    return ret;
}

template <typename T>
bool
kernel_test()
{
    bool ret = true;
    {
        sycl::queue deviceQueue = TestUtils::get_test_queue();
        sycl::range<1> numOfItems{1};
        sycl::buffer<bool, 1> buf1(&ret, numOfItems);

        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ptr1 = buf1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<KernelTest2<T>>([=]() {
                typedef dpl::array<T, 10> C;
                ptr1[0] = (dpl::is_same<typename C::reference, T&>::value == true);
                ptr1[0] &= (dpl::is_same<typename C::const_reference, const T&>::value == true);
                ptr1[0] &= (dpl::is_same<typename C::pointer, T*>::value == true);
                ptr1[0] &= (dpl::is_same<typename C::const_pointer, const T*>::value == true);
                ptr1[0] &= (dpl::is_same<typename C::size_type, dpl::size_t>::value == true);
                ptr1[0] &= (dpl::is_same<typename C::difference_type, dpl::ptrdiff_t>::value == true);
                ptr1[0] &=
                    (dpl::is_same<typename C::reverse_iterator, dpl::reverse_iterator<typename C::iterator>>::value ==
                     true);
                ptr1[0] &= (dpl::is_same<typename C::const_reverse_iterator,
                                         dpl::reverse_iterator<typename C::const_iterator>>::value == true);
                ptr1[0] &= (dpl::is_signed<typename C::difference_type>::value == true);
                ptr1[0] &= (dpl::is_unsigned<typename C::size_type>::value == true);
                ptr1[0] &=
                    (dpl::is_same<typename C::difference_type,
                                  typename dpl::iterator_traits<typename C::iterator>::difference_type>::value == true);
                ptr1[0] &=
                    (dpl::is_same<typename C::difference_type,
                                  typename dpl::iterator_traits<typename C::const_iterator>::difference_type>::value ==
                     true);
            });
        });
    }
    return ret;
}

int
main()
{
    auto ret = true;
    typedef dpl::array<float, 10> C;
    ret &= test_iterators<C>();
    ret &= kernel_test<int*>();
    ret &= kernel_test<float>();
    EXPECT_TRUE(ret, "Wrong result of work in test_iterators of kernel_test");

    return TestUtils::done();
}
