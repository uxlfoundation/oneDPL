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

// <iterator>
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());
// template <class C> constexpr auto cbegin(const C& c) ->
// decltype(std::begin(c)); // C++14 template <class C> constexpr auto
// cend(const C& c) -> decltype(std::end(c));     // C++14 template <class C>
// constexpr auto end  (C& c) -> decltype(c.end()); template <class C> constexpr
// auto end  (const C& c) -> decltype(c.end()); template <class E> constexpr
// reverse_iterator<const E*> rbegin(initializer_list<E> il); template <class E>
// constexpr reverse_iterator<const E*> rend  (initializer_list<E> il);
//
// template <class C> auto constexpr rbegin(C& c) -> decltype(c.rbegin()); //
// C++14 template <class C> auto constexpr rbegin(const C& c) ->
// decltype(c.rbegin());           // C++14 template <class C> auto constexpr
// rend(C& c) -> decltype(c.rend());                     // C++14 template
// <class C> constexpr auto rend(const C& c) -> decltype(c.rend()); // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rbegin(T
// (&array)[N]);      // C++14 template <class T, size_t N> reverse_iterator<T*>
// constexpr rend(T (&array)[N]);        // C++14 template <class C> constexpr
// auto crbegin(const C& c) -> decltype(std::rbegin(c));      // C++14 template
// <class C> constexpr auto crend(const C& c) -> decltype(std::rend(c)); //
// C++14
//
//  All of these are constexpr in C++17

#include "support/test_config.h"

#include <oneapi/dpl/iterator>
#include <oneapi/dpl/array>

#include <initializer_list>

#include "support/utils.h"

template <typename C>
class ConstContainerTest1;
template <typename C>
class ContainerTest1;
template <typename C>
class ConstArrayTest;

template <typename C>
bool
test_const_container(const C& c, typename C::value_type val)
{
    sycl::queue q = TestUtils::get_test_queue();
    auto ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<C, 1> buffer1(&c, numOfItems);
    sycl::buffer<bool, 1> buffer2(&ret, numOfItems);
    q.submit([&](sycl::handler& cgh) {
        auto c_access = buffer1.template get_access<sycl::access::mode::write>(cgh);
        auto ret_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<ConstContainerTest1<C>>([=]() {
            ret_access[0] &= (dpl::begin(c_access[0]) == c_access[0].begin());
            ret_access[0] &= (*dpl::begin(c_access[0]) == val);
            ret_access[0] &= (dpl::begin(c_access[0]) != c_access[0].end());
            ret_access[0] &= (dpl::end(c_access[0]) == c_access[0].end());

            ret_access[0] &= (dpl::cbegin(c_access[0]) == c_access[0].cbegin());
            ret_access[0] &= (dpl::cbegin(c_access[0]) != c_access[0].cend());
            ret_access[0] &= (dpl::cend(c_access[0]) == c_access[0].cend());
            ret_access[0] &= (dpl::rbegin(c_access[0]) == c_access[0].rbegin());
            ret_access[0] &= (dpl::rbegin(c_access[0]) != c_access[0].rend());
            ret_access[0] &= (dpl::rend(c_access[0]) == c_access[0].rend());
            ret_access[0] &= (dpl::crbegin(c_access[0]) == c_access[0].crbegin());
            ret_access[0] &= (dpl::crbegin(c_access[0]) != c_access[0].crend());
            ret_access[0] &= (dpl::crend(c_access[0]) == c_access[0].crend());
        });
    });
    return ret;
}

bool
test_initializer_list()
{
    sycl::queue q = TestUtils::get_test_queue();
    auto ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    q.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelTest1>([=]() {
            {
                std::initializer_list<int> il = {4};
                ret_access[0] &= (dpl::begin(il) == il.begin());
                ret_access[0] &= (*dpl::begin(il) == 4);
                ret_access[0] &= (dpl::begin(il) != il.end());
                ret_access[0] &= (dpl::end(il) == il.end());
            }
        });
    });
    return ret;
}

template <typename C>
bool
test_container(C& c, typename C::value_type val)
{
    sycl::queue q = TestUtils::get_test_queue();
    auto ret = true;
    sycl::range<1> numOfItems{1};
    sycl::buffer<C, 1> buffer1(&c, numOfItems);
    sycl::buffer<bool, 1> buffer2(&ret, numOfItems);
    q.submit([&](sycl::handler& cgh) {
        auto c_access = buffer1.template get_access<sycl::access::mode::write>(cgh);
        auto ret_access = buffer2.template get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<ContainerTest1<C>>([=]() {
            ret_access[0] &= (dpl::begin(c_access[0]) == c_access[0].begin());
            ret_access[0] &= (*dpl::begin(c_access[0]) == val);
            ret_access[0] &= (dpl::begin(c_access[0]) != c_access[0].end());
            ret_access[0] &= (dpl::end(c_access[0]) == c_access[0].end());

            ret_access[0] &= (dpl::cbegin(c_access[0]) == c_access[0].cbegin());
            ret_access[0] &= (dpl::cbegin(c_access[0]) != c_access[0].cend());
            ret_access[0] &= (dpl::cend(c_access[0]) == c_access[0].cend());
            ret_access[0] &= (dpl::rbegin(c_access[0]) == c_access[0].rbegin());
            ret_access[0] &= (dpl::rbegin(c_access[0]) != c_access[0].rend());
            ret_access[0] &= (dpl::rend(c_access[0]) == c_access[0].rend());
            ret_access[0] &= (dpl::crbegin(c_access[0]) == c_access[0].crbegin());
            ret_access[0] &= (dpl::crbegin(c_access[0]) != c_access[0].crend());
            ret_access[0] &= (dpl::crend(c_access[0]) == c_access[0].crend());
        });
    });
    return ret;
}

void
kernel_test()
{
    sycl::queue q = TestUtils::get_test_queue();
    q.submit([&](sycl::handler& cgh) {
        cgh.single_task<class KernelTest>([=]() {
            {
                typedef std::array<int, 5> C;
                constexpr const C c{0, 1, 2, 3, 4};

                static_assert(c.begin() == dpl::begin(c));
                static_assert(c.cbegin() == dpl::cbegin(c));
                static_assert(c.end() == dpl::end(c));
                static_assert(c.cend() == dpl::cend(c));

                static_assert(c.rbegin() == dpl::rbegin(c));
                static_assert(c.crbegin() == dpl::crbegin(c));
                static_assert(c.rend() == dpl::rend(c));
                static_assert(c.crend() == dpl::crend(c));

                static_assert(dpl::begin(c) != dpl::end(c));
                static_assert(dpl::rbegin(c) != dpl::rend(c));
                static_assert(dpl::cbegin(c) != dpl::cend(c));
                static_assert(dpl::crbegin(c) != dpl::crend(c));

                static_assert(*c.begin() == 0);
                static_assert(*c.rbegin() == 4);

                static_assert(*dpl::begin(c) == 0);
                static_assert(*dpl::cbegin(c) == 0);
                static_assert(*dpl::rbegin(c) == 4);
                static_assert(*dpl::crbegin(c) == 4);
            }

            {
                static constexpr const int c[] = {0, 1, 2, 3, 4};

                static_assert(*dpl::begin(c) == 0);
                static_assert(*dpl::cbegin(c) == 0);
                static_assert(*dpl::rbegin(c) == 4);
                static_assert(*dpl::crbegin(c) == 4);
            }
        });
    });
}

int
main()
{
    std::array<int, 1> a;
    a[0] = 3;

    auto ret = test_container(a, 3);
    ret &= test_const_container(a, 3);
    ret &= test_initializer_list();

    kernel_test();

    EXPECT_TRUE(
        ret, "Wrong result of dpl::begin / dpl::end in test_container, test_const_container or test_initializer_list");

    return TestUtils::done();
}
