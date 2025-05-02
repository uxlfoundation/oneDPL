// -*- C++ -*-
//===-- transform_iterator_constructible.pass.cpp -------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

using namespace TestUtils;

struct noop
{
    int
    operator()(int a) const
    {
        return a;
    }
};

struct noop_nodefault
{
    noop_nodefault() = delete;
    noop_nodefault(int) {}
    int
    operator()(int a) const
    {
        return a;
    }
};

struct stateful_functor
{
    int __x;
    stateful_functor(int x) : __x(x) {}
    int
    operator()(int a) const
    {
        return a + __x;
    }
};

struct stateful_functor_no_copy_assign
{
    int __x;
    stateful_functor_no_copy_assign(int x) : __x(x) {}

    stateful_functor_no_copy_assign&
    operator=(const stateful_functor_no_copy_assign&) = delete;
    int
    operator()(int a) const
    {
        return a + __x;
    }
};

void
test_copy_assignment()
{
    auto transformation = [](int) { return 0; };

    oneapi::dpl::counting_iterator<int> count(0);
    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, decltype(transformation)> trans1{
        count, transformation};
    static_assert((std::is_copy_assignable_v<decltype(trans1)>),
                  "transform_iterator with lambda is not copy assignable");

    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, noop> trans2{count, noop{}};
    static_assert(std::is_copy_assignable_v<decltype(trans2)>,
                  "transform_iterator with noop functor is not copy assignable");

    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, stateful_functor> trans3{count,
                                                                                                  stateful_functor{1}};
    static_assert(std::is_copy_assignable_v<decltype(trans3)>,
                  "transform_iterator with stateful functor is not copy assignable");

    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, stateful_functor> trans4{count,
                                                                                                  stateful_functor{2}};

    EXPECT_EQ(3, trans4[1], "transform_iterator returns the incorrect result");

    //should copy __x state of functor
    trans4 = trans3;

    EXPECT_EQ(2, trans4[1],
              "transform_iterator assignment with copy assignable functor does not successfully copy functor");

    //Note that trans5 uses count incremented by 100 as its base iterator
    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, stateful_functor_no_copy_assign> trans5{
        count + 100, stateful_functor_no_copy_assign{3}};
    static_assert(std::is_copy_assignable_v<decltype(trans5)>,
                  "transform_iterator with non-copy-assignable functor is not copy assignable");

    oneapi::dpl::transform_iterator<oneapi::dpl::counting_iterator<int>, stateful_functor_no_copy_assign> trans6{
        count, stateful_functor_no_copy_assign{4}};

    EXPECT_EQ(9, trans6[5], "transform_iterator returns the incorrect result");

    //should NOT copy __x state of functor (but still allows assignment of iterator)
    trans6 = trans5;

    //trans6 functor.__x remains the same, but iterator has been updated to be 100 elements later in the counting iter
    EXPECT_EQ(109, trans6[5], "transform_iterator assignment with non-copy-assignable functor copies functor");
}

void
test_default_constructible()
{
    auto transformation = [](int) { return 0; };

    int* ptr = nullptr;
    oneapi::dpl::transform_iterator<int*, decltype(transformation)> trans1{ptr, transformation};
    //default constructibility of lambdas depends on c++ standard, we want transform iterator to match its template args
    static_assert((std::is_default_constructible_v<decltype(transformation)> ==
                   std::is_default_constructible_v<decltype(trans1)>),
                  "transform_iterator with lambda does not match default constructibility trait of the lambda itself");

    //both types are default constructible
    oneapi::dpl::transform_iterator<int*, noop> trans2{ptr, noop{}};
    static_assert(std::is_default_constructible_v<decltype(trans2)>,
                  "transform_iterator with default constructible functor is seen to be non-default constructible");

    //functor is not default constructible
    oneapi::dpl::transform_iterator<int*, noop_nodefault> trans3{ptr, noop_nodefault{1}};
    static_assert(!std::is_default_constructible_v<decltype(trans3)>,
                  "transform_iterator with non-default constructible functor is seen to be default constructible");

    oneapi::dpl::transform_iterator<decltype(trans3), noop> trans4{trans3, noop{}};
    static_assert(
        !std::is_default_constructible_v<decltype(trans4)>,
        "transform_iterator with non-default constructible iterator source is seen to be default constructible");

    oneapi::dpl::transform_iterator<int*, noop> a(ptr);
    static_assert(std::is_constructible_v<oneapi::dpl::transform_iterator<int*, noop>, int*>,
                  "transform_iterator with default constructible functor is not constructible from its source iterator "
                  "type alone");
    static_assert(!std::is_constructible_v<oneapi::dpl::transform_iterator<int*, noop_nodefault>, int*>,
                  "transform_iterator is not constructible from its source iterator type alone");
}

std::int32_t
main()
{
    test_default_constructible();
    test_copy_assignment();

    return TestUtils::done();
}
