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

/*
  Warning  'result_type' is deprecated: warning STL4007: Many result_type typedefs and all argument_type, first_argument_type, and second_argument_type typedefs are deprecated in C++17.
  You can define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 */
#define _SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "support/test_config.h"

#include <oneapi/dpl/functional>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"

// dpl::reference_wrapper::result_type is removed since C++20
#if TEST_STD_VER == 17
class KernelWeakResultTest;

template <class Arg, class Result>
struct my_unary_function
{ // std::unary_function was removed in C++17
    typedef Arg argument_type;
    typedef Result result_type;
};

template <class Arg1, class Arg2, class Result>
struct my_binary_function
{ // std::binary_function was removed in C++17
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};

class functor1 : public my_unary_function<int, char>
{
};

class functor2 : public my_binary_function<char, int, float>
{
};

class functor3 : public my_unary_function<char, int>, public my_binary_function<char, int, float>
{
  public:
    typedef float result_type;
};

class functor4 : public my_unary_function<char, int>, public my_binary_function<char, int, float>
{
  public:
};

class C
{
};

template <class T>
struct has_result_type
{
  private:
    template <class U>
    static int
    test(...);
    template <class U>
    static char
    test(typename U::result_type* = 0);

  public:
    static const bool value = sizeof(test<T>(0)) == 1;
};

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    bool ret = false;
    sycl::range<1> numOfItems{1};
    sycl::buffer<bool, 1> buffer1(&ret, numOfItems);
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto ret_access = buffer1.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class KernelWeakResultTest>([=]() {
            // Static assert check...
            static_assert(dpl::is_same<dpl::reference_wrapper<functor1>::result_type, char>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<functor2>::result_type, float>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<functor3>::result_type, float>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<void()>::result_type, void>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int*(float*)>::result_type, int*>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<void (*)()>::result_type, void>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int* (*)(float*)>::result_type, int*>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<int* (C::*)(float*)>::result_type, int*>::value);
            static_assert(
                dpl::is_same<dpl::reference_wrapper<int (C::*)(float*) const volatile>::result_type, int>::value);
            static_assert(dpl::is_same<dpl::reference_wrapper<C()>::result_type, C>::value);
            static_assert(has_result_type<dpl::reference_wrapper<functor3>>::value);
            static_assert(!has_result_type<dpl::reference_wrapper<functor4>>::value);
            static_assert(!has_result_type<dpl::reference_wrapper<C>>::value);

            // Runtime check...

            ret_access[0] = dpl::is_same<dpl::reference_wrapper<functor1>::result_type, char>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<functor2>::result_type, float>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<functor3>::result_type, float>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<void()>::result_type, void>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int*(float*)>::result_type, int*>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<void (*)()>::result_type, void>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int* (*)(float*)>::result_type, int*>::value;

            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<int* (C::*)(float*)>::result_type, int*>::value;
            ret_access[0] &=
                dpl::is_same<dpl::reference_wrapper<int (C::*)(float*) const volatile>::result_type, int>::value;
            ret_access[0] &= dpl::is_same<dpl::reference_wrapper<C()>::result_type, C>::value;
            ret_access[0] &= has_result_type<dpl::reference_wrapper<functor3>>::value;
            ret_access[0] &= !has_result_type<dpl::reference_wrapper<functor4>>::value;
            ret_access[0] &= !has_result_type<dpl::reference_wrapper<C>>::value;
        });
    });

    auto ret_access_host = buffer1.get_host_access(sycl::read_only);
    EXPECT_TRUE(ret_access_host[0], "Error in work with weak results");
}
#endif // TEST_STD_VER

int
main()
{
#if TEST_STD_VER == 17
    kernel_test();
#endif // TEST_STD_VER

    return TestUtils::done(TEST_STD_VER == 17);
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif
