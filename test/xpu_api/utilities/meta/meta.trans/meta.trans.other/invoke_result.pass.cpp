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
  Warning  'result_of<S (int)>' is deprecated: warning STL4014: std::result_of and std::result_of_t are deprecated in C++17.
  They are superseded by std::invoke_result and std::invoke_result_t. You can define _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING or _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS to suppress this warning.
 */
#define _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/type_traits>

#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"

#include "has_type_member.h"

struct S
{
    typedef short (*FreeFunc)(long);
    operator FreeFunc() const;
    double
    operator()(char, int&);
    double const&
    operator()(char, int&) const;
    double volatile&
    operator()(char, int&) volatile;
    double const volatile&
    operator()(char, int&) const volatile;
};

struct SD : public S
{
};

struct NotDerived
{
};

template <class T>
using HasType = has_type_member<T>;

template <typename T, typename U>
struct test_invoke_result;

template <typename Fn, typename... Args, typename Ret>
struct test_invoke_result<Fn(Args...), Ret>
{
    static void
    call()
    {
        static_assert(dpl::is_invocable<Fn, Args...>::value);
        static_assert(dpl::is_invocable_r<Ret, Fn, Args...>::value);
        ASSERT_SAME_TYPE(Ret, typename dpl::invoke_result<Fn, Args...>::type);
        ASSERT_SAME_TYPE(Ret, dpl::invoke_result_t<Fn, Args...>);
    }
};

template <class KernelTest, class T, class U>
void
test_result_of(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
// dpl::result_of is removed since C++20
#if TEST_STD_VER == 17
            ASSERT_SAME_TYPE(U, typename dpl::result_of<T>::type);
#endif // TEST_STD_VER
            test_invoke_result<T, U>::call();
        });
    });
}

template <typename T>
struct test_invoke_no_result;

template <typename Fn, typename... Args>
struct test_invoke_no_result<Fn(Args...)>
{
    static void
    call()
    {
        static_assert(dpl::is_invocable<Fn, Args...>::value == false);
        static_assert(!HasType<dpl::invoke_result<Fn, Args...>>::value);
    }
};

template <class KernelTest, class T>
void
test_no_result(sycl::queue& deviceQueue)
{
    deviceQueue.submit([&](sycl::handler& cgh) {
        cgh.single_task<KernelTest>([=]() {
// dpl::result_of is removed since C++20
#if TEST_STD_VER == 17
            static_assert(!HasType<dpl::result_of<T>>::value);
#endif // TEST_STD_VER
            test_invoke_no_result<T>::call();
        });
    });
}

class KernelTest1;
class KernelTest2;
class KernelTest3;
class KernelTest4;
class KernelTest5;
class KernelTest6;
class KernelTest7;
class KernelTest8;
class KernelTest9;
class KernelTest10;
class KernelTest11;
class KernelTest12;
class KernelTest13;
class KernelTest14;
class KernelTest15;
class KernelTest16;
class KernelTest17;
class KernelTest18;
class KernelTest19;
class KernelTest20;
class KernelTest21;

void
kernel_test()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    { // functor object
        test_result_of<KernelTest1, S(int), short>(deviceQueue);

        const auto device = deviceQueue.get_device();
        if (TestUtils::has_type_support<double>(device))
        {
            test_result_of<KernelTest2, S&(unsigned char, int&), double>(deviceQueue);
            test_result_of<KernelTest3, S const&(unsigned char, int&), double const&>(deviceQueue);
            test_result_of<KernelTest4, S volatile&(unsigned char, int&), double volatile&>(deviceQueue);
            test_result_of<KernelTest5, S const volatile&(unsigned char, int&), double const volatile&>(deviceQueue);
        }
    }
    { // pointer to member data
        typedef char S::*PMD;
        test_result_of<KernelTest6, PMD(S&), char&>(deviceQueue);
        test_result_of<KernelTest7, PMD(S*), char&>(deviceQueue);
        test_result_of<KernelTest8, PMD(S* const), char&>(deviceQueue);
        test_result_of<KernelTest9, PMD(const S&), const char&>(deviceQueue);
        test_result_of<KernelTest10, PMD(const S*), const char&>(deviceQueue);
        test_result_of<KernelTest11, PMD(volatile S&), volatile char&>(deviceQueue);
        test_result_of<KernelTest12, PMD(volatile S*), volatile char&>(deviceQueue);
        test_result_of<KernelTest13, PMD(const volatile S&), const volatile char&>(deviceQueue);
        test_result_of<KernelTest14, PMD(const volatile S*), const volatile char&>(deviceQueue);
        test_result_of<KernelTest15, PMD(SD&), char&>(deviceQueue);
        test_result_of<KernelTest16, PMD(SD const&), const char&>(deviceQueue);
        test_result_of<KernelTest17, PMD(SD*), char&>(deviceQueue);
        test_result_of<KernelTest18, PMD(const SD*), const char&>(deviceQueue);
        test_result_of<KernelTest19, PMD(dpl::reference_wrapper<S>), char&>(deviceQueue);
        test_result_of<KernelTest20, PMD(dpl::reference_wrapper<S const>), const char&>(deviceQueue);
        test_no_result<KernelTest21, PMD(NotDerived&)>(deviceQueue);
    }
}

int
main()
{
    kernel_test();

    return TestUtils::done();
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif
