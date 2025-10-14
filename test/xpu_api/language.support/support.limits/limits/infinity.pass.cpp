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

// test numeric_limits

// infinity()

#include "support/test_complex.h"
#include "support/test_macros.h"

#include <oneapi/dpl/limits>
#include <cfloat>
#include <cassert>

template <class T>
void
test(T expected)
{
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#endif

    assert(dpl::numeric_limits<T>::infinity() == expected);
    assert(dpl::numeric_limits<const T>::infinity() == expected);
    assert(dpl::numeric_limits<volatile T>::infinity() == expected);
    assert(dpl::numeric_limits<const volatile T>::infinity() == expected);

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

ONEDPL_TEST_NUM_MAIN
{
    test<bool>(false);
    test<char>(0);
    test<signed char>(0);
    test<unsigned char>(0);
    test<wchar_t>(0);
#ifndef TEST_HAS_NO_CHAR8_T
    test<char8_t>(0);
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
    test<char16_t>(0);
    test<char32_t>(0);
#endif
    test<short>(0);
    test<unsigned short>(0);
    test<int>(0);
    test<unsigned int>(0);
    test<long>(0);
    test<unsigned long>(0);
    test<long long>(0);
    test<unsigned long long>(0);
#ifndef TEST_HAS_NO_INT128
    test<__int128_t>(0);
    test<__uint128_t>(0);
#endif
    float zero = 0;
    test<float>(1.f/zero);
    if constexpr (HasDoubleSupportInRuntime{})
    {
        test<double>(1. / zero);
    }
    if constexpr (HasLongDoubleSupportInCompiletime{})
    {
        // This lambda required to avoid compile errors like
        // test<long double>' requires 128 bit size 'long double' type support, but target 'spir64-unknown-unknown' does not support it
        auto fnc = [&zero] { test<long double>(1. / zero); };
        fnc();
    }

  return 0;
}
