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

#ifndef _TEST_BESSEL_H
#define _TEST_BESSEL_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#ifdef __SYCL_DEVICE_ONLY__

// Required to define these before all includes
namespace std
{
    SYCL_EXTERNAL void __throw_domain_error(const char*)
    {
    }
};

// Required to define these before all includes
namespace std
{
    SYCL_EXTERNAL void __throw_runtime_error(const char*)
    {
    }
};

#endif // __SYCL_DEVICE_ONLY__

#include "test_complex.h"
#include "specfun_testcase.h"

#include <limits>
#include <cmath>
#include <stdexcept>

#endif // _TEST_BESSEL_H
