// -*- C++ -*-
//===-- type_requirements.pass.cpp ---------------------------------------===//
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

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>

template <typename... Args>
void CheckTuple() {
    static_assert(::std::is_trivially_copyable_v<oneapi::dpl::__internal::tuple<Args...>>);
    static_assert(::std::is_standard_layout_v<oneapi::dpl::__internal::tuple<Args...>>);
}

#endif

#include "support/utils.h"

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    CheckTuple<int>();
    CheckTuple<int, long, float>();
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
