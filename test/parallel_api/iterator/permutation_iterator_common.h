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

#ifndef _PERMUTATION_ITERATOR_COMMON_H
#define _PERMUTATION_ITERATOR_COMMON_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

#include "support/utils.h"
#include "support/utils_test_base.h"

#include <vector>

#include "permutation_iterator.h"

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;
#endif // TEST_DPCPP_BACKEND_PRESENT
using namespace TestUtils;

////////////////////////////////////////////////////////////////////////////////
//
// Table: Current test state
//
// +------------------------+-----------------------+-----------+--------------------------------------+-------------+---------------+
// +       Test name        |     Algorithm         + Is modify +         Pattern                      + Host policy + Hetero policy +
// +------------------------+-----------------------+-----------+--------------------------------------+-------------+---------------+
// | test_transform         | dpl::transform        |     N     | __parallel_for                       |     +       |       +       |
// | test_transform_reduce  | dpl::transform_reduce |     N     | __parallel_transform_reduce          |     +       |       +       |
// | test_find              | dpl::find             |     N     | __parallel_find -> _parallel_find_or |     +       |       +       |
// | test_is_heap           | dpl::is_heap          |     N     | __parallel_or -> _parallel_find_or   |     +       |       +       |
// | test_merge             | dpl::merge            |     N     | __parallel_merge                     |     +       |       +       |
// | test_sort              | dpl::sort             |     Y     | __parallel_stable_sort               |     +       |       +       |
// | test_partial_sort      | dpl::partial_sort     |     Y     | __parallel_partial_sort              |     +       |       +       |
// | test_remove_if         | dpl::remove_if        |     Y     | __parallel_transform_scan            |     +       |       +       |
// +------------------------+-----------------------+-----------+--------------------------------------+-------------+---------------+

template <typename ExecutionPolicy>
void
wait_and_throw([[maybe_unused]] ExecutionPolicy&& exec)
{
#if TEST_DPCPP_BACKEND_PRESENT
    if constexpr (oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<ExecutionPolicy>>::value)
    {
        exec.queue().wait_and_throw();
    }
#endif // _PSTL_SYCL_TEST_USM
}

// DEFINE_TEST_PERM_IT should be used to declare permutation iterator tests
#define DEFINE_TEST_PERM_IT(TestClassName, TemplateParams)                                                             \
    template <typename TestValueType, typename TemplateParams>                                                         \
    struct TestClassName : TestUtils::test_base<TestValueType>

// DEFINE_TEST_PERM_IT_CONSTRUCTOR should be used to declare permutation iterator tests constructor
#define DEFINE_TEST_PERM_IT_CONSTRUCTOR(TestClassName, ScaleStepValue, ScaleMaxNValue)                                 \
    TestClassName(test_base_data<TestValueType>& _test_base_data)                                                      \
        : TestUtils::test_base<TestValueType>(_test_base_data)                                                         \
    {                                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    template <UDTKind kind, typename Size>                                                                             \
    using TestDataTransfer = typename TestUtils::test_base<TestValueType>::template TestDataTransfer<kind, Size>;      \
                                                                                                                       \
    using UsedValueType = TestValueType;                                                                               \
    static constexpr float ScaleMax = ScaleMaxNValue;                                                                  \
    static constexpr float ScaleStep = ScaleStepValue;

#endif // _PERMUTATION_ITERATOR_COMMON_H
