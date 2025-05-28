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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_EACH_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_EACH_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _ForwardIterator, class _Fp>
void
__parallel_for_each_body(_ForwardIterator __first, _ForwardIterator __last, _Fp __f)
{
    using DifferenceType = typename std::iterator_traits<_ForwardIterator>::difference_type;
    // TODO: Think of an approach to remove the std::distance call
    auto __size = std::distance(__first, __last);

    _ONEDPL_PRAGMA(omp taskloop untied mergeable)
    for (DifferenceType __index = 0; __index < __size; ++__index)
    {
        // TODO: Think of an approach to remove the increment here each time.
        auto __iter = std::next(__first, __index);
        __f(*__iter);
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _ForwardIterator __first,
                    _ForwardIterator __last, _Fp __f)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        oneapi::dpl::__omp_backend::__parallel_for_each_body(__first, __last, __f);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _ONEDPL_PRAGMA(omp parallel)
        _ONEDPL_PRAGMA(omp single nowait) { oneapi::dpl::__omp_backend::__parallel_for_each_body(__first, __last, __f); }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_EACH_H
