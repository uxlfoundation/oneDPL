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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_MERGE_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_MERGE_H

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessIterator3,
          typename _Compare, typename _LeafMerge>
void
__parallel_merge_body(std::size_t __size_x, std::size_t __size_y, _RandomAccessIterator1 __xs,
                      _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye,
                      _RandomAccessIterator3 __zs, _Compare __comp, _LeafMerge __leaf_merge)
{

    if (__size_x + __size_y <= oneapi::dpl::__omp_backend::__default_chunk_size)
    {
        __leaf_merge(__xs, __xe, __ys, __ye, __zs, __comp);
        return;
    }

    _RandomAccessIterator1 __xm;
    _RandomAccessIterator2 __ym;

    if (__size_x < __size_y)
    {
        __ym = __ys + (__size_y / 2);
        __xm = std::upper_bound(__xs, __xe, *__ym, __comp);
    }
    else
    {
        __xm = __xs + (__size_x / 2);
        __ym = std::lower_bound(__ys, __ye, *__xm, __comp);
    }

    auto __zm = __zs + (__xm - __xs) + (__ym - __ys);

    _ONEDPL_PRAGMA(omp task untied mergeable default(none)
                     firstprivate(__xs, __xm, __ys, __ym, __zs, __comp, __leaf_merge))
    oneapi::dpl::__omp_backend::__parallel_merge_body(__xm - __xs, __ym - __ys, __xs, __xm, __ys, __ym, __zs, __comp,
                                                      __leaf_merge);

    _ONEDPL_PRAGMA(omp task untied mergeable default(none)
                     firstprivate(__xm, __xe, __ym, __ye, __zm, __comp, __leaf_merge))
    oneapi::dpl::__omp_backend::__parallel_merge_body(__xe - __xm, __ye - __ym, __xm, __xe, __ym, __ye, __zm, __comp,
                                                      __leaf_merge);

    _ONEDPL_PRAGMA(omp taskwait)
}

template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
void
__parallel_merge(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&& /*__exec*/, _RandomAccessIterator1 __xs,
                 _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys, _RandomAccessIterator2 __ye,
                 _RandomAccessIterator3 __zs, _Compare __comp, _LeafMerge __leaf_merge)
{
    std::size_t __size_x = __xe - __xs;
    std::size_t __size_y = __ye - __ys;

    /*
     * Run the merge in parallel by chunking it up. Use the smaller range (if any) as the iteration range, and the
     * larger range as the search range.
     */

    if (omp_in_parallel())
    {
        oneapi::dpl::__omp_backend::__parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp,
                                                          __leaf_merge);
    }
    else
    {
        _ONEDPL_PRAGMA(omp parallel)
        {
            _ONEDPL_PRAGMA(omp single nowait)
            oneapi::dpl::__omp_backend::__parallel_merge_body(__size_x, __size_y, __xs, __xe, __ys, __ye, __zs, __comp,
                                                              __leaf_merge);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_MERGE_H
