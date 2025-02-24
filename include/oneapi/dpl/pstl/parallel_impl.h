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

#ifndef _ONEDPL_PARALLEL_IMPL_H
#define _ONEDPL_PARALLEL_IMPL_H

#include <atomic>
// This header defines the minimum set of parallel routines required to support Parallel STL,
// implemented on top of Intel(R) Threading Building Blocks (Intel(R) TBB) library

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// parallel_find
//-----------------------------------------------------------------------
/** Return extremum value returned by brick f[i,j) for subranges [i,j) of [first,last)
Each f[i,j) must return a value in [i,j). */
template <class _IsVector, class _ExecutionPolicy, class _Index, class _Brick, class _IsFirst>
_Index
__parallel_find(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _Index __first, _Index __last, _Brick __f,
                _IsFirst)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    typedef typename ::std::iterator_traits<_Index>::difference_type _DifferenceType;
    const _DifferenceType __n = __last - __first;
    _DifferenceType __initial_dist = _IsFirst::value ? __n : -1;

    constexpr auto __comp = ::std::conditional_t<_IsFirst::value, __pstl_less, __pstl_greater>{};

    ::std::atomic<_DifferenceType> __extremum(__initial_dist);
    // TODO: find out what is better here: parallel_for or parallel_reduce
    __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                  [__comp, __f, __first, &__extremum](_Index __i, _Index __j) {
                                      // See "Reducing Contention Through Priority Updates", PPoPP '13, for discussion of
                                      // why using a shared variable scales fairly well in this situation.
                                      if (__comp(__i - __first, __extremum))
                                      {
                                          _Index __res = __f(__i, __j);
                                          // If not '__last' returned then we found what we want so put this to extremum
                                          if (__res != __j)
                                          {
                                              const _DifferenceType __k = __res - __first;
                                              for (_DifferenceType __old = __extremum; __comp(__k, __old);
                                                   __old = __extremum)
                                              {
                                                  __extremum.compare_exchange_weak(__old, __k);
                                              }
                                          }
                                      }
                                  });
    return __extremum != __initial_dist ? __first + __extremum : __last;
}

//------------------------------------------------------------------------
// parallel_or
//------------------------------------------------------------------------
//! Return true if brick f[i,j) returns true for some subrange [i,j) of [first,last)
template <class _IsVector, class _ExecutionPolicy, class _Index, class _Brick>
bool
__parallel_or(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _Index __first, _Index __last, _Brick __f)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    ::std::atomic<bool> __found(false);
    __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                  [__f, &__found](_Index __i, _Index __j) {
                                      if (!__found.load(::std::memory_order_relaxed) && __f(__i, __j))
                                      {
                                          __found.store(true, ::std::memory_order_relaxed);
                                          __par_backend::__cancel_execution(__backend_tag{});
                                      }
                                  });
    return __found;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_IMPL_H
