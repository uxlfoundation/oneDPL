// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_MEMORY_IMPL_HETERO_H
#define _ONEDPL_MEMORY_IMPL_HETERO_H

#include "algorithm_impl_hetero.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// uninitialized_walk1
//------------------------------------------------------------------------

template <class _BackendTag, class _ExecutionPolicy, class _ForwardIterator, class _Function>
void
__pattern_uninitialized_walk1(__hetero_tag<_BackendTag> tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                              _ForwardIterator __last, _Function __f)
{
    oneapi::dpl::__internal::__pattern_hetero_walk1<sycl::access_mode::write, /*_IsNoInitRequested=*/false>(
        tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __f);
}

//------------------------------------------------------------------------
// uninitialized_walk1_n
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Size,
          typename _Function>
_ForwardIterator
__pattern_uninitialized_walk1_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                                _Size __n, _Function __f)
{
    oneapi::dpl::__internal::__pattern_hetero_walk1<sycl::access_mode::write, /*_IsNoInitRequested=*/false>(
        __tag, std::forward<_ExecutionPolicy>(__exec), __first, __first + __n, __f);
    return __first + __n;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_MEMORY_IMPL_HETERO_H
