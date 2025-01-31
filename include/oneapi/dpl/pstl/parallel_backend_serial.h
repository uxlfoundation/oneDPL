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

// This header guard is used to check inclusion of serial backend.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_SERIAL_H
#define _ONEDPL_PARALLEL_BACKEND_SERIAL_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <type_traits>
#include "parallel_backend_utils.h"

namespace oneapi
{
namespace dpl
{
namespace __serial_backend
{

namespace __detail
{

template <typename _ValueType>
struct __enumerable_thread_local_storage
{
    template <typename... _LocalArgs>
    __enumerable_thread_local_storage(_LocalArgs&&... __args) : __storage(std::forward<_LocalArgs>(__args)...)
    {
    }

    std::size_t
    size() const
    {
        return std::size_t{1};
    }

    _ValueType&
    get_for_current_thread()
    {
        return __storage;
    }

    _ValueType&
    get_with_id(std::size_t /*__i*/)
    {
        return get_for_current_thread();
    }

    _ValueType __storage;
};

} //namespace __detail

// enumerable thread local storage should only be created from make function
template <typename _ValueType, typename... Args>
__detail::__enumerable_thread_local_storage<_ValueType>
__make_enumerable_tls(Args&&... __args)
{
    return __detail::__enumerable_thread_local_storage<_ValueType>(std::forward<Args>(__args)...);
}

template <typename _ExecutionPolicy, typename _Tp>
using __buffer = oneapi::dpl::__utils::__buffer_impl<std::decay_t<_ExecutionPolicy>, _Tp, std::allocator>;

inline void
__cancel_execution(oneapi::dpl::__internal::__serial_backend_tag)
{
}

template <class _ExecutionPolicy, class _Index, class _Fp>
void
__parallel_for(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last,
               _Fp __f, std::size_t /*__grainsize*/ = 1)
{
    __f(__first, __last);
}

template <class _ExecutionPolicy, class _Value, class _Index, typename _RealBody, typename _Reduction>
_Value
__parallel_reduce(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last,
                  const _Value& __identity, const _RealBody& __real_body, const _Reduction&)
{
    if (__first == __last)
    {
        return __identity;
    }
    else
    {
        return __real_body(__first, __last, __identity);
    }
}

template <class _ExecutionPolicy, class _Index, class _UnaryOp, class _Tp, class _BinaryOp, class _Reduce>
_Tp
__parallel_transform_reduce(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _Index __first,
                            _Index __last, _UnaryOp, _Tp __init, _BinaryOp, _Reduce __reduce)
{
    return __reduce(__first, __last, __init);
}

template <class _ExecutionPolicy, typename _Index, typename _Tp, typename _Rp, typename _Cp, typename _Sp, typename _Ap>
void
__parallel_strict_scan(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _Index __n, _Tp __initial,
                       _Rp __reduce, _Cp __combine, _Sp __scan, _Ap __apex)
{
    _Tp __sum = __initial;
    if (__n)
        __sum = __combine(__sum, __reduce(_Index(0), __n));
    __apex(__sum);
    if (__n)
        __scan(_Index(0), __n, __initial);
}

template <class _ExecutionPolicy, class _Index, class _UnaryOp, class _Tp, class _BinaryOp, class _Reduce, class _Scan>
_Tp
__parallel_transform_scan(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _Index __n, _UnaryOp,
                          _Tp __init, _BinaryOp, _Reduce, _Scan __scan)
{
    return __scan(_Index(0), __n, __init);
}

template <class _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare, typename _LeafSort>
void
__parallel_stable_sort(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _RandomAccessIterator __first,
                       _RandomAccessIterator __last, _Compare __comp, _LeafSort __leaf_sort, ::std::size_t = 0)
{
    __leaf_sort(__first, __last, __comp);
}

template <class _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3, typename _Compare, typename _LeafMerge>
void
__parallel_merge(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _RandomAccessIterator1 __first1,
                 _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                 _RandomAccessIterator3 __outit, _Compare __comp, _LeafMerge __leaf_merge)
{
    __leaf_merge(__first1, __last1, __first2, __last2, __outit, __comp);
}

template <class _ExecutionPolicy, typename _F1, typename _F2>
void
__parallel_invoke(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _F1&& __f1, _F2&& __f2)
{
    ::std::forward<_F1>(__f1)();
    ::std::forward<_F2>(__f2)();
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Fp>
void
__parallel_for_each(oneapi::dpl::__internal::__serial_backend_tag, _ExecutionPolicy&&, _ForwardIterator __begin,
                    _ForwardIterator __end, _Fp __f)
{
    for (auto __iter = __begin; __iter != __end; ++__iter)
        __f(*__iter);
}

} // namespace __serial_backend
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SERIAL_H
