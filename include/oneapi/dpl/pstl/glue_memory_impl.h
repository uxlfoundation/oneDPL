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

#ifndef _ONEDPL_GLUE_MEMORY_IMPL_H
#define _ONEDPL_GLUE_MEMORY_IMPL_H

#include "execution_defs.h"
#include "utils.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_impl_hetero.h"
#endif

#include "memory_fwd.h"
#include "algorithm_fwd.h"

#include "execution_impl.h"

#include <type_traits>

namespace oneapi
{
namespace dpl
{

// [uninitialized.copy]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _OutValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference  _OutRefType;
    typedef typename std::iterator_traits<_InputIterator>::reference    _InRefType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, _InRefType> && // required operation is trivial
                  std::is_trivially_default_constructible_v<_OutValueType> &&     // actual operations are trivial
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
            oneapi::dpl::__internal::__brick_copy<decltype(__dispatch_tag)>{});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk2(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
            oneapi::dpl::__internal::__op_uninitialized_copy<std::decay_t<_ExecutionPolicy>>{});
    }
}

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_copy_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _OutValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference  _OutRefType;
    typedef typename std::iterator_traits<_InputIterator>::reference    _InRefType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, _InRefType> && // required operation is trivial
                  std::is_trivially_default_constructible_v<_OutValueType> &&     // actual operations are trivial
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
            oneapi::dpl::__internal::__brick_copy_n<decltype(__dispatch_tag)>{});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk2_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
            oneapi::dpl::__internal::__op_uninitialized_copy<std::decay_t<_ExecutionPolicy>>{});
    }
}

// [uninitialized.move]

template <class _ExecutionPolicy, class _InputIterator, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move(_ExecutionPolicy&& __exec, _InputIterator __first, _InputIterator __last, _ForwardIterator __result)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _OutValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference  _OutRefType;
    typedef typename std::iterator_traits<_InputIterator>::reference    _InRefType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, std::remove_reference_t<_InRefType>&&> &&
                  std::is_trivially_default_constructible_v<_OutValueType> &&
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
            oneapi::dpl::__internal::__brick_copy<decltype(__dispatch_tag)>{});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk2(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
            oneapi::dpl::__internal::__op_uninitialized_move<std::decay_t<_ExecutionPolicy>>{});
    }
}

template <class _ExecutionPolicy, class _InputIterator, class _Size, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_move_n(_ExecutionPolicy&& __exec, _InputIterator __first, _Size __n, _ForwardIterator __result)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _OutValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference  _OutRefType;
    typedef typename std::iterator_traits<_InputIterator>::reference    _InRefType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first, __result);

    if constexpr (std::is_trivially_constructible_v<_OutValueType, std::remove_reference_t<_InRefType>&&> &&
                  std::is_trivially_default_constructible_v<_OutValueType> &&
                  std::is_trivially_assignable_v<_OutRefType, _InRefType>)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
            oneapi::dpl::__internal::__brick_copy_n<decltype(__dispatch_tag)>{});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk2_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result,
            oneapi::dpl::__internal::__op_uninitialized_move<std::decay_t<_ExecutionPolicy>>{});
    }
}

// [uninitialized.fill]

template <class _ExecutionPolicy, class _ForwardIterator, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_fill(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    if constexpr (std::is_trivially_constructible_v<_ValueType, _Tp> &&    // required operation is trivial
                  std::is_trivially_default_constructible_v<_ValueType> && // actual operations are trivial
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        oneapi::dpl::__internal::__pattern_walk_brick(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            oneapi::dpl::__internal::__brick_fill<decltype(__dispatch_tag), _ValueType>{_ValueType(__value)});
    }
    else
    {
        oneapi::dpl::__internal::__pattern_walk1(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            oneapi::dpl::__internal::__op_uninitialized_fill<_Tp, std::decay_t<_ExecutionPolicy>>{__value});
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_fill_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n, const _Tp& __value)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    if constexpr (std::is_trivially_constructible_v<_ValueType, _Tp> &&    // required operation is trivial
                  std::is_trivially_default_constructible_v<_ValueType> && // actual operations are trivial
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        return oneapi::dpl::__internal::__pattern_walk_brick_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n,
            oneapi::dpl::__internal::__brick_fill_n<decltype(__dispatch_tag), _ValueType>{_ValueType(__value)});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk1_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n,
            oneapi::dpl::__internal::__op_uninitialized_fill<_Tp, std::decay_t<_ExecutionPolicy>>{__value});
    }
}

#if (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)

inline const oneapi::dpl::execution::parallel_policy&
get_unvectorized_policy(const oneapi::dpl::execution::parallel_unsequenced_policy&)
{
    return oneapi::dpl::execution::par;
}

inline const oneapi::dpl::execution::sequenced_policy&
get_unvectorized_policy(const oneapi::dpl::execution::unsequenced_policy&)
{
    return oneapi::dpl::execution::seq;
}

template <typename _ExecutionPolicy>
const _ExecutionPolicy&
get_unvectorized_policy(const _ExecutionPolicy& __exec)
{
    return __exec;
}

#endif // (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)

namespace __internal
{
template <typename _ValueType, typename _ReferenceType>
struct __destroy_fn
{
    void
    operator()(_ReferenceType __val) const
    {
        __val.~_ValueType();
    }
};
}; // namespace __internal

// [specialized.destroy]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
destroy(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference _ReferenceType;

    if constexpr (!std::is_trivially_destructible_v<_ValueType>)
    {
        const auto __dispatch_tag =
#if (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)
            oneapi::dpl::__internal::__select_backend(get_unvectorized_policy(__exec), __first);
#else
            oneapi::dpl::__internal::__select_backend(__exec, __first);
#endif

        oneapi::dpl::__internal::__pattern_walk1(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first,
                                                 __last, __internal::__destroy_fn<_ValueType, _ReferenceType>{});
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
destroy_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;
    typedef typename std::iterator_traits<_ForwardIterator>::reference _ReferenceType;

    if constexpr (std::is_trivially_destructible_v<_ValueType>)
    {
        return oneapi::dpl::__internal::__pstl_next(__first, __n);
    }
    else
    {
        const auto __dispatch_tag =
#if (_PSTL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN || _ONEDPL_ICPX_OMP_SIMD_DESTROY_WINDOWS_BROKEN)
            oneapi::dpl::__internal::__select_backend(get_unvectorized_policy(__exec), __first);
#else
            oneapi::dpl::__internal::__select_backend(__exec, __first);
#endif

        return oneapi::dpl::__internal::__pattern_walk1_n(__dispatch_tag, std::forward<_ExecutionPolicy>(__exec),
                                                          __first, __n,
                                                          __internal::__destroy_fn<_ValueType, _ReferenceType>{});
    }
}

// [uninitialized.construct.default]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_default_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    if constexpr (!std::is_trivially_default_constructible_v<_ValueType>)
    {
        const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

        oneapi::dpl::__internal::__pattern_walk1(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            oneapi::dpl::__internal::__op_uninitialized_default_construct<std::decay_t<_ExecutionPolicy>>{});
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_default_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    if constexpr (std::is_trivially_default_constructible_v<_ValueType>)
    {
        return oneapi::dpl::__internal::__pstl_next(__first, __n);
    }
    else
    {
        const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

        return oneapi::dpl::__internal::__pattern_walk1_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n,
            oneapi::dpl::__internal::__op_uninitialized_default_construct<std::decay_t<_ExecutionPolicy>>{});
    }
}

// [uninitialized.construct.value]

template <class _ExecutionPolicy, class _ForwardIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy>
uninitialized_value_construct(_ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    if constexpr (std::is_trivially_default_constructible_v<_ValueType> &&
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        oneapi::dpl::__internal::__pattern_walk_brick(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            oneapi::dpl::__internal::__brick_fill<decltype(__dispatch_tag), _ValueType>{_ValueType()});
    }
    else
    {
        oneapi::dpl::__internal::__pattern_walk1(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            oneapi::dpl::__internal::__op_uninitialized_value_construct<std::decay_t<_ExecutionPolicy>>{});
    }
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Size>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _ForwardIterator>
uninitialized_value_construct_n(_ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n)
{
    typedef typename std::iterator_traits<_ForwardIterator>::value_type _ValueType;

    auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(__exec, __first);

    if constexpr (std::is_trivially_default_constructible_v<_ValueType> &&
                  std::is_trivially_copy_assignable_v<_ValueType>)
    {
        return oneapi::dpl::__internal::__pattern_walk_brick_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n,
            oneapi::dpl::__internal::__brick_fill_n<decltype(__dispatch_tag), _ValueType>{_ValueType()});
    }
    else
    {
        return oneapi::dpl::__internal::__pattern_walk1_n(
            __dispatch_tag, std::forward<_ExecutionPolicy>(__exec), __first, __n,
            oneapi::dpl::__internal::__op_uninitialized_value_construct<std::decay_t<_ExecutionPolicy>>{});
    }
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_GLUE_MEMORY_IMPL_H
