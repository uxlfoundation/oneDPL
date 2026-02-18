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

#ifndef _ONEDPL_ALGORITHM_IMPL_H
#define _ONEDPL_ALGORITHM_IMPL_H

#include <iterator>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "algorithm_fwd.h"

#include "execution_impl.h"
#include "memory_impl.h"
#include "parallel_backend_utils.h"
#include "unseq_backend_simd.h"

#include "parallel_backend.h"
#include "parallel_impl.h"
#include "iterator_impl.h"
#include "functional_impl.h" // for oneapi::dpl::identity, std::invoke

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/algorithm_impl_hetero.h" // for __pattern_fill_n, __pattern_generate_n
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <class _ForwardIterator, class _Pred>
bool
__brick_any_of(const _ForwardIterator __first, const _ForwardIterator __last, _Pred __pred,
               /*__is_vector=*/::std::false_type) noexcept
{
    return ::std::any_of(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _Pred>
bool
__brick_any_of(const _RandomAccessIterator __first, const _RandomAccessIterator __last, _Pred __pred,
               /*__is_vector=*/::std::true_type) noexcept
{
    return __unseq_backend::__simd_or(__first, __last - __first, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Pred>
bool
__pattern_any_of(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Pred __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_any_of(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Pred>
bool
__pattern_any_of(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                 _RandomAccessIterator __last, _Pred __pred)
{
    return __internal::__except_handler([&]() {
        return __internal::__parallel_or(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                         [__pred](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                                             return __internal::__brick_any_of(__i, __j, __pred, _IsVector{});
                                         });
    });
}

// [alg.foreach]
// for_each_n with no policy

template <class _ForwardIterator, class _Size, class _Function>
_ForwardIterator
__for_each_n_it_serial(_ForwardIterator __first, _Size __n, _Function __f)
{
    for (; __n > 0; ++__first, (void)--__n)
        __f(__first);
    return __first;
}

//------------------------------------------------------------------------
// walk1 (pseudo)
//
// walk1 evaluates f(x) for each dereferenced value x drawn from [first,last)
//------------------------------------------------------------------------
template <class _ForwardIterator, class _Function>
void
__brick_walk1(_ForwardIterator __first, _ForwardIterator __last, _Function __f, /*vector=*/::std::false_type)
{
    ::std::for_each(__first, __last, __f);
}

template <class _RandomAccessIterator, class _Function>
void
__brick_walk1(_RandomAccessIterator __first, _RandomAccessIterator __last, _Function __f,
              /*vector=*/::std::true_type)
{
    __unseq_backend::__simd_walk_n(__last - __first, __f, __first);
}

template <class _DifferenceType, class _Function>
void
__brick_walk1(_DifferenceType __n, _Function __f, ::std::false_type) noexcept
{
    for (_DifferenceType __i = 0; __i < __n; ++__i)
        __f(__i);
}

template <class _DifferenceType, class _Function>
void
__brick_walk1(_DifferenceType __n, _Function __f, ::std::true_type) noexcept
{
    // TODO: when using this overload the correctness of the vectorization depends on that functor is provided.
    // To avoid possible bugs we need to add a restriction on the functor so only the ones which would be
    // correctly vectorizes are used passed here. But for now, just re-direct to serial version.
    oneapi::dpl::__internal::__brick_walk1(__n, __f, ::std::false_type{});
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Function>
void
__pattern_walk1(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Function __f)
{
    static_assert(__is_serial_tag_v<_Tag>);

    __internal::__brick_walk1(__first, __last, __f, typename _Tag::__is_vector{});
}

template <class _ExecutionPolicy, class _ForwardIterator, class _Function>
void
__pattern_walk1(__parallel_forward_tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                _Function __f)
{
    using __backend_tag = typename __parallel_forward_tag::__backend_tag;

    using _ReferenceType = typename std::iterator_traits<_ForwardIterator>::reference;
    auto __func = [&__f](_ReferenceType arg) { __f(arg); };
    __internal::__except_handler([&]() {
        __par_backend::__parallel_for_each(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                           __func);
    });
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Function>
void
__pattern_walk1(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                _RandomAccessIterator __last, _Function __f)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                      [__f](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                                          __internal::__brick_walk1(__i, __j, __f, _IsVector{});
                                      });
    });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Brick>
void
__pattern_walk_brick(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                     _Brick __brick) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __brick(__first, __last, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Brick>
void
__pattern_walk_brick(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                     _RandomAccessIterator __last, _Brick __brick)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__brick](_RandomAccessIterator __i, _RandomAccessIterator __j) { __brick(__i, __j, _IsVector{}); });
    });
}

//------------------------------------------------------------------------
// walk1_n
//------------------------------------------------------------------------
template <class _ForwardIterator, class _Size, class _Function>
_ForwardIterator
__brick_walk1_n(_ForwardIterator __first, _Size __n, _Function __f, /*_IsVectorTag=*/::std::false_type)
{
    return __internal::__for_each_n_it_serial(__first, __n,
                                              [&__f](_ForwardIterator __it) { __f(*__it); }); // calling serial version
}

template <class _RandomAccessIterator, class _DifferenceType, class _Function>
_RandomAccessIterator
__brick_walk1_n(_RandomAccessIterator __first, _DifferenceType __n, _Function __f,
                /*vectorTag=*/::std::true_type)
{
    return __unseq_backend::__simd_walk_n(__n, __f, __first);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Function>
_ForwardIterator
__pattern_walk1_n(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _Size __n, _Function __f)
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_walk1_n(__first, __n, __f, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Size, class _Function>
_RandomAccessIterator
__pattern_walk1_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first, _Size __n,
                  _Function __f)
{
    oneapi::dpl::__internal::__pattern_walk1(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __n,
                                             __f);
    return __first + __n;
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Brick>
_ForwardIterator
__pattern_walk_brick_n(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _Size __n, _Brick __brick) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __brick(__first, __n, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Size, class _Brick>
_RandomAccessIterator
__pattern_walk_brick_n(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first, _Size __n,
                       _Brick __brick)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __n,
            [__brick](_RandomAccessIterator __i, _RandomAccessIterator __j) { __brick(__i, __j - __i, _IsVector{}); });
        return __first + __n;
    });
}

//------------------------------------------------------------------------
// walk2 (pseudo)
//
// walk2 evaluates f(x,y) for deferenced values (x,y) drawn from [first1,last1) and [first2,...)
//------------------------------------------------------------------------
template <class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2
__brick_walk2(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f,
              /*vector=*/::std::false_type) noexcept
{
    for (; __first1 != __last1; ++__first1, (void)++__first2)
        __f(*__first1, *__first2);
    return __first2;
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Function>
_RandomAccessIterator2
__brick_walk2(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
              _Function __f,
              /*vector=*/::std::true_type) noexcept
{
    return __unseq_backend::__simd_walk_n(__last1 - __first1, __f, __first1, __first2);
}

template <class _ForwardIterator1, class _Size, class _ForwardIterator2, class _Function>
_ForwardIterator2
__brick_walk2_n(_ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2, _Function __f,
                /*vector=*/::std::false_type) noexcept
{
    for (; __n > 0; --__n, (void)++__first1, ++__first2)
        __f(*__first1, *__first2);
    return __first2;
}

template <class _RandomAccessIterator1, class _Size, class _RandomAccessIterator2, class _Function>
_RandomAccessIterator2
__brick_walk2_n(_RandomAccessIterator1 __first1, _Size __n, _RandomAccessIterator2 __first2, _Function __f,
                /*vector=*/::std::true_type) noexcept
{
    return __unseq_backend::__simd_walk_n(__n, __f, __first1, __first2);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2
__pattern_walk2(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _Function __f) noexcept
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __internal::__brick_walk2(__first1, __last1, __first2, __f, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Function>
_RandomAccessIterator2
__pattern_walk2(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Function __f)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__f, __first1, __first2](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __internal::__brick_walk2(__i, __j, __first2 + (__i - __first1), __f, _IsVector{});
            });
        return __first2 + (__last1 - __first1);
    });
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Function>
_ForwardIterator2
__pattern_walk2(__parallel_forward_tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f)
{
    using __backend_tag = typename __parallel_forward_tag::__backend_tag;

    return __internal::__except_handler([&]() {
        using _iterator_tuple = zip_forward_iterator<_ForwardIterator1, _ForwardIterator2>;
        auto __begin = _iterator_tuple(__first1, __first2);
        auto __end = _iterator_tuple(__last1, /*dummy parameter*/ _ForwardIterator2());

        using _ReferenceType1 = typename std::iterator_traits<_ForwardIterator1>::reference;
        using _ReferenceType2 = typename std::iterator_traits<_ForwardIterator2>::reference;

        __par_backend::__parallel_for_each(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __begin, __end,
                                           [&__f](::std::tuple<_ReferenceType1, _ReferenceType2> __val) {
                                               __f(::std::get<0>(__val), ::std::get<1>(__val));
                                           });

        //TODO: parallel_for_each does not allow to return correct iterator value according to the ::std::transform
        // implementation. Therefore, iterator value is calculated separately.
        for (; __begin != __end; ++__begin)
            ;
        return ::std::get<1>(__begin.base());
    });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _Size, class _ForwardIterator2,
          class _Function>
_ForwardIterator2
__pattern_walk2_n(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _Size __n, _ForwardIterator2 __first2,
                  _Function __f) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_walk2_n(__first1, __n, __first2, __f, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _Size,
          class _RandomAccessIterator2, class _Function>
_RandomAccessIterator2
__pattern_walk2_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                  _Size __n, _RandomAccessIterator2 __first2, _Function __f)
{
    return __internal::__pattern_walk2(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n,
                                       __first2, __f);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Brick>
_ForwardIterator2
__pattern_walk2_brick(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                      _ForwardIterator2 __first2, _Brick __brick) noexcept
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __brick(__first1, __last1, __first2, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Brick>
_RandomAccessIterator2
__pattern_walk2_brick(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                      _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Brick __brick)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__first1, __first2, __brick](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __brick(__i, __j, __first2 + (__i - __first1), _IsVector{});
            });
        return __first2 + (__last1 - __first1);
    });
}

//TODO: it postponed till adding more or less effective parallel implementation
template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Brick>
_ForwardIterator2
__pattern_walk2_brick(__parallel_forward_tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                      _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Brick __brick)
{
    using __backend_tag = typename __parallel_forward_tag::__backend_tag;

    using _iterator_tuple = zip_forward_iterator<_ForwardIterator1, _ForwardIterator2>;
    auto __begin = _iterator_tuple(__first1, __first2);
    auto __end = _iterator_tuple(__last1, /*dummy parameter*/ _ForwardIterator2());

    using _ReferenceType1 = typename std::iterator_traits<_ForwardIterator1>::reference;
    using _ReferenceType2 = typename std::iterator_traits<_ForwardIterator2>::reference;

    return __except_handler([&]() {
        __par_backend::__parallel_for_each(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __begin, __end,
                                           [__brick](::std::tuple<_ReferenceType1, _ReferenceType2> __val) {
                                               __brick(::std::get<0>(__val),
                                                       ::std::forward<_ReferenceType2>(::std::get<1>(__val)));
                                           });

        //TODO: parallel_for_each does not allow to return correct iterator value according to the ::std::transform
        // implementation. Therefore, iterator value is calculated separately.
        for (; __begin != __end; ++__begin)
            ;
        return ::std::get<1>(__begin.base());
    });
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _Size,
          class _RandomAccessIterator2, class _Brick>
_RandomAccessIterator2
__pattern_walk2_brick_n(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                        _Size __n, _RandomAccessIterator2 __first2, _Brick __brick)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n,
            [__first1, __first2, __brick](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __brick(__i, __j - __i, __first2 + (__i - __first1), _IsVector{});
            });
        return __first2 + __n;
    });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _Size, class _ForwardIterator2,
          class _Brick>
_ForwardIterator2
__pattern_walk2_brick_n(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _Size __n,
                        _ForwardIterator2 __first2, _Brick __brick) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __brick(__first1, __n, __first2, typename _Tag::__is_vector{});
}

//------------------------------------------------------------------------
// walk3 (pseudo)
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------
template <class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3, class _Function>
_ForwardIterator3
__brick_walk3(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator3 __first3, _Function __f, /*vector=*/::std::false_type) noexcept
{
    for (; __first1 != __last1; ++__first1, (void)++__first2, ++__first3)
        __f(*__first1, *__first2, *__first3);
    return __first3;
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Function>
_RandomAccessIterator3
__brick_walk3(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
              _RandomAccessIterator3 __first3, _Function __f, /*vector=*/::std::true_type) noexcept
{
    return __unseq_backend::__simd_walk_n(__last1 - __first1, __f, __first1, __first2, __first3);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3,
          class _Function>
_ForwardIterator3
__pattern_walk3(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f) noexcept
{
    static_assert(__is_serial_tag_v<_Tag>);

    return __internal::__brick_walk3(__first1, __last1, __first2, __first3, __f, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _Function>
_RandomAccessIterator3
__pattern_walk3(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator3 __first3,
                _Function __f)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__f, __first1, __first2, __first3](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                __internal::__brick_walk3(__i, __j, __first2 + (__i - __first1), __first3 + (__i - __first1), __f,
                                          _IsVector{});
            });
        return __first3 + (__last1 - __first1);
    });
}

template <class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardIterator3,
          class _Function>
_ForwardIterator3
__pattern_walk3(__parallel_forward_tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f)
{
    using __backend_tag = typename __parallel_forward_tag::__backend_tag;

    return __internal::__except_handler([&]() {
        using _iterator_tuple = zip_forward_iterator<_ForwardIterator1, _ForwardIterator2, _ForwardIterator3>;
        auto __begin = _iterator_tuple(__first1, __first2, __first3);
        auto __end = _iterator_tuple(__last1, /*dummy parameter*/ _ForwardIterator2(),
                                     /*dummy parameter*/ _ForwardIterator3());

        using _ReferenceType1 = typename std::iterator_traits<_ForwardIterator1>::reference;
        using _ReferenceType2 = typename std::iterator_traits<_ForwardIterator2>::reference;
        using _ReferenceType3 = typename std::iterator_traits<_ForwardIterator3>::reference;

        __par_backend::__parallel_for_each(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __begin, __end,
                                           [&](::std::tuple<_ReferenceType1, _ReferenceType2, _ReferenceType3> __val) {
                                               __f(::std::get<0>(__val), ::std::get<1>(__val), ::std::get<2>(__val));
                                           });

        //TODO: parallel_for_each does not allow to return correct iterator value according to the ::std::transform
        // implementation. Therefore, iterator value is calculated separately.
        for (; __begin != __end; ++__begin)
            ;
        return ::std::get<2>(__begin.base());
    });
}

//------------------------------------------------------------------------
// transform_if
//------------------------------------------------------------------------

template <class _Tag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Function>
_ForwardIterator2
__pattern_walk2_transform_if(_Tag __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                             _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __func) noexcept
{
    static_assert(__is_host_dispatch_tag_v<_Tag>);

    return __pattern_walk2(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __func);
}

template <class _Tag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _ForwardIterator3, typename _Function>
_ForwardIterator3
__pattern_walk3_transform_if(_Tag __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                             _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3,
                             _Function __func) noexcept
{
    static_assert(__is_host_dispatch_tag_v<_Tag>);

    return __pattern_walk3(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __first3,
                           __func);
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool
__brick_equal(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator2 __last2, _BinaryPredicate __p, /* IsVector = */ ::std::false_type) noexcept
{
    return ::std::equal(__first1, __last1, __first2, __last2, __p);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
bool
__brick_equal(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
              _RandomAccessIterator2 __last2, _BinaryPredicate __p, /* is_vector = */ ::std::true_type) noexcept
{
    if (__last1 - __first1 != __last2 - __first2)
        return false;

    return __unseq_backend::__simd_first(__first1, __last1 - __first1, __first2, __not_pred<_BinaryPredicate&>(__p))
               .first == __last1;
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool
__pattern_equal(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __p) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_equal(__first1, __last1, __first2, __last2, __p, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
bool
__pattern_equal(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                _BinaryPredicate __p)
{
    if (__last1 - __first1 != __last2 - __first2)
        return false;

    if (__last1 - __first1 == 0)
        return true;

    return __internal::__except_handler([&]() {
        return !__internal::__parallel_or(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__first1, __first2, __p](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                return !__internal::__brick_equal(__i, __j, __first2 + (__i - __first1), __first2 + (__j - __first1),
                                                  __p, _IsVector{});
            });
    });
}

//------------------------------------------------------------------------
// equal version for sequences with equal length
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool
__brick_equal(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _BinaryPredicate __p,
              /* IsVector = */ ::std::false_type) noexcept
{
    return ::std::equal(__first1, __last1, __first2, __p);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
bool
__brick_equal(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
              _BinaryPredicate __p, /* is_vector = */ ::std::true_type) noexcept
{
    return __unseq_backend::__simd_first(__first1, __last1 - __first1, __first2, __not_pred<_BinaryPredicate&>(__p))
               .first == __last1;
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
bool
__pattern_equal(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _BinaryPredicate __p) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_equal(__first1, __last1, __first2, __p, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
bool
__pattern_equal(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _BinaryPredicate __p)
{
    return __internal::__except_handler([&]() {
        return !__internal::__parallel_or(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__first1, __first2, __p](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                return !__internal::__brick_equal(__i, __j, __first2 + (__i - __first1), __p, _IsVector{});
            });
    });
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------
template <class _ForwardIterator, class _Predicate>
_ForwardIterator
__brick_find_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred,
                /*is_vector=*/::std::false_type) noexcept
{
    return ::std::find_if(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _Predicate>
_RandomAccessIterator
__brick_find_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate __pred,
                /*is_vector=*/::std::true_type) noexcept
{
    using _SizeType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
    return __unseq_backend::__simd_first(__first, _SizeType(0), __last - __first, __internal::__pred_at_index{__pred});
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
_ForwardIterator
__pattern_find_if(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                  _Predicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_find_if(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Predicate>
_RandomAccessIterator
__pattern_find_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                  _RandomAccessIterator __last, _Predicate __pred)
{
    return __except_handler([&]() {
        return __parallel_find(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__pred](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                return __brick_find_if(__i, __j, __pred, _IsVector{});
            },
            ::std::true_type{});
    });
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

// find the first occurrence of the subsequence [s_first, s_last)
//   or the  last occurrence of the subsequence in the range [first, last)
// b_first determines what occurrence we want to find (first or last)
template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate, class _IsVector>
_RandomAccessIterator1
__find_subrange(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator1 __global_last,
                _RandomAccessIterator2 __s_first, _RandomAccessIterator2 __s_last, _BinaryPredicate __pred,
                bool __b_first, _IsVector __is_vector) noexcept
{
    auto __n2 = __s_last - __s_first;
    if (__n2 < 1)
    {
        return __b_first ? __first : __last;
    }

    auto __n1 = __global_last - __first;
    if (__n1 < __n2)
    {
        return __last;
    }

    auto __cur = __last;
    while (__first != __last && (__global_last - __first >= __n2))
    {
        // find position of *s_first in [first, last) (it can be start of subsequence)
        auto __u_pred =
            [__pred, __s_first](auto&& __val) mutable { return __pred(std::forward<decltype(__val)>(__val), *__s_first); };
        __first = __internal::__brick_find_if(__first, __last, __u_pred, __is_vector);

        // if position that was found previously is the start of subsequence
        // then we can exit the loop (b_first == true) or keep the position
        // (b_first == false)
        if (__first != __last && (__global_last - __first >= __n2) &&
            __internal::__brick_equal(__s_first + 1, __s_last, __first + 1, __pred, __is_vector))
        {
            if (__b_first)
            {
                return __first;
            }
            else
            {
                __cur = __first;
            }
        }
        else if (__first == __last)
        {
            break;
        }
        else
        {
        }

        // in case of b_first == false we try to find new start position
        // for the next subsequence
        ++__first;
    }
    return __cur;
}

template <class _RandomAccessIterator, class _Size, class _Tp, class _BinaryPredicate, class _IsVector>
_RandomAccessIterator
__find_subrange(_RandomAccessIterator __first, _RandomAccessIterator __last, _RandomAccessIterator __global_last,
                _Size __count, const _Tp& __value, _BinaryPredicate __pred, _IsVector __is_vector) noexcept
{
    if (__count < 1)
    {
        return __first; // According to the standard std::search_n shall return first when count < 1
    }

    if (static_cast<_Size>(__global_last - __first) < __count)
    {
        return __last;
    }

    auto __unary_pred =
        [__pred, &__value](auto&& __val) mutable { return __pred(std::forward<decltype(__val)>(__val), __value); };
    while (__first != __last && (static_cast<_Size>(__global_last - __first) >= __count))
    {
        __first = __internal::__brick_find_if(__first, __last, __unary_pred, __is_vector);

        // check that all of elements in [first+1, first+count) equal to value
        if (__first != __last && (__global_last - __first >= __count) &&
            !__internal::__brick_any_of(__first + 1, __first + __count,
                                        __not_pred<decltype(__unary_pred)&>(__unary_pred), __is_vector))
        {
            return __first;
        }
        else if (__first == __last)
        {
            break;
        }
        else
        {
            ++__first;
        }
    }
    return __last;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__brick_find_end(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                 _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/::std::false_type) noexcept
{
    return ::std::find_end(__first, __last, __s_first, __s_last, __pred);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
_RandomAccessIterator1
__brick_find_end(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __s_first,
                 _RandomAccessIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/::std::true_type) noexcept
{
    return __find_subrange(__first, __last, __last, __s_first, __s_last, __pred, false, ::std::true_type());
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__pattern_find_end(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                   _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_find_end(__first, __last, __s_first, __s_last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
_RandomAccessIterator1
__pattern_find_end(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                   _RandomAccessIterator1 __last, _RandomAccessIterator2 __s_first, _RandomAccessIterator2 __s_last,
                   _BinaryPredicate __pred)
{
    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __internal::__pattern_equal(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                       __s_first, __pred);
        return __res ? __first : __last;
    }
    else
    {
        return __internal::__except_handler([&]() {
            return __internal::__parallel_find(
                __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                [__last, __s_first, __s_last, __pred](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                    return __internal::__find_subrange(__i, __j, __last, __s_first, __s_last, __pred, false,
                                                       _IsVector{});
                },
                ::std::false_type{});
        });
    }
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------
template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__brick_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                      _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/::std::false_type) noexcept
{
    return ::std::find_first_of(__first, __last, __s_first, __s_last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__brick_find_first_of(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
                      _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*__is_vector=*/::std::true_type) noexcept
{
    return __unseq_backend::__simd_find_first_of(__first, __last, __s_first, __s_last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__pattern_find_first_of(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                        _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_find_first_of(__first, __last, __s_first, __s_last, __pred,
                                             typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
_RandomAccessIterator1
__pattern_find_first_of(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                        _RandomAccessIterator1 __last, _RandomAccessIterator2 __s_first,
                        _RandomAccessIterator2 __s_last, _BinaryPredicate __pred)
{
    return __internal::__except_handler([&]() {
        return __internal::__parallel_find(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__s_first, __s_last, &__pred](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                return __internal::__brick_find_first_of(__i, __j, __s_first, __s_last, __pred, _IsVector{});
            },
            ::std::true_type{});
    });
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------
template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__brick_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
               _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*vector=*/::std::false_type) noexcept
{
    return ::std::search(__first, __last, __s_first, __s_last, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__brick_search(_ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __s_first,
               _ForwardIterator2 __s_last, _BinaryPredicate __pred, /*vector=*/::std::true_type) noexcept
{
    return __internal::__find_subrange(__first, __last, __last, __s_first, __s_last, __pred, true, ::std::true_type());
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_ForwardIterator1
__pattern_search(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first, _ForwardIterator1 __last,
                 _ForwardIterator2 __s_first, _ForwardIterator2 __s_last, _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_search(__first, __last, __s_first, __s_last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
_RandomAccessIterator1
__pattern_search(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                 _RandomAccessIterator1 __last, _RandomAccessIterator2 __s_first, _RandomAccessIterator2 __s_last,
                 _BinaryPredicate __pred)
{
    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __internal::__pattern_equal(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                                       __s_first, __pred);
        return __res ? __first : __last;
    }
    else
    {
        return __internal::__except_handler([&]() {
            return __internal::__parallel_find(
                __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                [__last, __s_first, __s_last, __pred](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                    return __internal::__find_subrange(__i, __j, __last, __s_first, __s_last, __pred, true,
                                                       _IsVector{});
                },
                /*_IsFirst=*/::std::true_type{});
        });
    }
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------
template <class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_ForwardIterator
__brick_search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value,
                 _BinaryPredicate __pred, /*vector=*/::std::false_type) noexcept
{
    return ::std::search_n(__first, __last, __count, __value, __pred);
}

template <class _RandomAccessIterator, class _Size, class _Tp, class _BinaryPredicate>
_RandomAccessIterator
__brick_search_n(_RandomAccessIterator __first, _RandomAccessIterator __last, _Size __count, const _Tp& __value,
                 _BinaryPredicate __pred, /*vector=*/::std::true_type) noexcept
{
    return __internal::__find_subrange(__first, __last, __last, __count, __value, __pred, ::std::true_type());
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_ForwardIterator
__pattern_search_n(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Size __count,
                   const _Tp& __value, _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_search_n(__first, __last, __count, __value, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Size, class _Tp,
          class _BinaryPredicate>
_RandomAccessIterator
__pattern_search_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                   _RandomAccessIterator __last, _Size __count, const _Tp& __value, _BinaryPredicate __pred)
{
    if (static_cast<_Size>(__last - __first) == __count)
    {
        const bool __result = !__internal::__pattern_any_of(__tag, std::forward<_ExecutionPolicy>(__exec), __first,
            __last, [&__value, __pred](auto&& __val) mutable { return !__pred(std::forward<decltype(__val)>(__val),
            __value); });
        return __result ? __first : __last;
    }
    else
    {
        return __internal::__except_handler([__tag, &__exec, __first, __last, __count, &__value, __pred]() {
            return __internal::__parallel_find(
                __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                [__last, __count, &__value, __pred](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                    return __internal::__find_subrange(__i, __j, __last, __count, __value, __pred, _IsVector{});
                },
                ::std::true_type{});
        });
    }
}

//------------------------------------------------------------------------
// copy_n
//------------------------------------------------------------------------
// It might be possible to share more between copy and copy_n, but it's not
// clear that doing so is worth the trouble and extra layers of call chain.
// Sometimes a little duplication for sake of regularity is better than the alternative.

template <class _Tag>
struct __brick_copy_n<_Tag, std::enable_if_t<oneapi::dpl::__internal::__is_host_dispatch_tag_v<_Tag>>>
{
    template <typename _RandomAccessIterator1, typename _Size, typename _RandomAccessIterator2>
    _RandomAccessIterator2
    operator()(_RandomAccessIterator1 __first, _Size __n, _RandomAccessIterator2 __result,
               /*vec*/ ::std::true_type) const
    {
        return __unseq_backend::__simd_assign(
            __first, __n, __result,
            [](_RandomAccessIterator1 __first, _RandomAccessIterator2 __result) { *__result = *__first; });
    }

    template <typename _Iterator, typename _Size, typename _OutputIterator>
    _OutputIterator
    operator()(_Iterator __first, _Size __n, _OutputIterator __result, /*vec*/ ::std::false_type) const
    {
        return ::std::copy_n(__first, __n, __result);
    }
};

//------------------------------------------------------------------------
// copy
//------------------------------------------------------------------------

template <class _Tag>
struct __brick_copy<_Tag, std::enable_if_t<__is_host_dispatch_tag_v<_Tag>>>
{
    template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
    _RandomAccessIterator2
    operator()(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
               /*vec*/ ::std::true_type) const
    {
        return __unseq_backend::__simd_assign(
            __first, __last - __first, __result,
            [](_RandomAccessIterator1 __first, _RandomAccessIterator2 __result) { *__result = *__first; });
    }

    template <typename _Iterator, typename _OutputIterator>
    _OutputIterator
    operator()(_Iterator __first, _Iterator __last, _OutputIterator __result, /*vec*/ ::std::false_type) const
    {
        return ::std::copy(__first, __last, __result);
    }

    template <typename _ReferenceType1, typename _ReferenceType2>
    void
    operator()(_ReferenceType1 __val, _ReferenceType2&& __result) const
    {
        __result = __val;
    }
};

//------------------------------------------------------------------------
// move
//------------------------------------------------------------------------

template <class _Tag>
struct __brick_move<_Tag, std::enable_if_t<__is_host_dispatch_tag_v<_Tag>>>
{
    template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
    _RandomAccessIterator2
    operator()(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
               /*vec*/ ::std::true_type) const
    {
        return __unseq_backend::__simd_assign(
            __first, __last - __first, __result,
            [](_RandomAccessIterator1 __first, _RandomAccessIterator2 __result) { *__result = ::std::move(*__first); });
    }

    template <typename _Iterator, typename _OutputIterator>
    _OutputIterator
    operator()(_Iterator __first, _Iterator __last, _OutputIterator __result, /*vec*/ ::std::false_type) const
    {
        return ::std::move(__first, __last, __result);
    }

    template <typename _ReferenceType1, typename _ReferenceType2>
    void
    operator()(_ReferenceType1&& __val, _ReferenceType2&& __result) const
    {
        __result = ::std::move(__val);
    }
};

template <class _Tag, typename = std::enable_if_t<__is_host_dispatch_tag_v<_Tag>>>
struct __brick_move_destroy
{
    template <typename _RandomAccessIterator1, typename _RandomAccessIterator2>
    _RandomAccessIterator2
    operator()(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
               /*vec*/ ::std::true_type) const
    {
        using _IteratorValueType = typename ::std::iterator_traits<_RandomAccessIterator1>::value_type;

        return __unseq_backend::__simd_assign(__first, __last - __first, __result,
                                              [](_RandomAccessIterator1 __first, _RandomAccessIterator2 __result) {
                                                  *__result = ::std::move(*__first);
                                                  (*__first).~_IteratorValueType();
                                              });
    }

    template <typename _Iterator, typename _OutputIterator>
    _OutputIterator
    operator()(_Iterator __first, _Iterator __last, _OutputIterator __result, /*vec*/ ::std::false_type) const
    {
        using _IteratorValueType = typename ::std::iterator_traits<_Iterator>::value_type;

        for (; __first != __last; ++__first, (void)++__result)
        {
            *__result = ::std::move(*__first);
            (*__first).~_IteratorValueType();
        }
        return __result;
    }
};

//------------------------------------------------------------------------
// swap_ranges
//------------------------------------------------------------------------
template <class _ForwardIterator, class _OutputIterator>
_OutputIterator
__brick_swap_ranges(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                    /*vector=*/::std::false_type) noexcept
{
    return ::std::swap_ranges(__first, __last, __result);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2>
_RandomAccessIterator2
__brick_swap_ranges(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
                    /*vector=*/::std::true_type) noexcept
{
    using ::std::iter_swap;
    return __unseq_backend::__simd_assign(__first, __last - __first, __result,
                                          iter_swap<_RandomAccessIterator1, _RandomAccessIterator2>);
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------
template <class _ForwardIterator, class _OutputIterator, class _UnaryPredicate>
_OutputIterator
__brick_copy_if(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result, _UnaryPredicate __pred,
                /*vector=*/std::false_type) noexcept
{
    return std::copy_if(__first, __last, __result, __pred);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _UnaryPredicate>
_RandomAccessIterator2
__brick_copy_if(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
                _UnaryPredicate __pred, /*vector=*/std::true_type) noexcept
{
    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    _DifferenceType __n = __last - __first;
    return __result + __unseq_backend::__simd_selective_copy</*bounded =*/ false>(
                          __first, __n, __result, __n, __internal::__pred_at_index{__pred}).second;
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _UnaryPredicate>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__brick_bounded_copy_if(_RandomAccessIterator1 __first,
                        typename std::iterator_traits<_RandomAccessIterator1>::difference_type __n,
                        _RandomAccessIterator2 __result,
                        typename std::iterator_traits<_RandomAccessIterator2>::difference_type __n_out,
                        _UnaryPredicate __pred, /*vector=*/std::true_type) noexcept
{
    using _DifferenceType = std::common_type_t<decltype(__n), decltype(__n_out)>;
    auto [__stop_in, __stop_out] = __unseq_backend::__simd_selective_copy</*bounded =*/ true>(
        __first, _DifferenceType(__n), __result, _DifferenceType(__n_out), __internal::__pred_at_index{__pred});
    return {__first + __stop_in, __result + __stop_out};
}

template <class _RandomAccessIterator, class _DifferenceType, class _IndexPredicate>
std::pair<_DifferenceType, _DifferenceType>
__brick_compute_mask(_RandomAccessIterator __first, _DifferenceType __len, _IndexPredicate __pred, bool* __mask,
                     /*vector=*/std::false_type) noexcept
{
    _DifferenceType __count_true = 0;
    for (_DifferenceType __i = 0; __i < __len; ++__i)
    {
        __mask[__i] = __pred(__first, __i);
        __count_true += __mask[__i];
    }
    return std::make_pair(__count_true, __len - __count_true);
}

template <class _RandomAccessIterator, class _DifferenceType, class _IndexPredicate>
std::pair<_DifferenceType, _DifferenceType>
__brick_compute_mask(_RandomAccessIterator __first, _DifferenceType __len, _IndexPredicate __pred, bool* __mask,
                     /*vector=*/std::true_type) noexcept
{
    auto __count_true = __unseq_backend::__simd_compute_mask(__first, __len, __pred, __mask);
    return std::make_pair(__count_true, __len - __count_true);
}

template <bool __Bounded, class _RandomAccessIterator1, class _RandomAccessIterator2, class _Bound, class _Assigner>
_Bound
__brick_copy_by_mask(_RandomAccessIterator1 __first, _Bound __in_len, _RandomAccessIterator2 __result, _Bound __out_len,
                     bool* __mask, _Assigner __assigner, /*vector=*/std::false_type) noexcept
{
    _Bound __i = 0, __j = 0;
    for (; __i < __in_len; ++__i)
    {
        if (__mask[__i])
        {
            if constexpr (__Bounded)
                if (__j == __out_len)
                    break;
            __assigner(__first + __i, __result + __j);
            ++__j;
        }
    }
    return __i;
}

template <bool __Bounded, class _RandomAccessIterator1, class _RandomAccessIterator2, class _Bound, class _Assigner>
_Bound
__brick_copy_by_mask(_RandomAccessIterator1 __first, _Bound __in_len, _RandomAccessIterator2 __result, _Bound __out_len,
                     bool* __mask, _Assigner __assigner, /*vector=*/std::true_type) noexcept
{
    return __unseq_backend::__simd_copy_by_mask<__Bounded>(__first, __in_len, __result, __out_len, __mask, __assigner);
}

template <class _ForwardIterator, class _OutputIterator1, class _OutputIterator2>
void
__brick_partition_by_mask(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true,
                          _OutputIterator2 __out_false, bool* __mask, /*vector=*/::std::false_type) noexcept
{
    for (; __first != __last; ++__first, (void)++__mask)
    {
        if (*__mask)
        {
            *__out_true = *__first;
            ++__out_true;
        }
        else
        {
            *__out_false = *__first;
            ++__out_false;
        }
    }
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3>
void
__brick_partition_by_mask(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                          _RandomAccessIterator2 __out_true, _RandomAccessIterator3 __out_false, bool* __mask,
                          /*vector=*/::std::true_type) noexcept
{
#if (_PSTL_MONOTONIC_PRESENT || _ONEDPL_MONOTONIC_PRESENT)
    __unseq_backend::__simd_partition_by_mask(__first, __last - __first, __out_true, __out_false, __mask);
#else
    __internal::__brick_partition_by_mask(__first, __last, __out_true, __out_false, __mask, ::std::false_type());
#endif
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _DifferenceType,
          class _RandomAccessIterator2, class _IndexPredicate>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__parallel_selective_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                          _DifferenceType __n, _RandomAccessIterator2 __result, _DifferenceType __n_out,
                          _IndexPredicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;
    __par_backend::__buffer<bool> __mask_buf(__n);
    bool* __mask = __mask_buf.get();

    return __internal::__except_handler([&__exec, __n, __first, __result, __pred, __mask, __n_out]() {
        _DifferenceType __stop_in{__n}, __stop_out{__n_out};
        __par_backend::__parallel_strict_scan(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), __n, _DifferenceType(0),
            [=](_DifferenceType __i, _DifferenceType __len) { // Reduce
                return __internal::__brick_compute_mask(__first + __i, __len, __pred, __mask + __i, _IsVector{}).first;
            },
            std::plus<_DifferenceType>(), // Combine
            [=, &__stop_in](_DifferenceType __i, _DifferenceType __len, _DifferenceType __initial) { // Scan
                if (__initial > __n_out) // The chunk has neither elements to write nor the stop position
                    return;
                auto __assign = [](_RandomAccessIterator1 __x, _RandomAccessIterator2 __z) { *__z = *__x; };
                _DifferenceType __space = __n_out - __initial;
                if (__space >= __len)
                {
                    __internal::__brick_copy_by_mask</*bounded =*/ false>(
                        __first + __i, __len, __result + __initial, __space, __mask + __i, __assign, _IsVector{});
                }
                else
                {
                    _DifferenceType __stop = __internal::__brick_copy_by_mask</*bounded =*/ true>(
                        __first + __i, __len, __result + __initial, __space, __mask + __i, __assign, _IsVector{});
                    if (__stop != __len)          // Found the position of the first element that cannot be copied
                        __stop_in = __i + __stop; // Since there is only one such position, there is no data race
                }
            },
            [&__stop_out](_DifferenceType __total) { // Apex
                if (__total < __stop_out) // Output size is bigger than needed
                    __stop_out = __total;
            });
        return std::make_pair(__first + __stop_in, __result + __stop_out);
    });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryPredicate>
_OutputIterator
__pattern_copy_if(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                  _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_copy_if(__first, __last, __result, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _UnaryPredicate>
_RandomAccessIterator2
__pattern_copy_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                  _RandomAccessIterator1 __last, _RandomAccessIterator2 __result, _UnaryPredicate __pred)
{
    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    const _DifferenceType __n = __last - __first;
    if (_DifferenceType(1) < __n)
    {
        return __parallel_selective_copy(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result, __n,
                                         __internal::__pred_at_index{__pred}).second;
    }
    // trivial sequence - use serial algorithm
    return __internal::__brick_copy_if(__first, __last, __result, __pred, _IsVector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _DifferenceType,
          class _RandomAccessIterator2, class _UnaryPredicate>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__pattern_bounded_copy_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                          _DifferenceType __n, _RandomAccessIterator2 __result, _DifferenceType __n_out,
                          _UnaryPredicate __pred)
{
    return __parallel_selective_copy(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result, __n_out,
                                     __internal::__pred_at_index{__pred});
}

//------------------------------------------------------------------------
// count
//------------------------------------------------------------------------
template <class _RandomAccessIterator, class _Predicate>
typename ::std::iterator_traits<_RandomAccessIterator>::difference_type
__brick_count(_RandomAccessIterator __first, _RandomAccessIterator __last, _Predicate __pred,
              /* is_vector = */ ::std::true_type) noexcept
{
    return __unseq_backend::__simd_count(__first, __last - __first, __pred);
}

template <class _ForwardIterator, class _Predicate>
typename ::std::iterator_traits<_ForwardIterator>::difference_type
__brick_count(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred,
              /* is_vector = */ ::std::false_type) noexcept
{
    return ::std::count_if(__first, __last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Predicate>
typename ::std::iterator_traits<_ForwardIterator>::difference_type
__pattern_count(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_count(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Predicate>
typename ::std::iterator_traits<_RandomAccessIterator>::difference_type
__pattern_count(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                _RandomAccessIterator __last, _Predicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _SizeType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;

    //trivial pre-checks
    if (__first == __last)
        return _SizeType(0);

    return __internal::__except_handler([&]() {
        return __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, _SizeType(0),
            [__pred](_RandomAccessIterator __begin, _RandomAccessIterator __end, _SizeType __value) -> _SizeType {
                return __value + __internal::__brick_count(__begin, __end, __pred, _IsVector{});
            },
            ::std::plus<_SizeType>());
    });
}

//------------------------------------------------------------------------
// unique
//------------------------------------------------------------------------

template <class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator
__brick_unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred,
               /*is_vector=*/::std::false_type) noexcept
{
    return ::std::unique(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _BinaryPredicate>
_RandomAccessIterator
__brick_unique(_RandomAccessIterator __first, _RandomAccessIterator __last, _BinaryPredicate __pred,
               /*is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return ::std::unique(__first, __last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator
__pattern_unique(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                 _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_unique(__first, __last, __pred, typename _Tag::__is_vector{});
}

// That function is shared between two algorithms - remove_if (__pattern_remove_if) and unique (pattern unique). But a mask calculation is different.
// So, a caller passes _CalcMask brick into remove_elements.
template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _CalcMask>
_RandomAccessIterator
__remove_elements(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                  _RandomAccessIterator __last, _CalcMask __calc_mask)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
    using _Tp = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    _DifferenceType __n = __last - __first;
    __par_backend::__buffer<bool> __mask_buf(__n);
    // 1. find a first iterator that should be removed
    return __internal::__except_handler([&]() {
        bool* __mask = __mask_buf.get();
        _DifferenceType __min = __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), _DifferenceType(0), __n, __n,
            [__first, __mask, &__calc_mask](_DifferenceType __i, _DifferenceType __j,
                                            _DifferenceType __local_min) -> _DifferenceType {
                // Create mask
                __calc_mask(__mask + __i, __mask + __j, __first + __i);

                // if minimum was found in a previous range we shouldn't do anymore
                if (__local_min < __i)
                {
                    return __local_min;
                }
                // find first iterator that should be removed
                bool* __result = __internal::__brick_find_if(
                    __mask + __i, __mask + __j, [](bool __val) { return !__val; }, _IsVector{});
                if (__result - __mask == __j)
                {
                    return __local_min;
                }
                return ::std::min(__local_min, _DifferenceType(__result - __mask));
            },
            [](_DifferenceType __local_min1, _DifferenceType __local_min2) -> _DifferenceType {
                return ::std::min(__local_min1, __local_min2);
            });

        // No elements to remove - exit
        if (__min == __n)
        {
            return __last;
        }
        __n -= __min;
        __first += __min;

        __par_backend::__buffer<_Tp> __buf(__n);
        _Tp* __result = __buf.get();
        __mask += __min;
        _DifferenceType __m{};
        // 2. Elements that doesn't satisfy pred are moved to result
        __par_backend::__parallel_strict_scan(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), __n, _DifferenceType(0),
            [__mask](_DifferenceType __i, _DifferenceType __len) {
                return __internal::__brick_count(
                    __mask + __i, __mask + __i + __len, [](bool __val) { return __val; }, _IsVector{});
            },
            ::std::plus<_DifferenceType>(),
            [=](_DifferenceType __i, _DifferenceType __len, _DifferenceType __initial) {
                __internal::__brick_copy_by_mask</*bounded*/ false>(
                    __first + __i, __len, __result + __initial, __len, __mask + __i,
                    [](_RandomAccessIterator __x, _Tp* __z) { ::new (std::addressof(*__z)) _Tp(std::move(*__x)); },
                    _IsVector{});
            },
            [&__m](_DifferenceType __total) { __m = __total; });

        // 3. Elements from result are moved to [first, last)
        __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __result,
                                      __result + __m, [__result, __first](_Tp* __i, _Tp* __j) {
                                          __brick_move_destroy<__parallel_tag<_IsVector>>{}(
                                              __i, __j, __first + (__i - __result), _IsVector{});
                                      });
        return __first + __m;
    });
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _BinaryPredicate>
_RandomAccessIterator
__pattern_unique(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                 _RandomAccessIterator __last, _BinaryPredicate __pred)
{
    using _ReferenceType = typename std::iterator_traits<_RandomAccessIterator>::reference;

    if (__first == __last)
    {
        return __last;
    }
    if (__first + 1 == __last || __first + 2 == __last)
    {
        // Trivial sequence - use serial algorithm
        return __internal::__brick_unique(__first, __last, __pred, _IsVector{});
    }
    return __internal::__remove_elements(
        __tag, ::std::forward<_ExecutionPolicy>(__exec), ++__first, __last,
        [&__pred](bool* __b, bool* __e, _RandomAccessIterator __it) {
            __internal::__brick_walk3(
                __b, __e, __it - 1, __it,
                [&__pred](bool& __x, _ReferenceType __y, _ReferenceType __z) { __x = !__pred(__y, __z); }, _IsVector{});
        });
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _BinaryPredicate>
_OutputIterator
__brick_unique_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                    _BinaryPredicate __pred, /*vector=*/std::false_type) noexcept
{
    return std::unique_copy(__first, __last, __result, __pred);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
_RandomAccessIterator2
__brick_unique_copy(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result,
                    _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept
{
    // Ensure the arguments to *_selective_copy are of signed type, to avoid wraparound in the predicate
    // when the index equals to zero
    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    _DifferenceType __n = __last - __first;
    if (__n == 0)
        return __result;

    *__result++ = *__first++; // Always copy the first element
    --__n;
    __result += __unseq_backend::__simd_selective_copy</*bounded =*/ false>(__first, __n, __result, __n,
                    __internal::__unique_at_index{__pred}).second;
    return __result;
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BinaryPredicate>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__brick_bounded_unique_copy(_RandomAccessIterator1 __first,
                            typename std::iterator_traits<_RandomAccessIterator1>::difference_type __n,
                            _RandomAccessIterator2 __result,
                            typename std::iterator_traits<_RandomAccessIterator2>::difference_type __n_out,
                            _BinaryPredicate __pred, /*vector=*/std::true_type) noexcept
{
    using _DifferenceType = std::common_type_t<decltype(__n), decltype(__n_out)>;
    if (__n == 0 || __n_out == 0)
        return {__first, __result};

    *__result++ = *__first++; // Always copy the first element
    --__n; --__n_out;
    auto [__stop_in, __stop_out] = __unseq_backend::__simd_selective_copy</*bounded =*/ true>(
        __first, _DifferenceType(__n), __result, _DifferenceType(__n_out), __internal::__unique_at_index{__pred});
    return {__first + __stop_in, __result + __stop_out};
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _BinaryPredicate>
_OutputIterator
__pattern_unique_copy(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                      _OutputIterator __result, _BinaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_unique_copy(__first, __last, __result, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryPredicate>
_RandomAccessIterator2
__pattern_unique_copy(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                      _RandomAccessIterator1 __last, _RandomAccessIterator2 __result, _BinaryPredicate __pred)
{
    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    _DifferenceType __n = __last - __first;
    if (_DifferenceType(2) < __n)
    {
        *__result++ = *__first++; // Always copy the first element
        --__n;
        return __parallel_selective_copy(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result, __n,
                                         __internal::__unique_at_index{__pred}).second;
    }
    // trivial sequence - use serial algorithm
    return __internal::__brick_unique_copy(__first, __last, __result, __pred, _IsVector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _DifferenceType,
          class _RandomAccessIterator2, class _BinaryPredicate>
std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__pattern_bounded_unique_copy(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                              _RandomAccessIterator1 __first, _DifferenceType __n, _RandomAccessIterator2 __result,
                              _DifferenceType __n_out, _BinaryPredicate __pred)
{
    if (__n == 0 || __n_out == 0)
        return {__first, __result};

    *__result++ = *__first++; // Always copy the first element
    --__n; --__n_out;
    return __parallel_selective_copy(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __n, __result, __n_out,
                                     __internal::__unique_at_index{__pred});
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------
template <class _BidirectionalIterator>
void
__brick_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last,
                /*__is_vector=*/::std::false_type) noexcept
{
    ::std::reverse(__first, __last);
}

template <class _RandomAccessIterator>
void
__brick_reverse(_RandomAccessIterator __first, _RandomAccessIterator __last,
                /*__is_vector=*/::std::true_type) noexcept
{
    using _ReferenceType = typename std::iterator_traits<_RandomAccessIterator>::reference;

    const auto __n = (__last - __first) / 2;
    __unseq_backend::__simd_walk_n(__n,
                                   [](_ReferenceType __x, _ReferenceType __y) {
                                       using ::std::swap;
                                       swap(__x, __y);
                                   },
                                   __first, std::reverse_iterator<_RandomAccessIterator>(__last));
}

// this brick is called in parallel version, so we can use iterator arithmetic
template <class _BidirectionalIterator>
void
__brick_reverse(_BidirectionalIterator __first, _BidirectionalIterator __last, _BidirectionalIterator __d_last,
                /*is_vector=*/::std::false_type) noexcept
{
    for (; __first != __last; ++__first)
    {
        using ::std::iter_swap;
        iter_swap(__first, --__d_last);
    }
}

// this brick is called in parallel version, so we can use iterator arithmetic
template <class _RandomAccessIterator>
void
__brick_reverse(_RandomAccessIterator __first, _RandomAccessIterator __last, _RandomAccessIterator __d_last,
                /*is_vector=*/::std::true_type) noexcept
{
    using _ReferenceType = typename std::iterator_traits<_RandomAccessIterator>::reference;

    __unseq_backend::__simd_walk_n(__last - __first,
                                   [](_ReferenceType __x, _ReferenceType __y) {
                                       using ::std::swap;
                                       swap(__x, __y);
                                   },
                                   __first, std::reverse_iterator<_RandomAccessIterator>(__d_last));
}

template <class _Tag, class _ExecutionPolicy, class _BidirectionalIterator>
void
__pattern_reverse(_Tag, _ExecutionPolicy&&, _BidirectionalIterator __first, _BidirectionalIterator __last) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __internal::__brick_reverse(__first, __last, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator>
void
__pattern_reverse(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                  _RandomAccessIterator __last)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    if (__first == __last)
        return;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + (__last - __first) / 2,
            [__first, __last](_RandomAccessIterator __inner_first, _RandomAccessIterator __inner_last) {
                __internal::__brick_reverse(__inner_first, __inner_last, __last - (__inner_first - __first),
                                            _IsVector{});
            });
    });
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------

template <class _BidirectionalIterator, class _OutputIterator>
_OutputIterator
__brick_reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __d_first,
                     /*is_vector=*/::std::false_type) noexcept
{
    return ::std::reverse_copy(__first, __last, __d_first);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2>
_RandomAccessIterator2
__brick_reverse_copy(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __d_first,
                     /*is_vector=*/::std::true_type) noexcept
{
    using _ReferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::reference;
    using _ReferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::reference;

    return __unseq_backend::__simd_walk_n(__last - __first,
        [](_ReferenceType1 __x, _ReferenceType2 __y) { __y = __x; },
        std::reverse_iterator<_RandomAccessIterator1>(__last), __d_first);
}

template <class _Tag, class _ExecutionPolicy, class _BidirectionalIterator, class _OutputIterator>
_OutputIterator
__pattern_reverse_copy(_Tag, _ExecutionPolicy&&, _BidirectionalIterator __first, _BidirectionalIterator __last,
                       _OutputIterator __d_first) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_reverse_copy(__first, __last, __d_first, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2>
_RandomAccessIterator2
__pattern_reverse_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                       _RandomAccessIterator1 __last, _RandomAccessIterator2 __d_first)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    auto __len = __last - __first;

    if (__len == 0)
        return __d_first;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__first, __len, __d_first](_RandomAccessIterator1 __inner_first, _RandomAccessIterator1 __inner_last) {
                __internal::__brick_reverse_copy(__inner_first, __inner_last,
                                                 __d_first + (__len - (__inner_last - __first)), _IsVector{});
            });
        return __d_first + __len;
    });
}

//------------------------------------------------------------------------
// rotate
//------------------------------------------------------------------------
template <class _ForwardIterator>
_ForwardIterator
__brick_rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last,
               /*is_vector=*/::std::false_type) noexcept
{
    return ::std::rotate(__first, __middle, __last);
}

template <class _RandomAccessIterator>
_RandomAccessIterator
__brick_rotate(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last,
               /*is_vector=*/::std::true_type) noexcept
{
    auto __n = __last - __first;
    auto __m = __middle - __first;
    const _RandomAccessIterator __ret = __first + (__last - __middle);

    bool __is_left = (__m <= __n / 2);
    if (!__is_left)
        __m = __n - __m;

    while (__n > 1 && __m > 0)
    {
        using ::std::iter_swap;
        const auto __m_2 = __m * 2;
        if (__is_left)
        {
            for (; __last - __first >= __m_2; __first += __m)
            {
                __unseq_backend::__simd_assign(__first, __m, __first + __m,
                                               iter_swap<_RandomAccessIterator, _RandomAccessIterator>);
            }
        }
        else
        {
            for (; __last - __first >= __m_2; __last -= __m)
            {
                __unseq_backend::__simd_assign(__last - __m, __m, __last - __m_2,
                                               iter_swap<_RandomAccessIterator, _RandomAccessIterator>);
            }
        }
        __is_left = !__is_left;
        __m = __n % __m;
        __n = __last - __first;
    }

    return __ret;
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator>
_ForwardIterator
__pattern_rotate(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __middle,
                 _ForwardIterator __last) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_rotate(__first, __middle, __last, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator>
_RandomAccessIterator
__pattern_rotate(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                 _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _Tp = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    auto __n = __last - __first;
    auto __m = __middle - __first;
    if (__m <= __n / 2)
    {
        __par_backend::__buffer<_Tp> __buf(__n - __m);
        return __internal::__except_handler([&__exec, __n, __m, __first, __middle, __last, &__buf]() {
            _Tp* __result = __buf.get();
            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __middle, __last,
                                          [__middle, __result](_RandomAccessIterator __b, _RandomAccessIterator __e) {
                                              __internal::__brick_uninitialized_move(
                                                  __b, __e, __result + (__b - __middle), _IsVector{});
                                          });

            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __middle,
                                          [__last, __middle](_RandomAccessIterator __b, _RandomAccessIterator __e) {
                                              __internal::__brick_move<__parallel_tag<_IsVector>>{}(
                                                  __b, __e, __b + (__last - __middle), _IsVector{});
                                          });

            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __result,
                                          __result + (__n - __m), [__first, __result](_Tp* __b, _Tp* __e) {
                                              __brick_move_destroy<__parallel_tag<_IsVector>>{}(
                                                  __b, __e, __first + (__b - __result), _IsVector{});
                                          });

            return __first + (__last - __middle);
        });
    }
    else
    {
        __par_backend::__buffer<_Tp> __buf(__m);
        return __internal::__except_handler([&__exec, __n, __m, __first, __middle, __last, &__buf]() {
            _Tp* __result = __buf.get();
            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __middle,
                                          [__first, __result](_RandomAccessIterator __b, _RandomAccessIterator __e) {
                                              __internal::__brick_uninitialized_move(
                                                  __b, __e, __result + (__b - __first), _IsVector{});
                                          });

            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __middle, __last,
                                          [__first, __middle](_RandomAccessIterator __b, _RandomAccessIterator __e) {
                                              __internal::__brick_move<__parallel_tag<_IsVector>>{}(
                                                  __b, __e, __first + (__b - __middle), _IsVector{});
                                          });

            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __result,
                                          __result + __m, [__n, __m, __first, __result](_Tp* __b, _Tp* __e) {
                                              __brick_move_destroy<__parallel_tag<_IsVector>>{}(
                                                  __b, __e, __first + ((__n - __m) + (__b - __result)), _IsVector{});
                                          });

            return __first + (__last - __middle);
        });
    }
}

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator>
_OutputIterator
__brick_rotate_copy(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __middle,
                    _ForwardIterator __last, _OutputIterator __result) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return ::std::rotate_copy(__first, __middle, __last, __result);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2>
_RandomAccessIterator2
__brick_rotate_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&&, _RandomAccessIterator1 __first,
                    _RandomAccessIterator1 __middle, _RandomAccessIterator1 __last,
                    _RandomAccessIterator2 __result) noexcept
{
    _RandomAccessIterator2 __res = __brick_copy<__parallel_tag<_IsVector>>{}(__middle, __last, __result);
    return __internal::__brick_copy<__parallel_tag<_IsVector>>{}(__first, __middle, __res);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator>
_OutputIterator
__pattern_rotate_copy(_Tag __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __middle,
                      _ForwardIterator __last, _OutputIterator __result) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_rotate_copy(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __middle, __last,
                                           __result);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2>
_RandomAccessIterator2
__pattern_rotate_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                      _RandomAccessIterator1 __middle, _RandomAccessIterator1 __last, _RandomAccessIterator2 __result)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__first, __last, __middle, __result](_RandomAccessIterator1 __b, _RandomAccessIterator1 __e) {
                __internal::__brick_copy<__parallel_tag<_IsVector>> __copy{};
                if (__b > __middle)
                {
                    __copy(__b, __e, __result + (__b - __middle), _IsVector{});
                }
                else
                {
                    _RandomAccessIterator2 __new_result = __result + ((__last - __middle) + (__b - __first));
                    if (__e < __middle)
                    {
                        __copy(__b, __e, __new_result, _IsVector{});
                    }
                    else
                    {
                        __copy(__b, __middle, __new_result, _IsVector{});
                        __copy(__middle, __e, __result, _IsVector{});
                    }
                }
            });
        return __result + (__last - __first);
    });
}

//------------------------------------------------------------------------
// is_partitioned
//------------------------------------------------------------------------

template <class _ForwardIterator, class _UnaryPredicate>
bool
__brick_is_partitioned(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred,
                       /*is_vector=*/::std::false_type) noexcept
{
    return ::std::is_partitioned(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _UnaryPredicate>
bool
__brick_is_partitioned(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryPredicate __pred,
                       /*is_vector=*/::std::true_type) noexcept
{
    using _SizeType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
    if (__first == __last)
    {
        return true;
    }
    else
    {
        _RandomAccessIterator __result = __unseq_backend::__simd_first(
            __first, _SizeType(0), __last - __first,
            [&__pred](_RandomAccessIterator __it, _SizeType __i) { return !__pred(__it[__i]); });
        if (__result == __last)
        {
            return true;
        }
        else
        {
            ++__result;
            return !__unseq_backend::__simd_or(__result, __last - __result, __pred);
        }
    }
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
bool
__pattern_is_partitioned(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                         _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_is_partitioned(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryPredicate>
bool
__pattern_is_partitioned(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                         _RandomAccessIterator __last, _UnaryPredicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    //trivial pre-checks
    if (__first == __last)
        return true;

    return __internal::__except_handler([&]() {
        // State of current range:
        // broken     - current range is not partitioned by pred
        // all_true   - all elements in current range satisfy pred
        // all_false  - all elements in current range don't satisfy pred
        // true_false - elements satisfy pred are placed before elements that don't satisfy pred
        enum _ReduceRes
        {
            __not_init = -1,
            __broken,
            __all_true,
            __all_false,
            __true_false
        };
        // Array with states that we'll have when state from the left branch is merged with state from the right branch.
        // State is calculated by formula: new_state = table[left_state * 4 + right_state]
        const _ReduceRes __table[] = {__broken,     __broken,     __broken,     __broken, __broken,    __all_true,
                                      __true_false, __true_false, __broken,     __broken, __all_false, __broken,
                                      __broken,     __broken,     __true_false, __broken};
        struct _ReduceType
        {
            _ReduceRes __val;
            _RandomAccessIterator __pos;
        };
        //a commutative combiner
        auto __combine = [&__table](_ReduceType __x, _ReduceType __y) {
            return __x.__pos > __y.__pos ? _ReduceType{__table[__y.__val * 4 + __x.__val], __y.__pos}
                                         : _ReduceType{__table[__x.__val * 4 + __y.__val], __x.__pos};
        };

        const _ReduceType __identity{__not_init, __last};

        _ReduceType __result = __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __identity,
            [&__pred, __combine](_RandomAccessIterator __i, _RandomAccessIterator __j,
                                 _ReduceType __value) -> _ReduceType {
                if (__value.__val == __broken)
                    return _ReduceType{__broken, __i};

                _ReduceType __res{__not_init, __i};
                // if first element satisfy pred
                if (__pred(*__i))
                {
                    // find first element that don't satisfy pred
                    _RandomAccessIterator __x =
                        __internal::__brick_find_if(__i + 1, __j, __not_pred<_UnaryPredicate&>(__pred), _IsVector{});
                    if (__x != __j)
                    {
                        // find first element after "x" that satisfy pred
                        _RandomAccessIterator __y = __internal::__brick_find_if(__x + 1, __j, __pred, _IsVector{});
                        // if it was found then range isn't partitioned by pred
                        if (__y != __j)
                            return _ReduceType{__broken, __i};

                        __res = _ReduceType{__true_false, __i};
                    }
                    else
                        __res = _ReduceType{__all_true, __i};
                }
                else
                { // if first element doesn't satisfy pred
                    // then we should find the first element that satisfy pred.
                    // If we found it then range isn't partitioned by pred
                    if (__internal::__brick_find_if(__i + 1, __j, __pred, _IsVector{}) != __j)
                        return _ReduceType{__broken, __i};

                    __res = _ReduceType{__all_false, __i};
                }
                // if we have value from left range then we should calculate the result
                return (__value.__val == __not_init) ? __res : __combine(__value, __res);
            },

            [__combine](_ReduceType __val1, _ReduceType __val2) -> _ReduceType {
                if (__val1.__val == __not_init)
                    return __val2;
                if (__val2.__val == __not_init)
                    return __val1;
                assert(__val1.__val != __not_init && __val2.__val != __not_init);

                if (__val1.__val == __broken || __val2.__val == __broken)
                    return _ReduceType{__broken, __val1.__pos};
                // calculate the result for new big range
                return __combine(__val1, __val2);
            });
        return __result.__val != __broken;
    });
}

//------------------------------------------------------------------------
// partition
//------------------------------------------------------------------------

template <class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator
__brick_partition(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred,
                  /*is_vector=*/::std::false_type) noexcept
{
    return ::std::partition(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__brick_partition(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryPredicate __pred,
                  /*is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return ::std::partition(__first, __last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator
__pattern_partition(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                    _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_partition(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__pattern_partition(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                    _RandomAccessIterator __last, _UnaryPredicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    // partitioned range: elements before pivot satisfy pred (true part),
    //                    elements after pivot don't satisfy pred (false part)
    struct _PartitionRange
    {
        _RandomAccessIterator __begin;
        _RandomAccessIterator __pivot;
        _RandomAccessIterator __end;
    };

    return __internal::__except_handler([&]() {
        _PartitionRange __init{__last, __last, __last};

        // lambda for merging two partitioned ranges to one partitioned range
        auto __reductor = [&__exec](_PartitionRange __val1, _PartitionRange __val2) -> _PartitionRange {
            auto __size1 = __val1.__end - __val1.__pivot;
            auto __size2 = __val2.__pivot - __val2.__begin;
            auto __new_begin = __val2.__begin - (__val1.__end - __val1.__begin);

            // if all elements in left range satisfy pred then we can move new pivot to pivot of right range
            if (__val1.__end == __val1.__pivot)
            {
                return {__new_begin, __val2.__pivot, __val2.__end};
            }
            // if true part of right range greater than false part of left range
            // then we should swap the false part of left range and last part of true part of right range
            else if (__size2 > __size1)
            {
                __par_backend::__parallel_for(
                    __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __val1.__pivot, __val1.__pivot + __size1,
                    [__val1, __val2, __size1](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                        __internal::__brick_swap_ranges(__i, __j, (__val2.__pivot - __size1) + (__i - __val1.__pivot),
                                                        _IsVector{});
                    });
                return {__new_begin, __val2.__pivot - __size1, __val2.__end};
            }
            // else we should swap the first part of false part of left range and true part of right range
            else
            {
                __par_backend::__parallel_for(
                    __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __val1.__pivot, __val1.__pivot + __size2,
                    [__val1, __val2](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                        __internal::__brick_swap_ranges(__i, __j, __val2.__begin + (__i - __val1.__pivot), _IsVector{});
                    });
                return {__new_begin, __val1.__pivot + __size2, __val2.__end};
            }
        };

        _PartitionRange __result = __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init,
            [__pred, __reductor](_RandomAccessIterator __i, _RandomAccessIterator __j,
                                 _PartitionRange __value) -> _PartitionRange {
                //1. serial partition
                _RandomAccessIterator __pivot = __internal::__brick_partition(__i, __j, __pred, _IsVector{});

                // 2. merging of two ranges (left and right respectively)
                return __reductor(__value, {__i, __pivot, __j});
            },
            __reductor);
        return __result.__pivot;
    });
}

//------------------------------------------------------------------------
// stable_partition
//------------------------------------------------------------------------

template <class _BidirectionalIterator, class _UnaryPredicate>
_BidirectionalIterator
__brick_stable_partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _UnaryPredicate __pred,
                         /*__is_vector=*/::std::false_type) noexcept
{
    return ::std::stable_partition(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__brick_stable_partition(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryPredicate __pred,
                         /*__is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return ::std::stable_partition(__first, __last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _BidirectionalIterator, class _UnaryPredicate>
_BidirectionalIterator
__pattern_stable_partition(_Tag, _ExecutionPolicy&&, _BidirectionalIterator __first, _BidirectionalIterator __last,
                           _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_stable_partition(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__pattern_stable_partition(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                           _RandomAccessIterator __last, _UnaryPredicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    // partitioned range: elements before pivot satisfy pred (true part),
    //                    elements after pivot don't satisfy pred (false part)
    struct _PartitionRange
    {
        _RandomAccessIterator __begin;
        _RandomAccessIterator __pivot;
        _RandomAccessIterator __end;
    };

    return __internal::__except_handler([&]() {
        _PartitionRange __init{__last, __last, __last};

        // lambda for merging two partitioned ranges to one partitioned range
        auto __reductor = [](_PartitionRange __val1, _PartitionRange __val2) -> _PartitionRange {
            auto __size1 = __val1.__end - __val1.__pivot;
            auto __new_begin = __val2.__begin - (__val1.__end - __val1.__begin);

            // if all elements in left range satisfy pred then we can move new pivot to pivot of right range
            if (__val1.__end == __val1.__pivot)
            {
                return {__new_begin, __val2.__pivot, __val2.__end};
            }
            // if true part of right range greater than false part of left range
            // then we should swap the false part of left range and last part of true part of right range
            else
            {
                __internal::__brick_rotate(__val1.__pivot, __val2.__begin, __val2.__pivot, _IsVector{});
                return {__new_begin, __val2.__pivot - __size1, __val2.__end};
            }
        };

        _PartitionRange __result = __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __init,
            [&__pred, __reductor](_RandomAccessIterator __i, _RandomAccessIterator __j,
                                  _PartitionRange __value) -> _PartitionRange {
                //1. serial stable_partition
                _RandomAccessIterator __pivot = __internal::__brick_stable_partition(__i, __j, __pred, _IsVector{});

                // 2. merging of two ranges (left and right respectively)
                return __reductor(__value, {__i, __pivot, __j});
            },
            __reductor);
        return __result.__pivot;
    });
}

//------------------------------------------------------------------------
// partition_copy
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPredicate>
::std::pair<_OutputIterator1, _OutputIterator2>
__brick_partition_copy(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator1 __out_true,
                       _OutputIterator2 __out_false, _UnaryPredicate __pred, /*is_vector=*/::std::false_type) noexcept
{
    return ::std::partition_copy(__first, __last, __out_true, __out_false, __pred);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3,
          class _UnaryPredicate>
::std::pair<_RandomAccessIterator2, _RandomAccessIterator3>
__brick_partition_copy(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _RandomAccessIterator2 __out_true,
                       _RandomAccessIterator3 __out_false, _UnaryPredicate __pred,
                       /*is_vector=*/::std::true_type) noexcept
{
#if (_PSTL_MONOTONIC_PRESENT || _ONEDPL_MONOTONIC_PRESENT)
    return __unseq_backend::__simd_partition_copy(__first, __last - __first, __out_true, __out_false, __pred);
#else
    return ::std::partition_copy(__first, __last, __out_true, __out_false, __pred);
#endif
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator1, class _OutputIterator2,
          class _UnaryPredicate>
::std::pair<_OutputIterator1, _OutputIterator2>
__pattern_partition_copy(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                         _OutputIterator1 __out_true, _OutputIterator2 __out_false, _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_partition_copy(__first, __last, __out_true, __out_false, __pred,
                                              typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _UnaryPredicate>
::std::pair<_RandomAccessIterator2, _RandomAccessIterator3>
__pattern_partition_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                         _RandomAccessIterator1 __last, _RandomAccessIterator2 __out_true,
                         _RandomAccessIterator3 __out_false, _UnaryPredicate __pred)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _ReturnType = std::pair<_DifferenceType, _DifferenceType>;
    const _DifferenceType __n = __last - __first;
    if (_DifferenceType(1) < __n)
    {
        __par_backend::__buffer<bool> __mask_buf(__n);
        return __internal::__except_handler([&__exec, __n, __first, __out_true, __out_false, &__pred, &__mask_buf]() {
            bool* __mask = __mask_buf.get();
            _ReturnType __m{};
            __par_backend::__parallel_strict_scan(
                __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), __n,
                std::make_pair(_DifferenceType(0), _DifferenceType(0)),
                [=, &__pred](_DifferenceType __i, _DifferenceType __len) { // Reduce
                    return __internal::__brick_compute_mask(__first + __i, __len, __internal::__pred_at_index{__pred},
                                                            __mask + __i, _IsVector{});
                },
                [](const _ReturnType& __x, const _ReturnType& __y) -> _ReturnType {
                    return ::std::make_pair(__x.first + __y.first, __x.second + __y.second);
                },                                                                       // Combine
                [=](_DifferenceType __i, _DifferenceType __len, _ReturnType __initial) { // Scan
                    __internal::__brick_partition_by_mask(__first + __i, __first + (__i + __len),
                                                          __out_true + __initial.first, __out_false + __initial.second,
                                                          __mask + __i, _IsVector{});
                },
                [&__m](_ReturnType __total) { __m = __total; });
            return ::std::make_pair(__out_true + __m.first, __out_false + __m.second);
        });
    }
    // trivial sequence - use serial algorithm
    return __internal::__brick_partition_copy(__first, __last, __out_true, __out_false, __pred, _IsVector{});
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class _LeafSort>
void
__pattern_sort(_Tag, _ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
               _LeafSort __leaf_sort) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __leaf_sort(__first, __last, __comp);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare, class _LeafSort>
void
__pattern_sort(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
               _RandomAccessIterator __last, _Compare __comp, _LeafSort __leaf_sort)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_stable_sort(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp,
            [__leaf_sort](_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
                __leaf_sort(__first, __last, __comp);
            },
            __last - __first);
    });
}

//------------------------------------------------------------------------
// sort_by_key
//------------------------------------------------------------------------

template <typename _Tag, typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _Compare, typename _LeafSort>
void
__pattern_sort_by_key(_Tag, _ExecutionPolicy&&, _RandomAccessIterator1 __keys_first,
                      _RandomAccessIterator1 __keys_last, _RandomAccessIterator2 __values_first, _Compare __comp,
                      _LeafSort __leaf_sort) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    auto __beg = oneapi::dpl::make_zip_iterator(__keys_first, __values_first);
    auto __end = __beg + (__keys_last - __keys_first);
    auto __cmp_f = [__comp](const auto& __a, const auto& __b) {
        return std::invoke(__comp, std::get<0>(__a), std::get<0>(__b));
    };

    __leaf_sort(__beg, __end, __cmp_f);
}

template <typename _IsVector, typename _ExecutionPolicy, typename _RandomAccessIterator1,
          typename _RandomAccessIterator2, typename _Compare, typename _LeafSort>
void
__pattern_sort_by_key(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __keys_first,
                      _RandomAccessIterator1 __keys_last, _RandomAccessIterator2 __values_first, _Compare __comp,
                      _LeafSort __leaf_sort)
{
    auto __beg = oneapi::dpl::make_zip_iterator(__keys_first, __values_first);
    auto __end = __beg + (__keys_last - __keys_first);
    auto __cmp_f = [__comp](const auto& __a, const auto& __b) {
        return std::invoke(__comp, std::get<0>(__a), std::get<0>(__b));
    };

    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_stable_sort(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), __beg, __end, __cmp_f,
            [__leaf_sort](auto __first, auto __last, auto __cmp) { __leaf_sort(__first, __last, __cmp); },
            __end - __beg);
    });
}

//------------------------------------------------------------------------
// partial_sort
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
void
__pattern_partial_sort(_Tag, _ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __middle,
                       _RandomAccessIterator __last, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    ::std::partial_sort(__first, __middle, __last, __comp);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
void
__pattern_partial_sort(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                       _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    const auto __n = __middle - __first;
    if (__n == 0)
        return;

    __except_handler([&]() {
        __par_backend::__parallel_stable_sort(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp,
            [__n](_RandomAccessIterator __begin, _RandomAccessIterator __end, _Compare __comp) {
                if (__n < __end - __begin)
                    ::std::partial_sort(__begin, __begin + __n, __end, __comp);
                else
                    ::std::sort(__begin, __end, __comp);
            },
            __n);
    });
}

//------------------------------------------------------------------------
// partial_sort_copy
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator
__pattern_partial_sort_copy(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                            _RandomAccessIterator __d_first, _RandomAccessIterator __d_last, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return ::std::partial_sort_copy(__first, __last, __d_first, __d_last, __comp);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Compare>
_RandomAccessIterator2
__pattern_partial_sort_copy(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                            _RandomAccessIterator1 __last, _RandomAccessIterator2 __d_first,
                            _RandomAccessIterator2 __d_last, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    if (__last == __first || __d_last == __d_first)
    {
        return __d_first;
    }
    auto __n1 = __last - __first;
    auto __n2 = __d_last - __d_first;
    return __internal::__except_handler([&]() {
        if (__n2 >= __n1)
        {
            __par_backend::__parallel_stable_sort(
                __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __d_first, __d_first + __n1, __comp,
                [__first, __d_first](_RandomAccessIterator2 __i, _RandomAccessIterator2 __j, _Compare __comp) {
                    _RandomAccessIterator1 __i1 = __first + (__i - __d_first);
                    _RandomAccessIterator1 __j1 = __first + (__j - __d_first);

                    // 1. Copy elements from input to output
                    __brick_copy<__parallel_tag<_IsVector>>{}(__i1, __j1, __i, _IsVector{});
                    // 2. Sort elements in output sequence
                    ::std::sort(__i, __j, __comp);
                },
                __n1);
            return __d_first + __n1;
        }
        else
        {
            using _T1 = typename std::iterator_traits<_RandomAccessIterator1>::value_type;
            using _T2 = typename std::iterator_traits<_RandomAccessIterator2>::value_type;
            __par_backend::__buffer<_T1> __buf(__n1);
            _T1* __r = __buf.get();

            __par_backend::__parallel_stable_sort(
                __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __r, __r + __n1, __comp,
                [__n2, __first, __r](_T1* __i, _T1* __j, _Compare __comp) {
                    _RandomAccessIterator1 __it = __first + (__i - __r);

                    // 1. Copy elements from input to raw memory
                    for (_T1* __k = __i; __k != __j; ++__k, (void)++__it)
                    {
                        ::new (__k) _T2(*__it);
                    }

                    // 2. Sort elements in temporary buffer
                    if (__n2 < __j - __i)
                        ::std::partial_sort(__i, __i + __n2, __j, __comp);
                    else
                        ::std::sort(__i, __j, __comp);
                },
                __n2);

            // 3. Move elements from temporary buffer to output
            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __r, __r + __n2,
                                          [__r, __d_first](_T1* __i, _T1* __j) {
                                              __brick_move_destroy<__parallel_tag<_IsVector>>{}(
                                                  __i, __j, __d_first + (__i - __r), _IsVector{});
                                          });

            if constexpr (!::std::is_trivially_destructible_v<_T1>)
                __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __r + __n2,
                                              __r + __n1,
                                              [](_T1* __i, _T1* __j) { __brick_destroy(__i, __j, _IsVector{}); });

            return __d_first + __n2;
        }
    });
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------
template <class _RandomAccessIterator, class _BinaryPredicate>
_RandomAccessIterator
__brick_adjacent_find(_RandomAccessIterator __first, _RandomAccessIterator __last, _BinaryPredicate __pred,
                      /* IsVector = */ ::std::true_type, bool __or_semantic) noexcept
{
    return __unseq_backend::__simd_adjacent_find(__first, __last, __pred, __or_semantic);
}

template <class _ForwardIterator, class _BinaryPredicate>
_ForwardIterator
__brick_adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred,
                      /* IsVector = */ ::std::false_type, bool /* __or_semantic */) noexcept
{
    return ::std::adjacent_find(__first, __last, __pred);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _BinaryPredicate, class _Semantic>
_ForwardIterator
__pattern_adjacent_find(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                        _BinaryPredicate __pred, _Semantic) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_adjacent_find(__first, __last, __pred, typename _Tag::__is_vector{}, _Semantic::value);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _BinaryPredicate, class _Semantic>
_RandomAccessIterator
__pattern_adjacent_find(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                        _RandomAccessIterator __last, _BinaryPredicate __pred, _Semantic __or_semantic)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    if (__last - __first < 2)
        return __last;

    return __internal::__except_handler([&]() {
        return __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __last,
            [__last, __pred, __or_semantic](_RandomAccessIterator __begin, _RandomAccessIterator __end,
                                            _RandomAccessIterator __value) -> _RandomAccessIterator {
                // TODO: investigate performance benefits from the use of shared variable for the result,
                // checking (compare_and_swap idiom) its __value at __first.
                if (__or_semantic && __value < __last)
                { //found
                    return __value;
                }

                if (__value > __begin)
                {
                    // modify __end to check the predicate on the boundary __values;
                    // TODO: to use a custom range with boundaries overlapping
                    // TODO: investigate what if we remove "if" below and run algorithm on range [__first, __last-1)
                    // then check the pair [__last-1, __last)
                    if (__end != __last)
                        ++__end;

                    //correct the global result iterator if the "brick" returns a local "__last"
                    const _RandomAccessIterator __res =
                        __internal::__brick_adjacent_find(__begin, __end, __pred, _IsVector{}, __or_semantic);
                    if (__res < __end)
                        __value = __res;
                }
                return __value;
            },
            [](_RandomAccessIterator __x, _RandomAccessIterator __y) -> _RandomAccessIterator {
                return __x < __y ? __x : __y;
            } //reduce a __value
        );
    });
}

//------------------------------------------------------------------------
// nth_element
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
void
__pattern_nth_element(_Tag, _ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __nth,
                      _RandomAccessIterator __last, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    ::std::nth_element(__first, __nth, __last, __comp);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
void
__pattern_nth_element(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                      _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp)
{
    if (__first == __last || __nth == __last)
    {
        return;
    }

    using ::std::iter_swap;
    using _Tp = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    _RandomAccessIterator __x;
    do
    {
        __x = __internal::__pattern_partition(
            __tag, std::forward<_ExecutionPolicy>(__exec), __first + 1, __last,
            [&__comp, __first](const _Tp& __x) { return std::invoke(__comp, __x, *__first); });
        --__x;
        if (__x != __first)
        {
            iter_swap(__first, __x);
        }
        // if x > nth then our new range for partition is [first, x)
        if (__x - __nth > 0)
        {
            __last = __x;
        }
        // if x < nth then our new range for partition is [x, last)
        else if (__x - __nth < 0)
        {
            // if *x == *nth then we start the new partition at the next index where *x != *nth
            while (!std::invoke(__comp, *__nth, *__x) && !std::invoke(__comp, *__x, *__nth) && __x - __nth < 0)
            {
                ++__x;
            }
            iter_swap(__nth, __x);
            __first = __x;
        }
    } while (__x != __nth);
}

//------------------------------------------------------------------------
// fill, fill_n
//------------------------------------------------------------------------
template <class _Tag, typename _Tp>
struct __brick_fill<_Tag, _Tp, std::enable_if_t<__is_host_dispatch_tag_v<_Tag>>>
{
    const _Tp& __value;

    template <typename _RandomAccessIterator>
    void
    operator()(_RandomAccessIterator __first, _RandomAccessIterator __last,
               /* __is_vector = */ ::std::true_type) const noexcept
    {
        __unseq_backend::__simd_fill_n(__first, __last - __first, __value);
    }

    template <typename _ForwardIterator>
    void
    operator()(_ForwardIterator __first, _ForwardIterator __last,
               /* __is_vector = */ ::std::false_type) const noexcept
    {
        ::std::fill(__first, __last, __value);
    }
};

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Tp>
void
__pattern_fill(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __internal::__brick_fill<_Tag, _Tp>{__value}(__first, __last, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Tp>
_RandomAccessIterator
__pattern_fill(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
               _RandomAccessIterator __last, const _Tp& __value)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&__exec, __first, __last, &__value]() {
        __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                      [&__value](_RandomAccessIterator __begin, _RandomAccessIterator __end) {
                                          __internal::__brick_fill<__parallel_tag<_IsVector>, _Tp>{__value}(
                                              __begin, __end, _IsVector{});
                                      });
        return __last;
    });
}

template <class _Tag, typename _Tp>
struct __brick_fill_n<_Tag, _Tp, std::enable_if_t<__is_host_dispatch_tag_v<_Tag>>>
{
    const _Tp& __value;

    template <typename _RandomAccessIterator, typename _Size>
    _RandomAccessIterator
    operator()(_RandomAccessIterator __first, _Size __count,
               /* __is_vector = */ ::std::true_type) const noexcept
    {
        return __unseq_backend::__simd_fill_n(__first, __count, __value);
    }

    template <typename _OutputIterator, typename _Size>
    _OutputIterator
    operator()(_OutputIterator __first, _Size __count,
               /* __is_vector = */ ::std::false_type) const noexcept
    {
        return ::std::fill_n(__first, __count, __value);
    }
};

template <class _Tag, class _ExecutionPolicy, class _OutputIterator, class _Size, class _Tp>
_OutputIterator
__pattern_fill_n(_Tag, _ExecutionPolicy&&, _OutputIterator __first, _Size __count, const _Tp& __value) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_fill_n<_Tag, _Tp>{__value}(__first, __count, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Size, class _Tp>
_RandomAccessIterator
__pattern_fill_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                 _Size __count, const _Tp& __value)
{
    return __internal::__pattern_fill(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __count,
                                      __value);
}

//------------------------------------------------------------------------
// generate, generate_n
//------------------------------------------------------------------------
template <class _RandomAccessIterator, class _Generator>
void
__brick_generate(_RandomAccessIterator __first, _RandomAccessIterator __last, _Generator __g,
                 /* is_vector = */ ::std::true_type) noexcept
{
    __unseq_backend::__simd_generate_n(__first, __last - __first, __g);
}

template <class _ForwardIterator, class _Generator>
void
__brick_generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __g,
                 /* is_vector = */ ::std::false_type) noexcept
{
    ::std::generate(__first, __last, __g);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Generator>
void
__pattern_generate(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Generator __g) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __internal::__brick_generate(__first, __last, __g, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Generator>
_RandomAccessIterator
__pattern_generate(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                   _RandomAccessIterator __last, _Generator __g)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                                      [__g](_RandomAccessIterator __begin, _RandomAccessIterator __end) {
                                          __internal::__brick_generate(__begin, __end, __g, _IsVector{});
                                      });
        return __last;
    });
}

template <class _RandomAccessIterator, class Size, class _Generator>
_RandomAccessIterator
__brick_generate_n(_RandomAccessIterator __first, Size __count, _Generator __g,
                   /* is_vector = */ ::std::true_type) noexcept
{
    return __unseq_backend::__simd_generate_n(__first, __count, __g);
}

template <class OutputIterator, class Size, class _Generator>
OutputIterator
__brick_generate_n(OutputIterator __first, Size __count, _Generator __g, /* is_vector = */ ::std::false_type) noexcept
{
    return ::std::generate_n(__first, __count, __g);
}

template <class _Tag, class _ExecutionPolicy, class _OutputIterator, class _Size, class _Generator>
_OutputIterator
__pattern_generate_n(_Tag, _ExecutionPolicy&&, _OutputIterator __first, _Size __count, _Generator __g) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_generate_n(__first, __count, __g, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Size, class _Generator>
_RandomAccessIterator
__pattern_generate_n(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                     _Size __count, _Generator __g)
{
    static_assert(__is_random_access_iterator_v<_RandomAccessIterator>,
                  "Pattern-brick error. Should be a random access iterator.");
    return __internal::__pattern_generate(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __count,
                                          __g);
}

//------------------------------------------------------------------------
// remove
//------------------------------------------------------------------------

template <class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator
__brick_remove_if(_ForwardIterator __first, _ForwardIterator __last, _UnaryPredicate __pred,
                  /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::remove_if(__first, __last, __pred);
}

template <class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__brick_remove_if(_RandomAccessIterator __first, _RandomAccessIterator __last, _UnaryPredicate __pred,
                  /* __is_vector = */ ::std::true_type) noexcept
{
#if (_PSTL_MONOTONIC_PRESENT || _ONEDPL_MONOTONIC_PRESENT)
    return __unseq_backend::__simd_remove_if(__first, __last - __first, __pred);
#else
    return ::std::remove_if(__first, __last, __pred);
#endif
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _UnaryPredicate>
_ForwardIterator
__pattern_remove_if(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                    _UnaryPredicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_remove_if(__first, __last, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _UnaryPredicate>
_RandomAccessIterator
__pattern_remove_if(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                    _RandomAccessIterator __last, _UnaryPredicate __pred)
{
    using _ReferenceType = typename std::iterator_traits<_RandomAccessIterator>::reference;

    if (__first == __last || __first + 1 == __last)
    {
        // Trivial sequence - use serial algorithm
        return __internal::__brick_remove_if(__first, __last, __pred, _IsVector{});
    }

    return __internal::__remove_elements(
        __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
        [&__pred](bool* __b, bool* __e, _RandomAccessIterator __it) {
            __internal::__brick_walk2(
                __b, __e, __it, [&__pred](bool& __x, _ReferenceType __y) { __x = !__pred(__y); }, _IsVector{});
        });
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------
template <typename _Iterator1, typename _Iterator2, typename _Iterator3, typename _Comp, typename _Proj1,
          typename _Proj2>
std::pair<_Iterator1, _Iterator2>
__serial_merge_out_lim(_Iterator1 __x, _Iterator1 __x_e, _Iterator2 __y, _Iterator2 __y_e, _Iterator3 __out_b,
                       _Iterator3 __out_e, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    for (_Iterator3 __k = __out_b; __k != __out_e; ++__k)
    {
        if (__x == __x_e)
        {
            assert(__y != __y_e);
            *__k = *__y;
            ++__y;
        }
        else if (__y == __y_e)
        {
            assert(__x != __x_e);
            *__k = *__x;
            ++__x;
        }
        else if (std::invoke(__comp, std::invoke(__proj2, *__y), std::invoke(__proj1, *__x)))
        {
            *__k = *__y;
            ++__y;
        }
        else
        {
            *__k = *__x;
            ++__x;
        }
    }
    return {__x, __y};
}

template <class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_merge(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator2 __last2, _OutputIterator __d_first, _Compare __comp,
              /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::merge(__first1, __last1, __first2, __last2, __d_first, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__brick_merge(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
              _RandomAccessIterator2 __last2, _RandomAccessIterator3 __d_first, _Compare __comp,
              /* __is_vector = */ ::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return ::std::merge(__first1, __last1, __first2, __last2, __d_first, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator,
          class _Compare>
_OutputIterator
__pattern_merge(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __d_first,
                _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_merge(__first1, __last1, __first2, __last2, __d_first, __comp,
                                     typename _Tag::__is_vector{});
}

template <typename _Tag, typename _ExecutionPolicy, typename _It1, typename _Index1, typename _It2, typename _Index2,
          typename _OutIt, typename _Index3, typename _Comp, typename _Proj1, typename _Proj2>
std::pair<_It1, _It2>
___merge_path_out_lim(_Tag, _ExecutionPolicy&&, _It1 __it_1, _Index1 __n_1, _It2 __it_2, _Index2 __n_2,
                      _OutIt __it_out, _Index3 __n_out, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __serial_merge_out_lim(__it_1, __it_1 + __n_1, __it_2, __it_2 + __n_2, __it_out, __it_out + __n_out, __comp,
                                  __proj1, __proj2);
}

inline constexpr std::size_t __merge_path_cut_off = 2000;

template <typename _IsVector, typename _ExecutionPolicy, typename _It1, typename _Index1, typename _It2,
          typename _Index2, typename _OutIt, typename _Index3, typename _Comp, typename _Proj1, typename _Proj2>
std::pair<_It1, _It2>
___merge_path_out_lim(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _It1 __it_1, _Index1 __n_1, _It2 __it_2,
                      _Index2 __n_2, _OutIt __it_out, _Index3 __n_out, _Comp __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    _It1 __it_res_1;
    _It2 __it_res_2;

    __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), _Index3(0), __n_out,
            [=, &__it_res_1, &__it_res_2](_Index3 __i, _Index3 __j) {
                //a start merging point on the merge path; for each thread
                _Index1 __r = 0; //row index
                _Index2 __c = 0; //column index

                if (__i > 0)
                {
                    //calc merge path intersection:
                    const _Index3 __d_size =
                        std::abs(std::max<_Index2>(0, __i - __n_2) - (std::min<_Index1>(__i, __n_1) - 1)) + 1;

                    auto __get_row = [__i, __n_1](auto __d) { return std::min<_Index1>(__i, __n_1) - __d - 1; };
                    auto __get_column = [__i, __n_1](auto __d) {
                        return std::max<_Index1>(0, __i - __n_1 - 1) + __d + (__i / (__n_1 + 1) > 0 ? 1 : 0);
                    };

                    oneapi::dpl::counting_iterator<_Index3> __it_d(0);

                    auto __res_d = *std::lower_bound(__it_d, __it_d + __d_size, 1, [&](auto __d, auto __val) {
                        auto __r = __get_row(__d);
                        auto __c = __get_column(__d);

                        const auto __res = std::invoke(__comp, std::invoke(__proj2, __it_2[__c]),
                                                       std::invoke(__proj1, __it_1[__r])) ? 0 : 1;

                        return __res < __val;
                    });

                    //intersection point
                    __r = __get_row(__res_d);
                    __c = __get_column(__res_d);
                    ++__r; //to get a merge matrix ceil, lying on the current diagonal
                }

                //serial merge n elements, starting from input x and y, to [i, j) output range
                auto [__res1, __res2] = __serial_merge_out_lim(__it_1 + __r, __it_1 + __n_1, __it_2 + __c,
                                                               __it_2 + __n_2, __it_out + __i, __it_out + __j, __comp,
                                                               __proj1, __proj2);

                if (__j == __n_out)
                {
                    __it_res_1 = __res1;
                    __it_res_2 = __res2;
                }
            },
            __merge_path_cut_off); //grainsize
    });

    return {__it_res_1, __it_res_2};
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__pattern_merge(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                _RandomAccessIterator3 __d_first, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_merge(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __d_first,
            __comp,
            [](_RandomAccessIterator1 __f1, _RandomAccessIterator1 __l1, _RandomAccessIterator2 __f2,
               _RandomAccessIterator2 __l2, _RandomAccessIterator3 __f3, _Compare __comp) {
                return __internal::__brick_merge(__f1, __l1, __f2, __l2, __f3, __comp, _IsVector{});
            });
        return __d_first + (__last1 - __first1) + (__last2 - __first2);
    });
}

//------------------------------------------------------------------------
// inplace_merge
//------------------------------------------------------------------------
template <class _BidirectionalIterator, class _Compare>
void
__brick_inplace_merge(_BidirectionalIterator __first, _BidirectionalIterator __middle, _BidirectionalIterator __last,
                      _Compare __comp, /* __is_vector = */ ::std::false_type) noexcept
{
    ::std::inplace_merge(__first, __middle, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare>
void
__brick_inplace_merge(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last,
                      _Compare __comp, /* __is_vector = */ ::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial")
    ::std::inplace_merge(__first, __middle, __last, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _BidirectionalIterator, class _Compare>
void
__pattern_inplace_merge(_Tag, _ExecutionPolicy&&, _BidirectionalIterator __first, _BidirectionalIterator __middle,
                        _BidirectionalIterator __last, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    __internal::__brick_inplace_merge(__first, __middle, __last, __comp, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
void
__pattern_inplace_merge(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                        _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    if (__first == __last || __first == __middle || __middle == __last)
    {
        return;
    }

    using _Tp = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    auto __n = __last - __first;
    __par_backend::__buffer<_Tp> __buf(__n);
    _Tp* __r = __buf.get();
    __internal::__except_handler([&]() {
        auto __move_values = [](_RandomAccessIterator __x, _Tp* __z) {
            ::new (std::addressof(*__z)) _Tp(std::move(*__x));
        };

        auto __move_sequences = [](_RandomAccessIterator __first1, _RandomAccessIterator __last1, _Tp* __first2) {
            return __internal::__brick_uninitialized_move(__first1, __last1, __first2, _IsVector{});
        };

        __par_backend::__parallel_merge(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __middle, __middle, __last, __r, __comp,
            [__n, __move_values, __move_sequences](_RandomAccessIterator __f1, _RandomAccessIterator __l1,
                                                   _RandomAccessIterator __f2, _RandomAccessIterator __l2, _Tp* __f3,
                                                   _Compare __comp) {
                (__utils::__serial_move_merge(__n))(__f1, __l1, __f2, __l2, __f3, __comp, __move_values, __move_values,
                                                    __move_sequences, __move_sequences);
                return __f3 + (__l1 - __f1) + (__l2 - __f2);
            });
        __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __r, __r + __n,
                                      [__r, __first](_Tp* __i, _Tp* __j) {
                                          __brick_move_destroy<__parallel_tag<_IsVector>>{}(
                                              __i, __j, __first + (__i - __r), _IsVector{});
                                      });
    });
}

//------------------------------------------------------------------------
// includes
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool
__pattern_includes(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                   _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return ::std::includes(__first1, __last1, __first2, __last2, __comp);
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Compare>
bool
__pattern_includes(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                   _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                   _Compare __comp)
{
    if (__first2 == __last2)
        return true;

    //optimization; {1} - the first sequence, {2} - the second sequence
    //{1} is empty or size_of{2} > size_of{1}
    if (__first1 == __last1 || __last2 - __first2 > __last1 - __first1 ||
        // {1}:     [**********]     or   [**********]
        // {2}: [***********]                   [***********]
        std::invoke(__comp, *__first2, *__first1) || std::invoke(__comp, *(__last1 - 1), *(__last2 - 1)))
        return false;

    __first1 = ::std::lower_bound(__first1, __last1, *__first2, __comp);
    if (__first1 == __last1)
        return false;

    if (__last2 - __first2 == 1)
        return !std::invoke(__comp, *__first1, *__first2) && !std::invoke(__comp, *__first2, *__first1);

    return __internal::__except_handler([&]() {
        return !__internal::__parallel_or(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first2, __last2,
            [__first1, __last1, __first2, __last2, &__comp](_RandomAccessIterator2 __i, _RandomAccessIterator2 __j) {
                assert(__j > __i);
                //assert(__j - __i > 1);

                //1. moving boundaries to "consume" subsequence of equal elements
                auto __is_equal_sorted = [&__comp](_RandomAccessIterator2 __a, _RandomAccessIterator2 __b) -> bool {
                    //enough one call of __comp due to compared couple belongs to one sorted sequence
                    return !std::invoke(__comp, *__a, *__b);
                };

                //1.1 left bound, case "aaa[aaaxyz...]" - searching "x"
                if (__i > __first2 && __is_equal_sorted(__i - 1, __i))
                {
                    //whole subrange continues to have equal elements - return "no op"
                    if (__is_equal_sorted(__i, __j - 1))
                        return false;

                    __i = ::std::upper_bound(__i, __last2, *__i, __comp);
                }

                //1.2 right bound, case "[...aaa]aaaxyz" - searching "x"
                if (__j < __last2 && __is_equal_sorted(__j - 1, __j))
                    __j = ::std::upper_bound(__j, __last2, *__j, __comp);

                //2. testing is __a subsequence of the second range included into the first range
                auto __b = ::std::lower_bound(__first1, __last1, *__i, __comp);

                assert(!std::invoke(__comp, *(__last1 - 1), *__b));
                assert(!std::invoke(__comp, *(__j - 1), *__i));
                return !::std::includes(__b, __last1, __i, __j, __comp);
            });
    });
}

template <typename Size>
constexpr bool
__is_set_algo_cutoff_exceeded(Size size)
{
    // 1000 is chosen as a cut-off value based on benchmarking source data sizes
    constexpr Size __set_algo_cut_off = 1000;
    return size > __set_algo_cut_off;
}

// Describes a data window in the temporary buffer and corresponding positions in the output range
template <bool __Bounded, typename _DifferenceType1, typename _DifferenceType2, typename _DifferenceTypeMask,
          typename _DifferenceTypeOut, typename _DifferenceType>
struct _SetRangeImpl
{
  protected:
    static constexpr std::size_t _DataIndex = 0;
    static constexpr std::size_t _MaskDataIndex = 1;
    static constexpr std::size_t _SourceDataOffsetsIndex = 2;

  public:
    struct _Data
    {
        //                                       [.........................)
        // Temporary windowed buffer:        TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
        //                                       ^                         ^
        //                                       +<-(__buf_pos)            +<-(__buf_pos + __len)
        //                                       |                         |
        //                                       +--+                      +--+
        //                                          |                         |
        //                                          |<-(__pos)                |<-(__pos + __len)
        //                                          V                         V
        // Result buffer:                 OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

        _DifferenceType __pos{};     // Offset in output range w/o limitation to output data size
        _DifferenceType __len{};     // The length of data pack: the same for windowed and result buffers
        _DifferenceType __buf_pos{}; // Offset in temporary buffer w/o limitation to output data size

        bool
        empty() const
        {
            return __len == 0;
        }

        static bool is_left(const _Data& __a, const _Data& __b)
        {
            return __a.__buf_pos < __b.__buf_pos ||
                   (__b.__buf_pos == __a.__buf_pos && !__b.empty());
        }

        template <typename _DataType>
        static std::pair<_DataType&&, _DataType&&>
        get_left_right(_DataType&& __a, _DataType&& __b)
        {
            if (is_left(__a, __b))
                return {std::forward<_DataType>(__a), std::forward<_DataType>(__b)};

            return {std::forward<_DataType>(__b), std::forward<_DataType>(__a)};
        }

        template <typename _DataType>
        static _Data combine_with(_DataType&& __a, _DataType&& __b)
        {
            auto&& [__left, __right] = get_left_right(std::forward<_DataType>(__a), std::forward<_DataType>(__b));

            return _Data{__left.__pos + __left.__len + __right.__pos, __right.__len, __right.__buf_pos};
        }
    };

    using _MaskData = _Data;

    struct _SourceProcessingDataOffsets
    {
        //                                             __start_offset1                 __start_offset2
        //                                                   |                               |
        //                                                   |                               |
        // First input range:                 [..............V...............................|.........)
        // Second input range:                [..............................................V...............................)
        //                                                          [.........................)
        // Temporary windowed result buffer:  TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
        //                                                          ^                         ^
        //                                                          +<-(__buf_pos)            +<-(__buf_pos + __len)

        _DifferenceType1 __start_offset1 = {}; // Offset in the first input range to start processing data in the window
        _DifferenceType2 __start_offset2 = {}; // Offset in the second input range to start processing data in the window

        static _SourceProcessingDataOffsets
        combine_with(const _SourceProcessingDataOffsets& __a, const _SourceProcessingDataOffsets& __b)
        {
            return {std::max(__a.__start_offset1, __b.__start_offset1),
                    std::max(__a.__start_offset2, __b.__start_offset2)};
        }
    };

    const _Data&
    get_data_part() const
    {
        if constexpr (!__Bounded)
            return __data;
        else
            return std::get<_DataIndex>(__data);
    }

    const _MaskData&
    get_mask_part() const
    {
        static_assert(__Bounded, "Mask data part is available only for bounded set operations");
        return std::get<_MaskDataIndex>(__data);
    }

    const _SourceProcessingDataOffsets&
    get_source_data_offsets_part() const
    {
        static_assert(__Bounded, "Source data offsets part is available only for bounded set operations");
        return std::get<_SourceDataOffsetsIndex>(__data);
    }

    using _DataStorage = std::conditional_t<!__Bounded, _Data, std::tuple<_Data, _MaskData, _SourceProcessingDataOffsets>>;
    _DataStorage __data;
};

template <bool __Bounded, typename _DifferenceType1, typename _DifferenceType2, typename _DifferenceTypeMask,
          typename _DifferenceTypeOut, typename _DifferenceType>
struct _SetRangeCombiner
{
    using _SetRange = _SetRangeImpl<__Bounded, _DifferenceType1, _DifferenceType2, _DifferenceTypeMask, _DifferenceTypeOut, _DifferenceType>;

    _SetRange
    operator()(const _SetRange& __a, const _SetRange& __b) const
    {
        const typename _SetRange::_Data __new_processing_data = _SetRange::_Data::combine_with(__a.get_data_part(), __b.get_data_part());

        if constexpr (!__Bounded)
        {
            return _SetRange{__new_processing_data};
        }
        else
        {
            typename _SetRange::_MaskData __new_mask_data = _SetRange::_MaskData::combine_with(__a.get_mask_part(), __b.get_mask_part());

            const typename _SetRange::_SourceProcessingDataOffsets& ___source_data_offsets_a = __a.get_source_data_offsets_part();
            const typename _SetRange::_SourceProcessingDataOffsets& ___source_data_offsets_b = __b.get_source_data_offsets_part();

            typename _SetRange::_SourceProcessingDataOffsets __new_offsets_to_processing_data =
                _SetRange::_SourceProcessingDataOffsets::combine_with(___source_data_offsets_a, ___source_data_offsets_b);

            typename _SetRange::_DataStorage _ds{
                __new_processing_data,
                __new_mask_data,
                __new_offsets_to_processing_data }; 

            return _SetRange{_ds};
        }
    }

    const _DifferenceTypeOut __n_out = {};  // Size of output range
};

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _OutputIterator>
using __parallel_set_op_return_t =
    oneapi::dpl::__utils::__set_operations_result<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator>;

template <bool __Bounded>
struct __mask_buffers;

template <>
struct __mask_buffers<true>
{
    using _mask_ptr_t = oneapi::dpl::__utils::__parallel_set_op_mask*;
    using _difference_t = std::iterator_traits<_mask_ptr_t>::difference_type;

    __mask_buffers(std::size_t __mask_buf_size) : __buf_mask_rng(__mask_buf_size)
    {
    }

    // Get pointer to the windowed mask buffer with offset
    _mask_ptr_t
    get_buf_mask_rng_data(std::size_t __offset = 0) const
    {
        return __buf_mask_rng.get() + __offset;
    }

    using _MaskBuffer = __par_backend::__buffer<oneapi::dpl::__utils::__parallel_set_op_mask>;
    _MaskBuffer __buf_mask_rng;     // Temporary (windowed) buffer for the input range1 + range2 item usage mask
};

template <>
struct __mask_buffers<false>
{
    using _mask_ptr_t = std::nullptr_t;
    using _difference_t = std::ptrdiff_t;

    __mask_buffers(std::size_t) {}

    // Get pointer to the windowed mask buffer with offset
    _mask_ptr_t
    get_buf_mask_rng_data(std::size_t = 0) const
    {
        return nullptr;
    }
};

template <bool __Bounded, class _IsVector, typename ProcessingDataPointer, typename MaskDataPointer, typename _OutputIterator, typename _DifferenceType1, typename _DifferenceType2>
struct _ScanPred
{
    __parallel_tag<_IsVector> __tag;
    ProcessingDataPointer     __buf_pos_begin;
    ProcessingDataPointer     __buf_pos_end;
    MaskDataPointer           __temporary_mask_buf;     // Pointer to the windowed mask buffer
    _OutputIterator           __result1;
    _OutputIterator           __result2;

    _DifferenceType1&         __res_reachedPos1;
    _DifferenceType2&         __res_reachedPos2;

    template <typename _SetRange>
    void
    operator()(_DifferenceType1, _DifferenceType1, const _SetRange& __s) const
    {
        if constexpr (!__Bounded)
        {
            // Processed data
            __brick_move_destroy<decltype(__tag)>{}(__buf_pos_begin + __s.get_data_part().__buf_pos,
                                                    __buf_pos_begin + __s.get_data_part().__buf_pos +
                                                        __s.get_data_part().__len,
                                                    __result1 + __s.get_data_part().__pos, _IsVector{});
        }
        else
        {
            // 1. Copy source data (bounded)
            const typename _SetRange::_Data& __s_data_part = __s.get_data_part();
            {
                // Evalueate output range boundaries for current data chunk
                const auto __result_from = __advance_clamped(__result1, __s_data_part.__pos, __result2);
                const auto __result_to =
                    __advance_clamped(__result1, __s_data_part.__pos + __s_data_part.__len, __result2);

                const auto __result_remaining = __result_to - __result_from;
                assert(__result_remaining <= __s_data_part.__len);
                if (__result_remaining > 0)
                {
                    // Evaluate pointers to current data chunk in temporary buffer
                    const auto __buf_pos_from =
                        __advance_clamped(__buf_pos_begin, __s_data_part.__buf_pos, __buf_pos_end);
                    const auto __buf_pos_to =
                        __advance_clamped(__buf_pos_begin, __s_data_part.__buf_pos + __result_remaining, __buf_pos_end);

                    // Copy results data into results range to have final output
                    __brick_move_destroy<decltype(__tag)>{}(__buf_pos_from, __buf_pos_to, __result_from, _IsVector{});
                }
            }

            // 2. Evaluate final positions in the first and the second input ranges
            {
                const typename _SetRange::_MaskData& __s_mask_data = __s.get_mask_part();
                const auto __n_out = __result2 - __result1;

                if (__s_data_part.__pos <= __n_out && (__s_data_part.__pos + __s_data_part.__len) > __n_out)
                {
                    // Process masks states in the output result (continious, not windowed) mask buffer
                    auto __mask_buffer_begin = __temporary_mask_buf + __s_mask_data.__buf_pos;
                    auto __mask_buffer_end = __temporary_mask_buf + __s_mask_data.__buf_pos + __s_mask_data.__len;

                    auto __pos_no = __s_data_part.__pos;
                    auto __mask_buffer_it = __mask_buffer_begin;

                    // 1. Pass positions which generates output
                    for (; __mask_buffer_it != __mask_buffer_end && __pos_no < __n_out; ++__mask_buffer_it)
                    {
                        if (oneapi::dpl::__utils::__test_parallel_set_op_mask_state<
                                oneapi::dpl::__utils::__parallel_set_op_mask::eDataOut>(*__mask_buffer_it))
                            ++__pos_no;
                    }

                    // 2. Take into account positions without generated output
                    while (__mask_buffer_it != __mask_buffer_end &&
                           !oneapi::dpl::__utils::__test_parallel_set_op_mask_state<
                               oneapi::dpl::__utils::__parallel_set_op_mask::eDataOut>(*__mask_buffer_it))
                    {
                        assert(*__mask_buffer_it == oneapi::dpl::__utils::__parallel_set_op_mask::eData1 ||
                               *__mask_buffer_it == oneapi::dpl::__utils::__parallel_set_op_mask::eData2 ||
                               *__mask_buffer_it == oneapi::dpl::__utils::__parallel_set_op_mask::eBoth);

                        ++__mask_buffer_it;
                    }

                    const typename _SetRange::_SourceProcessingDataOffsets& __source_data_offsets =
                        __s.get_source_data_offsets_part();

                    __res_reachedPos1 = std::count_if(
                        __mask_buffer_begin, __mask_buffer_it, [](oneapi::dpl::__utils::__parallel_set_op_mask __m) {
                            return oneapi::dpl::__utils::__test_parallel_set_op_mask_state<
                                oneapi::dpl::__utils::__parallel_set_op_mask::eData1>(__m);
                        });
                    __res_reachedPos1 += __source_data_offsets.__start_offset1;

                    __res_reachedPos2 = std::count_if(
                        __mask_buffer_begin, __mask_buffer_it, [](oneapi::dpl::__utils::__parallel_set_op_mask __m) {
                            return oneapi::dpl::__utils::__test_parallel_set_op_mask_state<
                                oneapi::dpl::__utils::__parallel_set_op_mask::eData2>(__m);
                        });
                    __res_reachedPos2 += __source_data_offsets.__start_offset2;
                }
            }
        }
    }

  protected:

    // Move it1 forward by n, but not beyond it2
    template <typename _RandomAccessIterator,
              typename Size = typename std::iterator_traits<_RandomAccessIterator>::difference_type>
    _RandomAccessIterator
    __advance_clamped(_RandomAccessIterator it1, Size __size, _RandomAccessIterator it2) const
    {
        assert(it1 <= it2);
        return it1 + std::min(it2 - it1, __size);
    }
};

template <bool __Bounded, typename _SetRange, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _OutputIterator, typename _SizeFunction, typename _MaskSizeFunction, typename _SetUnionOp,
          typename _Compare, typename _Proj1, typename _Proj2, typename _T>
struct _ParallelSetOpStrictScanPred
{
    _RandomAccessIterator1 __first1, __last1;
    _RandomAccessIterator2 __first2, __last2;
    _SizeFunction __size_func;
    _MaskSizeFunction __mask_size_func;
    _SetUnionOp __set_union_op;

    _Compare __comp;
    _Proj1 __proj1;
    _Proj2 __proj2;

    _T* __buf_raw_data_begin = nullptr;
    __mask_buffers<__Bounded>& __mask_bufs;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceTypeOutput = typename std::iterator_traits<_OutputIterator>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2, _DifferenceTypeOutput>;

    _SetRange
    operator()(_DifferenceType1 __i, _DifferenceType1 __len) const
    {
        //[__b; __e) - a subrange of the first sequence, to reduce
        _RandomAccessIterator1 __b = __first1 + __i;
        _RandomAccessIterator1 __e = __first1 + (__i + __len);

        //try searching for the first element which not equal to *__b
        if (__b != __first1)
            __b += __internal::__pstl_upper_bound(__b, _DifferenceType1{0}, __last1 - __b, __b, __comp, __proj1, __proj1);

        //try searching for the first element which not equal to *__e
        if (__e != __last1)
            __e += __internal::__pstl_upper_bound(__e, _DifferenceType1{0}, __last1 - __e, __e, __comp, __proj1, __proj1);

        //check is [__b; __e) empty
        if (__e - __b < 1)
        {
            _RandomAccessIterator2 __bb = __last2;
            if (__b != __last1)
                __bb = __first2 + __internal::__pstl_lower_bound(__first2, _DifferenceType2{0}, __last2 - __first2, __b, __comp, __proj2, __proj1);

            typename _SetRange::_Data __new_processing_data{
                0,                                                          // Offset in output range w/o limitation to output data size
                0,                                                          // The length of data pack: the same for windowed and result buffers
                __size_func((__b - __first1), (__bb - __first2))};          // Offset in temporary buffer w/o limitation to output data size

            if constexpr (!__Bounded)
            {
                return _SetRange{__new_processing_data};
            }
            else
            {
                typename _SetRange::_MaskData __new_mask_data{
                    0,                                                      // Offset in output range w/o limitation to output data size
                    0,                                                      // The length of data pack: the same for windowed and result buffers
                    __mask_size_func((__b - __first1), (__bb - __first2))}; // Offset in temporary buffer w/o limitation to output data size

                typename _SetRange::_DataStorage _ds{
                    __new_processing_data,                                  // Describes data
                    __new_mask_data,                                        // Describes mask
                    typename _SetRange::_SourceProcessingDataOffsets{} };   // Describes offsets to processing data

                return _SetRange{_ds};
            }
        }

        //try searching for "corresponding" subrange [__bb; __ee) in the second sequence
        _RandomAccessIterator2 __bb = __first2;
        if (__b != __first1)
            __bb = __first2 + __internal::__pstl_lower_bound(__first2, _DifferenceType2{0}, __last2 - __first2, __b,
                                                             __comp, __proj2, __proj1);

        _RandomAccessIterator2 __ee = __last2;
        if (__e != __last1)
            __ee = __bb + __internal::__pstl_lower_bound(__bb, _DifferenceType2{0}, __last2 - __bb, __e, __comp,
                                                         __proj2, __proj1);

        const _DifferenceType __buf_pos = __size_func(__b - __first1, __bb - __first2);

        auto __buffer_b = __buf_raw_data_begin + __buf_pos;

        const _DifferenceType __mask_buf_pos = __mask_size_func(__b - __first1, __bb - __first2);

        auto __mask_b = __mask_bufs.get_buf_mask_rng_data(__mask_buf_pos);

        auto [__it1_reached, __it2_reached, __output_reached, __mask_reached] =
            __set_union_op(__b, __e, __bb, __ee, __buffer_b, __mask_b, __comp, __proj1, __proj2);

        // Prepare processed data info
        const typename _SetRange::_Data __new_processing_data{
            0,                                  // Offset in output range w/o limitation to output data size
            __output_reached - __buffer_b,      // The length of data pack: the same for windowed and result buffers
            __buf_pos};                         // Offset in temporary buffer w/o limitation to output data size

        if constexpr (!__Bounded)
        {
            return _SetRange{__new_processing_data};
        }
        else
        {
            // Prepare processed mask info
            typename _SetRange::_MaskData __new_mask_data{
                0,                                  // Offset in output range w/o limitation to output data size
                __mask_reached - __mask_b,          // The length of data pack: the same for windowed and result buffers
                __mask_buf_pos };                   // Offset in temporary buffer w/o limitation to output data size

            typename _SetRange::_SourceProcessingDataOffsets __new_offsets_to_processing_data{
                __b - __first1,                     // Absolute offset to processing data in the first data set
                __bb - __first2 };                  // Absolute offset to processing data in the second data set

            typename _SetRange::_DataStorage _ds{
                __new_processing_data,              // Describes data
                __new_mask_data,                    // Describes mask
                __new_offsets_to_processing_data }; // Describes offsets to processing data

            return _SetRange{_ds};
        }
    }
};

template <bool __Bounded, class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1,
          class _RandomAccessIterator2, class _OutputIterator, class _SizeFunction, class _MaskSizeFunction,
          class _SetUnionOp, class _Compare, class _Proj1, class _Proj2>
__parallel_set_op_return_t<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator>
__parallel_set_op(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                  _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                  _OutputIterator __result1, _OutputIterator __result2, _SizeFunction __size_func,
                  _MaskSizeFunction __mask_size_func, _SetUnionOp __set_union_op, _Compare __comp, _Proj1 __proj1,
                  _Proj2 __proj2)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceTypeOutput = typename std::iterator_traits<_OutputIterator>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2, _DifferenceTypeOutput>;
    using _T = typename std::iterator_traits<_OutputIterator>::value_type;

    const _DifferenceType1   __n1 = __last1 - __first1;
    const _DifferenceType2   __n2 = __last2 - __first2;
    const _DifferenceType __n_out = __result2 - __result1;

    const auto __buf_size = __size_func(__n1, __n2);
    const auto __mask_buf_size = __mask_size_func(__n1, __n2);

    __par_backend::__buffer<_T> __buf(__buf_size);                                              // Temporary (windowed) buffer for result preparation
    __mask_buffers<__Bounded>   __mask_bufs(__mask_buf_size);                                   // Temporary (windowed) buffer + result buffer for mask preparation
    using __mask_difference_type_t = typename __mask_buffers<__Bounded>::_difference_t;
    using _mask_ptr_t = typename __mask_buffers<__Bounded>::_mask_ptr_t;

    using _SetRange = _SetRangeImpl<__Bounded, _DifferenceType1, _DifferenceType2, __mask_difference_type_t, _DifferenceTypeOutput, _DifferenceType>;

    return __internal::__except_handler([__tag, &__exec, __n1, __n2, __n_out, __first1, __last1, __first2, __last2,
                                         __result1, __result2, __comp, __proj1, __proj2, __size_func, __mask_size_func,
                                         __set_union_op, &__buf, &__mask_bufs, __buf_size, __mask_buf_size]() 
    {
        // Buffer raw data begin/end pointers
        const auto __buf_raw_data_begin = __buf.get();
        const auto __buf_raw_data_end = __buf_raw_data_begin + __buf_size;

        _DifferenceType1 __res_reachedPos1 = __n1; // offset to the first unprocessed item from the first input range
        _DifferenceType2 __res_reachedPos2 = __n2; // offset to the first unprocessed item from the second input range
        _DifferenceType __res_reachedPosOut = 0;   // offset to the first unprocessed item from output range

        _SetRangeCombiner<__Bounded, _DifferenceType1, _DifferenceType2, __mask_difference_type_t, _DifferenceTypeOutput, _DifferenceType> __combine_pred{__n_out};

        // Scan predicate
        _ScanPred<__Bounded, _IsVector, _T*, _mask_ptr_t, _OutputIterator, _DifferenceType1, _DifferenceType2>
            __scan_pred{
                __tag,     __buf_raw_data_begin, __buf_raw_data_end, __mask_bufs.get_buf_mask_rng_data(), __result1,
                __result2, __res_reachedPos1,    __res_reachedPos2};

        _ParallelSetOpStrictScanPred<__Bounded, _SetRange, _RandomAccessIterator1, _RandomAccessIterator2,
                                     _OutputIterator, _SizeFunction, _MaskSizeFunction, _SetUnionOp, _Compare, _Proj1,
                                     _Proj2, _T>
            __reduce_pred{__first1,       __last1, __first2, __last2, __size_func,          __mask_size_func,
                          __set_union_op, __comp,  __proj1,  __proj2, __buf_raw_data_begin, __mask_bufs};

        auto __apex_pred = [__n_out, __result1, __result2, &__res_reachedPos1, &__res_reachedPos2, &__res_reachedPosOut,
                            &__scan_pred](const _SetRange& __total) {
            //final scan
            __scan_pred(/* 0 */ _DifferenceType1{}, /* 0 */ _DifferenceType1{}, __total);

            __res_reachedPosOut = std::min(__total.get_data_part().__pos + __total.get_data_part().__len, __n_out);
        };

        __par_backend::__parallel_strict_scan(__backend_tag{}, __exec, __n1, _SetRange(), __reduce_pred, __combine_pred,
                                              __scan_pred, __apex_pred);

        return __parallel_set_op_return_t<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator>{
            __first1 + __res_reachedPos1, __first2 + __res_reachedPos2, __result1 + __res_reachedPosOut};
    });
}

//a shared parallel pattern for '__pattern_set_union' and '__pattern_set_symmetric_difference'
template <bool __Bounded, class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1,
          class _RandomAccessIterator2, class _OutputIterator, class _SetUnionOp, class _Compare, class _Proj1,
          class _Proj2>
oneapi::dpl::__utils::__set_operations_result<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator>
__parallel_set_union_op(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                        _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                        _OutputIterator __result1, _OutputIterator __result2, _SetUnionOp __set_union_op,
                        _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2>;

    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;
    const auto __n_out = __result2 - __result1;

    __brick_copy<__parallel_tag<_IsVector>> __copy_range{};

    // {1} {}: parallel copying just first sequence
    if (__n2 == 0)
    {
        auto __last1_tmp = !__Bounded ? __last1 : __first1 + std::min(__n1, __n_out);

        _OutputIterator __result_finish = __internal::__pattern_walk2_brick(
            __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1_tmp, __result1, __copy_range);

        return {__last1_tmp, __first2, __result_finish};
    }

    // {} {2}: parallel copying just second sequence
    if (__n1 == 0)
    {
        auto __last2_tmp = !__Bounded ? __last2 : __first2 + std::min(__n2, __n_out);

        _OutputIterator __result_finish = __internal::__pattern_walk2_brick(
            __tag, std::forward<_ExecutionPolicy>(__exec), __first2, __last2_tmp, __result1, __copy_range);

        return {__first1, __last2_tmp, __result_finish};
    }

    // testing  whether the sequences are intersected
    _RandomAccessIterator1 __left_bound_seq_1 =
        __first1 + __internal::__pstl_lower_bound(__first1, _DifferenceType1{0}, __last1 - __first1, __first2, __comp,
                                                  __proj1, __proj2);

    if (__left_bound_seq_1 == __last1)
    {
        auto __last1_tmp = !__Bounded ? __last1 : __first1 + std::min(__n1, __n_out);
        auto __n1_tmp = __last1_tmp - __first1;

        auto __last2_tmp =
            !__Bounded ? __last2 : __first2 + std::min(__n2, __n_out > __n1_tmp ? __n_out - __n1_tmp : 0);
        auto __n2_tmp = __last2_tmp - __first2;

        //{1} < {2}: seq2 is wholly greater than seq1, so, do parallel copying seq1 and seq2
        __par_backend::__parallel_invoke(
            __backend_tag{}, __exec,
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first1, __last1_tmp, __result1, __copy_range);
            },
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first2, __last2_tmp, __result1 + __n1_tmp,
                                                  __copy_range);
            });

        return {__last1_tmp, __last2_tmp, __result1 + __n1_tmp + __n2_tmp};
    }

    // testing  whether the sequences are intersected
    _RandomAccessIterator2 __left_bound_seq_2 =
        __first2 + __internal::__pstl_lower_bound(__first2, _DifferenceType2{0}, __last2 - __first2, __first1, __comp,
                                                  __proj2, __proj1);

    if (__left_bound_seq_2 == __last2)
    {
        auto __last2_tmp = !__Bounded ? __last2 : __first2 + std::min(__n2, __n_out);
        auto __n2_tmp = __last2_tmp - __first2;

        auto __last1_tmp =
            !__Bounded ? __last1 : __first1 + std::min(__n1, __n_out > __n2_tmp ? __n_out - __n2_tmp : 0);
        auto __n1_tmp = __last1_tmp - __first1;

        //{2} < {1}: seq2 is wholly greater than seq1, so, do parallel copying seq1 and seq2
        __par_backend::__parallel_invoke(
            __backend_tag{}, __exec,
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first2, __last2_tmp, __result1, __copy_range);
            },
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first1, __last1_tmp, __result1 + __n2_tmp,
                                                  __copy_range);
            });

        return {__last1_tmp, __last2_tmp, __result1 + __n1_tmp + __n2_tmp};
    }

    auto __size_fnc = [](_DifferenceType __n, _DifferenceType __m) { return __n + __m; };
    auto __mask_size_fnc = __size_fnc;

    const auto __m1 = __left_bound_seq_1 - __first1;
    if (oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__m1))
    {
        oneapi::dpl::__utils::__set_operations_result<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator> __finish;

        auto __res_or = __result1;
        __result1 += __m1;                                                             //we know proper offset due to [first1; left_bound_seq_1) < [first2; last2)
        __par_backend::__parallel_invoke(
            __backend_tag{}, __exec,
            //do parallel copying of [first1; left_bound_seq_1)
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first1, __left_bound_seq_1, __res_or, __copy_range);
            },
            [=, &__exec, &__finish]
            {
                __finish = __internal::__parallel_set_op<__Bounded>(
                    __tag, __exec, __left_bound_seq_1, __last1, __first2, __last2, __result1, __result2, __size_fnc,
                    __mask_size_fnc, __set_union_op, __comp, __proj1, __proj2);
            });
        return __finish;
    }

    const auto __m2 = __left_bound_seq_2 - __first2;
    assert(__m1 == 0 || __m2 == 0);
    if (oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__m2))
    {
        oneapi::dpl::__utils::__set_operations_result<_RandomAccessIterator1, _RandomAccessIterator2, _OutputIterator> __finish;

        auto __res_or = __result1;
        __result1 += __m2;                                                              //we know proper offset due to [first2; left_bound_seq_2) < [first1; last1)
        __par_backend::__parallel_invoke(
            __backend_tag{}, __exec,
            //do parallel copying of [first2; left_bound_seq_2)
            [=, &__exec] {
                __internal::__pattern_walk2_brick(__tag, __exec, __first2, __left_bound_seq_2, __res_or, __copy_range);
            },
            [=, &__exec, &__finish]
            {
                __finish = __internal::__parallel_set_op<__Bounded>(
                    __tag, __exec, __first1, __last1, __left_bound_seq_2, __last2, __result1, __result2, __size_fnc,
                    __mask_size_fnc, __set_union_op, __comp, __proj1, __proj2);
            });
        return __finish;
    }

    return __internal::__parallel_set_op<__Bounded>(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                    __first2, __last2, __result1, __result2, __size_fnc,
                                                    __mask_size_fnc, __set_union_op, __comp, __proj1, __proj2);
}

//------------------------------------------------------------------------
// set_union
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_set_union(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                  _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                  /*__is_vector=*/::std::false_type) noexcept
{
    return std::set_union(__first1, __last1, __first2, __last2, __result, __comp);
}

template <typename _IsVector>
struct __BrickCopyConstruct
{
    template <typename _ForwardIterator, typename _OutputIterator>
    _OutputIterator
    operator()(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result)
    {
        return __brick_uninitialized_copy(__first, __last, __result, _IsVector());
    }
};

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_set_union(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                  _RandomAccessIterator2 __last2, _OutputIterator __result, _Compare __comp,
                  /*__is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_union(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator,
          class _Compare>
_OutputIterator
__pattern_set_union(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                    _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                    _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_set_union(__first1, __last1, __first2, __last2, __result, __comp,
                                         typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _OutputIterator, class _Compare>
_OutputIterator
__pattern_set_union(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                    _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                    _OutputIterator __result, _Compare __comp)
{
    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;

    // use serial algorithm
    if (!oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__n1 + __n2))
        return std::set_union(__first1, __last1, __first2, __last2, __result, __comp);

    using _Tp = typename std::iterator_traits<_OutputIterator>::value_type;
    return __parallel_set_union_op</*__Bounded*/ false>(
               __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
               __result + __n1 + __n2,
               [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                  _RandomAccessIterator2 __last2, _Tp* __result, auto __mask, _Compare __comp, oneapi::dpl::identity,
                  oneapi::dpl::identity) {
                   return oneapi::dpl::__utils::__set_union_construct(__first1, __last1, __first2, __last2, __result,
                                                                      __mask, __BrickCopyConstruct<_IsVector>(), __comp,
                                                                      oneapi::dpl::identity{}, oneapi::dpl::identity{});
               },
               __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{})
        .__get_reached_out();
}

//------------------------------------------------------------------------
// set_intersection
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_set_intersection(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                         _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                         /*__is_vector=*/::std::false_type) noexcept
{
    return std::set_intersection(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__brick_set_intersection(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                         _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                         _RandomAccessIterator3 __result, _Compare __comp,
                         /*__is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_intersection(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator,
          class _Compare>
_OutputIterator
__pattern_set_intersection(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                           _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                           _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_set_intersection(__first1, __last1, __first2, __last2, __result, __comp,
                                                typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__pattern_set_intersection(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                           _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                           _RandomAccessIterator2 __last2, _RandomAccessIterator3 __result, _Compare __comp)
{
    using _T = typename std::iterator_traits<_RandomAccessIterator3>::value_type;

    using _DifferenceType1 = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;
    using _DifferenceType2 = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;
    using _DifferenceType = std::common_type_t<_DifferenceType1, _DifferenceType2>;

    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;

    // intersection is empty
    if (__n1 == 0 || __n2 == 0)
        return __result;

    // testing  whether the sequences are intersected
    _RandomAccessIterator1 __left_bound_seq_1 = std::lower_bound(__first1, __last1, *__first2, __comp);
    //{1} < {2}: seq 2 is wholly greater than seq 1, so, the intersection is empty
    if (__left_bound_seq_1 == __last1)
        return __result;

    // testing  whether the sequences are intersected
    _RandomAccessIterator2 __left_bound_seq_2 = std::lower_bound(__first2, __last2, *__first1, __comp);
    //{2} < {1}: seq 1 is wholly greater than seq 2, so, the intersection is empty
    if (__left_bound_seq_2 == __last2)
        return __result;

    const auto __m1 = __last1 - __left_bound_seq_1 + __n2;
    if (oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__m1))
    {
        //we know proper offset due to [first1; left_bound_seq_1) < [first2; last2)
        return __internal::__except_handler([&]() {
            return __internal::__parallel_set_op</*__Bounded*/ false>(
                       __tag, std::forward<_ExecutionPolicy>(__exec), __left_bound_seq_1, __last1, __first2, __last2,
                       __result, __result + __n1 + __n2,
                       [](_DifferenceType __n, _DifferenceType __m) { return std::min(__n, __m); },
                       [](_DifferenceType __n, _DifferenceType __m) { return __n + __m; },
                       [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                          _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _T* __result, auto __mask,
                          _Compare __comp, oneapi::dpl::identity, oneapi::dpl::identity) {
                           return oneapi::dpl::__utils::__set_intersection_construct(
                               __first1, __last1, __first2, __last2, __result, __mask,
                               oneapi::dpl::__internal::__op_uninitialized_copy<_ExecutionPolicy>{}, __comp,
                               oneapi::dpl::identity{}, oneapi::dpl::identity{});
                       },
                       __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{})
                .__get_reached_out();
        });
    }

    const auto __m2 = __last2 - __left_bound_seq_2 + __n1;
    if (oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__m2))
    {
        //we know proper offset due to [first2; left_bound_seq_2) < [first1; last1)
        return __internal::__except_handler([&]() {
            return __internal::__parallel_set_op</*__Bounded*/ false>(
                       __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __left_bound_seq_2, __last2,
                       __result, __result + __n1 + __n2,
                       [](_DifferenceType __n, _DifferenceType __m) { return std::min(__n, __m); },
                       [](_DifferenceType __n, _DifferenceType __m) { return __n + __m; },
                       [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                          _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _T* __result, auto __mask,
                          _Compare __comp, oneapi::dpl::identity, oneapi::dpl::identity) {
                           return oneapi::dpl::__utils::__set_intersection_construct(
                               __first1, __last1, __first2, __last2, __result, __mask,
                               oneapi::dpl::__internal::__op_uninitialized_copy<_ExecutionPolicy>{}, __comp,
                               oneapi::dpl::identity{}, oneapi::dpl::identity{});
                       },
                       __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{})
                .__get_reached_out();
        });
    }

    // [left_bound_seq_1; last1) and [left_bound_seq_2; last2) - use serial algorithm
    return std::set_intersection(__left_bound_seq_1, __last1, __left_bound_seq_2, __last2, __result, __comp);
}

//------------------------------------------------------------------------
// set_difference
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_set_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                       _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                       /*__is_vector=*/::std::false_type) noexcept
{
    return std::set_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__brick_set_difference(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                       _RandomAccessIterator2 __last2, _RandomAccessIterator3 __result, _Compare __comp,
                       /*__is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator,
          class _Compare>
_OutputIterator
__pattern_set_difference(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                         _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_set_difference(__first1, __last1, __first2, __last2, __result, __comp,
                                              typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__pattern_set_difference(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                         _RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                         _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                         _RandomAccessIterator3 __result,
                         _Compare __comp)
{
    using _T = typename std::iterator_traits<_RandomAccessIterator3>::value_type;
    using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator1>::difference_type;

    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;

    // {} \ {2}: the difference is empty
    if (__n1 == 0)
        return __result;

    // {1} \ {}: parallel copying just first sequence
    if (__n2 == 0)
        return __pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __result,
                                     __internal::__brick_copy<__parallel_tag<_IsVector>>{});

    // testing  whether the sequences are intersected
    _RandomAccessIterator1 __left_bound_seq_1 = std::lower_bound(__first1, __last1, *__first2, __comp);
    //{1} < {2}: seq 2 is wholly greater than seq 1, so, parallel copying just first sequence
    if (__left_bound_seq_1 == __last1)
        return __pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __result,
                                     __internal::__brick_copy<__parallel_tag<_IsVector>>{});

    // testing  whether the sequences are intersected
    _RandomAccessIterator2 __left_bound_seq_2 = std::lower_bound(__first2, __last2, *__first1, __comp);
    //{2} < {1}: seq 1 is wholly greater than seq 2, so, parallel copying just first sequence
    if (__left_bound_seq_2 == __last2)
        return __internal::__pattern_walk2_brick(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                 __result, __brick_copy<__parallel_tag<_IsVector>>{});

    if (oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__n1 + __n2))
    {
        return __parallel_set_op</*__Bounded*/ false>(
                   __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
                   __result + __n1 + __n2, [](_DifferenceType __n, _DifferenceType) { return __n; },
                   [](_DifferenceType __n, _DifferenceType __m) { return __n + __m; },
                   [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                      _RandomAccessIterator2 __last2, _T* __result, auto __mask, _Compare __comp, oneapi::dpl::identity,
                      oneapi::dpl::identity) {
                       return oneapi::dpl::__utils::__set_difference_construct(
                           __first1, __last1, __first2, __last2, __result, __mask, __BrickCopyConstruct<_IsVector>(),
                           __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{});
                   },
                   __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{})
            .__get_reached_out();
    }

    // use serial algorithm
    return std::set_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

//------------------------------------------------------------------------
// set_symmetric_difference
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator, class _Compare>
_OutputIterator
__brick_set_symmetric_difference(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                 _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                                 /*__is_vector=*/::std::false_type) noexcept
{
    return std::set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__brick_set_symmetric_difference(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                                 _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                                 _RandomAccessIterator3 __result, _Compare __comp,
                                 /*__is_vector=*/::std::true_type) noexcept
{
    _PSTL_PRAGMA_MESSAGE("Vectorized algorithm unimplemented, redirected to serial");
    return std::set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _OutputIterator,
          class _Compare>
_OutputIterator
__pattern_set_symmetric_difference(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                   _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                                   _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp,
                                                        typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _RandomAccessIterator3, class _Compare>
_RandomAccessIterator3
__pattern_set_symmetric_difference(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                                   _RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                                   _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                                   _RandomAccessIterator3 __result, _Compare __comp)
{
    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;

    // use serial algorithm
    if (!oneapi::dpl::__internal::__is_set_algo_cutoff_exceeded(__n1 + __n2))
        return std::set_symmetric_difference(__first1, __last1, __first2, __last2, __result, __comp);

    using _Tp = typename std::iterator_traits<_RandomAccessIterator3>::value_type;
    return __internal::__except_handler([&]() {
        return __internal::__parallel_set_union_op</*__Bounded*/ false>(
                   __tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2, __last2, __result,
                   __result + __n1 + __n2,
                   [](_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                      _RandomAccessIterator2 __last2, _Tp* __result, auto __mask, _Compare __comp,
                      oneapi::dpl::identity, oneapi::dpl::identity) {
                       return oneapi::dpl::__utils::__set_symmetric_difference_construct(
                           __first1, __last1, __first2, __last2, __result, __mask, __BrickCopyConstruct<_IsVector>(),
                           __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{});
                   },
                   __comp, oneapi::dpl::identity{}, oneapi::dpl::identity{})
            .__get_reached_out();
    });
}

//------------------------------------------------------------------------
// is_heap_until
//------------------------------------------------------------------------

template <class _RandomAccessIterator, class _Compare>
_RandomAccessIterator
__brick_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                      /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::is_heap_until(__first, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare>
_RandomAccessIterator
__brick_is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                      /* __is_vector = */ ::std::true_type) noexcept
{
    using _SizeType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
    return __unseq_backend::__simd_first(__first, _SizeType(0), __last - __first,
                                         [&__comp](_RandomAccessIterator __it, _SizeType __i) {
                                             return std::invoke(__comp, __it[(__i - 1) / 2], __it[__i]);
                                         });
}

template <class _Tag, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator
__pattern_is_heap_until(_Tag, _ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                        _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_is_heap_until(__first, __last, __comp, typename _Tag::__is_vector{});
}

template <class _RandomAccessIterator, class _DifferenceType, class _Compare>
_RandomAccessIterator
__is_heap_until_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
                      /* __is_vector = */ ::std::false_type) noexcept
{
    _DifferenceType __i = __begin;
    for (; __i < __end; ++__i)
        if (std::invoke(__comp, __first[(__i - 1) / 2], __first[__i]))
            break;
    return __first + __i;
}

template <class _RandomAccessIterator, class _DifferenceType, class _Compare>
_RandomAccessIterator
__is_heap_until_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
                      /* __is_vector = */ ::std::true_type) noexcept
{
    return __unseq_backend::__simd_first(__first, __begin, __end,
                                         [&__comp](_RandomAccessIterator __it, _DifferenceType __i) {
                                             return std::invoke(__comp, __it[(__i - 1) / 2], __it[__i]);
                                         });
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
_RandomAccessIterator
__pattern_is_heap_until(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                        _RandomAccessIterator __last, _Compare __comp)
{
    return __internal::__except_handler([&]() {
        return __parallel_find(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__first, __comp](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                return __internal::__is_heap_until_local(__first, __i - __first, __j - __first, __comp, _IsVector{});
            },
            ::std::true_type{});
    });
}

//------------------------------------------------------------------------
// is_heap
//------------------------------------------------------------------------

template <class _RandomAccessIterator, class _Compare>
bool
__brick_is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::is_heap(__first, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare>
bool
__brick_is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                /* __is_vector = */ ::std::true_type) noexcept
{
    return !__unseq_backend::__simd_or_iter(__first, __last - __first, [__first, &__comp](_RandomAccessIterator __it) {
        return std::invoke(__comp, *(__first + (__it - __first - 1) / 2), *__it);
    });
}

template <class _RandomAccessIterator, class _DifferenceType, class _Compare>
bool
__is_heap_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
                /* __is_vector = */ ::std::false_type) noexcept
{
    return __internal::__is_heap_until_local(__first, __begin, __end, __comp, ::std::false_type{}) == __first + __end;
}

template <class _RandomAccessIterator, class _DifferenceType, class _Compare>
bool
__is_heap_local(_RandomAccessIterator __first, _DifferenceType __begin, _DifferenceType __end, _Compare __comp,
                /* __is_vector = */ ::std::true_type) noexcept
{
    return !__unseq_backend::__simd_or_iter(
        __first + __begin, __end - __begin, [__first, &__comp](_RandomAccessIterator __it) {
            return std::invoke(__comp, *(__first + (__it - __first - 1) / 2), *__it);
        });
}

template <class _Tag, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
bool
__pattern_is_heap(_Tag, _ExecutionPolicy&&, _RandomAccessIterator __first, _RandomAccessIterator __last,
                  _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_is_heap(__first, __last, __comp, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Compare>
bool
__pattern_is_heap(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                  _RandomAccessIterator __last, _Compare __comp)
{
    return __internal::__except_handler([&]() {
        return !__parallel_or(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
                              [__first, __comp](_RandomAccessIterator __i, _RandomAccessIterator __j) {
                                  return !__internal::__is_heap_local(__first, __i - __first, __j - __first, __comp,
                                                                      _IsVector{});
                              });
    });
}

//------------------------------------------------------------------------
// min_element
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _Compare>
_ForwardIterator
__brick_min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp,
                    /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::min_element(__first, __last, __comp);
}

template <typename _RandomAccessIterator, typename _Compare>
_RandomAccessIterator
__brick_min_element(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                    /* __is_vector = */ ::std::true_type) noexcept
{
#if _ONEDPL_UDR_PRESENT // _PSTL_UDR_PRESENT
    return __unseq_backend::__simd_min_element(__first, __last - __first, __comp);
#else
    return ::std::min_element(__first, __last, __comp);
#endif
}

template <class _Tag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Compare>
_ForwardIterator
__pattern_min_element(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                      _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_min_element(__first, __last, __comp, typename _Tag::__is_vector{});
}

template <typename _IsVector, typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
_RandomAccessIterator
__pattern_min_element(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                      _RandomAccessIterator __last, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    // a trivial case pre-check
    if (__last - __first < 2)
        return __first;

    return __internal::__except_handler([&]() {
        return __par_backend::__parallel_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, /*identity*/ __last,
            [=](_RandomAccessIterator __begin, _RandomAccessIterator __end,
                _RandomAccessIterator __init) -> _RandomAccessIterator {
                const _RandomAccessIterator __subresult =
                    __internal::__brick_min_element(__begin, __end, __comp, _IsVector{});
                return __init == __last ? __subresult
                                        : __internal::__cmp_iterators_by_values(__init, __subresult, __comp,
                                                                                oneapi::dpl::__internal::__pstl_less());
            },
            [=](_RandomAccessIterator __it1, _RandomAccessIterator __it2) -> _RandomAccessIterator {
                if (__it1 == __last)
                    return __it2;
                if (__it2 == __last)
                    return __it1;
                return __internal::__cmp_iterators_by_values(__it1, __it2, __comp,
                                                             oneapi::dpl::__internal::__pstl_less());
            });
    });
}

//------------------------------------------------------------------------
// minmax_element
//------------------------------------------------------------------------

template <typename _ForwardIterator, typename _Compare>
::std::pair<_ForwardIterator, _ForwardIterator>
__brick_minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp,
                       /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::minmax_element(__first, __last, __comp);
}

template <typename _RandomAccessIterator, typename _Compare>
::std::pair<_RandomAccessIterator, _RandomAccessIterator>
__brick_minmax_element(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
                       /* __is_vector = */ ::std::true_type) noexcept
{
#if _ONEDPL_UDR_PRESENT // _PSTL_UDR_PRESENT
    return __unseq_backend::__simd_minmax_element(__first, __last - __first, __comp);
#else
    return ::std::minmax_element(__first, __last, __comp);
#endif
}

template <class _Tag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Compare>
::std::pair<_ForwardIterator, _ForwardIterator>
__pattern_minmax_element(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                         _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_minmax_element(__first, __last, __comp, typename _Tag::__is_vector{});
}

template <typename _IsVector, typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
::std::pair<_RandomAccessIterator, _RandomAccessIterator>
__pattern_minmax_element(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                         _RandomAccessIterator __last, _Compare __comp)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    // a trivial case pre-check
    if (__last - __first < 2)
        return ::std::make_pair(__first, __first);

    return __internal::__except_handler([&]() {
        using _Result = std::pair<_RandomAccessIterator, _RandomAccessIterator>;

        return __par_backend::__parallel_reduce(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), __first, __last,
            /*identity*/ std::make_pair(__last, __last),
            [=, &__comp](_RandomAccessIterator __begin, _RandomAccessIterator __end, _Result __init) -> _Result {
                const _Result __subresult = __internal::__brick_minmax_element(__begin, __end, __comp, _IsVector{});
                if (__init.first == __last) // = identity
                    return __subresult;
                return ::std::make_pair(
                    __internal::__cmp_iterators_by_values(__init.first, __subresult.first, __comp,
                                                          oneapi::dpl::__internal::__pstl_less()),
                    __internal::__cmp_iterators_by_values(__init.second, __subresult.second,
                                                          oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
                                                          oneapi::dpl::__internal::__pstl_greater()));
            },
            [=, &__comp](_Result __p1, _Result __p2) -> _Result {
                if (__p1.first == __last)
                    return __p2;
                if (__p2.first == __last)
                    return __p1;
                return ::std::make_pair(
                    __internal::__cmp_iterators_by_values(__p1.first, __p2.first, __comp,
                                                          oneapi::dpl::__internal::__pstl_less()),
                    __internal::__cmp_iterators_by_values(__p1.second, __p2.second,
                                                          oneapi::dpl::__internal::__reorder_pred<_Compare>(__comp),
                                                          oneapi::dpl::__internal::__pstl_greater()));
            });
    });
}

//------------------------------------------------------------------------
// mismatch
//------------------------------------------------------------------------
template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
::std::pair<_ForwardIterator1, _ForwardIterator2>
__mismatch_serial(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                  _ForwardIterator2 __last2, _BinaryPredicate __pred)
{
    return ::std::mismatch(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
::std::pair<_ForwardIterator1, _ForwardIterator2>
__brick_mismatch(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                 _ForwardIterator2 __last2, _Predicate __pred, /* __is_vector = */ ::std::false_type) noexcept
{
    return __mismatch_serial(__first1, __last1, __first2, __last2, __pred);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Predicate>
::std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__brick_mismatch(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2,
                 _RandomAccessIterator2 __last2, _Predicate __pred, /* __is_vector = */ ::std::true_type) noexcept
{
    auto __n = ::std::min(__last1 - __first1, __last2 - __first2);
    return __unseq_backend::__simd_first(__first1, __n, __first2, __not_pred<_Predicate&>(__pred));
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
::std::pair<_ForwardIterator1, _ForwardIterator2>
__pattern_mismatch(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                   _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Predicate __pred) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_mismatch(__first1, __last1, __first2, __last2, __pred, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Predicate>
::std::pair<_RandomAccessIterator1, _RandomAccessIterator2>
__pattern_mismatch(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                   _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                   _Predicate __pred)
{
    if (__last1 - __first1 == 0 || __last2 - __first2 == 0)
        return {__first1, __first2};

    return __internal::__except_handler([&]() {
        auto __n = ::std::min(__last1 - __first1, __last2 - __first2);
        auto __result = __internal::__parallel_find(
            __tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n,
            [__first1, __first2, __pred](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                return __internal::__brick_mismatch(__i, __j, __first2 + (__i - __first1), __first2 + (__j - __first1),
                                                    __pred, _IsVector{})
                    .first;
            },
            ::std::true_type{});
        return ::std::make_pair(__result, __first2 + (__result - __first1));
    });
}

//------------------------------------------------------------------------
// lexicographical_compare
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool
__brick_lexicographical_compare(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                _ForwardIterator2 __last2, _Compare __comp,
                                /* __is_vector = */ ::std::false_type) noexcept
{
    return ::std::lexicographical_compare(__first1, __last1, __first2, __last2, __comp);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Compare>
bool
__brick_lexicographical_compare(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                                _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _Compare __comp,
                                /* __is_vector = */ ::std::true_type) noexcept
{
    if (__first2 == __last2)
    { // if second sequence is empty
        return false;
    }
    else if (__first1 == __last1)
    { // if first sequence is empty
        return true;
    }
    else
    {
        using ref_type1 = typename std::iterator_traits<_RandomAccessIterator1>::reference;
        using ref_type2 = typename std::iterator_traits<_RandomAccessIterator2>::reference;
        --__last1;
        --__last2;
        auto __n = ::std::min(__last1 - __first1, __last2 - __first2);
        ::std::pair<_RandomAccessIterator1, _RandomAccessIterator2> __result = __unseq_backend::__simd_first(
            __first1, __n, __first2, [__comp](const ref_type1 __x, const ref_type2 __y) mutable {
                return std::invoke(__comp, __x, __y) || std::invoke(__comp, __y, __x);
            });

        if (__result.first == __last1 && __result.second != __last2)
        { // if first sequence shorter than second
            return !std::invoke(__comp, *__result.second, *__result.first);
        }
        else
        { // if second sequence shorter than first or both have the same number of elements
            return std::invoke(__comp, *__result.first, *__result.second);
        }
    }
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Compare>
bool
__pattern_lexicographical_compare(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                  _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_lexicographical_compare(__first1, __last1, __first2, __last2, __comp,
                                                       typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Compare>
bool
__pattern_lexicographical_compare(__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                                  _RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                                  _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2,
                                  _Compare __comp) noexcept
{
    if (__first2 == __last2)
    { // if second sequence is empty
        return false;
    }
    else if (__first1 == __last1)
    { // if first sequence is empty
        return true;
    }
    else
    {
        using _RefType1 = typename std::iterator_traits<_RandomAccessIterator1>::reference;
        using _RefType2 = typename std::iterator_traits<_RandomAccessIterator2>::reference;

        return __internal::__except_handler([&]() {
            --__last1;
            --__last2;
            auto __n = ::std::min(__last1 - __first1, __last2 - __first2);
            auto __result = __internal::__parallel_find(
                __tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n,
                [__first1, __first2, &__comp](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j) {
                    return __internal::__brick_mismatch(
                               __i, __j, __first2 + (__i - __first1), __first2 + (__j - __first1),
                               [&__comp](const _RefType1 __x, const _RefType2 __y) {
                                   return !std::invoke(__comp, __x, __y) && !std::invoke(__comp, __y, __x);
                               },
                               _IsVector{})
                        .first;
                },
                ::std::true_type{});

            if (__result == __last1 && __first2 + (__result - __first1) != __last2)
            { // if first sequence shorter than second
                return !std::invoke(__comp, *(__first2 + (__result - __first1)), *__result);
            }
            else
            { // if second sequence shorter than first or both have the same number of elements
                return std::invoke(__comp, *__result, *(__first2 + (__result - __first1)));
            }
        });
    }
}

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2>
_ForwardIterator2
__pattern_swap(_Tag __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
               _ForwardIterator2 __first2)
{
    static_assert(__is_host_dispatch_tag_v<_Tag>);

    return __pattern_walk2(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                           oneapi::dpl::__internal::__swap_fn{});
}

//------------------------------------------------------------------------
// shift_left
//------------------------------------------------------------------------

template <class _ForwardIterator>
_ForwardIterator
__brick_shift_left(_ForwardIterator __first, _ForwardIterator __last,
                   typename ::std::iterator_traits<_ForwardIterator>::difference_type __n,
                   /*is_vector=*/::std::false_type) noexcept
{
#if _ONEDPL_CPP20_SHIFT_LEFT_RIGHT_PRESENT
    return ::std::shift_left(__first, __last, __n);
#else
    //If (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first. Otherwise, returns last.
    if (__n <= 0)
        return __last;

    //seek for (first + n)
    auto __it = oneapi::dpl::__internal::__next_to_last()(__first, __last, __n);
    if (__it == __last) // n >= last - first;
        return __first;

    //Moving the rest elements from a position number n to the begin of the sequence.
    for (; __it != __last; ++__it, (void)++__first)
        *__first = ::std::move(*__it);

    return __first;
#endif
}

template <class _ForwardIterator>
_ForwardIterator
__brick_shift_left(_ForwardIterator __first, _ForwardIterator __last,
                   typename ::std::iterator_traits<_ForwardIterator>::difference_type __n,
                   /*is_vector=*/::std::true_type) noexcept
{
    //If (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first. Otherwise, returns last.
    if (__n <= 0)
        return __last;
    auto __size = __last - __first;
    if (__n >= __size)
        return __first;

    using _DiffType = typename ::std::iterator_traits<_ForwardIterator>::difference_type;
    using _ReferenceType = typename ::std::iterator_traits<_ForwardIterator>::reference;

    _DiffType __mid = __size / 2 + __size % 2;
    _DiffType __size_res = __size - __n;

    //1. n >= size/2; there is enough memory to 'total' parallel (SIMD) copying
    if (__n >= __mid)
    {
        __unseq_backend::__simd_walk_n(__size_res,
                                       [](_ReferenceType __x, _ReferenceType __y) { __y = ::std::move(__x); },
                                       __first + __n, __first);
    }
    else //2. n < size/2; there is not enough memory to parallel (SIMD) copying; doing SIMD copying by n elements
    {
        for (auto __k = __n; __k < __size; __k += __n)
        {
            auto __end = ::std::min(__k + __n, __size);
            __unseq_backend::__simd_walk_n(__end - __k,
                                           [](_ReferenceType __x, _ReferenceType __y) { __y = ::std::move(__x); },
                                           __first + __k, __first + __k - __n);
        }
    }

    return __first + __size_res;
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator>
_ForwardIterator
__pattern_shift_left(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                     typename ::std::iterator_traits<_ForwardIterator>::difference_type __n) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __brick_shift_left(__first, __last, __n, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator>
_RandomAccessIterator
__pattern_shift_left(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                     _RandomAccessIterator __last,
                     typename ::std::iterator_traits<_RandomAccessIterator>::difference_type __n)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    //If (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first. Otherwise, returns last.
    if (__n <= 0)
        return __last;
    auto __size = __last - __first;
    if (__n >= __size)
        return __first;

    using _DiffType = typename ::std::iterator_traits<_RandomAccessIterator>::difference_type;

    _DiffType __mid = __size / 2 + __size % 2;
    _DiffType __size_res = __size - __n;

    return __internal::__except_handler([&]() {
        //1. n >= size/2; there is enough memory to 'total' parallel copying
        if (__n >= __mid)
        {
            __par_backend::__parallel_for(__backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __n, __size,
                                          [__first, __n](_DiffType __i, _DiffType __j) {
                                              __brick_move<__parallel_tag<_IsVector>>{}(
                                                  __first + __i, __first + __j, __first + __i - __n, _IsVector{});
                                          });
        }
        else //2. n < size/2; there is not enough memory to parallel copying; doing parallel copying by n elements
        {
            //TODO: to consider parallel processing by the 'internal' loop (but we may probably get cache locality issues)
            for (auto __k = __n; __k < __size; __k += __n)
            {
                auto __end = ::std::min(__k + __n, __size);
                __par_backend::__parallel_for(__backend_tag{}, __exec, __k, __end,
                                              [__first, __n](_DiffType __i, _DiffType __j) {
                                                  __brick_move<__parallel_tag<_IsVector>>{}(
                                                      __first + __i, __first + __j, __first + __i - __n, _IsVector{});
                                              });
            }
        }

        return __first + __size_res;
    });
}

template <class _Tag, class _ExecutionPolicy, class _BidirectionalIterator>
_BidirectionalIterator
__pattern_shift_right(_Tag __tag, _ExecutionPolicy&& __exec, _BidirectionalIterator __first,
                      _BidirectionalIterator __last,
                      typename ::std::iterator_traits<_BidirectionalIterator>::difference_type __n)
{
    static_assert(__is_host_dispatch_tag_v<_Tag>);

    using _ReverseIterator = typename ::std::reverse_iterator<_BidirectionalIterator>;

    auto __res = oneapi::dpl::__internal::__pattern_shift_left(
        __tag, ::std::forward<_ExecutionPolicy>(__exec), _ReverseIterator(__last), _ReverseIterator(__first), __n);

    return __res.base();
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ALGORITHM_IMPL_H
