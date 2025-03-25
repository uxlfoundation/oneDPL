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

#ifndef _ONEDPL_NUMERIC_IMPL_H
#define _ONEDPL_NUMERIC_IMPL_H

#include <cassert>
#include <iterator>
#include <type_traits>
#include <numeric>

#include "execution_impl.h"
#include "unseq_backend_simd.h"
#include "algorithm_fwd.h"

#include "parallel_backend.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions, according to draft N4659)
//------------------------------------------------------------------------

template <class _ForwardIterator1, class _ForwardIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp
__brick_transform_reduce(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Tp __init,
                         _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2,
                         /*is_vector=*/::std::false_type) noexcept
{
    return ::std::inner_product(__first1, __last1, __first2, __init, __binary_op1, __binary_op2);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _Tp, class _BinaryOperation1,
          class _BinaryOperation2>
_Tp
__brick_transform_reduce(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
                         _RandomAccessIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1,
                         _BinaryOperation2 __binary_op2,
                         /*is_vector=*/::std::true_type) noexcept
{
    typedef typename ::std::iterator_traits<_RandomAccessIterator1>::difference_type _DifferenceType;
    return __unseq_backend::__simd_transform_reduce(
        __last1 - __first1, __init, __binary_op1,
        [=, &__binary_op2](_DifferenceType __i) { return __binary_op2(__first1[__i], __first2[__i]); });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator1, class _ForwardIterator2, class _Tp,
          class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(_Tag, _ExecutionPolicy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                           _ForwardIterator2 __first2, _Tp __init, _BinaryOperation1 __binary_op1,
                           _BinaryOperation2 __binary_op2) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __brick_transform_reduce(__first1, __last1, __first2, __init, __binary_op1, __binary_op2,
                                    typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_Tp
__pattern_transform_reduce(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                           _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init,
                           _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        return __par_backend::__parallel_transform_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
            [__first1, __first2, __binary_op2](_RandomAccessIterator1 __i) mutable {
                return __binary_op2(*__i, *(__first2 + (__i - __first1)));
            },
            __init,
            __binary_op1, // Combine
            [__first1, __first2, __binary_op1, __binary_op2](_RandomAccessIterator1 __i, _RandomAccessIterator1 __j,
                                                             _Tp __init) -> _Tp {
                return __internal::__brick_transform_reduce(__i, __j, __first2 + (__i - __first1), __init, __binary_op1,
                                                            __binary_op2, _IsVector{});
            });
    });
}

//------------------------------------------------------------------------
// transform_reduce (version with unary and binary functions)
//------------------------------------------------------------------------

template <class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
_Tp
__brick_transform_reduce(_ForwardIterator __first, _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                         _UnaryOperation __unary_op, /*is_vector=*/::std::false_type) noexcept
{
    for (; __first != __last; ++__first)
    {
        __init = __binary_op(__init, __unary_op(*__first));
    }
    return __init;
}

template <class _RandomAccessIterator, class _Tp, class _UnaryOperation, class _BinaryOperation>
_Tp
__brick_transform_reduce(_RandomAccessIterator __first, _RandomAccessIterator __last, _Tp __init,
                         _BinaryOperation __binary_op, _UnaryOperation __unary_op,
                         /*is_vector=*/::std::true_type) noexcept
{
    typedef typename ::std::iterator_traits<_RandomAccessIterator>::difference_type _DifferenceType;
    return __unseq_backend::__simd_transform_reduce(
        __last - __first, __init, __binary_op,
        [=, &__unary_op](_DifferenceType __i) { return __unary_op(__first[__i]); });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
_Tp
__pattern_transform_reduce(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last, _Tp __init,
                           _BinaryOperation __binary_op, _UnaryOperation __unary_op) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_transform_reduce(__first, __last, __init, __binary_op, __unary_op,
                                                typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _Tp, class _BinaryOperation,
          class _UnaryOperation>
_Tp
__pattern_transform_reduce(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                           _RandomAccessIterator __last, _Tp __init, _BinaryOperation __binary_op,
                           _UnaryOperation __unary_op)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    return __internal::__except_handler([&]() {
        return __par_backend::__parallel_transform_reduce(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last,
            [__unary_op](_RandomAccessIterator __i) mutable { return __unary_op(*__i); }, __init, __binary_op,
            [__unary_op, __binary_op](_RandomAccessIterator __i, _RandomAccessIterator __j, _Tp __init) {
                return __internal::__brick_transform_reduce(__i, __j, __init, __binary_op, __unary_op, _IsVector{});
            });
    });
}

//------------------------------------------------------------------------
// transform_exclusive_scan
//
// walk3 evaluates f(x,y,z) for (x,y,z) drawn from [first1,last1), [first2,...), [first3,...)
//------------------------------------------------------------------------

// Exclusive form
template <class _ForwardIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
::std::pair<_OutputIterator, _Tp>
__brick_transform_scan(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __result,
                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op,
                       /*Inclusive*/ ::std::false_type, /*is_vector=*/::std::false_type) noexcept
{
    for (; __first != __last; ++__first, ++__result)
    {
        // Copy the value pointed to by __first to avoid overwriting it when __result == __first
        _Tp __temp = *__first;
        *__result = __init;
        _ONEDPL_PRAGMA_FORCEINLINE
        __init = __binary_op(__init, __unary_op(__temp));
    }
    return ::std::make_pair(__result, __init);
}

// Inclusive form
template <class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation>
::std::pair<_OutputIterator, _Tp>
__brick_transform_scan(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result,
                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op,
                       /*Inclusive*/ ::std::true_type, /*is_vector=*/::std::false_type) noexcept
{
    for (; __first != __last; ++__first, ++__result)
    {
        _ONEDPL_PRAGMA_FORCEINLINE
        __init = __binary_op(__init, __unary_op(*__first));
        *__result = __init;
    }
    return ::std::make_pair(__result, __init);
}

// type is arithmetic and binary operation is a user defined operation.
template <typename _Tp, typename _BinaryOperation>
using is_arithmetic_udop = ::std::integral_constant<bool, ::std::is_arithmetic_v<_Tp> &&
                                                              !::std::is_same_v<_BinaryOperation, ::std::plus<_Tp>> &&
                                                              !::std::is_same_v<_BinaryOperation, ::std::plus<void>>>;

// [restriction] - T shall be DefaultConstructible.
// [violation] - default ctor of T shall set the identity value for binary_op.
template <class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation,
          class _Inclusive>
::std::enable_if_t<!is_arithmetic_udop<_Tp, _BinaryOperation>::value, ::std::pair<_OutputIterator, _Tp>>
__brick_transform_scan(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result,
                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op, _Inclusive,
                       /*is_vector=*/::std::true_type) noexcept
{
#if _ONEDPL_UDS_PRESENT // PSTL_UDS_PRESENT
    if (_Inclusive() || !oneapi::dpl::__internal::__iterators_possibly_equal(__first, __result))
    {
        return __unseq_backend::__simd_scan(__first, __last - __first, __result, __unary_op, __init, __binary_op,
                                            _Inclusive());
    }
#endif
    // We need to call serial brick here to call function for inclusive and exclusive scan that depends on _Inclusive() value
    return __internal::__brick_transform_scan(__first, __last, __result, __unary_op, __init, __binary_op, _Inclusive(),
                                              /*is_vector=*/::std::false_type());
}

template <class _RandomAccessIterator, class _OutputIterator, class _UnaryOperation, class _Tp, class _BinaryOperation,
          class _Inclusive>
::std::enable_if_t<is_arithmetic_udop<_Tp, _BinaryOperation>::value, ::std::pair<_OutputIterator, _Tp>>
__brick_transform_scan(_RandomAccessIterator __first, _RandomAccessIterator __last, _OutputIterator __result,
                       _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op, _Inclusive,
                       /*is_vector=*/::std::true_type) noexcept
{
    return __internal::__brick_transform_scan(__first, __last, __result, __unary_op, __init, __binary_op, _Inclusive(),
                                              /*is_vector=*/::std::false_type());
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation,
          class _Tp, class _BinaryOperation, class _Inclusive>
_OutputIterator
__pattern_transform_scan(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                         _OutputIterator __result, _UnaryOperation __unary_op, _Tp __init, _BinaryOperation __binary_op,
                         _Inclusive) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_transform_scan(__first, __last, __result, __unary_op, __init, __binary_op, _Inclusive(),
                                              typename _Tag::__is_vector{})
        .first;
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator,
          class _UnaryOperation, class _Tp, class _BinaryOperation, class _Inclusive>
::std::enable_if_t<!::std::is_floating_point_v<_Tp>, _OutputIterator>
__pattern_transform_scan(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                         _RandomAccessIterator __last, _OutputIterator __result, _UnaryOperation __unary_op, _Tp __init,
                         _BinaryOperation __binary_op, _Inclusive)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    typedef typename ::std::iterator_traits<_RandomAccessIterator>::difference_type _DifferenceType;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_transform_scan(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __last - __first,
            [__first, __unary_op](_DifferenceType __i) mutable { return __unary_op(__first[__i]); }, __init,
            __binary_op,
            [__first, __unary_op, __binary_op](_DifferenceType __i, _DifferenceType __j, _Tp __init) {
                // Execute serial __brick_transform_reduce, due to the explicit SIMD vectorization (reduction) requires a commutative operation for the guarantee of correct scan.
                return __internal::__brick_transform_reduce(__first + __i, __first + __j, __init, __binary_op,
                                                            __unary_op,
                                                            /*__is_vector*/ ::std::false_type());
            },
            [__first, __unary_op, __binary_op, __result](_DifferenceType __i, _DifferenceType __j, _Tp __init) {
                return __internal::__brick_transform_scan(__first + __i, __first + __j, __result + __i, __unary_op,
                                                          __init, __binary_op, _Inclusive(), _IsVector{})
                    .second;
            });
        return __result + (__last - __first);
    });
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator, class _OutputIterator,
          class _UnaryOperation, class _Tp, class _BinaryOperation, class _Inclusive>
::std::enable_if_t<::std::is_floating_point_v<_Tp>, _OutputIterator>
__pattern_transform_scan(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                         _RandomAccessIterator __last, _OutputIterator __result, _UnaryOperation __unary_op, _Tp __init,
                         _BinaryOperation __binary_op, _Inclusive)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    typedef typename ::std::iterator_traits<_RandomAccessIterator>::difference_type _DifferenceType;
    _DifferenceType __n = __last - __first;

    if (__n <= 0)
    {
        return __result;
    }

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_strict_scan(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __n, __init,
            [__first, __unary_op, __binary_op, __result](_DifferenceType __i, _DifferenceType __len) {
                return __internal::__brick_transform_scan(__first + __i, __first + (__i + __len), __result + __i,
                                                          __unary_op, _Tp{}, __binary_op, _Inclusive(), _IsVector{})
                    .second;
            },
            __binary_op,
            [__result, &__binary_op](_DifferenceType __i, _DifferenceType __len, _Tp __initial) {
                return *(::std::transform(__result + __i, __result + __i + __len, __result + __i,
                                          [&__initial, &__binary_op](const _Tp& __x) {
                                              _ONEDPL_PRAGMA_FORCEINLINE
                                              return __binary_op(__initial, __x);
                                          }) -
                         1);
            },
            [](_Tp) {});
        return __result + (__last - __first);
    });
}

// transform_scan without initial element
template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _UnaryOperation,
          class _BinaryOperation, class _Inclusive>
_OutputIterator
__pattern_transform_scan(_Tag __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                         _OutputIterator __result, _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive)
{
    static_assert(__is_host_dispatch_tag_v<_Tag>);

    typedef typename ::std::iterator_traits<_ForwardIterator>::value_type _ValueType;
    if (__first != __last)
    {
        _ValueType __tmp = __unary_op(*__first);
        *__result = __tmp;

        return __pattern_transform_scan(__tag, ::std::forward<_ExecutionPolicy>(__exec), ++__first, __last, ++__result,
                                        __unary_op, __tmp, __binary_op, _Inclusive());
    }
    else
    {
        return __result;
    }
}

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

template <class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator
__brick_adjacent_difference(_ForwardIterator __first, _ForwardIterator __last, _OutputIterator __d_first,
                            _BinaryOperation __op, /*is_vector*/ ::std::false_type) noexcept
{
    return ::std::adjacent_difference(__first, __last, __d_first, __op);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class BinaryOperation>
_RandomAccessIterator2
__brick_adjacent_difference(_RandomAccessIterator1 __first, _RandomAccessIterator1 __last,
                            _RandomAccessIterator2 __d_first, BinaryOperation __op,
                            /*is_vector=*/::std::true_type) noexcept
{
    assert(__first != __last);

    typedef typename ::std::iterator_traits<_RandomAccessIterator1>::reference _ReferenceType1;
    typedef typename ::std::iterator_traits<_RandomAccessIterator2>::reference _ReferenceType2;

    auto __n = __last - __first;
    *__d_first = *__first;
    return __unseq_backend::__simd_walk_3(
        __first + 1, __n - 1, __first, __d_first + 1,
        [&__op](_ReferenceType1 __x, _ReferenceType1 __y, _ReferenceType2 __z) { __z = __op(__x, __y); });
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _OutputIterator, class _BinaryOperation>
_OutputIterator
__pattern_adjacent_difference(_Tag, _ExecutionPolicy&&, _ForwardIterator __first, _ForwardIterator __last,
                              _OutputIterator __d_first, _BinaryOperation __op) noexcept
{
    static_assert(__is_serial_tag_v<_Tag> || __is_parallel_forward_tag_v<_Tag>);

    return __internal::__brick_adjacent_difference(__first, __last, __d_first, __op, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _RandomAccessIterator2,
          class _BinaryOperation>
_RandomAccessIterator2
__pattern_adjacent_difference(__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first,
                              _RandomAccessIterator1 __last, _RandomAccessIterator2 __d_first, _BinaryOperation __op)
{
    using __backend_tag = typename __parallel_tag<_IsVector>::__backend_tag;

    assert(__first != __last);
    typedef typename ::std::iterator_traits<_RandomAccessIterator1>::reference _ReferenceType1;
    typedef typename ::std::iterator_traits<_RandomAccessIterator2>::reference _ReferenceType2;

    *__d_first = *__first;

    return __internal::__except_handler([&]() {
        __par_backend::__parallel_for(
            __backend_tag{}, ::std::forward<_ExecutionPolicy>(__exec), __first, __last - 1,
            [&__op, __d_first, __first](_RandomAccessIterator1 __b, _RandomAccessIterator1 __e) {
                _RandomAccessIterator2 __d_b = __d_first + (__b - __first);
                __internal::__brick_walk3(
                    __b, __e, __b + 1, __d_b + 1,
                    [&__op](_ReferenceType1 __x, _ReferenceType1 __y, _ReferenceType2 __z) { __z = __op(__y, __x); },
                    _IsVector{});
            });
        return __d_first + (__last - __first);
    });
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_NUMERIC_IMPL_H
