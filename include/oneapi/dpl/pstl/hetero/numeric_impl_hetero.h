// -*- C++ -*-
//===-- numeric_impl_hetero.h ---------------------------------------------===//
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

#ifndef _ONEDPL_NUMERIC_IMPL_HETERO_H
#define _ONEDPL_NUMERIC_IMPL_HETERO_H

#include <iterator>
#include "../parallel_backend.h"
#include "../utils.h"
#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "algorithm_impl_hetero.h" // to use __pattern_walk2_brick
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _RandomAccessIterator1,
          typename _RandomAccessIterator2, typename _Tp, typename _BinaryOperation1, typename _BinaryOperation2>
_Tp
__pattern_transform_reduce(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _RandomAccessIterator1 __first1,
                           _RandomAccessIterator1 __last1, _RandomAccessIterator2 __first2, _Tp __init,
                           _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    if (__first1 == __last1)
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _BinaryOperation2>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __n = __last1 - __first1;
    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _RandomAccessIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __binary_op1, _Functor{__binary_op2},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               __buf1.all_view(), __buf2.all_view())
        .get();
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Tp,
          typename _BinaryOperation, typename _UnaryOperation>
_Tp
__pattern_transform_reduce(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                           _ForwardIterator __last, _Tp __init, _BinaryOperation __binary_op,
                           _UnaryOperation __unary_op)
{
    if (__first == __last)
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _RepackedTp = __par_backend_hetero::__repacked_tuple_t<_Tp>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __binary_op, _Functor{__unary_op},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               __buf.all_view())
        .get();
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------
template <typename T>
struct ExecutionPolicyWrapper;

#if _ONEDPL_BACKEND_SYCL
template <sycl::access::mode _Mode1, sycl::access::mode _Mode2, typename _T, typename _Allocator>
bool
__iterators_possibly_equal(const sycl_iterator<_Mode1, _T, _Allocator>& __it1,
                           const sycl_iterator<_Mode2, _T, _Allocator>& __it2)
{
    const auto buf1 = __it1.get_buffer();
    const auto buf2 = __it2.get_buffer();

    // If two different sycl iterators belongs to the different sycl buffers, they are different
    if (buf1 != buf2)
        return false;

    // We are unable to compare two sycl_iterator's if one of them is sub_buffer and assume that
    // two different sycl iterators are equal.
    if (buf1.is_sub_buffer() || buf2.is_sub_buffer())
        return true;

    return __it1 == __it2;
}
#endif // _ONEDPL_BACKEND_SYCL

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _UnaryOperation, typename _InitType, typename _BinaryOperation, typename _Inclusive>
_Iterator2
__pattern_transform_scan_base(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first,
                              _Iterator1 __last, _Iterator2 __result, _UnaryOperation __unary_op, _InitType __init,
                              _BinaryOperation __binary_op, _Inclusive)
{
    if (__first == __last)
        return __result;

    const auto __n = __last - __first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);

    // This is a temporary workaround for an in-place exclusive scan while the SYCL backend scan pattern is not fixed.
    const bool __is_scan_inplace_exclusive = __n > 1 && !_Inclusive{} && __iterators_possibly_equal(__first, __result);
    if (!__is_scan_inplace_exclusive)
    {
        auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
        auto __buf2 = __keep2(__result, __result + __n);

        oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(), __buf2.all_view(), __n,
            __unary_op, __init, __binary_op, _Inclusive{})
            .__deferrable_wait();
    }
    else
    {
        assert(__n > 1);
        assert(!_Inclusive{});
        assert(__iterators_possibly_equal(__first, __result));

        using _Type = typename _InitType::__value_type;

        auto __policy =
            __par_backend_hetero::make_wrapped_policy<ExecutionPolicyWrapper>(::std::forward<_ExecutionPolicy>(__exec));
        using _NewExecutionPolicy = decltype(__policy);

        // Create temporary buffer
        oneapi::dpl::__par_backend_hetero::__buffer<_NewExecutionPolicy, _Type> __tmp_buf(__policy, __n);
        auto __first_tmp = __tmp_buf.get();
        auto __last_tmp = __first_tmp + __n;
        auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
        auto __buf2 = __keep2(__first_tmp, __last_tmp);

        // Run main algorithm and save data into temporary buffer
        oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(_BackendTag{}, __policy, __buf1.all_view(),
                                                                     __buf2.all_view(), __n, __unary_op, __init,
                                                                     __binary_op, _Inclusive{})
            .wait();

        // Move data from temporary buffer into results
        oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag, ::std::move(__policy), __first_tmp, __last_tmp, __result,
            oneapi::dpl::__internal::__brick_move<__hetero_tag<_BackendTag>, _ExecutionPolicy>{});

        //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    }

    return __result + __n;
}
template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _UnaryOperation, typename _Type, typename _BinaryOperation, typename _Inclusive>
_Iterator2
__pattern_transform_scan(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first,
                         _Iterator1 __last, _Iterator2 __result, _UnaryOperation __unary_op, _Type __init,
                         _BinaryOperation __binary_op, _Inclusive)
{
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__init_value<_RepackedType>;

    return __pattern_transform_scan_base(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                         __unary_op, _InitType{__init}, __binary_op, _Inclusive{});
}

// scan without initial element
template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _UnaryOperation, typename _BinaryOperation, typename _Inclusive>
_Iterator2
__pattern_transform_scan(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first,
                         _Iterator1 __last, _Iterator2 __result, _UnaryOperation __unary_op,
                         _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = typename ::std::iterator_traits<_Iterator1>::value_type;
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__no_init_value<_RepackedType>;

    return __pattern_transform_scan_base(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __result,
                                         __unary_op, _InitType{}, __binary_op, _Inclusive{});
}

//------------------------------------------------------------------------
// adjacent_difference
//------------------------------------------------------------------------

// a wrapper for the policy is required to avoid the kernel naming issue
template <typename Name>
struct adjacent_difference_wrapper
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _BinaryOperation>
_ForwardIterator2
__pattern_adjacent_difference(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first,
                              _ForwardIterator1 __last, _ForwardIterator2 __d_first, _BinaryOperation __op)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __d_first;

    using _It1ValueT = typename ::std::iterator_traits<_ForwardIterator1>::value_type;
    using _It2ValueTRef = typename ::std::iterator_traits<_ForwardIterator2>::reference;

    _ForwardIterator2 __d_last = __d_first + __n;

#if !__SYCL_UNNAMED_LAMBDA__
    // if we have the only element, just copy it according to the specification
    if (__n == 1)
    {
        auto __wrapped_policy = __par_backend_hetero::make_wrapped_policy<adjacent_difference_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec));

        __internal::__pattern_walk2_brick(__tag, __wrapped_policy, __first, __last, __d_first,
                                          __internal::__brick_copy<__hetero_tag<_BackendTag>, _ExecutionPolicy>{});
    }
    else
#endif
    {
        auto __fn = [__op](_It1ValueT __in1, _It1ValueT __in2, _It2ValueTRef __out1) {
            __out1 = __op(__in2, __in1); // This move assignment is allowed by the C++ standard draft N4810
        };

        auto __keep1 =
            oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
        auto __buf1 = __keep1(__first, __last);
        auto __keep2 =
            oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator2>();
        auto __buf2 = __keep2(__d_first, __d_last);

        using _Function = unseq_backend::walk_adjacent_difference<_ExecutionPolicy, decltype(__fn)>;

        oneapi::dpl::__par_backend_hetero::__parallel_for(_BackendTag{}, __exec, _Function{__fn}, __n,
                                                          __buf1.all_view(), __buf2.all_view())
            .__deferrable_wait();
    }

    return __d_last;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_NUMERIC_IMPL_HETERO_H
