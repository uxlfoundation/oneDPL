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

#ifndef _ONEDPL_NUMERIC_RANGES_IMPL_HETERO_H
#define _ONEDPL_NUMERIC_RANGES_IMPL_HETERO_H

#include "../numeric_fwd.h"
#include "../parallel_backend.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/utils_ranges_sycl.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

namespace oneapi
{
namespace dpl
{
namespace __internal
{
namespace __ranges
{

//------------------------------------------------------------------------
// transform_reduce (version with two binary functions)
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Tp,
          typename _BinaryOperation1, typename _BinaryOperation2>
_Tp
__pattern_transform_reduce(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                           _Tp __init, _BinaryOperation1 __binary_op1, _BinaryOperation2 __binary_op2)
{
    if (__rng1.empty())
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _BinaryOperation2>;
    using _RepackedTp = oneapi::dpl::__par_backend_hetero::__repacked_tuple_t<_Tp>;

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __binary_op1, _Functor{__binary_op2},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2))
        .get();
}

//------------------------------------------------------------------------
// transform_reduce (with unary and binary functions)
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Tp, typename _BinaryOperation,
          typename _UnaryOperation>
_Tp
__pattern_transform_reduce(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range&& __rng, _Tp __init,
                           _BinaryOperation __binary_op, _UnaryOperation __unary_op)
{
    if (__rng.empty())
        return __init;

    using _Functor = unseq_backend::walk_n<_ExecutionPolicy, _UnaryOperation>;
    using _RepackedTp = oneapi::dpl::__par_backend_hetero::__repacked_tuple_t<_Tp>;

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_RepackedTp,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __binary_op, _Functor{__unary_op},
               unseq_backend::__init_value<_RepackedTp>{__init}, // initial value
               ::std::forward<_Range>(__rng))
        .get();
}

//------------------------------------------------------------------------
// transform_scan
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _InitType, typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_transform_scan_base(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                              _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    oneapi::dpl::__internal::__difference_t<_Range2> __n = __rng1.size();
    if (__n == 0)
        return 0;

    oneapi::dpl::__par_backend_hetero::__parallel_transform_scan(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), std::forward<_Range1>(__rng1),
        std::forward<_Range2>(__rng2), __n, __unary_op, __init, __binary_op, _Inclusive{})
        .__deferrable_wait();
    return __n;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _Type, typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_transform_scan(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                         _UnaryOperation __unary_op, _Type __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__init_value<_RepackedType>;

    return __pattern_transform_scan_base(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                                         ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2), __unary_op,
                                         _InitType{__init}, __binary_op, _Inclusive{});
}

// scan without initial element
template <typename _BackendTag, typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation,
          typename _BinaryOperation, typename _Inclusive>
oneapi::dpl::__internal::__difference_t<_Range2>
__pattern_transform_scan(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Range1&& __rng1, _Range2&& __rng2,
                         _UnaryOperation __unary_op, _BinaryOperation __binary_op, _Inclusive)
{
    using _Type = oneapi::dpl::__internal::__value_t<_Range1>;
    using _RepackedType = __par_backend_hetero::__repacked_tuple_t<_Type>;
    using _InitType = unseq_backend::__no_init_value<_RepackedType>;

    return __pattern_transform_scan_base(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                                         ::std::forward<_Range1>(__rng1), ::std::forward<_Range2>(__rng2), __unary_op,
                                         _InitType{}, __binary_op, _Inclusive{});
}

} // namespace __ranges
} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_NUMERIC_RANGES_IMPL_HETERO_H
