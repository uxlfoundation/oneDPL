// -*- C++ -*-
//===-- for_loop.h --------------------------------------------------------===//
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

#ifndef _ONEDPL_EXPERIMENTAL_FOR_LOOP_H
#define _ONEDPL_EXPERIMENTAL_FOR_LOOP_H

#include <tuple>

#include "../../onedpl_config.h"
#include "../../execution_impl.h"
#include "../../utils.h"
#include "for_loop_impl.h"

namespace oneapi
{
namespace dpl
{
namespace experimental
{
inline namespace parallelism_v2
{

// TODO: type_identity should be available in type_traits starting from C++20
// Perhaps we need to add an internal structure if PSTL is used with older versions
template <typename _Tp>
struct type_identity
{
    using type = _Tp;
};
template <typename _Tp>
using type_identity_t = typename type_identity<_Tp>::type;

//hide vector policy till the algorithms implementation
// template <>
// struct __policy_traits<oneapi::dpl::execution::vector_policy>
// {
//     typedef ::std::false_type __allow_parallel;
//     typedef ::std::false_type __allow_unsequenced;
//     typedef ::std::true_type __allow_vector;
// };

// TODO: add static asserts for parameters according to the requirements
template <typename _ExecutionPolicy, typename _Ip, typename... _Rest>
void
for_loop(_ExecutionPolicy&& __exec, type_identity_t<_Ip> __start, _Ip __finish, _Rest&&... __rest)
{
    static_assert(oneapi::dpl::__internal::__is_host_execution_policy<::std::decay_t<_ExecutionPolicy>>::value,
                  "for_loop is implemented for the host policies only");

    oneapi::dpl::__internal::__for_loop_repack(__exec, __start, __finish,
                                               oneapi::dpl::__internal::__single_stride_type{},
                                               ::std::forward_as_tuple(::std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Sp, typename... _Rest>
void
for_loop_strided(_ExecutionPolicy&& __exec, type_identity_t<_Ip> __start, _Ip __finish, _Sp __stride, _Rest&&... __rest)
{
    static_assert(oneapi::dpl::__internal::__is_host_execution_policy<::std::decay_t<_ExecutionPolicy>>::value,
                  "for_loop_strided is implemented for the host policies only");

    oneapi::dpl::__internal::__for_loop_repack(__exec, __start, __finish, __stride,
                                               ::std::forward_as_tuple(::std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename... _Rest>
void
for_loop_n(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Rest&&... __rest)
{
    static_assert(oneapi::dpl::__internal::__is_host_execution_policy<::std::decay_t<_ExecutionPolicy>>::value,
                  "for_loop_n is implemented for the host policies only");

    oneapi::dpl::__internal::__for_loop_repack_n(__exec, __start, __n,
                                                 oneapi::dpl::__internal::__single_stride_type{},
                                                 ::std::forward_as_tuple(::std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Sp, typename... _Rest>
void
for_loop_n_strided(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Sp __stride, _Rest&&... __rest)
{
    static_assert(oneapi::dpl::__internal::__is_host_execution_policy<::std::decay_t<_ExecutionPolicy>>::value,
                  "for_loop_n_strided is implemented for the host policies only");

    oneapi::dpl::__internal::__for_loop_repack_n(__exec, __start, __n, __stride,
                                                 ::std::forward_as_tuple(::std::forward<_Rest>(__rest)...));
}

// Serial implementations
template <typename _Ip, typename... _Rest>
void
for_loop(type_identity_t<_Ip> __start, _Ip __finish, _Rest&&... __rest)
{
    oneapi::dpl::experimental::parallelism_v2::for_loop(oneapi::dpl::execution::v1::seq, __start, __finish,
                                                        ::std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Sp, typename... _Rest>
void
for_loop_strided(type_identity_t<_Ip> __start, _Ip __finish, _Sp __stride, _Rest&&... __rest)
{
    oneapi::dpl::experimental::parallelism_v2::for_loop_strided(oneapi::dpl::execution::v1::seq, __start, __finish,
                                                                __stride, ::std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Size, typename... _Rest>
void
for_loop_n(_Ip __start, _Size __n, _Rest&&... __rest)
{
    oneapi::dpl::experimental::parallelism_v2::for_loop_n(oneapi::dpl::execution::v1::seq, __start, __n,
                                                          ::std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Size, typename _Sp, typename... _Rest>
void
for_loop_n_strided(_Ip __start, _Size __n, _Sp __stride, _Rest&&... __rest)
{
    oneapi::dpl::experimental::parallelism_v2::for_loop_n_strided(oneapi::dpl::execution::v1::seq, __start, __n,
                                                                  __stride, ::std::forward<_Rest>(__rest)...);
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_EXPERIMENTAL_FOR_LOOP_H
