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

// This file contains some specialization SYCL traits for some oneDPL types.
//
// Fancy iterators and internal functors which are device copyable when their
// template arguments are also device copyable should be explicitly specialized
// as such. This is important when template argument member variables may be
// device copyable but not trivially copyable.
// Include this header before a kernel submit SYCL code.

#ifndef _ONEDPL_SYCL_TRAITS_H
#define _ONEDPL_SYCL_TRAITS_H

#if _ONEDPL_SYCL_DEVICE_COPYABLE_SPECIALIZATION_BROKEN
// Prior to the particular version of SYCL library implementation, sycl::is_device_copyable relied upon a second
// template parameter to resolve ambiguity with the general is_trivially_copyable trait. This does not follow the SYCL
// specification, but is required to support older versions of SYCL library. Without this workaround, the user will face a
// compile time error of ambiguous partial template specialization.
#    define _ONEDPL_SPECIALIZE_FOR(TYPE, ...)                                                                          \
        TYPE<__VA_ARGS__>, std::enable_if_t<!std::is_trivially_copyable_v<TYPE<__VA_ARGS__>>>
#else
#    define _ONEDPL_SPECIALIZE_FOR(TYPE, ...) TYPE<__VA_ARGS__>
#endif

namespace oneapi::dpl::__internal
{

template <typename... _Ts>
struct __are_all_device_copyable : std::conjunction<sycl::is_device_copyable<_Ts>...>
{
};

template <typename _Pred>
class __not_pred;

template <typename _Pred>
class __reorder_pred;

template <typename _Tp>
class __equal_value;

template <typename _Tp>
class __not_equal_value;

template <typename _Comp, typename _Proj>
struct __compare;

template <typename _Pred>
class __transform_functor;

template <typename _UnaryOper, typename _UnaryPred>
class __transform_if_unary_functor;

template <typename _BinaryOper, typename _BinaryPred>
class __transform_if_binary_functor;

template <typename _Tp, typename _Pred>
class __replace_functor;

template <typename _Tp, typename _Pred>
class __replace_copy_functor;

template <typename _SourceT>
struct fill_functor;

template <typename _Generator>
struct generate_functor;

template <typename _Pred>
struct equal_predicate;

template <typename _Tp, typename _Pred>
struct __search_n_unary_predicate;

template <typename _Predicate>
struct adjacent_find_fn;

template <class _Comp>
struct __is_heap_check;

template <typename _Predicate, typename _ValueType>
struct __create_mask_unique_copy;

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename>
struct __brick_fill;

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename>
struct __brick_fill_n;

template <typename... T>
struct tuple;

} // namespace oneapi::dpl::__internal

template <typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__not_pred, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__reorder_pred, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _Tp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__equal_value, _Tp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp>
{
};

template <typename _Tp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__not_equal_value, _Tp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp>
{
};

template <typename _Comp, typename _Proj>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__compare, _Comp, _Proj)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Comp, _Proj>
{
};

template <typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__transform_functor, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _UnaryOper, typename _UnaryPred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__transform_if_unary_functor,
                                                       _UnaryOper, _UnaryPred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_UnaryOper, _UnaryPred>
{
};

template <typename _BinaryOper, typename _BinaryPred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__transform_if_binary_functor,
                                                       _BinaryOper, _BinaryPred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOper, _BinaryPred>
{
};

template <typename _Tp, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__replace_functor, _Tp, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp, _Pred>
{
};

template <typename _Tp, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__replace_copy_functor, _Tp, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp, _Pred>
{
};

template <typename _SourceT>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::fill_functor, _SourceT)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_SourceT>
{
};

template <typename _Generator>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::generate_functor, _Generator)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Generator>
{
};

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename _EnableIfPlaceholder>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__brick_fill, _Tag, _ExecutionPolicy,
                                                       _Tp, _EnableIfPlaceholder)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp>
{
};

template <class _Tag, typename _ExecutionPolicy, typename _Tp, typename _EnableIfPlaceholder>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__brick_fill_n, _Tag, _ExecutionPolicy,
                                                       _Tp, _EnableIfPlaceholder)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp>
{
};

template <typename _Tp, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__search_n_unary_predicate, _Tp, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Tp, _Pred>
{
};

template <class _Comp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__is_heap_check, _Comp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Comp>
{
};

template <typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::equal_predicate, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _Predicate>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::adjacent_find_fn, _Predicate)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Predicate>
{
};

template <typename _Predicate, typename _ValueType>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::__create_mask_unique_copy, _Predicate,
                                                       _ValueType)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Predicate>
{
};

template <typename... _Types>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__internal::tuple, _Types...)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Types...>
{
};

namespace oneapi::dpl::__par_backend_hetero
{

template <typename _UnaryOp>
struct __gen_transform_input;

template <typename _BinaryPred>
struct __gen_red_by_seg_reduce_input;

template <typename _BinaryPred>
struct __gen_red_by_seg_scan_input;

template <typename _Predicate, typename _RangeTransform>
struct __gen_mask;

template <typename _BinaryPredicate>
struct __gen_unique_mask;

template <typename _GenMask>
struct __gen_count_mask;

template <typename _GenMask, typename _RangeTransform>
struct __gen_expand_count_mask;

template <int32_t __offset, typename _Assign>
struct __write_to_id_if;

template <typename _Assign>
struct __write_to_id_if_else;

template <typename _BinaryPred>
struct __write_red_by_seg;

template <typename _ExecutionPolicy, typename _Pred>
struct __early_exit_find_or;

template <typename _Range, typename _Compare>
struct __leaf_sorter;

template <typename _BinaryOp>
struct __red_by_seg_op;

} // namespace oneapi::dpl::__par_backend_hetero

template <typename _UnaryOp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_transform_input,
                                                       _UnaryOp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_UnaryOp>
{
};

template <typename _Predicate, typename _RangeTransform>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_mask, _Predicate,
                                                       _RangeTransform)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Predicate>
{
};

template <typename _BinaryPred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_red_by_seg_reduce_input,
                                                       _BinaryPred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryPred>
{
};

template <typename _BinaryPred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_red_by_seg_scan_input,
                                                       _BinaryPred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryPred>
{
};

template <typename _BinaryPredicate>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_unique_mask,
                                                       _BinaryPredicate)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryPredicate>
{
};
template <typename _GenMask>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_count_mask, _GenMask)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_GenMask>
{
};

template <typename _GenMask, typename _RangeTransform>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__gen_expand_count_mask,
                                                       _GenMask, _RangeTransform)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_GenMask>
{
};

template <int32_t __offset, typename _Assign>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__write_to_id_if, __offset,
                                                       _Assign)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Assign>
{
};

template <typename _Assign>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__write_to_id_if_else,
                                                       _Assign)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Assign>
{
};

template <typename _BinaryPred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__write_red_by_seg,
                                                       _BinaryPred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryPred>
{
};

template <typename _ExecutionPolicy, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__early_exit_find_or,
                                                       _ExecutionPolicy, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _Range, typename _Compare>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__leaf_sorter, _Range,
                                                       _Compare)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Range, _Compare>
{
};

template <typename _BinaryOp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::__par_backend_hetero::__red_by_seg_op, _BinaryOp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOp>
{
};

namespace oneapi::dpl::unseq_backend
{

template <typename _ExecutionPolicy, typename _F>
struct walk_n;

template <typename _ExecutionPolicy, typename _F, typename _Range>
struct walk1_vector_or_scalar;

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2>
struct walk2_vectors_or_scalars;

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2, typename _Range3>
struct walk3_vectors_or_scalars;

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2>
struct walk_adjacent_difference;

template <typename _ExecutionPolicy, typename _Operation1, typename _Operation2, typename _Tp, typename _Commutative,
          std::uint8_t _VecSize>
struct transform_reduce;

template <typename _ExecutionPolicy, typename _BinaryOperation1, typename _Tp>
struct reduce_over_group;

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred_by_idx;

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred;

template <typename _ExecutionPolicy, typename _Pred>
struct multiple_match_pred;

template <typename _ExecutionPolicy, typename _Pred, typename _Tp, typename _Size>
struct n_elem_match_pred;

template <typename _ExecutionPolicy, typename _Pred>
struct first_match_pred;

template <typename _Pred, typename _Tp>
struct __create_mask;

template <typename _BinaryOp, typename _Assigner, typename _Inclusive, std::size_t N>
struct __copy_by_mask;

template <typename _BinaryOp, typename _Inclusive>
struct __partition_by_mask;

template <typename _Inclusive, typename _BinaryOp, typename _InitType>
struct __global_scan_functor;

template <typename _InitType>
struct __init_value;

template <typename _Inclusive, typename _ExecutionPolicy, typename _BinaryOperation, typename _UnaryOp,
          typename _WgAssigner, typename _GlobalAssigner, typename _DataAccessor, typename _InitType>
struct __scan;

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2>
struct __brick_includes;

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2, typename _IsOpDifference>
class __brick_set_op;

template <typename _BinaryOperator, typename _Size, typename _Range>
struct __brick_reduce_idx;

} // namespace oneapi::dpl::unseq_backend

template <typename _ExecutionPolicy, typename _F>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::walk_n, _ExecutionPolicy, _F)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_F>
{
};

template <typename _ExecutionPolicy, typename _F, typename _Range>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::walk1_vector_or_scalar,
                                                       _ExecutionPolicy, _F, _Range)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_F>
{
};

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::walk2_vectors_or_scalars,
                                                       _ExecutionPolicy, _F, _Range1, _Range2)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_F>
{
};

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2, typename _Range3>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::walk3_vectors_or_scalars,
                                                       _ExecutionPolicy, _F, _Range1, _Range2, _Range3)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_F>
{
};

template <typename _ExecutionPolicy, typename _F, typename _Range1, typename _Range2>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::walk_adjacent_difference,
                                                       _ExecutionPolicy, _F, _Range1, _Range2)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_F>
{
};

template <typename _ExecutionPolicy, typename _Operation1, typename _Operation2, typename _Tp, typename _Commutative,
          std::uint8_t _VecSize>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::transform_reduce, _ExecutionPolicy,
                                                       _Operation1, _Operation2, _Tp, _Commutative, _VecSize)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Operation1, _Operation2, _Tp>
{
};

template <typename _ExecutionPolicy, typename _BinaryOperation1, typename _Tp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::reduce_over_group, _ExecutionPolicy,
                                                       _BinaryOperation1, _Tp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOperation1, _Tp>
{
};

template <typename _ExecutionPolicy, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::single_match_pred_by_idx,
                                                       _ExecutionPolicy, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _ExecutionPolicy, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::single_match_pred, _ExecutionPolicy,
                                                       _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _ExecutionPolicy, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::multiple_match_pred,
                                                       _ExecutionPolicy, _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _ExecutionPolicy, typename _Pred, typename _Tp, typename _Size>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::n_elem_match_pred, _ExecutionPolicy,
                                                       _Pred, _Tp, _Size)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred, _Tp, _Size>
{
};

template <typename _ExecutionPolicy, typename _Pred>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::first_match_pred, _ExecutionPolicy,
                                                       _Pred)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred>
{
};

template <typename _Pred, typename _Tp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__create_mask, _Pred, _Tp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Pred, _Tp>
{
};

template <typename _BinaryOp, typename _Assigner, typename _Inclusive, std::size_t N>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__copy_by_mask, _BinaryOp, _Assigner,
                                                       _Inclusive, N)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOp, _Assigner>
{
};

template <typename _BinaryOp, typename _Inclusive>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__partition_by_mask, _BinaryOp,
                                                       _Inclusive)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOp>
{
};

template <typename _Inclusive, typename _BinaryOp, typename _InitType>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__global_scan_functor, _Inclusive,
                                                       _BinaryOp, _InitType)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOp, _InitType>
{
};

template <typename _InitType>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__init_value, _InitType)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_InitType>
{
};

template <typename _Inclusive, typename _ExecutionPolicy, typename _BinaryOperation, typename _UnaryOp,
          typename _WgAssigner, typename _GlobalAssigner, typename _DataAccessor, typename _InitType>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__scan, _Inclusive, _ExecutionPolicy,
                                                       _BinaryOperation, _UnaryOp, _WgAssigner, _GlobalAssigner,
                                                       _DataAccessor, _InitType)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOperation, _UnaryOp, _WgAssigner, _GlobalAssigner,
                                                         _DataAccessor, _InitType>
{
};

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__brick_includes, _ExecutionPolicy,
                                                       _Compare, _Size1, _Size2)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Compare, _Size1, _Size2>
{
};

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2, typename _IsOpDifference>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__brick_set_op, _ExecutionPolicy,
                                                       _Compare, _Size1, _Size2, _IsOpDifference)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Compare, _Size1, _Size2>
{
};

template <typename _BinaryOperator, typename _Size, typename _Range>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::unseq_backend::__brick_reduce_idx, _BinaryOperator,
                                                       _Size, _Range)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_BinaryOperator, _Size>
{
};

namespace oneapi::dpl::internal
{

enum class search_algorithm;

template <typename Comp, typename T, typename _Range, search_algorithm func>
struct __custom_brick;

template <typename T, typename Predicate>
struct replace_if_fun;

template <typename ValueType, typename FlagType, typename BinaryOp>
struct scan_by_key_fun;

template <typename ValueType, typename FlagType, typename BinaryOp>
struct segmented_scan_fun;

template <typename Output1, typename Output2>
class scatter_and_accumulate_fun;

template <typename T, typename Predicate, typename UnaryOperation>
class transform_if_stencil_fun;

} // namespace oneapi::dpl::internal

template <typename Comp, typename T, typename _Range, oneapi::dpl::internal::search_algorithm func>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::__custom_brick, Comp, T, _Range, func)>
    : oneapi::dpl::__internal::__are_all_device_copyable<Comp, T>
{
};

template <typename T, typename Predicate>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::replace_if_fun, T, Predicate)>
    : oneapi::dpl::__internal::__are_all_device_copyable<T, Predicate>
{
};

template <typename ValueType, typename FlagType, typename BinaryOp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::scan_by_key_fun, ValueType, FlagType,
                                                       BinaryOp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<BinaryOp>
{
};

template <typename ValueType, typename FlagType, typename BinaryOp>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::segmented_scan_fun, ValueType, FlagType,
                                                       BinaryOp)>
    : oneapi::dpl::__internal::__are_all_device_copyable<BinaryOp>
{
};

template <typename Output1, typename Output2>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::scatter_and_accumulate_fun, Output1,
                                                       Output2)>
    : oneapi::dpl::__internal::__are_all_device_copyable<Output1, Output2>
{
};

template <typename T, typename Predicate, typename UnaryOperation>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::internal::transform_if_stencil_fun, T, Predicate,
                                                       UnaryOperation)>
    : oneapi::dpl::__internal::__are_all_device_copyable<Predicate, UnaryOperation>
{
};

namespace oneapi::dpl
{

template <typename... _Types>
class zip_iterator;

template <typename _Iter, typename _UnaryFunc>
class transform_iterator;

template <typename SourceIterator, typename _Permutation>
class permutation_iterator;

} // namespace oneapi::dpl

template <typename... _Types>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::zip_iterator, _Types...)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Types...>
{
};

template <typename _Iter, typename _UnaryFunc>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::transform_iterator, _Iter, _UnaryFunc)>
    : oneapi::dpl::__internal::__are_all_device_copyable<_Iter, _UnaryFunc>
{
};

template <typename SourceIterator, typename _Permutation>
struct sycl::is_device_copyable<_ONEDPL_SPECIALIZE_FOR(oneapi::dpl::permutation_iterator, SourceIterator, _Permutation)>
    : oneapi::dpl::__internal::__are_all_device_copyable<SourceIterator, _Permutation>
{
};

#undef _ONEDPL_SPECIALIZE_FOR

#endif // _ONEDPL_SYCL_TRAITS_H
