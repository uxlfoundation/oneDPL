// -*- C++ -*-
//===-- unseq_backend_sycl.h ----------------------------------------------===//
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

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL
#ifndef _ONEDPL_UNSEQ_BACKEND_SYCL_H
#define _ONEDPL_UNSEQ_BACKEND_SYCL_H

#include <type_traits>

#include "../../onedpl_config.h"
#include "../../utils.h"
#include "sycl_defs.h"

namespace oneapi
{
namespace dpl
{
namespace unseq_backend
{

#if _USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)
//This optimization depends on Intel(R) oneAPI DPC++ Compiler implementation such as support of binary operators from std namespace.
//We need to use defined(SYCL_IMPLEMENTATION_INTEL) macro as a guard.

template <typename _Tp>
inline constexpr bool __can_use_known_identity =
#    if ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION
    // When ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION is defined as non-zero, we avoid using known identity for 64-bit arithmetic data types
    !(::std::is_arithmetic_v<_Tp> && sizeof(_Tp) == sizeof(::std::uint64_t));
#    else
    true;
#    endif // ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION

//TODO: To change __has_known_identity implementation as soon as the Intel(R) oneAPI DPC++ Compiler implementation issues related to
//std::multiplies, std::bit_or, std::bit_and and std::bit_xor operations will be fixed.
//std::logical_and and std::logical_or are not supported in Intel(R) oneAPI DPC++ Compiler to be used in sycl::inclusive_scan_over_group and sycl::reduce_over_group
template <typename _BinaryOp, typename _Tp>
using __has_known_identity = ::std::conditional_t<
    __can_use_known_identity<_Tp>,
#    if _ONEDPL_LIBSYCL_VERSION >= 50200
    typename ::std::disjunction<
        __dpl_sycl::__has_known_identity<_BinaryOp, _Tp>,
        ::std::conjunction<::std::is_arithmetic<_Tp>,
                           ::std::disjunction<::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<_Tp>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<void>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<_Tp>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<void>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__minimum<_Tp>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__minimum<void>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__maximum<_Tp>>,
                                              ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__maximum<void>>>>>,
#    else  //_ONEDPL_LIBSYCL_VERSION >= 50200
    typename ::std::conjunction<
        ::std::is_arithmetic<_Tp>,
        ::std::disjunction<::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<_Tp>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<void>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<_Tp>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<void>>>>,
#    endif //_ONEDPL_LIBSYCL_VERSION >= 50200
    ::std::false_type>;     // This is for the case of __can_use_known_identity<_Tp>==false

#else //_USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)

template <typename _BinaryOp, typename _Tp>
using __has_known_identity = std::false_type;

#endif //_USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)

template <typename _BinaryOp, typename _Tp>
struct __known_identity_for_plus
{
    static_assert(::std::is_same_v<::std::decay_t<_BinaryOp>, ::std::plus<_Tp>> ||
                  ::std::is_same_v<::std::decay_t<_BinaryOp>, ::std::plus<void>> ||
                  ::std::is_same_v<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<_Tp>> ||
                  ::std::is_same_v<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<void>>);
    static constexpr _Tp value = 0;
};

template <typename _BinaryOp, typename _Tp>
inline constexpr _Tp __known_identity =
#if _ONEDPL_LIBSYCL_VERSION >= 50200
    __dpl_sycl::__known_identity<_BinaryOp, _Tp>::value;
#else  //_ONEDPL_LIBSYCL_VERSION >= 50200
    __known_identity_for_plus<_BinaryOp, _Tp>::value; //for plus only
#endif //_ONEDPL_LIBSYCL_VERSION >= 50200

template <typename _ExecutionPolicy, typename _F>
struct walk_n
{
    _F __f;

    template <typename _ItemId, typename... _Ranges>
    auto
    operator()(const _ItemId __idx, _Ranges&&... __rngs) const -> decltype(__f(__rngs[__idx]...))
    {
        return __f(__rngs[__idx]...);
    }
};

// If read accessor returns temporary value then __no_op returns lvalue reference to it.
// After temporary value destroying it will be a reference on invalid object.
// So let's don't call functor in case of __no_op
template <typename _ExecutionPolicy>
struct walk_n<_ExecutionPolicy, oneapi::dpl::__internal::__no_op>
{
    oneapi::dpl::__internal::__no_op __f;

    template <typename _ItemId, typename _Range>
    auto
    operator()(const _ItemId __idx, _Range&& __rng) const -> decltype(__rng[__idx])
    {
        return __rng[__idx];
    }
};

//------------------------------------------------------------------------
// walk_adjacent_difference
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _F>
struct walk_adjacent_difference
{
    _F __f;

    template <typename _ItemId, typename _Acc1, typename _Acc2>
    void
    operator()(const _ItemId __idx, const _Acc1& _acc_src, _Acc2& _acc_dst) const
    {
        using ::std::get;

        // just copy an element if it is the first one
        if (__idx == 0)
            _acc_dst[__idx] = _acc_src[__idx];
        else
            __f(_acc_src[__idx + (-1)], _acc_src[__idx], _acc_dst[__idx]);
    }
};

// the C++ stuff types to distinct "init vs. no init"
template <typename _InitType>
struct __init_value
{
    _InitType __value;
    using __value_type = _InitType;
};

template <typename _InitType = void>
struct __no_init_value
{
    using __value_type = _InitType;
};

// structure for the correct processing of the initial scan element
template <typename _InitType>
struct __init_processing
{
    template <typename _Tp>
    void
    operator()(const __init_value<_InitType>& __init, _Tp&& __value) const
    {
        __value = __init.__value;
    }
    template <typename _Tp>
    void
    operator()(const __no_init_value<_InitType>&, _Tp&&) const
    {
    }

    template <typename _Tp, typename _BinaryOp>
    void
    operator()(const __init_value<_InitType>& __init, _Tp&& __value, _BinaryOp __bin_op) const
    {
        __value = __bin_op(__init.__value, __value);
    }
    template <typename _Tp, typename _BinaryOp>
    void
    operator()(const __no_init_value<_InitType>&, _Tp&&, _BinaryOp) const
    {
    }
    template <typename _Tp, typename _BinaryOp>
    void
    operator()(const __no_init_value<void>&, _Tp&&, _BinaryOp) const
    {
    }
};

//------------------------------------------------------------------------
// transform_reduce
//------------------------------------------------------------------------

// Load elements consecutively from global memory, transform them, and apply a local reduction. Each local result is
// stored in local memory.
template <typename _ExecutionPolicy, typename _Operation1, typename _Operation2, typename _Tp, typename _Commutative,
          std::uint8_t _VecSize>
struct transform_reduce
{
    _Operation1 __binary_op;
    _Operation2 __unary_op;

    template <typename _Size, typename _Res, typename... _Acc>
    void
    vectorized_reduction_first(const _Size __start_idx, _Res& __res, const _Acc&... __acc) const
    {
        __res.__setup(__unary_op(__start_idx, __acc...));
        _ONEDPL_PRAGMA_UNROLL
        for (_Size __i = 1; __i < _VecSize; ++__i)
            __res.__v = __binary_op(__res.__v, __unary_op(__start_idx + __i, __acc...));
    }

    template <typename _Size, typename _Res, typename... _Acc>
    void
    vectorized_reduction_remainder(const _Size __start_idx, _Res& __res, const _Acc&... __acc) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (_Size __i = 0; __i < _VecSize; ++__i)
            __res.__v = __binary_op(__res.__v, __unary_op(__start_idx + __i, __acc...));
    }

    template <typename _Size, typename _Res, typename... _Acc>
    void
    scalar_reduction_remainder(const _Size __start_idx, const _Size __adjusted_n, _Res& __res,
                               const _Acc&... __acc) const
    {
        // The boundary checks are done in the caller, i.e., __start_idx <= __adjusted_n
        const _Size __no_iters = __adjusted_n - __start_idx;
        for (_Size __idx = 0; __idx < __no_iters; ++__idx)
            __res.__v = __binary_op(__res.__v, __unary_op(__start_idx + __idx, __acc...));
    }

    template <typename _NDItemId, typename _Size, typename _Res, typename... _Acc>
    void
    operator()(const _NDItemId& __item_id, const _Size& __n, const _Size& __iters_per_work_item,
               const _Size& __global_offset, const bool __is_full, const _Size __n_groups, _Res& __res,
               const _Acc&... __acc) const
    {
        const _Size __global_idx = __item_id.get_global_id(0);
        // Check if there is any work to do
        if (__global_idx >= __n)
            return;
        if (__iters_per_work_item == 1)
        {
            __res.__setup(__unary_op(__global_idx, __acc...));
            return;
        }
        const _Size __local_range = __item_id.get_local_range(0);
        const _Size __no_vec_ops = __iters_per_work_item / _VecSize;
        const _Size __adjusted_n = __global_offset + __n;
        constexpr _Size __vec_size_minus_one = _VecSize - 1;

        _Size __stride = _VecSize; // sequential loads with _VecSize-wide vectors
        _Size __adjusted_global_id = __global_offset;
        if constexpr (_Commutative{})
        {
            __stride *= __local_range; // coalesced loads with _VecSize-wide vectors
            _Size __local_idx = __item_id.get_local_id(0);
            _Size __group_idx = __item_id.get_group_linear_id();
            __adjusted_global_id += __group_idx * __local_range * __iters_per_work_item + __local_idx * _VecSize;
        }
        else
            __adjusted_global_id += __iters_per_work_item * __global_idx;

        // Groups are full if n is evenly divisible by the number of elements processed per work-group.
        // Multi group reductions will be full for all groups before the last group.
        _Size __group_idx = __item_id.get_group(0);
        _Size __n_groups_minus_one = __n_groups - 1;

        // _VecSize-wide vectorized path (__iters_per_work_item are multiples of _VecSize)
        if (__is_full || (__group_idx < __n_groups_minus_one))
        {
            vectorized_reduction_first(__adjusted_global_id, __res, __acc...);
            for (_Size __i = 1; __i < __no_vec_ops; ++__i)
                vectorized_reduction_remainder(__adjusted_global_id + __i * __stride, __res, __acc...);
        }
        // At least one vector operation
        else if (__adjusted_global_id + __vec_size_minus_one < __adjusted_n)
        {
            vectorized_reduction_first(__adjusted_global_id, __res, __acc...);
            if (__no_vec_ops > 1)
            {
                _Size __n_diff = __adjusted_n - __adjusted_global_id - _VecSize;
                _Size __no_iters = __n_diff / __stride;
                _Size __no_vec_ops_minus_one = __no_vec_ops - 1;
                bool __excess_scalar_elements = false;
                if (__no_iters >= __no_vec_ops_minus_one)
                {
                    // Completely full work item
                    __no_iters = __no_vec_ops_minus_one;
                    __excess_scalar_elements = false;
                }
                else
                {
                    // Partially full work item, but we need to consider if it's next iteration after its last
                    // vector instruction begins within the sequence
                    __excess_scalar_elements = __adjusted_global_id + (__no_iters + 1) * __stride < __adjusted_n;
                }
                _Size __base_idx = __adjusted_global_id + __stride;
                for (_Size __i = 1; __i <= __no_iters; ++__i)
                {
                    vectorized_reduction_remainder(__base_idx, __res, __acc...);
                    __base_idx += __stride;
                }
                if (__excess_scalar_elements)
                    scalar_reduction_remainder(__base_idx, __adjusted_n, __res, __acc...);
            }
        }
        // Scalar remainder
        else if (__adjusted_global_id < __adjusted_n)
        {
            __res.__setup(__unary_op(__adjusted_global_id, __acc...));
            const _Size __adjusted_global_id_plus_one = __adjusted_global_id + 1;
            scalar_reduction_remainder(__adjusted_global_id_plus_one, __adjusted_n, __res, __acc...);
        }
    }

    template <typename _Size>
    _Size
    output_size(const _Size __n, const _Size __work_group_size, const _Size __iters_per_work_item) const
    {
        if (__iters_per_work_item == 1)
            return __n;
        if constexpr (_Commutative{})
        {
            _Size __items_per_work_group = __work_group_size * __iters_per_work_item;
            _Size __full_group_contrib = (__n / __items_per_work_group) * __work_group_size;
            _Size __last_wg_remainder = __n % __items_per_work_group;
            // Adjust remainder and wg size for vector size
            _Size __last_wg_vec = oneapi::dpl::__internal::__dpl_ceiling_div(__last_wg_remainder, _VecSize);
            _Size __last_wg_contrib = std::min(__last_wg_vec, __work_group_size);
            return __full_group_contrib + __last_wg_contrib;
        }
        // else (if not commutative)
        return oneapi::dpl::__internal::__dpl_ceiling_div(__n, __iters_per_work_item);
    }
};

// Reduce local reductions of each work item to a single reduced element per work group. The local reductions are held
// in local memory. sycl::reduce_over_group is used for supported data types and operations. All other operations are
// processed in order and without a known identity.
template <typename _ExecutionPolicy, typename _BinaryOperation1, typename _Tp>
struct reduce_over_group
{
    _BinaryOperation1 __bin_op1;

    // Reduce on local memory with subgroups
    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    reduce_impl(const _NDItemId __item_id, const _Size __n, const _Tp& __val, const _AccLocal& /*__local_mem*/,
                std::true_type /*has_known_identity*/) const
    {
        const _Size __global_idx = __item_id.get_global_id(0);
        return __dpl_sycl::__reduce_over_group(
            __item_id.get_group(), __global_idx >= __n ? __known_identity<_BinaryOperation1, _Tp> : __val, __bin_op1);
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    reduce_impl(const _NDItemId __item_id, const _Size __n, const _Tp& __val, const _AccLocal& __local_mem,
                std::false_type /*has_known_identity*/) const
    {
        auto __local_idx = __item_id.get_local_id(0);
        const _Size __global_idx = __item_id.get_global_id(0);
        auto __group_size = __item_id.get_local_range().size();

        __local_mem[__local_idx] = __val;
        for (std::uint32_t __power_2 = 1; __power_2 < __group_size; __power_2 *= 2)
        {
            __dpl_sycl::__group_barrier(__item_id);
            if ((__local_idx & (2 * __power_2 - 1)) == 0 && __local_idx + __power_2 < __group_size &&
                __global_idx + __power_2 < __n)
            {
                __local_mem[__local_idx] = __bin_op1(__local_mem[__local_idx], __local_mem[__local_idx + __power_2]);
            }
        }
        return __local_mem[__local_idx];
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    operator()(const _NDItemId __item_id, const _Size __n, const _Tp& __val, const _AccLocal& __local_mem) const
    {
        return reduce_impl(__item_id, __n, __val, __local_mem, __has_known_identity<_BinaryOperation1, _Tp>{});
    }

    template <typename _InitType, typename _Result>
    void
    apply_init(const _InitType& __init, _Result&& __result) const
    {
        __init_processing<_Tp>{}(__init, __result, __bin_op1);
    }

    inline std::size_t
    local_mem_req(const std::uint16_t& __work_group_size) const
    {
        if constexpr (__has_known_identity<_BinaryOperation1, _Tp>{})
            return 0;

        return __work_group_size;
    }
};

// Matchers for early_exit_or and early_exit_find

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred_by_idx
{
    _Pred __pred;

    template <typename _Idx, typename _Acc>
    bool
    operator()(const _Idx __shifted_idx, _Acc& __acc) const
    {
        return __pred(__shifted_idx, __acc);
    }
};

template <typename _ExecutionPolicy, typename _Pred>
struct single_match_pred : single_match_pred_by_idx<_ExecutionPolicy, walk_n<_ExecutionPolicy, _Pred>>
{
    single_match_pred(_Pred __p) : single_match_pred_by_idx<_ExecutionPolicy, walk_n<_ExecutionPolicy, _Pred>>{__p} {}
};

template <typename _ExecutionPolicy, typename _Pred>
struct multiple_match_pred
{
    _Pred __pred;

    template <typename _Idx, typename _Acc1, typename _Acc2>
    bool
    operator()(const _Idx __shifted_idx, _Acc1& __acc, const _Acc2& __s_acc) const
    {
        // if __shifted_idx > __n - __s_n then subrange bigger than original range.
        // So the second range is not a subrange of the first range
        auto __n = __acc.size();
        auto __s_n = __s_acc.size();
        bool __result = __shifted_idx <= __n - __s_n;
        const auto __total_shift = __shifted_idx;

        using _Size2 = decltype(__s_n);
        // Moving __result out of the loop condition produces more optimized code
        if (__result)
        {
            for (_Size2 __ii = 0; __ii < __s_n; ++__ii)
            {
                __result = __pred(__acc[__total_shift + __ii], __s_acc[__ii]);
                if (!__result)
                    break;
            }
        }

        return __result;
    }
};

template <typename _ExecutionPolicy, typename _Pred, typename _Tp, typename _Size>
struct n_elem_match_pred
{
    _Pred __pred;
    _Tp __value;
    _Size __count;

    template <typename _Idx, typename _Acc>
    bool
    operator()(const _Idx __shifted_idx, const _Acc& __acc) const
    {

        bool __result = ((__shifted_idx + __count) <= __acc.size());
        const auto __total_shift = __shifted_idx;

        for (auto __idx = 0; __idx < __count && __result; ++__idx)
            __result = __pred(__acc[__total_shift + __idx], __value);

        return __result;
    }
};

template <typename _ExecutionPolicy, typename _Pred>
struct first_match_pred
{
    _Pred __pred;

    template <typename _Idx, typename _Acc1, typename _Acc2>
    bool
    operator()(const _Idx __shifted_idx, const _Acc1& __acc, const _Acc2& __s_acc) const
    {

        // assert: __shifted_idx < __n
        const auto __elem = __acc[__shifted_idx];
        auto __s_n = __s_acc.size();

        for (auto __idx = 0; __idx < __s_n; ++__idx)
            if (__pred(__elem, __s_acc[__idx]))
                return true;

        return false;
    }
};

//------------------------------------------------------------------------
// scan
//------------------------------------------------------------------------

// mask assigner for tuples
template <::std::size_t N>
struct __mask_assigner
{
    template <typename _Acc, typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(_Acc& __acc, _OutAcc&, const _OutIdx __out_idx, const _InAcc& __in_acc, const _InIdx __in_idx) const
    {
        using ::std::get;
        get<N>(__acc[__out_idx]) = __in_acc[__in_idx];
    }
};

// data assigners and accessors for transform_scan
struct __scan_assigner
{
    template <typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    std::enable_if_t<!std::is_pointer_v<_OutAcc>>
    operator()(_OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, _InIdx __in_idx) const
    {
        __out_acc[__out_idx] = __in_acc[__in_idx];
    }

    template <typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    std::enable_if_t<std::is_pointer_v<_OutAcc>>
    operator()(_OutAcc __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, _InIdx __in_idx) const
    {
        __out_acc[__out_idx] = __in_acc[__in_idx];
    }

    template <typename _Acc, typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(_Acc&, _OutAcc& __out_acc, const _OutIdx __out_idx, const _InAcc& __in_acc, _InIdx __in_idx) const
    {
        __out_acc[__out_idx] = __in_acc[__in_idx];
    }
};

struct __scan_no_assign
{
    template <typename _OutAcc, typename _OutIdx, typename _InAcc, typename _InIdx>
    void
    operator()(_OutAcc&, const _OutIdx, const _InAcc&, const _InIdx) const
    {
    }
};

// create mask
template <typename _Pred, typename _Tp>
struct __create_mask
{
    _Pred __pred;

    template <typename _Idx, typename _Input>
    _Tp
    operator()(const _Idx __idx, const _Input& __input) const
    {
        using ::std::get;
        // 1. apply __pred
        auto __temp = __pred(get<0>(__input[__idx]));
        // 2. initialize mask
        get<1>(__input[__idx]) = __temp;
        return _Tp(__temp);
    }
};

// functors for scan
template <typename _BinaryOp, typename _Assigner, typename _Inclusive, ::std::size_t N>
struct __copy_by_mask
{
    _BinaryOp __binary_op;
    _Assigner __assigner;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsPtr, typename _RetPtr, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, _OutAcc& __out_acc, const _InAcc& __in_acc, _WgSumsPtr* __wg_sums_ptr, _RetPtr* __ret_ptr,
               _Size __n, _SizePerWg __size_per_wg) const
    {
        using ::std::get;
        auto __item_idx = __item.get_linear_id();
        if (__item_idx < __n && get<N>(__in_acc[__item_idx]))
        {
            auto __out_idx = get<N>(__in_acc[__item_idx]) - 1;

            using __tuple_type =
                typename __internal::__get_tuple_type<::std::decay_t<decltype(get<0>(__in_acc[__item_idx]))>,
                                                      ::std::decay_t<decltype(__out_acc[__out_idx])>>::__type;

            // calculation of position for copy
            if (__item_idx >= __size_per_wg)
            {
                auto __wg_sums_idx = __item_idx / __size_per_wg - 1;
                __out_idx = __binary_op(__out_idx, __wg_sums_ptr[__wg_sums_idx]);
            }
            if (__item_idx % __size_per_wg == 0 || (get<N>(__in_acc[__item_idx]) != get<N>(__in_acc[__item_idx - 1])))
                // If we work with tuples we might have a situation when internal tuple is assigned to ::std::tuple
                // (e.g. returned by user-provided lambda).
                // For internal::tuple<T...> we have a conversion operator to ::std::tuple<T..>. The problem here
                // is that the types of these 2 tuples may be different but still convertible to each other.
                // Technically this should be solved by adding to internal::tuple<T..> an additional conversion
                // operator to ::std::tuple<U...>, but for some reason this doesn't work(conversion from
                // ::std::tuple<T...> to ::std::tuple<U..> fails). What does work is the explicit cast below:
                // for internal::tuple<T..> we define a field that provides a corresponding ::std::tuple<T..>
                // with matching types. We get this type(see __typle_type definition above) and use it
                // for static cast to explicitly convert internal::tuple<T..> -> ::std::tuple<T..>.
                // Now we have the following assignment ::std::tuple<U..> = ::std::tuple<T..> which works as expected.
                // NOTE: we only need this explicit conversion when we have internal::tuple and
                // ::std::tuple as operands, in all the other cases this is not necessary and no conversion
                // is performed(i.e. __typle_type is the same type as its operand).
                __assigner(static_cast<__tuple_type>(get<0>(__in_acc[__item_idx])), __out_acc[__out_idx]);
        }
        if (__item_idx == 0)
        {
            //copy final result to output
            *__ret_ptr = __wg_sums_ptr[(__n - 1) / __size_per_wg];
        }
    }
};

template <typename _BinaryOp, typename _Inclusive>
struct __partition_by_mask
{
    _BinaryOp __binary_op;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsPtr, typename _RetPtr, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, _OutAcc& __out_acc, const _InAcc& __in_acc, _WgSumsPtr* __wg_sums_ptr, _RetPtr* __ret_ptr,
               _Size __n, _SizePerWg __size_per_wg) const
    {
        auto __item_idx = __item.get_linear_id();
        if (__item_idx < __n)
        {
            using ::std::get;
            using __in_type = ::std::decay_t<decltype(get<0>(__in_acc[__item_idx]))>;
            auto __wg_sums_idx = __item_idx / __size_per_wg;
            bool __not_first_wg = __item_idx >= __size_per_wg;
            if (get<1>(__in_acc[__item_idx]) &&
                (__item_idx % __size_per_wg == 0 || get<1>(__in_acc[__item_idx]) != get<1>(__in_acc[__item_idx - 1])))
            {
                auto __out_idx = get<1>(__in_acc[__item_idx]) - 1;
                using __tuple_type = typename __internal::__get_tuple_type<
                    __in_type, ::std::decay_t<decltype(get<0>(__out_acc[__out_idx]))>>::__type;

                if (__not_first_wg)
                    __out_idx = __binary_op(__out_idx, __wg_sums_ptr[__wg_sums_idx - 1]);
                get<0>(__out_acc[__out_idx]) = static_cast<__tuple_type>(get<0>(__in_acc[__item_idx]));
            }
            else
            {
                auto __out_idx = __item_idx - get<1>(__in_acc[__item_idx]);
                using __tuple_type = typename __internal::__get_tuple_type<
                    __in_type, ::std::decay_t<decltype(get<1>(__out_acc[__out_idx]))>>::__type;

                if (__not_first_wg)
                    __out_idx -= __wg_sums_ptr[__wg_sums_idx - 1];
                get<1>(__out_acc[__out_idx]) = static_cast<__tuple_type>(get<0>(__in_acc[__item_idx]));
            }
        }
        if (__item_idx == 0)
        {
            //copy final result to output
            *__ret_ptr = __wg_sums_ptr[(__n - 1) / __size_per_wg];
        }
    }
};

template <typename _Inclusive, typename _BinaryOp, typename _InitType>
struct __global_scan_functor
{
    _BinaryOp __binary_op;
    _InitType __init;

    template <typename _Item, typename _OutAcc, typename _InAcc, typename _WgSumsPtr, typename _RetPtr, typename _Size,
              typename _SizePerWg>
    void
    operator()(_Item __item, _OutAcc& __out_acc, const _InAcc&, _WgSumsPtr* __wg_sums_ptr, _RetPtr*, _Size __n,
               _SizePerWg __size_per_wg) const
    {
        constexpr auto __shift = _Inclusive{} ? 0 : 1;
        auto __item_idx = __item.get_linear_id();
        // skip the first group scanned locally
        if (__item_idx >= __size_per_wg && __item_idx + __shift < __n)
        {
            auto __wg_sums_idx = __item_idx / __size_per_wg - 1;
            // an initial value precedes the first group for the exclusive scan
            __item_idx += __shift;
            auto __bin_op_result = __binary_op(__wg_sums_ptr[__wg_sums_idx], __out_acc[__item_idx]);
            using __out_type = ::std::decay_t<decltype(__out_acc[__item_idx])>;
            using __in_type = ::std::decay_t<decltype(__bin_op_result)>;
            __out_acc[__item_idx] =
                static_cast<typename __internal::__get_tuple_type<__in_type, __out_type>::__type>(__bin_op_result);
        }
        if constexpr (!_Inclusive::value)
            //store an initial value to the output first position should be done as postprocess (for in-place scanning)
            if (__item_idx == 0)
            {
                using _Tp = typename _InitType::__value_type;
                __init_processing<_Tp> __use_init{};
                __use_init(__init, __out_acc[__item_idx]);
            }
    }
};

template <typename _Inclusive, typename _ExecutionPolicy, typename _BinaryOperation, typename _UnaryOp,
          typename _WgAssigner, typename _GlobalAssigner, typename _DataAccessor, typename _InitType>
struct __scan
{
    using _Tp = typename _InitType::__value_type;
    _BinaryOperation __bin_op;
    _UnaryOp __unary_op;
    _WgAssigner __wg_assigner;
    _GlobalAssigner __gl_assigner;
    _DataAccessor __data_acc;

    template <typename _NDItemId, typename _Size, typename _AccLocal, typename _InAcc, typename _OutAcc,
              typename _WGSumsPtr, typename _SizePerWG, typename _WGSize, typename _ItersPerWG>
    void
    scan_impl(_NDItemId __item, _Size __n, _AccLocal& __local_acc, const _InAcc& __acc, _OutAcc& __out_acc,
              _WGSumsPtr* __wg_sums_ptr, _SizePerWG __size_per_wg, _WGSize __wgroup_size, _ItersPerWG __iters_per_wg,
              _InitType __init, std::false_type /*has_known_identity*/) const
    {
        ::std::size_t __group_id = __item.get_group(0);
        ::std::size_t __global_id = __item.get_global_id(0);
        ::std::size_t __local_id = __item.get_local_id(0);
        __init_processing<_Tp> __use_init{};

        constexpr ::std::size_t __shift = _Inclusive{} ? 0 : 1;

        ::std::size_t __adjusted_global_id = __local_id + __size_per_wg * __group_id;
        auto __adder = __local_acc[0];
        for (_ItersPerWG __iter = 0; __iter < __iters_per_wg; ++__iter, __adjusted_global_id += __wgroup_size)
        {
            if (__adjusted_global_id < __n)
            {
                // get input data
                __local_acc[__local_id] = __data_acc(__adjusted_global_id, __acc);
                // apply unary op
                __local_acc[__local_id] = __unary_op(__local_id, __local_acc);
            }
            if (__local_id == 0 && __iter > 0)
                __local_acc[0] = __bin_op(__adder, __local_acc[0]);
            else if (__global_id == 0)
                __use_init(__init, __local_acc[__global_id], __bin_op);

            // 1. reduce
            ::std::size_t __k = 1;
            // TODO: use adjacent work items for better SIMD utilization
            // Consider the example with the mask of work items performing reduction:
            // iter    now         proposed
            // 1:      01010101    11110000
            // 2:      00010001    11000000
            // 3:      00000001    10000000
            do
            {
                __dpl_sycl::__group_barrier(__item);

                if (__adjusted_global_id < __n && __local_id % (2 * __k) == 2 * __k - 1)
                {
                    __local_acc[__local_id] = __bin_op(__local_acc[__local_id - __k], __local_acc[__local_id]);
                }
                __k *= 2;
            } while (__k < __wgroup_size);
            __dpl_sycl::__group_barrier(__item);

            // 2. scan
            auto __partial_sums = __local_acc[__local_id];
            __k = 2;
            do
            {
                // use signed type to avoid overflowing
                ::std::int32_t __shifted_local_id = __local_id - __local_id % __k - 1;
                if (__shifted_local_id >= 0 && __adjusted_global_id < __n && __local_id % (2 * __k) >= __k &&
                    __local_id % (2 * __k) < 2 * __k - 1)
                {
                    __partial_sums = __bin_op(__local_acc[__shifted_local_id], __partial_sums);
                }
                __k *= 2;
            } while (__k < __wgroup_size);
            __dpl_sycl::__group_barrier(__item);

            __local_acc[__local_id] = __partial_sums;
            __dpl_sycl::__group_barrier(__item);
            __adder = __local_acc[__wgroup_size - 1];

            if (__adjusted_global_id + __shift < __n)
                __gl_assigner(__acc, __out_acc, __adjusted_global_id + __shift, __local_acc, __local_id);

            if (__adjusted_global_id == __n - 1)
                __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
        }

        if (__local_id == __wgroup_size - 1 && __adjusted_global_id - __wgroup_size < __n)
            __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal, typename _InAcc, typename _OutAcc,
              typename _WGSumsPtr, typename _SizePerWG, typename _WGSize, typename _ItersPerWG>
    void
    scan_impl(_NDItemId __item, _Size __n, _AccLocal& __local_acc, const _InAcc& __acc, _OutAcc& __out_acc,
              _WGSumsPtr* __wg_sums_ptr, _SizePerWG __size_per_wg, _WGSize __wgroup_size, _ItersPerWG __iters_per_wg,
              _InitType __init, std::true_type /*has_known_identity*/) const
    {
        auto __group_id = __item.get_group(0);
        auto __local_id = __item.get_local_id(0);
        auto __use_init = __init_processing<_Tp>{};

        constexpr auto __shift = _Inclusive{} ? 0 : 1;

        auto __adjusted_global_id = __local_id + __size_per_wg * __group_id;
        auto __adder = __local_acc[0];
        for (auto __iter = 0; __iter < __iters_per_wg; ++__iter, __adjusted_global_id += __wgroup_size)
        {
            if (__adjusted_global_id < __n)
                __local_acc[__local_id] = __data_acc(__adjusted_global_id, __acc);
            else
                __local_acc[__local_id] = _Tp{__known_identity<_BinaryOperation, _Tp>};

            // the result of __unary_op must be convertible to _Tp
            _Tp __old_value = __unary_op(__local_id, __local_acc);
            if (__iter > 0 && __local_id == 0)
                __old_value = __bin_op(__adder, __old_value);
            else if (__adjusted_global_id == 0)
                __use_init(__init, __old_value, __bin_op);

            __local_acc[__local_id] =
                __dpl_sycl::__inclusive_scan_over_group(__item.get_group(), __old_value, __bin_op);
            __dpl_sycl::__group_barrier(__item);

            __adder = __local_acc[__wgroup_size - 1];

            if (__adjusted_global_id + __shift < __n)
                __gl_assigner(__acc, __out_acc, __adjusted_global_id + __shift, __local_acc, __local_id);

            if (__adjusted_global_id == __n - 1)
                __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
        }

        if (__local_id == __wgroup_size - 1 && __adjusted_global_id - __wgroup_size < __n)
            __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal, typename _InAcc, typename _OutAcc,
              typename _WGSumsPtr, typename _SizePerWG, typename _WGSize, typename _ItersPerWG>
    void operator()(_NDItemId __item, _Size __n, _AccLocal& __local_acc, const _InAcc& __acc, _OutAcc& __out_acc,
                    _WGSumsPtr* __wg_sums_ptr, _SizePerWG __size_per_wg, _WGSize __wgroup_size,
                    _ItersPerWG __iters_per_wg,
                    _InitType __init = __no_init_value<typename _InitType::__value_type>{}) const
    {
        scan_impl(__item, __n, __local_acc, __acc, __out_acc, __wg_sums_ptr, __size_per_wg, __wgroup_size,
                  __iters_per_wg, __init, __has_known_identity<_BinaryOperation, _Tp>{});
    }
};

//------------------------------------------------------------------------
// __brick_includes
//------------------------------------------------------------------------

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2>
struct __brick_includes
{
    _Compare __comp;
    _Size1 __na;
    _Size2 __nb;

    __brick_includes(_Compare __c, _Size1 __n1, _Size2 __n2) : __comp(__c), __na(__n1), __nb(__n2) {}

    template <typename _ItemId, typename _Acc1, typename _Acc2>
    bool
    operator()(_ItemId __idx, const _Acc1& __b_acc, const _Acc2& __a_acc) const
    {
        using ::std::get;

        auto __a = __a_acc;
        auto __b = __b_acc;

        auto __a_beg = _Size1(0);
        auto __a_end = __na;

        auto __b_beg = _Size2(0);
        auto __b_end = __nb;

        // testing __comp(*__first2, *__first1) or __comp(*(__last1 - 1), *(__last2 - 1))
        if ((__idx == 0 && __comp(__b[__b_beg + 0], __a[__a_beg + 0])) ||
            (__idx == __nb - 1 && __comp(__a[__a_end - 1], __b[__b_end - 1])))
            return true; //__a doesn't include __b

        const auto __idx_b = __b_beg + __idx;
        const auto __val_b = __b[__idx_b];
        auto __res = __internal::__pstl_lower_bound(__a, __a_beg, __a_end, __val_b, __comp);

        // {a} < {b} or __val_b != __a[__res]
        if (__res == __a_end || __comp(__val_b, __a[__res]))
            return true; //__a doesn't include __b

        auto __val_a = __a[__res];

        //searching number of duplication
        const auto __count_a = __internal::__pstl_right_bound(__a, __res, __a_end, __val_a, __comp) - __res + __res -
                               __internal::__pstl_left_bound(__a, __a_beg, __res, __val_a, __comp);

        const auto __count_b = __internal::__pstl_right_bound(__b, _Size2(__idx_b), __b_end, __val_b, __comp) -
                               __idx_b + __idx_b -
                               __internal::__pstl_left_bound(__b, __b_beg, _Size2(__idx_b), __val_b, __comp);

        return __count_b > __count_a; //false means __a includes __b
    }
};

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------
template <typename _Size>
struct __reverse_functor
{
    _Size __size;
    template <typename _Idx, typename _Accessor>
    void
    operator()(const _Idx __idx, _Accessor& __acc) const
    {
        using ::std::swap;
        swap(__acc[__idx], __acc[__size - __idx - 1]);
    }
};

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __reverse_copy
{
    _Size __size;
    template <typename _Idx, typename _AccessorSrc, typename _AccessorDst>
    void
    operator()(const _Idx __idx, const _AccessorSrc& __acc1, _AccessorDst& __acc2) const
    {
        __acc2[__idx] = __acc1[__size - __idx - 1];
    }
};

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __rotate_copy
{
    _Size __size;
    _Size __shift;
    template <typename _Idx, typename _AccessorSrc, typename _AccessorDst>
    void
    operator()(const _Idx __idx, const _AccessorSrc& __acc1, _AccessorDst& __acc2) const
    {
        __acc2[__idx] = __acc1[(__shift + __idx) % __size];
    }
};

//------------------------------------------------------------------------
// brick_set_op for difference and intersection operations
//------------------------------------------------------------------------
struct _IntersectionTag : public ::std::false_type
{
};
struct _DifferenceTag : public ::std::true_type
{
};

template <typename _ExecutionPolicy, typename _Compare, typename _Size1, typename _Size2, typename _IsOpDifference>
class __brick_set_op
{
    _Compare __comp;
    _Size1 __na;
    _Size2 __nb;

  public:
    __brick_set_op(_Compare __c, _Size1 __n1, _Size2 __n2) : __comp(__c), __na(__n1), __nb(__n2) {}

    template <typename _ItemId, typename _Acc>
    bool
    operator()(_ItemId __idx, const _Acc& __inout_acc) const
    {
        using ::std::get;
        auto __a = get<0>(__inout_acc.tuple()); // first sequence
        auto __b = get<1>(__inout_acc.tuple()); // second sequence
        auto __c = get<2>(__inout_acc.tuple()); // mask buffer

        auto __a_beg = _Size1(0);
        auto __b_beg = _Size2(0);

        auto __idx_c = __idx;
        const auto __idx_a = __idx;
        auto __val_a = __a[__a_beg + __idx_a];

        auto __res = __internal::__pstl_lower_bound(__b, _Size2(0), __nb, __val_a, __comp);

        bool bres = _IsOpDifference(); //initialization in true in case of difference operation; false - intersection.
        if (__res == __nb || __comp(__val_a, __b[__b_beg + __res]))
        {
            // there is no __val_a in __b, so __b in the difference {__a}/{__b};
        }
        else
        {
            auto __val_b = __b[__b_beg + __res];

            //Difference operation logic: if number of duplication in __a on left side from __idx > total number of
            //duplication in __b than a mask is 1

            //Intersection operation logic: if number of duplication in __a on left side from __idx <= total number of
            //duplication in __b than a mask is 1

            const _Size1 __count_a_left =
                __idx_a - __internal::__pstl_left_bound(__a, _Size1(0), _Size1(__idx_a), __val_a, __comp) + 1;

            const _Size2 __count_b = __internal::__pstl_right_bound(__b, _Size2(__res), __nb, __val_b, __comp) - __res +
                                     __res -
                                     __internal::__pstl_left_bound(__b, _Size2(0), _Size2(__res), __val_b, __comp);

            if constexpr (_IsOpDifference::value)
                bres = __count_a_left > __count_b; /*difference*/
            else
                bres = __count_a_left <= __count_b; /*intersection*/
        }
        __c[__idx_c] = bres; //store a mask
        return bres;
    }
};

template <typename _ExecutionPolicy, typename _DiffType>
struct __brick_shift_left
{
    _DiffType __size;
    _DiffType __n;

    template <typename _ItemId, typename _Range>
    void
    operator()(const _ItemId __idx, _Range&& __rng) const
    {
        const _DiffType __i = __idx - __n; //loop invariant
        for (_DiffType __k = __n; __k < __size; __k += __n)
        {
            if (__k + __idx < __size)
                __rng[__k + __i] = ::std::move(__rng[__k + __idx]);
        }
    }
};

struct __brick_assign_key_position
{
    // __a is a tuple {i, (i-1)-th key, i-th key}
    // __b is a tuple {key, index} that stores the key and index where a new segment begins
    template <typename _T1, typename _T2>
    void
    operator()(const _T1& __a, _T2&& __b) const
    {
        ::std::get<0>(::std::forward<_T2>(__b)) = ::std::get<2>(__a); // store new key value
        ::std::get<1>(::std::forward<_T2>(__b)) = ::std::get<0>(__a); // store index of new key
    }
};

// reduce the values in a segment associated with a key
template <typename _BinaryOperator, typename _Size>
struct __brick_reduce_idx
{
    __brick_reduce_idx(const _BinaryOperator& __b, const _Size __n_) : __binary_op(__b), __n(__n_) {}

    template <typename _Idx, typename _Values>
    auto
    reduce(_Idx __segment_begin, _Idx __segment_end, const _Values& __values) const
    {
        using __ret_type = oneapi::dpl::__internal::__decay_with_tuple_specialization_t<decltype(__values[0])>;
        __ret_type __res = __values[__segment_begin];

        for (++__segment_begin; __segment_begin < __segment_end; ++__segment_begin)
            __res = __binary_op(__res, __values[__segment_begin]);
        return __res;
    }

    template <typename _ItemId, typename _ReduceIdx, typename _Values, typename _OutValues>
    void
    operator()(const _ItemId __idx, const _ReduceIdx& __segment_starts, const _Values& __values,
               _OutValues& __out_values) const
    {
        using __value_type = decltype(__segment_starts[__idx]);
        __value_type __segment_end =
            (__idx == __segment_starts.size() - 1) ? __value_type(__n) : __segment_starts[__idx + 1];
        __out_values[__idx] = reduce(__segment_starts[__idx], __segment_end, __values);
    }

  private:
    _BinaryOperator __binary_op;
    _Size __n;
};

} // namespace unseq_backend
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UNSEQ_BACKEND_SYCL_H
