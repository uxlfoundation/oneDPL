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
#include "utils_ranges_sycl.h"
#include "parallel_backend_sycl_utils.h"
#include "../../functional_impl.h" // for oneapi::dpl::identity

#define _ONEDPL_SYCL_KNOWN_IDENTITY_PRESENT                                                                            \
    (_ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT || _ONEDPL_LIBSYCL_KNOWN_IDENTITY_PRESENT)

namespace oneapi
{
namespace dpl
{
namespace unseq_backend
{

#if _ONEDPL_USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)
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
#    if _ONEDPL_SYCL_KNOWN_IDENTITY_PRESENT
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
#    else
    typename ::std::conjunction<
        ::std::is_arithmetic<_Tp>,
        ::std::disjunction<::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<_Tp>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, ::std::plus<void>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<_Tp>>,
                           ::std::is_same<::std::decay_t<_BinaryOp>, __dpl_sycl::__plus<void>>>>,
#    endif
    ::std::false_type>;     // This is for the case of __can_use_known_identity<_Tp>==false

#else //_ONEDPL_USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)

template <typename _BinaryOp, typename _Tp>
using __has_known_identity = std::false_type;

#endif //_ONEDPL_USE_GROUP_ALGOS && defined(SYCL_IMPLEMENTATION_INTEL)

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
#if _ONEDPL_SYCL_KNOWN_IDENTITY_PRESENT
    __dpl_sycl::__known_identity<_BinaryOp, _Tp>::value;
#else
    __known_identity_for_plus<_BinaryOp, _Tp>::value; //for plus only
#endif

template <typename _F>
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

// If read accessor returns temporary value then oneapi::dpl::identity returns lvalue reference to it.
// After temporary value destroying it will be a reference to an invalid object.
// So let's not call the functor in case of oneapi::dpl::identity
template <>
struct walk_n<oneapi::dpl::identity>
{
    oneapi::dpl::identity __f; // only needed for uniform initialization

    template <typename _ItemId, typename _Range>
    auto
    operator()(const _ItemId __idx, _Range&& __rng) const -> decltype(__rng[__idx])
    {
        return __rng[__idx];
    }
};

using __unchanged = walk_n<oneapi::dpl::identity>;

// walk_n_vectors_or_scalars
template <typename _F>
struct walk_n_vectors_or_scalars
{
  private:
    //'mutable' is to relax the requirements for a user functor/lambda type operator() may be non-const
    mutable _F __f;
    std::size_t __n;
    template <typename _IsFull, typename _Params, typename _InRng, typename _OutRng,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    __vector_impl(_IsFull __is_full, const std::size_t __idx, _Params, _InRng&& __in_rng, _OutRng&& __out_rng) const
    {
        using _InValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        _InValueType __in_rng_vector[_Params::__vector_size];
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<_F> __store_op{__f};
        // 1. Load input into a vector
        __vec_load(__is_full, __idx, __load_op, __in_rng, __in_rng_vector);
        // 2. Apply functor to vector and store into global memory
        __vec_store(__is_full, __idx, __store_op, __in_rng_vector, __out_rng);
    }
    template <typename _IsFull, typename _Params, typename _InRng1, typename _InRng2, typename _OutRng,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    __vector_impl(_IsFull __is_full, const std::size_t __idx, _Params, _InRng1&& __in_rng1, _InRng2&& __in_rng2,
                  _OutRng&& __out_rng) const
    {
        using _InValueType1 = oneapi::dpl::__internal::__value_t<_InRng1>;
        using _InValueType2 = oneapi::dpl::__internal::__value_t<_InRng2>;

        _InValueType1 __in_rng1_vector[_Params::__vector_size];
        _InValueType2 __in_rng2_vector[_Params::__vector_size];

        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<_F> __store_op{__f};

        // 1. Load inputs into vectors
        __vec_load(__is_full, __idx, __load_op, __in_rng1, __in_rng1_vector);
        __vec_load(__is_full, __idx, __load_op, __in_rng2, __in_rng2_vector);
        // 2. Apply binary functor to vector and store into global memory
        __vec_store(__is_full, __idx, __store_op, __in_rng1_vector, __in_rng2_vector, __out_rng);
    }

  public:
    walk_n_vectors_or_scalars(_F __f, std::size_t __n) : __f(std::move(__f)), __n(__n) {}

    template <typename _IsFull, typename _Params, typename... _Ranges,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull __is_full, const std::size_t __idx, _Params, _Ranges&&... __rngs) const
    {
        constexpr std::size_t __num_ranges = sizeof...(__rngs);
        static_assert(__num_ranges <= 3,
                      "walk_n_vectors_or_scalars only supports up to 3 range packs with vectorization enabled");
        if constexpr (__num_ranges == 1)
        {
            using oneapi::dpl::__par_backend_hetero::__vector_walk;
            __vector_walk<_Params::__vector_size>{__n}(__is_full, __idx, __f, std::forward<_Ranges>(__rngs)...);
        }
        else
        {
            __vector_impl(__is_full, __idx, _Params{}, std::forward<_Ranges>(__rngs)...);
        }
    }

    // _IsFull is ignored here. We assume that boundary checking has been already performed for this index.
    template <typename _IsFull, typename _Params, typename... _Ranges,
              std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Ranges&&... __rngs) const
    {
        __f(__rngs[__idx]...);
    }
};

//------------------------------------------------------------------------
// walk_adjacent_difference
//------------------------------------------------------------------------

template <typename _F>
struct walk_adjacent_difference
{
  private:
    _F __f;
    std::size_t __n;
    oneapi::dpl::__internal::__pstl_assign __assigner;

  public:
    walk_adjacent_difference(_F __f, std::size_t __n) : __f(std::move(__f)), __n(__n) {}

    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        // just copy an element if it is the first one
        if (__idx == 0)
            __assigner(__rng1[__idx], __rng2[__idx]);
        else
            __f(__rng1[__idx + (-1)], __rng1[__idx], __rng2[__idx]);
    }
    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull __is_full, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        _ValueType __rng1_vector[_Params::__vector_size + 1];
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<_F> __store_op{__f};
        // 1. Establish a vector of __preferred_vector_size + 1 where a scalar load is performed on the first element
        // followed by a vector load of the specified length.
        __assigner(__idx != 0 ? __rng1[__idx - 1] : __rng1[0], __rng1_vector[0]);
        __vec_load(__is_full, __idx, __load_op, __rng1, &__rng1_vector[1]);
        // 2. Perform a vector store of __preferred_vector_size adjacent differences.
        __vec_store(__is_full, __idx, __store_op, __rng1_vector, &__rng1_vector[1], __rng2);
        // A dummy value is first written to global memory followed by an overwrite for the first index. Pulling the vector loads / stores into an if branch
        // to better handle this results in performance degradation.
        if (__idx == 0)
            __assigner(__rng1_vector[0], __rng2[0]);
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
template <typename _Operation1, typename _Operation2, typename _Tp, typename _Commutative, std::uint8_t _VecSize>
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
    operator()(const _NDItemId& __item, const _Size& __n, const _Size& __iters_per_work_item,
               const _Size& __global_offset, const bool __is_full, const _Size __n_groups, _Res& __res,
               const _Acc&... __acc) const
    {
        const _Size __global_idx = __item.get_global_id(0);
        // Check if there is any work to do
        if (__global_idx >= __n)
            return;
        if (__iters_per_work_item == 1)
        {
            __res.__setup(__unary_op(__global_idx, __acc...));
            return;
        }
        const _Size __local_range = __item.get_local_range(0);
        const _Size __no_vec_ops = __iters_per_work_item / _VecSize;
        const _Size __adjusted_n = __global_offset + __n;
        constexpr _Size __vec_size_minus_one = _VecSize - 1;

        _Size __stride = _VecSize; // sequential loads with _VecSize-wide vectors
        _Size __adjusted_global_id = __global_offset;
        if constexpr (_Commutative{})
        {
            __stride *= __local_range; // coalesced loads with _VecSize-wide vectors
            _Size __local_idx = __item.get_local_id(0);
            _Size __group_idx = __item.get_group_linear_id();
            __adjusted_global_id += __group_idx * __local_range * __iters_per_work_item + __local_idx * _VecSize;
        }
        else
            __adjusted_global_id += __iters_per_work_item * __global_idx;

        // Groups are full if n is evenly divisible by the number of elements processed per work-group.
        // Multi group reductions will be full for all groups before the last group.
        _Size __group_idx = __item.get_group(0);
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
// processed in order and without a known identity. For the local-memory path, only work-item 0 returns the reduced
// value.
template <typename _BinaryOperation1, typename _Tp>
struct reduce_over_group
{
    _BinaryOperation1 __bin_op1;

    // Reduce on local memory with subgroups
    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    reduce_impl(const _NDItemId __item, const _Size __n, oneapi::dpl::__internal::__lazy_ctor_storage<_Tp>& __val,
                const _AccLocal& /*__local_mem*/, std::true_type /*has_known_identity*/) const
    {
        const _Size __global_idx = __item.get_global_id(0);
        return __dpl_sycl::__reduce_over_group(
            __item.get_group(), __global_idx >= __n ? __known_identity<_BinaryOperation1, _Tp> : __val.__v, __bin_op1);
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    reduce_impl(const _NDItemId __item, const _Size __n, oneapi::dpl::__internal::__lazy_ctor_storage<_Tp>& __val,
                const _AccLocal& __local_mem, std::false_type /*has_known_identity*/) const
    {
        auto __local_idx = __item.get_local_id(0);
        auto __group_size = __item.get_local_range().size();

        const _Size __global_idx = __item.get_global_id(0);
        if (__global_idx < __n)
            __local_mem[__local_idx] = __val.__v;

        const _Size __group_start = __item.get_group(0) * __group_size;
        _Size __active_count = __n > __group_start ? __n - __group_start : 0;
        if (__active_count > __group_size)
            __active_count = __group_size;

        for (std::uint32_t __power_2 = 1; __power_2 < __group_size; __power_2 *= 2)
        {
            sycl::group_barrier(__item.get_group());
            if ((__local_idx & (2 * __power_2 - 1)) == 0 && __local_idx + __power_2 < __active_count)
            {
                __local_mem[__local_idx] = __bin_op1(__local_mem[__local_idx], __local_mem[__local_idx + __power_2]);
            }
        }
        return __local_mem[0];
    }

    template <typename _NDItemId, typename _Size, typename _AccLocal>
    _Tp
    operator()(const _NDItemId __item, const _Size __n, oneapi::dpl::__internal::__lazy_ctor_storage<_Tp>& __val,
               const _AccLocal& __local_mem) const
    {
        return reduce_impl(__item, __n, __val, __local_mem, __has_known_identity<_BinaryOperation1, _Tp>{});
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

template <typename _Pred>
struct single_match_pred_by_idx
{
    _Pred __pred;

    template <typename _Idx, typename... _Acc>
    bool
    operator()(const _Idx __shifted_idx, const _Acc&... __acc) const
    {
        return __pred(__shifted_idx, __acc...);
    }
};

template <typename _Pred>
struct single_match_pred : single_match_pred_by_idx<walk_n<_Pred>>
{
    single_match_pred(_Pred __p) : single_match_pred_by_idx<walk_n<_Pred>>{__p} {}
};

template <typename _Pred>
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

template <typename _Pred, typename _Tp, typename _Size>
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

template <typename _Pred>
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
// __brick_includes
//------------------------------------------------------------------------

template <typename _SizeA, typename _SizeB, typename _Compare, typename _ProjA, typename _ProjB>
struct __brick_includes
{
    _SizeA __na;
    _SizeB __nb;
    _Compare __comp;
    _ProjA __projA;
    _ProjB __projB;

    __brick_includes(_SizeA __na, _SizeB __nb, _Compare __comp, _ProjA __projA, _ProjB __projB)
        : __na(__na), __nb(__nb), __comp(__comp), __projA(__projA), __projB(__projB) {}

    template <typename _ItemId, typename __RngA, typename __RngB>
    bool
    operator()(_ItemId __idx, const __RngA& __rngA, const __RngB& __rngB) const
    {
        using std::get;

        const _SizeA __a_beg = 0;
        const _SizeA __a_end = __na;

        const _SizeB __b_beg = 0;
        const _SizeB __b_end = __nb;

        // testing __comp(*__first2, *__first1) or __comp(*(__last1 - 1), *(__last2 - 1))
        if ((__idx == 0 && std::invoke(__comp, std::invoke(__projB, __rngB[__b_beg + 0]),
                                               std::invoke(__projA, __rngA[__a_beg + 0]))) ||
            (__idx == __nb - 1 && std::invoke(__comp, std::invoke(__projA, __rngA[__a_end - 1]),
                                                      std::invoke(__projB, __rngB[__b_end - 1]))))
            return true; //__rngA doesn't include __rngB

        const _SizeB __idx_b = __b_beg + __idx;

        const _SizeA __res =
            __internal::__pstl_lower_bound_idx(__rngA, __a_beg, __a_end, __rngB, __idx_b, __comp, __projA, __projB);

        // {a} < {b} or __rngB[__idx_b] != __rngA[__res]
        if (__res == __a_end ||
            std::invoke(__comp, std::invoke(__projB, __rngB[__idx_b]), std::invoke(__projA, __rngA[__res])))
            return true; //__rngA doesn't include __rngB

        //searching number of duplication
        const auto __count_a =
            __internal::__pstl_right_bound_idx(__rngA, __res, __a_end, __rngA, __res, __comp, __projA, __projA) -
            __internal::__pstl_left_bound_idx(__rngA, __a_beg, __res, __rngA, __res, __comp, __projA, __projA);

        const auto __count_b =
            __internal::__pstl_right_bound_idx(__rngB, __idx_b, __b_end, __rngB, __idx_b, __comp, __projB, __projB) -
            __internal::__pstl_left_bound_idx(__rngB, __b_beg, __idx_b, __rngB, __idx_b, __comp, __projB, __projB);

        return __count_b > __count_a; //false means __rngA includes __rngB
    }
};

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------
struct __reverse_functor
{
  private:
    const std::size_t __begin;
    const std::size_t __end;

  public:
    // Construction parameters: the number of elements to reverse, the index of the first element (0 by default)
    __reverse_functor(std::size_t __n, std::size_t __shift = 0) : __begin(__shift), __end(__n + __shift) {}

    template <typename _IsFull, typename _Params, typename _Range, std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range&& __rng) const
    {
        assert(__idx < (__end - __begin) / 2);

        using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;
        _ValueType __rng_left_vector[_Params::__vector_size];
        _ValueType __rng_right_vector[_Params::__vector_size];

        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__end};
        oneapi::dpl::__par_backend_hetero::__vector_reverse<_Params::__vector_size> __vec_reverse;
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__end - __idx};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<oneapi::dpl::__internal::__pstl_assign>
            __store_op;

        const std::size_t __left_start_idx = __begin + __idx;

        if constexpr (_IsFull::value == false)
        {
            if (__left_start_idx + _Params::__vector_size >= __end - __idx)
            {
                // The remaining data to reverse fits into a single vector
                __vec_load(std::false_type{}, __left_start_idx, __load_op, __rng, __rng_left_vector);
                __vec_reverse(std::false_type{}, __end - __idx - __left_start_idx, __rng_left_vector);
                __vec_store(std::false_type{}, __left_start_idx, __store_op, __rng_left_vector, __rng);
                return;
            }
        }

        // In the below implementation, _IsFull is ignored in favor of std::true_type{} in all cases.
        // This relaxation is due to the fact that in-place reverse iterates only over the first half of the buffer.
        // Since there is more than a single vector of data to reverse, there is no OOB accesses or race condition.
        // There may exist a single point of double processing between left and right vectors in the last work-item
        // which reverses middle elements.

        const std::size_t __right_start_idx = __end - __idx - _Params::__vector_size;

        // 1. Load two vectors that we want to swap: one from the left half of the buffer and one from the right.
        // Note that due to indices we have chosen, there will always be a full vector of elements to load.
        __vec_load(std::true_type{}, __left_start_idx, __load_op, __rng, __rng_left_vector);
        __vec_load(std::true_type{}, __right_start_idx, __load_op, __rng, __rng_right_vector);
        // 2. Reverse vectors in registers.
        __vec_reverse(std::true_type{}, _Params::__vector_size, __rng_left_vector);
        __vec_reverse(std::true_type{}, _Params::__vector_size, __rng_right_vector);
        // 3. Store the left-half vector to the corresponding right-half indices and vice versa
        __vec_store(std::true_type{}, __right_start_idx, __store_op, __rng_left_vector, __rng);
        __vec_store(std::true_type{}, __left_start_idx, __store_op, __rng_right_vector, __rng);
    }
    template <typename _IsFull, typename _Params, typename _Range, std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range&& __rng) const
    {
        assert(__idx < (__end - __begin) / 2);
        using std::swap;
        swap(__rng[__begin + __idx], __rng[__end - __idx - 1]);
    }
};

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __reverse_copy
{
  private:
    _Size __size;
    oneapi::dpl::__internal::__pstl_assign __assigner;

  public:
    __reverse_copy(_Size __size) : __size(__size) {}

    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        __rng2[__idx] = __rng1[__size - __idx - 1];
    }
    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull __is_full, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        const std::size_t __n = __size;
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<oneapi::dpl::__internal::__pstl_assign>
            __store_op;
        oneapi::dpl::__par_backend_hetero::__vector_reverse<_Params::__vector_size> __vec_reverse;
        const std::size_t __remaining_elements = __n - __idx;
        const std::uint8_t __elements_to_process =
            std::min(static_cast<std::size_t>(_Params::__vector_size), __remaining_elements);
        const std::size_t __output_start = __size - __idx - __elements_to_process;
        // 1. Load vector to reverse
        _ValueType __rng1_vector[_Params::__vector_size];
        __vec_load(__is_full, __idx, __load_op, __rng1, __rng1_vector);
        // 2. Reverse in registers
        __vec_reverse(__is_full, __elements_to_process, __rng1_vector);
        // 3. Flip the location of the vector in the output buffer
        if constexpr (_IsFull::value)
        {
            __vec_store(std::true_type{}, __output_start, __store_op, __rng1_vector, __rng2);
        }
        else
        {
            // The non-full case is processed manually here due to the translation of indices in the reverse operation.
            // The last few elements in the buffer are reversed into the beginning of the buffer. However,
            // __vector_store would believe that we always have a full vector length of elements due to the starting
            // index having greater than __preferred_vector_size elements until the end of the buffer.
            for (std::uint8_t __i = 0; __i < __elements_to_process; ++__i)
                __assigner(__rng1_vector[__i], __rng2[__output_start + __i]);
        }
    }
};

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------
template <typename _Size>
struct __rotate_copy
{
  private:
    _Size __size;
    _Size __shift;
    oneapi::dpl::__internal::__pstl_assign __assigner;

  public:
    __rotate_copy(_Size __size, _Size __shift) : __size(__size), __shift(__shift) {}

    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull __is_full, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        const std::size_t __n = __size;
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<oneapi::dpl::__internal::__pstl_assign>
            __store_op;
        const std::size_t __shifted_idx = __shift + __idx;
        const std::size_t __wrapped_idx = __shifted_idx % __size;
        _ValueType __rng1_vector[_Params::__vector_size];
        //1. Vectorize loads only if we know the wrap around point is beyond the current vector elements to process
        if (__wrapped_idx + _Params::__vector_size <= __n)
        {
            __vec_load(__is_full, __wrapped_idx, __load_op, __rng1, __rng1_vector);
        }
        else
        {
            // A single point of non-contiguity within the rotation operation. Manually process two loops here:
            // the first before the wraparound point and the second after.
            const std::size_t __remaining_elements = __n - __idx;
            const std::uint8_t __elements_to_process =
                std::min(std::size_t{_Params::__vector_size}, __remaining_elements);
            // __n - __wrapped_idx can safely fit into a uint8_t due to the condition check above.
            const std::uint8_t __loop1_elements =
                std::min(__elements_to_process, static_cast<std::uint8_t>(__n - __wrapped_idx));
            const std::uint8_t __loop2_elements = __elements_to_process - __loop1_elements;
            std::uint8_t __i = 0;
            for (__i = 0; __i < __loop1_elements; ++__i)
                __assigner(__rng1[__wrapped_idx + __i], __rng1_vector[__i]);
            for (std::uint8_t __j = 0; __j < __loop2_elements; ++__j)
                __assigner(__rng1[__j], __rng1_vector[__i + __j]);
        }
        // 2. Store the rotation
        __vec_store(__is_full, __idx, __store_op, __rng1_vector, __rng2);
    }
    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        __rng2[__idx] = __rng1[(__shift + __idx) % __size];
    }
};

//------------------------------------------------------------------------
// brick_set_op for difference and intersection operations
//------------------------------------------------------------------------

struct _IntersectionTag
{
};

struct _DifferenceTag
{
};

struct _UnionTag
{
};

struct _SymmetricDifferenceTag
{
};

template <typename _DiffType>
struct __brick_shift_left
{
    // Multiple iterations per item are manually processed in the brick with a nd-range strided approach.
    constexpr static std::uint8_t __iters_per_item = 1;

    _DiffType __size;
    _DiffType __n;

    template <typename _IsFull, typename _Params, typename _Range, std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range&& __rng) const
    {
        using _ValueType = oneapi::dpl::__internal::__value_t<_Range>;
        const std::size_t __unsigned_size = __size;
        const _DiffType __i = __idx - __n;
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__unsigned_size};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__unsigned_size};
        oneapi::dpl::__par_backend_hetero::__scalar_load_op __load_op;
        oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<oneapi::dpl::__internal::__pstl_assign>
            __store_op;
        for (_DiffType __k = __n; __k < __size; __k += __n)
        {
            const _DiffType __read_offset = __k + __idx;
            const _DiffType __write_offset = __k + __i;
            if constexpr (_IsFull::value)
            {
                if (__read_offset + _Params::__vector_size <= __size)
                {
                    _ValueType __rng_vector[_Params::__vector_size];
                    __vec_load(std::true_type{}, __read_offset, __load_op, __rng, __rng_vector);
                    __vec_store(std::true_type{}, __write_offset, __store_op, __rng_vector, __rng);
                }
                else if (__read_offset < __size)
                {
                    const std::size_t __num_remaining = __size - __read_offset;
                    for (_DiffType __j = 0; __j < __num_remaining; ++__j)
                        __rng[__write_offset + __j] = __rng[__read_offset + __j];
                }
            }
            else
            {
                // Some items within a sub-group may still have a full vector length to process even if _IsFull is
                // false by intentional design of __stride_recommender. While these are vectorizable, this will result
                // in branch divergence and masked execution of both vectorized and serial paths for all items in the
                // sub-group which may worsen performance. Instead, have each item in the sub-group process its work
                // serially.
                for (_DiffType __j = 0; __j < std::min(std::size_t{_Params::__vector_size}, __n - __idx); ++__j)
                    if (__read_offset + __j < __size)
                        __rng[__write_offset + __j] = __rng[__read_offset + __j];
            }
        }
    }

    template <typename _IsFull, typename _Params, typename _Range, std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range&& __rng) const
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

    template <typename _Values>
    auto
    reduce(std::size_t __segment_begin, std::size_t __segment_end, const _Values& __values) const
    {
        using __ret_type = oneapi::dpl::__internal::__decay_with_tuple_specialization_t<decltype(__values[0])>;
        __ret_type __res = __values[__segment_begin];

        for (++__segment_begin; __segment_begin < __segment_end; ++__segment_begin)
            __res = __binary_op(__res, __values[__segment_begin]);
        return __res;
    }
    template <typename _IsFull, typename _Params, typename _ReduceIdx, typename _Values, typename _OutValues>
    void
    operator()(_IsFull, std::size_t __idx, _Params, const _ReduceIdx& __segment_starts, const _Values& __values,
               _OutValues& __out_values) const
    {
        using __value_type = decltype(__segment_starts[__idx]);
        const std::size_t __end = __segment_starts.size();
        __value_type __segment_end = (__idx == __end - 1) ? __value_type(__n) : __segment_starts[__idx + 1];
        __out_values[__idx] = reduce(__segment_starts[__idx], __segment_end, __values);

        if constexpr (_Params::__vector_size > 1)
        {
            // repeat for adjacent elements
            std::size_t __rest = std::min<std::size_t>(_Params::__vector_size, __end - __idx) - 1;
            for (++__idx; __rest > 0; ++__idx, --__rest)
            {
                __segment_end = (__idx == __end - 1) ? __value_type(__n) : __segment_starts[__idx + 1];
                __out_values[__idx] = reduce(__segment_starts[__idx], __segment_end, __values);
            }
        }
    }

  private:
    _BinaryOperator __binary_op;
    _Size __n;
};

// std::swap_ranges is unique in that both sets of provided ranges will be modified. Due to this,
// we define a separate functor from walk_n_vectors_or_scalars with a customized vectorization path.
template <typename _F>
struct __brick_swap
{
  private:
    _F __f;
    std::size_t __n;

  public:
    __brick_swap(_F __f, std::size_t __n) : __f(std::move(__f)), __n(__n) {}

    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull __is_full, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        // Copies are used in the vector path of swap due to the restriction to fundamental types.
        using _ValueType = oneapi::dpl::__internal::__value_t<_Range1>;
        _ValueType __rng_vector[_Params::__vector_size];
        oneapi::dpl::__par_backend_hetero::__vector_load<_Params::__vector_size> __vec_load{__n};
        oneapi::dpl::__par_backend_hetero::__vector_store<_Params::__vector_size> __vec_store{__n};
        // 1. Load elements from __rng1.
        __vec_load(__is_full, __idx, oneapi::dpl::__par_backend_hetero::__scalar_load_op{}, __rng1, __rng_vector);
        // 2. Swap the __rng1 elements in the vector with __rng2 elements from global memory. Note the store operation
        // updates __rng_vector due to the swap functor.
        __vec_store(__is_full, __idx, oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<_F>{__f},
                    __rng_vector, __rng2);
        // 3. Store __rng2 elements in the vector into __rng1.
        __vec_store(
            __is_full, __idx,
            oneapi::dpl::__par_backend_hetero::__scalar_store_transform_op<oneapi::dpl::__internal::__pstl_assign>{},
            __rng_vector, __rng1);
    }

    template <typename _IsFull, typename _Params, typename _Range1, typename _Range2,
              std::enable_if_t<!_Params::__can_vectorize, int> = 0>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Range1&& __rng1, _Range2&& __rng2) const
    {
        __f(__rng1[__idx], __rng2[__idx]);
    }
};

} // namespace unseq_backend
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UNSEQ_BACKEND_SYCL_H
