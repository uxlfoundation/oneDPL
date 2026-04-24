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

// This file contains SYCL specific macros and abstractions
// to support different versions of SYCL and to simplify its interfaces
//
// Include this header instead of sycl.hpp throughout the project

#ifndef _ONEDPL_UTILS_HETERO_H
#define _ONEDPL_UTILS_HETERO_H

#include "../utils.h"

#include <algorithm>   // for std::max
#include <tuple>       // for std::apply
#include <type_traits> // for std::decay_t

namespace oneapi
{
namespace dpl
{
namespace __internal
{

template <typename _Compare, typename _ReduceValueType>
struct __pattern_minmax_element_reduce_fn
{
    _Compare __comp;

    _ReduceValueType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        using std::get;
        auto __chosen_for_min = __a;
        auto __chosen_for_max = __b;

        if (std::invoke(__comp, get<2>(__b), get<2>(__a)))
            __chosen_for_min = std::move(__b);
        if (std::invoke(__comp, get<3>(__b), get<3>(__a)))
            __chosen_for_max = std::move(__a);
        return _ReduceValueType{get<0>(__chosen_for_min), get<1>(__chosen_for_max), get<2>(__chosen_for_min),
                                get<3>(__chosen_for_max)};
    }
};

template <typename _ReduceValueType, typename _Compare>
struct __pattern_min_element_reduce_fn
{
    _Compare __comp;

    _ReduceValueType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        using std::get;
        // TODO: Consider removing the non-commutative operator for SPIR-V targets when we see improved performance with the
        // non-sequential load path in transform_reduce.
        if constexpr (oneapi::dpl::__internal::__is_spirv_target_v)
        {
            // This operator doesn't track the lowest found index in case of equal min. or max. values. Thus, this operator is
            // not commutative.
            if (std::invoke(__comp, get<1>(__b), get<1>(__a)))
            {
                return __b;
            }
            return __a;
        }
        else
        {
            // This operator keeps track of the lowest found index in case of equal min. or max. values. Thus, this operator is
            // commutative.
            bool _is_a_lt_b = std::invoke(__comp, get<1>(__a), get<1>(__b));
            bool _is_b_lt_a = std::invoke(__comp, get<1>(__b), get<1>(__a));

            if (_is_b_lt_a || (!_is_a_lt_b && get<0>(__b) < get<0>(__a)))
            {
                return __b;
            }
            return __a;
        }
    }
};

template <typename _ReduceValueType>
struct __pattern_minmax_element_transform_fn
{
    template <typename _TGroupIdx, typename _TAcc>
    _ReduceValueType
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return _ReduceValueType{__gidx, __gidx, __acc[__gidx], __acc[__gidx]};
    }
};

template <typename _Predicate>
struct __pattern_count_transform_fn
{
    _Predicate __predicate;

    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    template <typename _TGroupIdx, typename _TAcc>
    int
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return (__predicate(__acc[__gidx]) ? 1 : 0);
    }
};

template <typename _ReduceValueType>
struct __pattern_min_element_transform_fn
{
    template <typename _TGroupIdx, typename _TAcc>
    _ReduceValueType
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return _ReduceValueType{__gidx, __acc[__gidx]};
    }
};

struct __pos_operations
{
    // We should call this operation without any runtime condition checks to avoid deadlocks
    template <typename _NDGroup, typename _TupleOfSizes>
    static _TupleOfSizes
    reduce_max_pos_over_group_elementwise(const _NDGroup& __group, _TupleOfSizes __pos)
    {
        __for_each_field(__pos, [__group](auto& __field) {
            using _Value = std::decay_t<decltype(__field)>;
            __field = __dpl_sycl::__reduce_over_group(__group, __field, __dpl_sycl::__maximum<_Value>());
        });
        
        return __pos;
    }

    template <typename _Tuple>
    static void
    fetch_min_pos_local_elementwise(_Tuple& __min_pos, const _Tuple& __pos)
    {
        __for_each_pair_of_fields(__min_pos, __pos, [](auto& __min_pos_field, const auto& __pos_field) {
            __min_pos_field = std::min(__min_pos_field, __pos_field);
        });
    }

    template <typename _Tuple>
    static void
    fetch_max_pos_local_elementwise(_Tuple& __max_pos, const _Tuple& __pos)
    {
        __for_each_pair_of_fields(__max_pos, __pos, [](auto& __max_pos_field, const auto& __pos_field) {
            __max_pos_field = std::max(__max_pos_field, __pos_field);
        });
    }

    // Precondition: __global_max_pos must refer to device global memory (e.g., a USM global allocation
    // or a SYCL buffer accessor with global_space). Passing a stack variable is undefined behavior
    // because sycl::atomic_ref requires global address space.
    template <typename _Tuple>
    static void
    fetch_max_pos_global_elementwise(_Tuple& __global_max_pos, const _Tuple& __pos)
    {
        // memory_order::relaxed is sufficient here because:
        //   - the atomic fetch_max itself is the only operation that must be race-free;
        //   - no other memory (output data, SLM, etc.) is being published through this atomic,
        //     so no acquire/release ordering is needed between work-groups;
        //   - the host reads the result only after the kernel completes, and kernel completion
        //     provides a full device-to-host memory barrier unconditionally.
        //
        // memory_scope::device is required because work item 0 of each work-group writes to the
        // same global location concurrently with work item 0 of every other work-group,
        // so the atomic must be visible across the entire device, not just within one work-group.

        __for_each_pair_of_fields(
            __global_max_pos, __pos, [](auto& __global_max_pos_field, const auto& __pos_field) {
                using _Value = std::decay_t<decltype(__global_max_pos_field)>;
                using _AtomicValueT = sycl::atomic_ref<_Value, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                       sycl::access::address_space::global_space>;

                _AtomicValueT __atomic(__global_max_pos_field);
                __atomic.fetch_max(__pos_field);
            });
    }

  protected:

    template <typename _Tuple, typename _F>
    static void
    __for_each_field(_Tuple& __tuple, _F&& __f)
    {
        std::apply([&](auto&... __fields) { (..., __f(__fields)); }, __tuple);
    }

    template <typename _Tuple1, typename _Tuple2, typename _F>
    static void
    __for_each_pair_of_fields(_Tuple1& __tuple1, const _Tuple2& __tuple2, _F&& __f)
    {
        std::apply(
            [&](auto&... __fields1) {
                std::apply([&](const auto&... __fields2) { (..., __f(__fields1, __fields2)); }, __tuple2);
            },
            __tuple1);
    }
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_UTILS_HETERO_H
