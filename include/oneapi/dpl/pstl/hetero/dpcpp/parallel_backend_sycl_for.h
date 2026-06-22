// -*- C++ -*-
//===-- parallel_backend_sycl_for.h ---------------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_FOR_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_FOR_H

#include <algorithm>
#include <cstdint>
#include <cassert>
#include <type_traits>
#include <tuple>

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"
#include "unseq_backend_sycl.h"

#include "../../utils_ranges.h" // __min_size_calc
#include "utils_ranges_sycl.h"

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

template <typename... _Ranges>
class __pfor_params
{
    using _ValueTypes = std::tuple<oneapi::dpl::__internal::__value_t<_Ranges>...>;
    constexpr static std::uint8_t __min_type_size = oneapi::dpl::__internal::__min_nested_type_size<_ValueTypes>::value;
    // Empirically determined 'bytes-in-flight' to maximize bandwidth utilization
    constexpr static std::uint8_t __bytes_per_item = 16;
    // Maximum size supported by compilers to generate vector instructions
    constexpr static std::uint8_t __max_vector_size = 4;

  public:
    constexpr static bool __can_vectorize =
        (std::is_fundamental_v<oneapi::dpl::__internal::__value_t<_Ranges>> && ...) && __min_type_size < 4;
    // Vectorize for small types, so we generate 128-byte load / stores in a sub-group
    constexpr static std::uint8_t __vector_size =
        __can_vectorize ? oneapi::dpl::__internal::__dpl_ceiling_div(__max_vector_size, __min_type_size) : 1;
    constexpr static std::uint8_t __iters_per_item = __bytes_per_item / (__min_type_size * __vector_size);
};

struct __pfor_params_simple
{
    constexpr static bool __can_vectorize = false;
    constexpr static std::uint8_t __vector_size = 1;
    constexpr static std::uint8_t __iters_per_item = 1;
};

template <typename _Brick, typename... _Ranges>
class __iterations_per_item
{
    template <typename _F>
    static std::integral_constant<std::uint8_t, _F::__iters_per_item>
    test(int);

    template <typename>
    static std::integral_constant<std::uint8_t, __pfor_params<_Ranges...>::__iters_per_item>
    test(...);

  public:
    constexpr static std::uint8_t value = decltype(test<_Brick>(0))::value;
};
template <typename _Brick, typename... _Ranges>
inline constexpr std::uint8_t __iterations_per_item_v = __iterations_per_item<_Brick, _Ranges...>::value;

template <typename... Name>
class __parallel_for_small_kernel;

template <typename... Name>
class __parallel_for_large_kernel;

//------------------------------------------------------------------------
// parallel_for - async pattern
//------------------------------------------------------------------------

// Use the trick with incomplete type and partial specialization to deduce the kernel name
// as the parameter pack that can be empty (for unnamed kernels) or contain exactly one
// type (for explicitly specified name by the user)
template <typename _KernelName>
struct __parallel_for_small_submitter;

template <typename... _Name>
struct __parallel_for_small_submitter<__internal::__optional_kernel_name<_Name...>>
{
    template <typename _Fp, typename _Index, typename... _Ranges>
    __future<sycl::event>
    operator()(sycl::queue& __q, _Fp __brick, _Index __count, _Ranges&&... __rngs) const
    {
        assert(oneapi::dpl::__ranges::__min_size_calc{}(__rngs...) > 0);
        assert(__count > 0);

        _PRINT_INFO_IN_DEBUG_MODE(__q);
        auto __event = __q.submit([__rngs..., __brick, __count](sycl::handler& __cgh) {

            //get an access to data under SYCL buffer:
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);

            __cgh.parallel_for<_Name...>(sycl::range</*dim=*/1>(__count), [=](sycl::item</*dim=*/1> __item) {
                // Simple loop and no vectorization within the brick, to evenly spread work across compute units.
                const std::size_t __idx = __item.get_linear_id();
                __brick(std::true_type{}, __idx, __pfor_params_simple{}, __rngs...);
            });
        });

        return __future{std::move(__event)};
    }
};

template <typename _KernelName>
struct __parallel_for_large_submitter;

template <typename... _Name>
struct __parallel_for_large_submitter<__internal::__optional_kernel_name<_Name...>>
{
    // Limit the work-group size to 512 which has empirically yielded the best results across different architectures.
    static constexpr std::uint16_t __work_group_size_limit = 512;

    // SPIR-V compilation targets show best performance with a stride of the sub-group size.
    // Other compilation targets perform best with a work-group size stride. This utility can only be called from the
    // device.
    static inline std::tuple<std::size_t, std::size_t, bool>
    __stride_recommender(const sycl::nd_item<1>& __item, std::size_t __count, std::size_t __iters_per_work_item,
                         std::size_t __adj_elements_per_work_item, std::size_t __group_size)
    {
        std::uint32_t __item_local_id;
        if constexpr (oneapi::dpl::__internal::__is_spirv_target_v)
        {
            const __dpl_sycl::__sub_group __sub_group = __item.get_sub_group();
            __group_size = __sub_group.get_local_linear_range();
            __item_local_id = __sub_group.get_local_linear_id();
        }
        else
        {
            __item_local_id = __item.get_local_linear_id();
        }
        const std::size_t __group_start_idx =
            __iters_per_work_item * __adj_elements_per_work_item * (__item.get_global_linear_id() - __item_local_id);
        const bool __is_full_group =
            __group_start_idx + __iters_per_work_item * __adj_elements_per_work_item * __group_size <= __count;
        const std::size_t __work_item_start_idx = __group_start_idx + __adj_elements_per_work_item * __item_local_id;
        return std::tuple(__work_item_start_idx, __adj_elements_per_work_item * __group_size, __is_full_group);
    }

    // Once there is enough work to launch a group on each compute unit with our chosen __iters_per_item,
    // then we should start using this code path.
    static inline std::size_t
    __minimal_useful_size(const sycl::queue& __q, std::size_t __iters_per_work_item)
    {
        const std::size_t __work_group_size =
            oneapi::dpl::__internal::__max_work_group_size(__q, __work_group_size_limit);
        const std::uint32_t __max_cu = oneapi::dpl::__internal::__max_compute_units(__q);
        return __work_group_size * __iters_per_work_item * __max_cu;
    }

    template <typename _Fp, typename _Index, typename... _Ranges>
    __future<sycl::event>
    operator()(sycl::queue& __q, _Fp __brick, _Index __count, _Ranges&&... __rngs) const
    {
        using __params_t = __pfor_params<_Ranges...>;
        const std::size_t __work_group_size =
            oneapi::dpl::__internal::__max_work_group_size(__q, __work_group_size_limit);
        _PRINT_INFO_IN_DEBUG_MODE(__q);
        auto __event = __q.submit([__rngs..., __brick, __work_group_size, __count](sycl::handler& __cgh) {
            //get an access to data under SYCL buffer:
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
            constexpr std::uint8_t __iters_per_work_item = __iterations_per_item_v<_Fp, _Ranges...>;
            constexpr std::uint8_t __vector_size = __params_t::__vector_size;
            const std::size_t __num_groups = oneapi::dpl::__internal::__dpl_ceiling_div(
                __count, (__work_group_size * __vector_size * __iters_per_work_item));
            __cgh.parallel_for<_Name...>(
                sycl::nd_range(sycl::range<1>(__num_groups * __work_group_size), sycl::range<1>(__work_group_size)),
                [=](sycl::nd_item</*dim=*/1> __item) {
                    __params_t __params;
                    const auto [__idx, __stride, __is_full] =
                        __stride_recommender(__item, __count, __iters_per_work_item, __vector_size, __work_group_size);
                    __strided_loop<__iters_per_work_item> __execute_loop{static_cast<std::size_t>(__count)};
                    if (__is_full)
                    {
                        __execute_loop(std::true_type{}, __idx, __stride, __brick, __params, __rngs...);
                    }
                    else
                    {
                        __execute_loop(std::false_type{}, __idx, __stride, __brick, __params, __rngs...);
                    }
                });
        });

        return __future{std::move(__event)};
    }
};

//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.
template <typename _CustomName, typename _Fp, typename _Index, typename... _Ranges>
__future<sycl::event>
__parallel_for_impl(sycl::queue& __q, _Fp __brick, _Index __count, _Ranges&&... __rngs)
{
    assert(oneapi::dpl::__ranges::__min_size_calc{}(__rngs...) > 0);
    assert(__count > 0);

    using _ForKernelSmall =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__parallel_for_small_kernel<_CustomName>>;
    using _ForKernelLarge =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__parallel_for_large_kernel<_CustomName>>;

    using __small_submitter = __parallel_for_small_submitter<_ForKernelSmall>;
    using __large_submitter = __parallel_for_large_submitter<_ForKernelLarge>;
    constexpr std::uint8_t __iters_per_work_item = __iterations_per_item_v<_Fp, _Ranges...>;
    // Compile two kernels: one for small-to-medium inputs and a second for large. This avoids runtime checks within a
    // single kernel that worsen performance for small cases. If the number of iterations of the large submitter is 1,
    // then only compile the basic kernel as the two versions are effectively the same.
    if constexpr (__iters_per_work_item > 1 || __pfor_params<_Ranges...>::__vector_size > 1)
    {
        if (__count >= __large_submitter::__minimal_useful_size(__q, __iters_per_work_item))
        {
            return __large_submitter{}(__q, __brick, __count, std::forward<_Ranges>(__rngs)...);
        }
    }
    return __small_submitter{}(__q, __brick, __count, std::forward<_Ranges>(__rngs)...);
}

//General version of parallel_for, one additional parameter - __count of iterations of loop __cgh.parallel_for,
//for some algorithms happens that size of processing range is n, but amount of iterations is n/2.
template <typename _ExecutionPolicy, typename _Fp, typename _Index, typename... _Ranges>
__future<sycl::event>
__parallel_for(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Fp __brick, _Index __count,
               _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;
    sycl::queue __q_local = __exec.queue();
    return oneapi::dpl::__par_backend_hetero::__parallel_for_impl<_CustomName>(__q_local, __brick, __count,
                                                                               std::forward<_Ranges>(__rngs)...);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif
