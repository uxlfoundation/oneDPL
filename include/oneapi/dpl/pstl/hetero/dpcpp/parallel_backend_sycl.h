// -*- C++ -*-
//===-- parallel_backend_sycl.h -------------------------------------------===//
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

// This header guard is used to check inclusion of DPC++ backend.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_H

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <functional>
#include <utility>
#include <cmath>
#include <limits>
#include <cstdint>

#include "../../iterator_impl.h"
#include "../../execution_impl.h"
#include "../../utils_ranges.h"
#include "../../ranges_defs.h"

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "parallel_backend_sycl_for.h"
#include "parallel_backend_sycl_reduce.h"
#include "parallel_backend_sycl_merge.h"
#include "parallel_backend_sycl_merge_sort.h"
#include "parallel_backend_sycl_reduce_by_segment.h"
#include "parallel_backend_sycl_reduce_then_scan.h"
#include "execution_sycl_defs.h"
#include "sycl_iterator.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#define _ONEDPL_USE_RADIX_SORT (_ONEDPL_USE_SUB_GROUPS && _ONEDPL_USE_GROUP_ALGOS)

#if _ONEDPL_USE_RADIX_SORT
#    include "parallel_backend_sycl_radix_sort.h"
#endif

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

//-----------------------------------------------------------------------------
//- iter_mode_resolver
//-----------------------------------------------------------------------------

// iter_mode_resolver resolves the situations when
// the access mode provided by a user differs (inMode) from
// the access mode required by an algorithm (outMode).
// In general case iter_mode_resolver accepts the only situations
// when inMode == outMode,
// whereas the template specializations describe cases with specific
// inMode and outMode and the preferred access mode between the two.
template <access_mode inMode, access_mode outMode>
struct iter_mode_resolver
{
    static_assert(inMode == outMode, "Access mode provided by user conflicts with the one required by the algorithm");
    static constexpr access_mode value = inMode;
};

template <>
struct iter_mode_resolver<access_mode::read, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::read;
};

template <>
struct iter_mode_resolver<access_mode::write, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::read_write, access_mode::read>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::read;
};

template <>
struct iter_mode_resolver<access_mode::read_write, access_mode::write>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::discard_write, access_mode::write>
{
    static constexpr access_mode value = access_mode::discard_write;
};

template <>
struct iter_mode_resolver<access_mode::discard_read_write, access_mode::write>
{
    //TODO: warn user that the access mode is changed
    static constexpr access_mode value = access_mode::write;
};

template <>
struct iter_mode_resolver<access_mode::discard_read_write, access_mode::read_write>
{
    static constexpr access_mode value = access_mode::discard_read_write;
};

//-----------------------------------------------------------------------------
//- iter_mode
//-----------------------------------------------------------------------------

// create iterator with different access mode
template <access_mode outMode>
struct iter_mode
{
    // for common heterogeneous iterator
    template <template <access_mode, typename...> class Iter, access_mode inMode, typename... Types>
    Iter<iter_mode_resolver<inMode, outMode>::value, Types...>
    operator()(const Iter<inMode, Types...>& it)
    {
        constexpr access_mode preferredMode = iter_mode_resolver<inMode, outMode>::value;
        if (inMode == preferredMode)
            return it;
        return Iter<preferredMode, Types...>(it);
    }
    // for ounting_iterator
    template <typename T>
    oneapi::dpl::counting_iterator<T>
    operator()(const oneapi::dpl::counting_iterator<T>& it)
    {
        return it;
    }
    // for zip_iterator
    template <typename... Iters>
    auto
    operator()(const oneapi::dpl::zip_iterator<Iters...>& it)
        -> decltype(oneapi::dpl::__internal::map_zip(*this, it.base()))
    {
        return oneapi::dpl::__internal::map_zip(*this, it.base());
    }
    // for common iterator
    template <typename Iter>
    Iter
    operator()(const Iter& it1)
    {
        return it1;
    }
    // for raw pointers
    template <typename T>
    T*
    operator()(T* ptr)
    {
        // it does not have any iter mode because of two factors:
        //   - since it is a raw pointer, kernel can read/write despite of access_mode
        //   - access_mode also serves for implicit synchronization for buffers to build graph dependency
        //     and since usm have only explicit synchronization and does not provide dependency resolution mechanism
        //     it does not require access_mode
        return ptr;
    }

    template <typename T>
    const T*
    operator()(const T* ptr)
    {
        return ptr;
    }
};

template <access_mode outMode, typename _Iterator>
auto
make_iter_mode(const _Iterator& __it) -> decltype(iter_mode<outMode>()(__it))
{
    return iter_mode<outMode>()(__it);
}

// set of class templates to name kernels

template <typename... _Name>
class __scan_local_kernel;

template <typename... _Name>
class __scan_group_kernel;

template <typename... _Name>
class __find_or_kernel_one_wg;

template <typename... _Name>
class __find_or_kernel;

template <typename... _Name>
class __scan_propagate_kernel;

template <typename... _Name>
class __scan_single_wg_kernel;

template <typename... _Name>
class __scan_single_wg_dynamic_kernel;

template <typename... Name>
class __scan_copy_single_wg_kernel;

//------------------------------------------------------------------------
// parallel_transform_scan - async pattern
//------------------------------------------------------------------------

// Please see the comment above __parallel_for_small_submitter for optional kernel name explanation
template <typename _CustomName, typename _PropagateScanName>
struct __parallel_scan_submitter;

// Even if this class submits three kernel optional name is allowed to be only for one of them
// because for two others we have to provide the name to get the reliable work group size
template <typename _CustomName, typename... _PropagateScanName>
struct __parallel_scan_submitter<_CustomName, __internal::__optional_kernel_name<_PropagateScanName...>>
{
    template <typename _Range1, typename _Range2, typename _InitType, typename _LocalScan, typename _GroupScan,
              typename _GlobalScan>
    __future<sycl::event, __result_and_scratch_storage<typename _InitType::__value_type>>
    operator()(sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _InitType __init, _LocalScan __local_scan,
               _GroupScan __group_scan, _GlobalScan __global_scan) const
    {
        using _Type = typename _InitType::__value_type;
        using _LocalScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __scan_local_kernel, _CustomName, _Range1, _Range2, _Type, _LocalScan, _GroupScan, _GlobalScan>;
        using _GroupScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_generator<
            __scan_group_kernel, _CustomName, _Range1, _Range2, _Type, _LocalScan, _GroupScan, _GlobalScan>;
        auto __n = __rng1.size();
        assert(__n > 0);

        auto __max_cu = oneapi::dpl::__internal::__max_compute_units(__q);
        // get the work group size adjusted to the local memory limit
        // TODO: find a way to generalize getting of reliable work-group sizes
        std::size_t __wgroup_size = oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q, sizeof(_Type));
        // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
        // This value matches the current practical limit for GPUs, but may need to be re-evaluated in the future.
        __wgroup_size = std::min(__wgroup_size, (std::size_t)1024);

#if _ONEDPL_COMPILE_KERNEL
        //Actually there is one kernel_bundle for the all kernels of the pattern.
        auto __kernels = __internal::__kernel_compiler<_LocalScanKernel, _GroupScanKernel>::__compile(__q);
        auto __kernel_1 = __kernels[0];
        auto __kernel_2 = __kernels[1];
        auto __wgroup_size_kernel_1 = oneapi::dpl::__internal::__kernel_work_group_size(__q, __kernel_1);
        auto __wgroup_size_kernel_2 = oneapi::dpl::__internal::__kernel_work_group_size(__q, __kernel_2);
        __wgroup_size = ::std::min({__wgroup_size, __wgroup_size_kernel_1, __wgroup_size_kernel_2});
#endif

        // Practically this is the better value that was found
        constexpr decltype(__wgroup_size) __iters_per_witem = 16;
        auto __size_per_wg = __iters_per_witem * __wgroup_size;
        auto __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __size_per_wg);
        // Storage for the results of scan for each workgroup

        using __result_and_scratch_storage_t = __result_and_scratch_storage<_Type>;
        __result_and_scratch_storage_t __result_and_scratch{__q, __n_groups + 1};

        _PRINT_INFO_IN_DEBUG_MODE(__q, __wgroup_size, __max_cu);

        // 1. Local scan on each workgroup
        auto __submit_event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
            auto __temp_acc = __result_and_scratch.template __get_scratch_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});
            __dpl_sycl::__local_accessor<_Type> __local_acc(__wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
            __cgh.use_kernel_bundle(__kernel_1.get_kernel_bundle());
#endif
            __cgh.parallel_for<_LocalScanKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
                __kernel_1,
#endif
                sycl::nd_range<1>(__n_groups * __wgroup_size, __wgroup_size), [=](sycl::nd_item<1> __item) {
                    auto __temp_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                    __local_scan(__item, __n, __local_acc, __rng1, __rng2, __temp_ptr, __size_per_wg, __wgroup_size,
                                 __iters_per_witem, __init);
                });
        });
        // 2. Scan for the entire group of values scanned from each workgroup (runs on a single workgroup)
        if (__n_groups > 1)
        {
            auto __iters_per_single_wg = oneapi::dpl::__internal::__dpl_ceiling_div(__n_groups, __wgroup_size);
            __submit_event = __q.submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__submit_event);
                auto __temp_acc = __result_and_scratch.template __get_scratch_acc<sycl::access_mode::read_write>(__cgh);
                __dpl_sycl::__local_accessor<_Type> __local_acc(__wgroup_size, __cgh);
#if _ONEDPL_COMPILE_KERNEL && _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
                __cgh.use_kernel_bundle(__kernel_2.get_kernel_bundle());
#endif
                __cgh.parallel_for<_GroupScanKernel>(
#if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT && _ONEDPL_LIBSYCL_PROGRAM_PRESENT
                    __kernel_2,
#endif
                    // TODO: try to balance work between several workgroups instead of one
                    sycl::nd_range<1>(__wgroup_size, __wgroup_size), [=](sycl::nd_item<1> __item) {
                        auto __temp_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                        __group_scan(__item, __n_groups, __local_acc, __temp_ptr, __temp_ptr,
                                     /*dummy*/ __temp_ptr, __n_groups, __wgroup_size, __iters_per_single_wg);
                    });
            });
        }

        // 3. Final scan for whole range
        auto __final_event = __q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__submit_event);
            oneapi::dpl::__ranges::__require_access(__cgh, __rng1, __rng2); //get an access to data under SYCL buffer
            auto __temp_acc = __result_and_scratch.template __get_scratch_acc<sycl::access_mode::read>(__cgh);
            auto __res_acc = __result_and_scratch.template __get_result_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});
            __cgh.parallel_for<_PropagateScanName...>(sycl::range<1>(__n_groups * __size_per_wg), [=](auto __item) {
                auto __temp_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                auto __res_ptr =
                    __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__res_acc, __n_groups + 1);
                __global_scan(__item, __rng2, __rng1, __temp_ptr, __res_ptr, __n, __size_per_wg);
            });
        });

        return __future{std::move(__final_event), std::move(__result_and_scratch)};
    }
};

template <typename _ValueType, bool _Inclusive, typename _Group, typename _Begin, typename _End, typename _OutIt,
          typename _BinaryOperation>
void
__scan_work_group(const _Group& __group, _Begin __begin, _End __end, _OutIt __out_it, _BinaryOperation __bin_op,
                  unseq_backend::__no_init_value<_ValueType>)
{
    if constexpr (_Inclusive)
        __dpl_sycl::__joint_inclusive_scan(__group, __begin, __end, __out_it, __bin_op);
    else
        __dpl_sycl::__joint_exclusive_scan(__group, __begin, __end, __out_it, __bin_op);
}

template <typename _ValueType, bool _Inclusive, typename _Group, typename _Begin, typename _End, typename _OutIt,
          typename _BinaryOperation>
void
__scan_work_group(const _Group& __group, _Begin __begin, _End __end, _OutIt __out_it, _BinaryOperation __bin_op,
                  unseq_backend::__init_value<_ValueType> __init)
{
    if constexpr (_Inclusive)
        __dpl_sycl::__joint_inclusive_scan(__group, __begin, __end, __out_it, __bin_op, __init.__value);
    else
        __dpl_sycl::__joint_exclusive_scan(__group, __begin, __end, __out_it, __init.__value, __bin_op);
}

template <bool _Inclusive, typename _KernelName>
struct __parallel_transform_scan_dynamic_single_group_submitter;

template <bool _Inclusive, typename... _ScanKernelName>
struct __parallel_transform_scan_dynamic_single_group_submitter<_Inclusive,
                                                                __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation, typename _UnaryOp>
    sycl::event
    operator()(sycl::queue& __q, _InRng&& __in_rng, _OutRng&& __out_rng, std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op, ::std::uint16_t __wg_size)
    {
        using _ValueType = typename _InitType::__value_type;

        const ::std::uint16_t __elems_per_item = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wg_size);
        const ::std::uint16_t __elems_per_wg = __elems_per_item * __wg_size;

        return __q.submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg}, __hdl);
            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(__wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                    const auto& __group = __self_item.get_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += __wg_size)
                    {
                        __lacc[__idx] = __unary_op(__in_rng[__idx]);
                    }

                    auto __ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    __scan_work_group<_ValueType, _Inclusive>(__group, __ptr, __ptr + __n, __ptr, __bin_op, __init);

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += __wg_size)
                    {
                        __out_rng[__idx] = __lacc[__idx];
                    }

                    const ::std::uint16_t __residual = __n % __wg_size;
                    const ::std::uint16_t __residual_start = __n - __residual;
                    if (__residual > 0 && __item_id < __residual)
                    {
                        auto __idx = __residual_start + __item_id;
                        __out_rng[__idx] = __lacc[__idx];
                    }
                });
        });
    }
};

template <bool _Inclusive, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename _KernelName>
struct __parallel_transform_scan_static_single_group_submitter;

template <bool _Inclusive, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename... _ScanKernelName>
struct __parallel_transform_scan_static_single_group_submitter<_Inclusive, _ElemsPerItem, _WGSize, _IsFullGroup,
                                                               __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation, typename _UnaryOp>
    sycl::event
    operator()(sycl::queue& __q, _InRng&& __in_rng, _OutRng&& __out_rng, std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op)
    {
        using _ValueType = typename _InitType::__value_type;

        constexpr ::uint32_t __elems_per_wg = _ElemsPerItem * _WGSize;

        return __q.submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg}, __hdl);

            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(_WGSize, _WGSize), [=](sycl::nd_item<1> __self_item) {
                    const auto& __group = __self_item.get_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();

                    auto __lacc_ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    for (std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                    {
                        __lacc[__idx] = __unary_op(__in_rng[__idx]);
                    }

                    __scan_work_group<_ValueType, _Inclusive>(__group, __lacc_ptr, __lacc_ptr + __n,
                                                              __lacc_ptr, __bin_op, __init);

                    for (std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                    {
                        __out_rng[__idx] = __lacc[__idx];
                    }

                    const std::uint16_t __residual = __n % _WGSize;
                    const std::uint16_t __residual_start = __n - __residual;
                    if (__item_id < __residual)
                    {
                        auto __idx = __residual_start + __item_id;
                        __out_rng[__idx] = __lacc[__idx];
                    }
                });
        });
    }
};

template <typename _Size, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename _KernelName>
struct __parallel_copy_if_static_single_group_submitter;

template <typename _Size, ::std::uint16_t _ElemsPerItem, ::std::uint16_t _WGSize, bool _IsFullGroup,
          typename... _ScanKernelName>
struct __parallel_copy_if_static_single_group_submitter<_Size, _ElemsPerItem, _WGSize, _IsFullGroup,
                                                        __internal::__optional_kernel_name<_ScanKernelName...>>
{
    template <typename _InRng, typename _OutRng, typename _InitType, typename _BinaryOperation, typename _UnaryOp,
              typename _Assign>
    __future<sycl::event, __result_and_scratch_storage<_Size>>
    operator()(sycl::queue& __q, _InRng&& __in_rng, _OutRng&& __out_rng, std::size_t __n, _InitType __init,
               _BinaryOperation __bin_op, _UnaryOp __unary_op, _Assign __assign)
    {
        using _ValueType = ::std::uint16_t;

        // This type is used as a workaround for when an internal tuple is assigned to ::std::tuple, such as
        // with zip_iterator
        using __tuple_type =
            typename ::oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__in_rng[0])>,
                                                                 std::decay_t<decltype(__out_rng[0])>>::__type;

        constexpr ::std::uint32_t __elems_per_wg = _ElemsPerItem * _WGSize;
        using __result_and_scratch_storage_t = __result_and_scratch_storage<_Size>;
        __result_and_scratch_storage_t __result{__q, 0};

        auto __event = __q.submit([&](sycl::handler& __hdl) {
            oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);

            // Local memory is split into two parts. The first half stores the result of applying the
            // predicate on each element of the input range. The second half stores the index of the output
            // range to copy elements of the input range.
            auto __lacc = __dpl_sycl::__local_accessor<_ValueType>(sycl::range<1>{__elems_per_wg * 2}, __hdl);
            auto __res_acc =
                __result.template __get_result_acc<sycl::access_mode::write>(__hdl, __dpl_sycl::__no_init{});

            __hdl.parallel_for<_ScanKernelName...>(
                sycl::nd_range<1>(_WGSize, _WGSize), [=](sycl::nd_item<1> __self_item) {
                    auto __res_ptr = __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__res_acc);
                    const auto& __group = __self_item.get_group();
                    // This kernel is only launched for sizes less than 2^16
                    const ::std::uint16_t __item_id = __self_item.get_local_linear_id();
                    auto __lacc_ptr = __dpl_sycl::__get_accessor_ptr(__lacc);
                    for (std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                    {
                        __lacc[__idx] = __unary_op(__in_rng[__idx]);
                    }

                    __scan_work_group<_ValueType, /* _Inclusive */ false>(
                        __group, __lacc_ptr, __lacc_ptr + __elems_per_wg, __lacc_ptr + __elems_per_wg, __bin_op,
                         __init);

                    for (::std::uint16_t __idx = __item_id; __idx < __n; __idx += _WGSize)
                    {
                        if (__lacc[__idx])
                            __assign(static_cast<__tuple_type>(__in_rng[__idx]),
                                     __out_rng[__lacc[__idx + __elems_per_wg]]);
                    }

                    const ::std::uint16_t __residual = __n % _WGSize;
                    const ::std::uint16_t __residual_start = __n - __residual;
                    if (__item_id < __residual)
                    {
                        auto __idx = __residual_start + __item_id;
                        if (__lacc[__idx])
                            __assign(static_cast<__tuple_type>(__in_rng[__idx]),
                                     __out_rng[__lacc[__idx + __elems_per_wg]]);
                    }

                    if (__item_id == 0)
                    {
                        // Add predicate of last element to account for the scan's exclusivity
                        *__res_ptr = __lacc[__elems_per_wg + __n - 1] + __lacc[__n - 1];
                    }
                });
        });

        return __future{std::move(__event), std::move(__result)};
    }
};

template <typename _CustomName, typename _InRng, typename _OutRng, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive>
sycl::event
__parallel_transform_scan_single_group(oneapi::dpl::__internal::__device_backend_tag, sycl::queue& __q,
                                       _InRng&& __in_rng, _OutRng&& __out_rng, ::std::size_t __n,
                                       _UnaryOperation __unary_op, _InitType __init, _BinaryOperation __binary_op,
                                       _Inclusive)
{
    std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__q);

    // Specialization for devices that have a max work-group size of 1024
    constexpr ::std::uint16_t __targeted_wg_size = 1024;

    if (__max_wg_size >= __targeted_wg_size)
    {
        auto __single_group_scan_f = [&](auto __size_constant) {
            constexpr ::std::uint16_t __size = decltype(__size_constant)::value;
            constexpr ::std::uint16_t __wg_size = ::std::min(__size, __targeted_wg_size);
            constexpr ::std::uint16_t __num_elems_per_item =
                oneapi::dpl::__internal::__dpl_ceiling_div(__size, __wg_size);
            const bool __is_full_group = __n == __wg_size;

            if (__is_full_group)
                return __parallel_transform_scan_static_single_group_submitter<
                    _Inclusive::value, __num_elems_per_item, __wg_size,
                    /* _IsFullGroup= */ true,
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_single_wg_kernel<
                        ::std::integral_constant<::std::uint16_t, __wg_size>,
                        ::std::integral_constant<::std::uint16_t, __num_elems_per_item>, _BinaryOperation,
                        /* _IsFullGroup= */ std::true_type, _Inclusive, _CustomName>>>()(
                    __q, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n, __init, __binary_op,
                    __unary_op);
            else
                return __parallel_transform_scan_static_single_group_submitter<
                    _Inclusive::value, __num_elems_per_item, __wg_size,
                    /* _IsFullGroup= */ false,
                    oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_single_wg_kernel<
                        ::std::integral_constant<::std::uint16_t, __wg_size>,
                        ::std::integral_constant<::std::uint16_t, __num_elems_per_item>, _BinaryOperation,
                        /* _IsFullGroup= */ ::std::false_type, _Inclusive, _CustomName>>>()(
                    __q, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n, __init, __binary_op,
                    __unary_op);
        };
        if (__n <= 16)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 16>{});
        else if (__n <= 32)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 32>{});
        else if (__n <= 64)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 64>{});
        else if (__n <= 128)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 128>{});
        else if (__n <= 256)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 256>{});
        else if (__n <= 512)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 512>{});
        else if (__n <= 1024)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 1024>{});
        else if (__n <= 2048)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 2048>{});
        else if (__n <= 4096)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 4096>{});
        else if (__n <= 8192)
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 8192>{});
        else
            return __single_group_scan_f(std::integral_constant<::std::uint16_t, 16384>{});
    }
    else
    {
        using _DynamicGroupScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __par_backend_hetero::__scan_single_wg_dynamic_kernel<_BinaryOperation, _CustomName>>;

        return __parallel_transform_scan_dynamic_single_group_submitter<_Inclusive::value, _DynamicGroupScanKernel>()(
            __q, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n, __init, __binary_op, __unary_op,
            __max_wg_size);
    }
}

template <typename _CustomName, typename _Range1, typename _Range2, typename _InitType, typename _LocalScan,
          typename _GroupScan, typename _GlobalScan>
__future<sycl::event, __result_and_scratch_storage<typename _InitType::__value_type>>
__parallel_transform_scan_base(oneapi::dpl::__internal::__device_backend_tag, sycl::queue& __q, _Range1&& __in_rng,
                               _Range2&& __out_rng, _InitType __init, _LocalScan __local_scan, _GroupScan __group_scan,
                               _GlobalScan __global_scan)
{
    using _PropagateKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__scan_propagate_kernel<_CustomName>>;

    return __parallel_scan_submitter<_CustomName, _PropagateKernel>()(__q, std::forward<_Range1>(__in_rng),
                                                                      std::forward<_Range2>(__out_rng), __init,
                                                                      __local_scan, __group_scan, __global_scan);
}

template <typename _Type>
bool
__group_scan_fits_in_slm(const sycl::queue& __q, std::size_t __n, std::size_t __n_uniform,
                         std::size_t __single_group_upper_limit)
{
    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const std::size_t __max_slm_size = __q.get_device().template get_info<sycl::info::device::local_mem_size>() / 2;
    const auto __req_slm_size = sizeof(_Type) * __n_uniform;

    return (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size);
}

//This is a temporary data structure which is used to store results to registers during a reduce then scan operation.
template <std::uint16_t elements, typename _ValueT>
struct __temp_data_array
{
    template <typename _ValueT2>
    void
    set(std::uint16_t __idx, const _ValueT2& __ele)
    {
        __data[__idx].__setup(__ele);
    }

    _ValueT
    get_and_destroy(std::uint16_t __idx)
    {
        _ValueT __ele = std::move(__data[__idx].__v);
        __data[__idx].__destroy();
        return __ele;
    }

    oneapi::dpl::__internal::__lazy_ctor_storage<_ValueT> __data[elements];
};

// This is a stand-in for a temporary data structure which is used to turn set() into a no-op. This is used in the case
// where no temporary register data is needed within reduce then scan kern
struct __noop_temp_data
{
    template <typename _ValueT>
    void
    set(std::uint16_t, const _ValueT&) const
    {
    }
};

template <typename _UnaryOp>
struct __gen_transform_input
{
    using TempData = __noop_temp_data;
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        // We explicitly convert __in_rng[__id] to the value type of _InRng to properly handle the case where we
        // process zip_iterator input where the reference type is a tuple of a references. This prevents the caller
        // from modifying the input range when altering the return of this functor.
        using _ValueType = oneapi::dpl::__internal::__value_t<_InRng>;
        return __unary_op(_ValueType{__in_rng[__id]});
    }
    _UnaryOp __unary_op;
};

template <typename _BinaryPred>
struct __gen_red_by_seg_reduce_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // (new_seg_mask, value)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        // The first segment start (index 0) is not marked with a 1. This is because we need the first
        // segment's key and value output index to be 0. We begin marking new segments only after the
        // first.
        const std::size_t __new_seg_mask = __id > 0 && !__binary_pred(__in_keys[__id - 1], __in_keys[__id]);
        return oneapi::dpl::__internal::make_tuple(__new_seg_mask, _ValueType{__in_vals[__id]});
    }
    _BinaryPred __binary_pred;
};

template <typename _BinaryPred>
struct __gen_red_by_seg_scan_input
{
    using TempData = __noop_temp_data;
    // Returns the following tuple:
    // ((new_seg_mask, value), output_value, next_key, current_key)
    // size_t new_seg_mask : 1 for a start of a new segment, 0 otherwise
    // ValueType value     : Current element's value for reduction
    // bool output_value   : Whether this work-item should write an output (end of segment)
    // KeyType next_key    : The key of the next segment to write if output_value is true
    // KeyType current_key : The current element's key. This is only ever used by work-item 0 to write the first key
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng, std::size_t __id, TempData&) const
    {
        const auto __in_keys = std::get<0>(__in_rng.tuple());
        const auto __in_vals = std::get<1>(__in_rng.tuple());
        using _KeyType = oneapi::dpl::__internal::__value_t<decltype(__in_keys)>;
        using _ValueType = oneapi::dpl::__internal::__value_t<decltype(__in_vals)>;
        const _KeyType& __current_key = __in_keys[__id];
        const _ValueType& __current_val = __in_vals[__id];
        // Ordering the most common condition first has yielded the best results.
        if (__id > 0 && __id < __n - 1)
        {
            const _KeyType& __prev_key = __in_keys[__id - 1];
            const _KeyType& __next_key = __in_keys[__id + 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
        else if (__id == __n - 1)
        {
            const _KeyType& __prev_key = __in_keys[__id - 1];
            const std::size_t __new_seg_mask = !__binary_pred(__prev_key, __current_key);
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(__new_seg_mask, __current_val), true, __current_key,
                __current_key); // Passing __current_key as the next key for the last element is a placeholder
        }
        else // __id == 0
        {
            const _KeyType& __next_key = __in_keys[__id + 1];
            return oneapi::dpl::__internal::make_tuple(
                oneapi::dpl::__internal::make_tuple(std::size_t{0}, __current_val),
                !__binary_pred(__current_key, __next_key), __next_key, __current_key);
        }
    }
    _BinaryPred __binary_pred;
    // For correctness of the function call operator, __n must be greater than 1.
    std::size_t __n;
};

template <typename _BinaryOp>
struct __red_by_seg_op
{
    // Consider the following segment / value pairs that would be processed in reduce-then-scan by a sub-group of size 8:
    // ----------------------------------------------------------
    // Keys:   0 0 1 1 2 2 2 2
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // The reduce and scan input generation phase flags new segments (excluding index 0) for use in the sub-group scan
    // operation. The above key, value pairs correspond to the following flag, value pairs:
    // ----------------------------------------------------------
    // Flags:  0 0 1 0 1 0 0 0
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // The sub-group scan operation looks back by powers-of-2 applying encountered prefixes. The __red_by_seg_op
    // operation performs a standard inclusive scan over the flags to compute output indices while performing a masked
    // scan over values to avoid applying a previous segment's partial reduction. Previous value elements are reduced
    // so long as the current index's flag is 0, indicating that input within its segment is still being processed
    // ----------------------------------------------------------
    // Start:
    // ----------------------------------------------------------
    // Flags:  0 0 1 0 1 0 0 0
    // Values: 1 1 1 1 1 1 1 1
    // ----------------------------------------------------------
    // After step 1 (apply the i-1th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 1 1 0 0
    // Values: 1 2 1 2 1 2 2 2
    // ----------------------------------------------------------
    // After step 2 (apply the i-2th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 2 2 1 1
    // Values: 1 2 1 2 1 2 3 4
    // ----------------------------------------------------------
    // After step 3 (apply the i-4th value if the ith flag is 0):
    // ----------------------------------------------------------
    // Flags:  0 0 1 1 2 2 2 2
    // Values: 1 2 1 2 1 2 3 4
    //           ^   ^       ^
    // ----------------------------------------------------------
    // Note that the scan of segment flags results in the desired output index of the reduce_by_segment operation in
    // each segment and the item corresponding to the final key in a segment contains its output reduction value. This
    // operation is first applied within a sub-group and then across sub-groups, work-groups, and blocks to
    // reduce-by-segment across the full input. The result of these operations combined with cached key data in
    // __gen_red_by_seg_scan_input enables the write phase to output keys and reduction values.
    // =>
    // Segments : 0 1 2
    // Values   : 2 2 4
    template <typename _Tup1, typename _Tup2>
    auto
    operator()(const _Tup1& __lhs_tup, const _Tup2& __rhs_tup) const
    {
        using std::get;
        using _OpReturnType = decltype(__binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        // The left-hand side has processed elements from the same segment, so update the reduction value.
        if (get<0>(__rhs_tup) == 0)
        {
            return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup),
                                                       __binary_op(get<1>(__lhs_tup), get<1>(__rhs_tup)));
        }
        // We are looking at elements from a previous segment so just update the output index.
        return oneapi::dpl::__internal::make_tuple(get<0>(__lhs_tup) + get<0>(__rhs_tup),
                                                   _OpReturnType{get<1>(__rhs_tup)});
    }
    _BinaryOp __binary_op;
};

template <typename _BinaryPred>
struct __write_red_by_seg
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _Tup>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _Tup& __tup, const _TempData&) const
    {
        using std::get;
        auto __out_keys = get<0>(__out_rng.tuple());
        auto __out_values = get<1>(__out_rng.tuple());

        const auto& __next_key = get<2>(__tup);
        const auto& __current_key = get<3>(__tup);
        const auto& __current_value = get<1>(get<0>(__tup));
        const bool __is_seg_end = get<1>(__tup);
        const std::size_t __out_idx = get<0>(get<0>(__tup));

        // With the exception of the first key which is output by index 0, the first key in each segment is written
        // by the work item that outputs the previous segment's reduction value. This is because the reduce_by_segment
        // API requires that the first key in a segment is output and is important for when keys in a segment might not
        // be the same (but satisfy the predicate). The last segment does not output a key as there are no future
        // segments process.
        if (__id == 0)
            __out_keys[0] = __current_key;
        if (__is_seg_end)
        {
            __out_values[__out_idx] = __current_value;
            if (__id != __n - 1)
                __out_keys[__out_idx + 1] = __next_key;
        }
    }
    _BinaryPred __binary_pred;
    std::size_t __n;
};

struct __simple_write_to_id
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, std::size_t __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__v)>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        __out_rng[__id] = static_cast<_ConvertedTupleType>(__v);
    }
};

template <typename _Predicate, typename _RangeTransform = oneapi::dpl::__internal::__no_op>
struct __gen_mask
{
    template <typename _InRng>
    bool
    operator()(_InRng&& __in_rng, std::size_t __id) const
    {
        return __pred((__rng_transform(std::forward<_InRng>(__in_rng)))[__id]);
    }
    _Predicate __pred;
    _RangeTransform __rng_transform;
};

template <typename _BinaryPredicate>
struct __gen_unique_mask
{
    template <typename _InRng>
    bool
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // Starting index is offset to 1 for "unique" patterns and 0th element
        // copy is handled separately, which allows us to do this without
        // branching each access to protect underflow
        return !__pred(__in_rng[__id], __in_rng[__id - 1]);
    }
    _BinaryPredicate __pred;
};

template <typename _IsOpDifference, typename _Compare>
struct __gen_set_mask
{
    template <typename _InRng>
    bool
    operator()(const _InRng& __in_rng, std::size_t __id) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __set_a = std::get<0>(__in_rng.tuple());    // first sequence
        auto __set_b = std::get<1>(__in_rng.tuple());    // second sequence
        auto __set_mask = std::get<2>(__in_rng.tuple()); // mask sequence

        std::size_t __nb = __set_b.size();

        auto __val_a = __set_a[__id];

        auto __res = oneapi::dpl::__internal::__pstl_lower_bound(__set_b, std::size_t{0}, __nb, __val_a, __comp);

        bool bres =
            _IsOpDifference::value; //initialization is true in case of difference operation; false - intersection.
        if (__res == __nb || __comp(__val_a, __set_b[__res]))
        {
            // there is no __val_a in __set_b, so __set_b in the difference {__set_a}/{__set_b};
        }
        else
        {
            auto __val_b = __set_b[__res];

            //Difference operation logic: if number of duplication in __set_a on left side from __id > total number of
            //duplication in __set_b then a mask is 1

            //Intersection operation logic: if number of duplication in __set_a on left side from __id <= total number of
            //duplication in __set_b then a mask is 1

            const std::size_t __count_a_left =
                __id - oneapi::dpl::__internal::__pstl_left_bound(__set_a, std::size_t{0}, __id, __val_a, __comp) + 1;

            const std::size_t __count_b =
                oneapi::dpl::__internal::__pstl_right_bound(__set_b, __res, __nb, __val_b, __comp) -
                oneapi::dpl::__internal::__pstl_left_bound(__set_b, std::size_t{0}, __res, __val_b, __comp);

            if constexpr (_IsOpDifference::value)
                bres = __count_a_left > __count_b; /*difference*/
            else
                bres = __count_a_left <= __count_b; /*intersection*/
        }
        __set_mask[__id] = bres;
        return bres;
    }
    _Compare __comp;
};

template <std::size_t _EleId>
struct __extract_range_from_zip
{
    template <typename _InRng>
    auto
    operator()(const _InRng& __in_rng) const
    {
        return std::get<_EleId>(__in_rng.tuple());
    }
};

template <typename _GenMask>
struct __gen_count_mask
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _SizeType>
    _SizeType
    operator()(_InRng&& __in_rng, _SizeType __id, TempData&) const
    {
        return __gen_mask(std::forward<_InRng>(__in_rng), __id) ? _SizeType{1} : _SizeType{0};
    }
    _GenMask __gen_mask;
};

template <typename _GenMask, typename _RangeTransform = oneapi::dpl::__internal::__no_op>
struct __gen_expand_count_mask
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _SizeType>
    auto
    operator()(_InRng&& __in_rng, _SizeType __id, TempData&) const
    {
        auto __transformed_input = __rng_transform(__in_rng);
        // Explicitly creating this element type is necessary to avoid modifying the input data when _InRng is a
        //  zip_iterator which will return a tuple of references when dereferenced. With this explicit type, we copy
        //  the values of zipped input types rather than their references.
        using _ElementType = oneapi::dpl::__internal::__value_t<decltype(__transformed_input)>;
        _ElementType ele = __transformed_input[__id];
        bool mask = __gen_mask(std::forward<_InRng>(__in_rng), __id);
        return std::tuple(mask ? _SizeType{1} : _SizeType{0}, mask, ele);
    }
    _GenMask __gen_mask;
    _RangeTransform __rng_transform;
};

// Locates and returns the "intersection" of a diagonal on the balanced path, based on merge path coordinates.
// It returns coordinates in each set of the intersection with a boolean representing if the diagonal is "starred",
// meaning that the balanced path "intersection" point does not lie directly on the diagonal, but one step forward in
// the second set.
// Some diagonals must be "starred" to ensure that matching elements between rng1 and rng2 are processed in pairs
// starting from the first of repeating value(s) in each range and a matched pair are not split between work-items.
template <typename _Rng1, typename _Rng2, typename _Index, typename _Compare>
auto
__find_balanced_path_start_point(const _Rng1& __rng1, const _Rng2& __rng2, const _Index __merge_path_rng1,
                                 const _Index __merge_path_rng2, _Compare __comp)
{
    // back up to balanced path divergence with a biased binary search
    bool __star = false;
    if (__merge_path_rng1 == 0 || __merge_path_rng2 == __rng2.size())
    {
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    auto __ele_val = __rng1[__merge_path_rng1 - 1];

    if (__comp(__ele_val, __rng2[__merge_path_rng2]))
    {
        // There is no chance that the balanced path differs from the merge path here, because the previous element of
        // rng1 does not match the next element of rng2. We can just return the merge path.
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    // find first element of repeating sequence in the first set of the previous element
    _Index __rng1_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
        __rng1, _Index{0}, __merge_path_rng1, __ele_val, __comp);
    // find first element of repeating sequence in the second set of the next element
    _Index __rng2_repeat_start = oneapi::dpl::__internal::__biased_lower_bound</*__last_bias=*/true>(
        __rng2, _Index{0}, __merge_path_rng2, __ele_val, __comp);

    _Index __rng1_repeats = __merge_path_rng1 - __rng1_repeat_start;
    _Index __rng2_repeats_bck = __merge_path_rng2 - __rng2_repeat_start;

    if (__rng2_repeats_bck >= __rng1_repeats)
    {
        // If we have at least as many repeated elements in rng2, we end up back on merge path
        return std::make_tuple(__merge_path_rng1, __merge_path_rng2, false);
    }

    // Calculate the number of "unmatched" repeats in the first set, add one and divide by two to round up for a
    // possible star diagonal.
    _Index __fwd_search_count = (__rng1_repeats - __rng2_repeats_bck + 1) / 2;

    // Calculate the max location to search in the second set for future repeats, limiting to the edge of the range
    _Index __fwd_search_bound = std::min(__merge_path_rng2 + __fwd_search_count, __rng2.size());

    _Index __balanced_path_intersection_rng2 =
        oneapi::dpl::__internal::__pstl_upper_bound(__rng2, __merge_path_rng2, __fwd_search_bound, __ele_val, __comp);

    // Calculate the number of matchable "future" repeats in the second set
    _Index __matchable_forward_ele_rng2 = __balanced_path_intersection_rng2 - __merge_path_rng2;
    _Index __total_matched_rng2 = __balanced_path_intersection_rng2 - __rng2_repeat_start;

    // Update balanced path intersection for rng1, must account for cases where there are more repeating elements in
    // rng1 than matched elements of rng2
    _Index __balanced_path_intersection_rng1 =
        __rng1_repeat_start + std::max(__total_matched_rng2, __rng1_repeats - __matchable_forward_ele_rng2);

    // If we needed to step off the diagonal to find the balanced path, mark the diagonal as "starred"
    __star =
        __balanced_path_intersection_rng1 + __balanced_path_intersection_rng2 != __merge_path_rng1 + __merge_path_rng2;

    return std::make_tuple(__balanced_path_intersection_rng1, __balanced_path_intersection_rng2, __star);
}

// Reduce then scan building block for set balanced path which is used in the reduction kernel to calculate the
// balanced path intersection, store it to temporary data with "star" status, then count the number of elements to write
// to the output for the reduction operation.
template <typename _SetOpCount, typename _Compare>
struct __gen_set_balanced_path
{
    using TempData = __noop_temp_data;
    template <typename _InRng, typename _IndexT>
    std::uint16_t
    operator()(const _InRng& __in_rng, _IndexT __id, TempData& __temp_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        auto __rng1_temp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence

        using _SizeType = decltype(__rng1.size());
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            return 0;
        //find merge path intersection
        auto [__rng1_pos, __rng2_pos] = oneapi::dpl::__par_backend_hetero::__find_start_point(
            __rng1, _SizeType{0}, __rng1.size(), __rng2, _SizeType{0}, __rng2.size(), __i_elem, __comp);

        //Find balanced path for diagonal start
        auto [__rng1_balanced_pos, __rng2_balanced_pos, __star_offset] =
            __find_balanced_path_start_point(__rng1, __rng2, __rng1_pos, __rng2_pos, __comp);

        // Use sign bit to represent star offset. Temp storage is a signed type equal to the difference_type of the
        // input iterator range. The index will fit into the positive portion of the type, so the sign may be used to
        // indicate the star offset.
        __rng1_temp_diag[__id] = __rng1_balanced_pos * (__star_offset ? -1 : 1);

        _SizeType __eles_to_process = std::min(__diagonal_spacing - (__star_offset ? _SizeType{1} : _SizeType{0}),
                                               __rng1.size() + __rng2.size() - (__i_elem - 1));

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_balanced_pos, __rng2_balanced_pos,
                                               __eles_to_process, __temp_data, __comp);
        return __count;
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _Compare __comp;
};

// Reduce then scan building block for set balanced path which is used in the scan kernel to decode the stored balanced
// path intersection, perform the serial set operation for the diagonal, counting the number of elements and writing
// the output to temporary data in registers to be ready for the scan and write operations to follow.
template <typename _SetOpCount, typename _TempData, typename _Compare>
struct __gen_set_op_from_known_balanced_path
{
    using TempData = _TempData;
    template <typename _InRng, typename _IndexT>
    auto
    operator()(const _InRng& __in_rng, _IndexT __id, _TempData& __output_data) const
    {
        // First we must extract individual sequences from zip iterator because they may not have the same length,
        // dereferencing is dangerous
        auto __rng1 = std::get<0>(__in_rng.tuple()); // first sequence
        auto __rng2 = std::get<1>(__in_rng.tuple()); // second sequence

        auto __rng1_temp_diag = std::get<2>(__in_rng.tuple()); // set a temp storage sequence, star value in sign bit
        using _SizeType = decltype(__rng1.size());
        _SizeType __i_elem = __id * __diagonal_spacing;
        if (__i_elem >= __rng1.size() + __rng2.size())
            return std::make_tuple(std::uint32_t{0}, std::uint16_t{0});
        _SizeType __star_offset = oneapi::dpl::__internal::__dpl_signbit(__rng1_temp_diag[__id]) ? 1 : 0;
        auto __rng1_temp_diag_abs = std::abs(__rng1_temp_diag[__id]);
        auto __rng2_temp_diag = __i_elem - __rng1_temp_diag_abs + __star_offset;

        _SizeType __eles_to_process =
            std::min(_SizeType{__diagonal_spacing} - __star_offset, __rng1.size() + __rng2.size() - (__i_elem - 1));

        std::uint16_t __count = __set_op_count(__rng1, __rng2, __rng1_temp_diag_abs, __rng2_temp_diag,
                                               __eles_to_process, __output_data, __comp);
        return std::make_tuple(std::uint32_t{__count}, __count);
    }
    _SetOpCount __set_op_count;
    std::uint16_t __diagonal_spacing;
    _Compare __comp;
};

// Returns by reference: iterations consumed, and the number of elements copied to temp output
template <bool _CopyMatch, bool _CopyDiffSetA, bool _CopyDiffSetB, bool _CheckBounds, typename _InRng1,
          typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare>
void
__set_generic_operation_iteration(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t& __idx1,
                                  std::size_t& __idx2, _SizeType __num_eles_min, _TempOutput& __temp_out,
                                  _SizeType& __idx, std::uint16_t& __count, _Compare __comp)
{
    using _ValueTypeRng1 = typename oneapi::dpl::__internal::__value_t<_InRng1>;
    using _ValueTypeRng2 = typename oneapi::dpl::__internal::__value_t<_InRng2>;

    if constexpr (_CheckBounds)
    {
        if (__idx1 == __in_rng1.size())
        {
            if constexpr (_CopyDiffSetB)
            {
                // If we are at the end of rng1, copy the rest of rng2 within our diagonal's bounds
                for (; __idx2 < __in_rng2.size() && __idx < __num_eles_min; ++__idx2, ++__idx)
                {
                    __temp_out.set(__count, __in_rng2[__idx2]);
                    ++__count;
                }
            }
            __idx = __num_eles_min;
            return;
        }
        if (__idx2 == __in_rng2.size())
        {
            if constexpr (_CopyDiffSetA)
            {
                // If we are at the end of rng2, copy the rest of rng1 within our diagonal's bounds
                for (; __idx1 < __in_rng1.size() && __idx < __num_eles_min; ++__idx1, ++__idx)
                {
                    __temp_out.set(__count, __in_rng1[__idx1]);
                    ++__count;
                }
            }
            __idx = __num_eles_min;
            return;
        }
    }

    const _ValueTypeRng1& __ele_rng1 = __in_rng1[__idx1];
    const _ValueTypeRng2& __ele_rng2 = __in_rng2[__idx2];
    if (__comp(__ele_rng1, __ele_rng2))
    {
        if constexpr (_CopyDiffSetA)
        {
            __temp_out.set(__count, __ele_rng1);
            ++__count;
        }
        ++__idx1;
        ++__idx;
    }
    else if (__comp(__ele_rng2, __ele_rng1))
    {
        if constexpr (_CopyDiffSetB)
        {
            __temp_out.set(__count, __ele_rng2);
            ++__count;
        }
        ++__idx2;
        ++__idx;
    }
    else // if neither element is less than the other, they are equal
    {
        if constexpr (_CopyMatch)
        {
            __temp_out.set(__count, __ele_rng1);
            ++__count;
        }
        ++__idx1;
        ++__idx2;
        __idx += 2;
    }
}

template <bool _CopyMatch, bool _CopyDiffSetA, bool _CopyDiffSetB>
struct __set_generic_operation
{
    template <typename _InRng1, typename _InRng2, typename _SizeType, typename _TempOutput, typename _Compare>
    std::uint16_t
    operator()(const _InRng1& __in_rng1, const _InRng2& __in_rng2, std::size_t __idx1, std::size_t __idx2,
               _SizeType __num_eles_min, _TempOutput& __temp_out, _Compare __comp) const
    {

        std::uint16_t __count = 0;
        _SizeType __idx = 0;
        bool __can_reach_rng1_end = __idx1 + __num_eles_min >= __in_rng1.size();
        bool __can_reach_rng2_end = __idx2 + __num_eles_min >= __in_rng2.size();

        if (!__can_reach_rng1_end && !__can_reach_rng2_end)
        {
            while (__idx < __num_eles_min)
            {
                // no bounds checking
                __set_generic_operation_iteration<_CopyMatch, _CopyDiffSetA, _CopyDiffSetB, false>(
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp);
            }
        }
        else
        {
            while (__idx < __num_eles_min)
            {
                //bounds check all
                __set_generic_operation_iteration<_CopyMatch, _CopyDiffSetA, _CopyDiffSetB, true>(
                    __in_rng1, __in_rng2, __idx1, __idx2, __num_eles_min, __temp_out, __idx, __count, __comp);
            }
        }
        return __count;
    }
};

using __set_intersection = __set_generic_operation<true, false, false>;
using __set_difference = __set_generic_operation<false, true, false>;
using __set_union = __set_generic_operation<true, true, true>;
using __set_symmetric_difference = __set_generic_operation<false, true, true>;

template <typename _SetTag>
struct __get_set_operation;

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_IntersectionTag<std::true_type>> : public __set_intersection
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_DifferenceTag<std::true_type>> : public __set_difference
{
};
template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_UnionTag<std::true_type>> : public __set_union
{
};

template <>
struct __get_set_operation<oneapi::dpl::unseq_backend::_SymmetricDifferenceTag<std::true_type>>
    : public __set_symmetric_difference
{
};

struct __get_zeroth_element
{
    template <typename _Tp>
    auto&
    operator()(_Tp&& __a) const
    {
        return std::get<0>(std::forward<_Tp>(__a));
    }
};

template <std::int32_t __offset, typename _Assign>
struct __write_to_id_if
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        if (std::get<1>(__v))
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), __out_rng[std::get<0>(__v) - 1 + __offset]);
    }
    _Assign __assign;
};

template <typename _Assign>
struct __write_to_id_if_else
{
    using _TempData = __noop_temp_data;
    template <typename _OutRng, typename _SizeType, typename _ValueType>
    void
    operator()(_OutRng& __out_rng, _SizeType __id, const _ValueType& __v, const _TempData&) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(std::get<2>(__v))>,
                                                               std::decay_t<decltype(__out_rng[__id])>>::__type;
        if (std::get<1>(__v))
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)), std::get<0>(__out_rng[std::get<0>(__v) - 1]));
        else
            __assign(static_cast<_ConvertedTupleType>(std::get<2>(__v)),
                     std::get<1>(__out_rng[__id - std::get<0>(__v)]));
    }
    _Assign __assign;
};

// Writes multiple elements from temp data to the output range. The values to write are stored in `__temp_data` from a
// previous operation, and must be written to the output range in the appropriate location. The zeroth element of `__v`
// will contain the index of one past the last element to write, and the first element of `__v` will contain the number
// of elements to write.
template <typename _Assign>
struct __write_multiple_to_id
{
    template <typename _OutRng, typename _SizeType, typename _ValueType, typename _TempData>
    void
    operator()(_OutRng& __out_rng, _SizeType, const _ValueType& __v, _TempData& __temp_data) const
    {
        // Use of an explicit cast to our internal tuple type is required to resolve conversion issues between our
        // internal tuple and std::tuple. If the underlying type is not a tuple, then the type will just be passed
        // through.
        using _ConvertedTupleType =
            typename oneapi::dpl::__internal::__get_tuple_type<std::decay_t<decltype(__temp_data.get_and_destroy(0))>,
                                                               std::decay_t<decltype(__out_rng[0])>>::__type;
        for (std::size_t __i = 0; __i < std::get<1>(__v); ++__i)
        {
            __assign(static_cast<_ConvertedTupleType>(__temp_data.get_and_destroy(__i)),
                     __out_rng[std::get<0>(__v) - std::get<1>(__v) + __i]);
        }
    }
    _Assign __assign;
};

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryOperation, typename _InitType,
          typename _BinaryOperation, typename _Inclusive>
__future<sycl::event, __result_and_scratch_storage<typename _InitType::__value_type>>
__parallel_transform_scan(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                          _Range1&& __in_rng, _Range2&& __out_rng, std::size_t __n, _UnaryOperation __unary_op,
                          _InitType __init, _BinaryOperation __binary_op, _Inclusive)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    using _Type = typename _InitType::__value_type;
    // Reduce-then-scan is dependent on sycl::shift_group_right which requires the underlying type to be trivially
    // copyable. If this is not met, then we must fallback to the multi pass scan implementation. The single
    // work-group implementation requires a fundamental type which must also be trivially copyable.
    if constexpr (std::is_trivially_copyable_v<_Type>)
    {
        bool __use_reduce_then_scan =
            oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__q_local);

        // TODO: Consider re-implementing single group scan to support types without known identities. This could also
        // allow us to use single wg scan for the last block of reduce-then-scan if it is sufficiently small.
        constexpr bool __can_use_group_scan = unseq_backend::__has_known_identity<_BinaryOperation, _Type>::value;
        if constexpr (__can_use_group_scan)
        {
            // Next power of 2 greater than or equal to __n
            std::size_t __n_uniform = oneapi::dpl::__internal::__dpl_bit_ceil(__n);

            // Empirically found values for reduce-then-scan and multi pass scan implementation for single wg cutoff
            std::size_t __single_group_upper_limit = __use_reduce_then_scan ? 2048 : 16384;
            if (__group_scan_fits_in_slm<_Type>(__q_local, __n, __n_uniform, __single_group_upper_limit))
            {
                auto __event = __parallel_transform_scan_single_group<_CustomName>(
                    __backend_tag, __q_local, std::forward<_Range1>(__in_rng), std::forward<_Range2>(__out_rng), __n,
                    __unary_op, __init, __binary_op, _Inclusive{});

                // Although we do not actually need result storage in this case, we need to construct
                // a placeholder here to match the return type of the non-single-work-group implementation
                __result_and_scratch_storage<_Type> __dummy_result_and_scratch{__q_local, 0};

                return __future{std::move(__event), std::move(__dummy_result_and_scratch)};
            }
        }
        if (__use_reduce_then_scan)
        {
            using _GenInput = oneapi::dpl::__par_backend_hetero::__gen_transform_input<_UnaryOperation>;
            using _ScanInputTransform = oneapi::dpl::__internal::__no_op;
            using _WriteOp = oneapi::dpl::__par_backend_hetero::__simple_write_to_id;

            _GenInput __gen_transform{__unary_op};

            const std::size_t __n = __in_rng.size();
            return __parallel_transform_reduce_then_scan<sizeof(typename _InitType::__value_type), _CustomName>(
                __backend_tag, __q_local, __n, std::forward<_Range1>(__in_rng), std::forward<_Range2>(__out_rng),
                __gen_transform, __binary_op, __gen_transform, _ScanInputTransform{}, _WriteOp{}, __init, _Inclusive{},
                /*_IsUniquePattern=*/std::false_type{});
        }
    }

    //else use multi pass scan implementation
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _UnaryFunctor = unseq_backend::walk_n<_UnaryOperation>;
    using _NoOpFunctor = unseq_backend::walk_n<oneapi::dpl::__internal::__no_op>;

    _Assigner __assign_op;
    _NoAssign __no_assign_op;
    _NoOpFunctor __get_data_op;

    return __parallel_transform_scan_base<_CustomName>(
        __backend_tag, __q_local, std::forward<_Range1>(__in_rng), std::forward<_Range2>(__out_rng), __init,
        // local scan
        unseq_backend::__scan<_Inclusive, _BinaryOperation, _UnaryFunctor, _Assigner, _Assigner, _NoOpFunctor,
                              _InitType>{__binary_op, _UnaryFunctor{__unary_op}, __assign_op, __assign_op,
                                         __get_data_op},
        // scan between groups
        unseq_backend::__scan</*inclusive=*/std::true_type, _BinaryOperation, _NoOpFunctor, _NoAssign, _Assigner,
                              _NoOpFunctor, unseq_backend::__no_init_value<_Type>>{
            __binary_op, _NoOpFunctor{}, __no_assign_op, __assign_op, __get_data_op},
        // global scan
        unseq_backend::__global_scan_functor<_Inclusive, _BinaryOperation, _InitType>{__binary_op, __init});
}

template <typename _CustomName, typename _SizeType>
struct __invoke_single_group_copy_if
{
    // Specialization for devices that have a max work-group size of at least 1024
    static constexpr ::std::uint16_t __targeted_wg_size = 1024;

    template <std::uint16_t _Size, typename _InRng, typename _OutRng, typename _Pred,
              typename _Assign = oneapi::dpl::__internal::__pstl_assign>
    auto
    operator()(sycl::queue& __q, std::size_t __n, _InRng&& __in_rng, _OutRng&& __out_rng, _Pred __pred,
               _Assign __assign)
    {
        constexpr ::std::uint16_t __wg_size = ::std::min(_Size, __targeted_wg_size);
        constexpr ::std::uint16_t __num_elems_per_item = ::oneapi::dpl::__internal::__dpl_ceiling_div(_Size, __wg_size);
        const bool __is_full_group = __n == __wg_size;

        using _InitType = unseq_backend::__no_init_value<::std::uint16_t>;
        using _ReduceOp = ::std::plus<::std::uint16_t>;
        if (__is_full_group)
        {
            using _FullKernel =
                __scan_copy_single_wg_kernel<std::integral_constant<std::uint16_t, __wg_size>,
                                             std::integral_constant<std::uint16_t, __num_elems_per_item>,
                                             /* _IsFullGroup= */ std::true_type, _CustomName>;
            using _FullKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<_FullKernel>;
            return __par_backend_hetero::__parallel_copy_if_static_single_group_submitter<
                _SizeType, __num_elems_per_item, __wg_size, true, _FullKernelName>()(
                __q, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n, _InitType{}, _ReduceOp{},
                __pred, __assign);
        }
        else
        {
            using _NonFullKernel =
                __scan_copy_single_wg_kernel<std::integral_constant<std::uint16_t, __wg_size>,
                                             std::integral_constant<std::uint16_t, __num_elems_per_item>,
                                             /* _IsFullGroup= */ std::false_type, _CustomName>;
            using _NonFullKernelName =
                oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<_NonFullKernel>;
            return __par_backend_hetero::__parallel_copy_if_static_single_group_submitter<
                _SizeType, __num_elems_per_item, __wg_size, false, _NonFullKernelName>()(
                __q, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng), __n, _InitType{}, _ReduceOp{},
                __pred, __assign);
        }
    }
};

template <typename _CustomName, typename _InRng, typename _OutRng, typename _Size, typename _GenMask, typename _WriteOp,
          typename _IsUniquePattern>
__future<sycl::event, __result_and_scratch_storage<_Size>>
__parallel_reduce_then_scan_copy(oneapi::dpl::__internal::__device_backend_tag __backend_tag, sycl::queue& __q,
                                 _InRng&& __in_rng, _OutRng&& __out_rng, _Size, _GenMask __generate_mask,
                                 _WriteOp __write_op, _IsUniquePattern __is_unique_pattern)
{
    using _GenReduceInput = oneapi::dpl::__par_backend_hetero::__gen_count_mask<_GenMask>;
    using _ReduceOp = std::plus<_Size>;
    using _GenScanInput = oneapi::dpl::__par_backend_hetero::__gen_expand_count_mask<_GenMask>;
    using _ScanInputTransform = oneapi::dpl::__par_backend_hetero::__get_zeroth_element;

    const std::size_t __n = __in_rng.size();
    return __parallel_transform_reduce_then_scan<sizeof(_Size), _CustomName>(
        __backend_tag, __q, __n, std::forward<_InRng>(__in_rng), std::forward<_OutRng>(__out_rng),
        _GenReduceInput{__generate_mask}, _ReduceOp{}, _GenScanInput{__generate_mask, {}}, _ScanInputTransform{},
        __write_op, oneapi::dpl::unseq_backend::__no_init_value<_Size>{},
        /*_Inclusive=*/std::true_type{}, __is_unique_pattern);
}

template <typename _CustomName, typename _InRng, typename _OutRng, typename _Size, typename _CreateMaskOp,
          typename _CopyByMaskOp>
__future<sycl::event, __result_and_scratch_storage<_Size>>
__parallel_scan_copy(oneapi::dpl::__internal::__device_backend_tag __backend_tag, sycl::queue& __q, _InRng&& __in_rng,
                     _OutRng&& __out_rng, _Size __n, _CreateMaskOp __create_mask_op, _CopyByMaskOp __copy_by_mask_op)
{
    using _ReduceOp = std::plus<_Size>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<1>;
    using _DataAcc = unseq_backend::walk_n<oneapi::dpl::__internal::__no_op>;
    using _InitType = unseq_backend::__no_init_value<_Size>;

    _Assigner __assign_op;
    _ReduceOp __reduce_op;
    _DataAcc __get_data_op;
    _MaskAssigner __add_mask_op;

    // temporary buffer to store boolean mask
    oneapi::dpl::__par_backend_hetero::__buffer<int32_t> __mask_buf(__n);

    return __parallel_transform_scan_base<_CustomName>(
        __backend_tag, __q,
        oneapi::dpl::__ranges::zip_view(
            __in_rng, oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                          __mask_buf.get_buffer())),
        std::forward<_OutRng>(__out_rng), _InitType{},
        // local scan
        unseq_backend::__scan</*inclusive*/ std::true_type, _ReduceOp, _DataAcc, _Assigner, _MaskAssigner,
                              _CreateMaskOp, _InitType>{__reduce_op, __get_data_op, __assign_op, __add_mask_op,
                                                        __create_mask_op},
        // scan between groups
        unseq_backend::__scan</*inclusive*/ std::true_type, _ReduceOp, _DataAcc, _NoAssign, _Assigner, _DataAcc,
                              _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op, __get_data_op},
        // global scan
        __copy_by_mask_op);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _BinaryPredicate>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range1>>>
__parallel_unique_copy(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                       _Range1&& __rng, _Range2&& __result, _BinaryPredicate __pred)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using _Assign = oneapi::dpl::__internal::__pstl_assign;
    oneapi::dpl::__internal::__difference_t<_Range1> __n = __rng.size();

    // We expect at least two elements to perform unique_copy.  With fewer we
    // can simply copy the input range to the output.
    assert(__n > 1);

    sycl::queue __q_local = __exec.queue();

    if (oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__q_local))
    {
        using _GenMask = oneapi::dpl::__par_backend_hetero::__gen_unique_mask<_BinaryPredicate>;
        using _WriteOp = oneapi::dpl::__par_backend_hetero::__write_to_id_if<1, _Assign>;

        return __parallel_reduce_then_scan_copy<_CustomName>(__backend_tag, __q_local, std::forward<_Range1>(__rng),
                                                             std::forward<_Range2>(__result), __n, _GenMask{__pred},
                                                             _WriteOp{_Assign{}},
                                                             /*_IsUniquePattern=*/std::true_type{});
    }
    else
    {
        using _ReduceOp = std::plus<decltype(__n)>;
        using _CreateOp =
            oneapi::dpl::__internal::__create_mask_unique_copy<oneapi::dpl::__internal::__not_pred<_BinaryPredicate>,
                                                               decltype(__n)>;
        using _CopyOp = unseq_backend::__copy_by_mask<_ReduceOp, _Assign, /*inclusive*/ std::true_type, 1>;

        return __parallel_scan_copy<_CustomName>(
            __backend_tag, __q_local, std::forward<_Range1>(__rng), std::forward<_Range2>(__result), __n,
            _CreateOp{oneapi::dpl::__internal::__not_pred<_BinaryPredicate>{__pred}}, _CopyOp{_ReduceOp{}, _Assign{}});
    }
}

template <typename _CustomName, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
__future<sycl::event, __result_and_scratch_storage<
                          oneapi::dpl::__internal::tuple<std::size_t, oneapi::dpl::__internal::__value_t<_Range2>>>>
__parallel_reduce_by_segment_reduce_then_scan(oneapi::dpl::__internal::__device_backend_tag __backend_tag,
                                              sycl::queue& __q, _Range1&& __keys, _Range2&& __values,
                                              _Range3&& __out_keys, _Range4&& __out_values,
                                              _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    // Flags new segments and passes input value through a 2-tuple
    using _GenReduceInput = __gen_red_by_seg_reduce_input<_BinaryPredicate>;
    // Operation that computes output indices and output reduction values per segment
    using _ReduceOp = __red_by_seg_op<_BinaryOperator>;
    // Returns 4-component tuple which contains flags, keys, value, and a flag to write output
    using _GenScanInput = __gen_red_by_seg_scan_input<_BinaryPredicate>;
    // Returns the first component from scan input which is scanned over
    using _ScanInputTransform = __get_zeroth_element;
    // Writes current segment's output reduction and the next segment's output key
    using _WriteOp = __write_red_by_seg<_BinaryPredicate>;
    using _ValueType = oneapi::dpl::__internal::__value_t<_Range2>;
    std::size_t __n = __keys.size();
    // __gen_red_by_seg_scan_input requires that __n > 1
    assert(__n > 1);
    return __parallel_transform_reduce_then_scan<sizeof(oneapi::dpl::__internal::tuple<std::size_t, _ValueType>),
                                                 _CustomName>(
        __backend_tag, __q, __n,
        oneapi::dpl::__ranges::make_zip_view(std::forward<_Range1>(__keys), std::forward<_Range2>(__values)),
        oneapi::dpl::__ranges::make_zip_view(std::forward<_Range3>(__out_keys), std::forward<_Range4>(__out_values)),
        _GenReduceInput{__binary_pred}, _ReduceOp{__binary_op}, _GenScanInput{__binary_pred, __n},
        _ScanInputTransform{}, _WriteOp{__binary_pred, __n},
        oneapi::dpl::unseq_backend::__no_init_value<oneapi::dpl::__internal::tuple<std::size_t, _ValueType>>{},
        /*Inclusive*/ std::true_type{}, /*_IsUniquePattern=*/std::false_type{});
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _UnaryPredicate>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range1>>>
__parallel_partition_copy(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                          _Range1&& __rng, _Range2&& __result, _UnaryPredicate __pred)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    oneapi::dpl::__internal::__difference_t<_Range1> __n = __rng.size();
    if (oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__q_local))
    {
        using _GenMask = oneapi::dpl::__par_backend_hetero::__gen_mask<_UnaryPredicate>;
        using _WriteOp =
            oneapi::dpl::__par_backend_hetero::__write_to_id_if_else<oneapi::dpl::__internal::__pstl_assign>;

        return __parallel_reduce_then_scan_copy<_CustomName>(__backend_tag, __q_local, std::forward<_Range1>(__rng),
                                                             std::forward<_Range2>(__result), __n, _GenMask{__pred, {}},
                                                             _WriteOp{},
                                                             /*_IsUniquePattern=*/std::false_type{});
    }
    else
    {
        using _ReduceOp = std::plus<decltype(__n)>;
        using _CreateOp = unseq_backend::__create_mask<_UnaryPredicate, decltype(__n)>;
        using _CopyOp = unseq_backend::__partition_by_mask<_ReduceOp, /*inclusive*/ std::true_type>;

        return __parallel_scan_copy<_CustomName>(__backend_tag, __q_local, std::forward<_Range1>(__rng),
                                                 std::forward<_Range2>(__result), __n, _CreateOp{__pred},
                                                 _CopyOp{_ReduceOp{}});
    }
}

template <typename _ExecutionPolicy, typename _InRng, typename _OutRng, typename _Size, typename _Pred,
          typename _Assign = oneapi::dpl::__internal::__pstl_assign>
__future<sycl::event, __result_and_scratch_storage<_Size>>
__parallel_copy_if(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                   _InRng&& __in_rng, _OutRng&& __out_rng, _Size __n, _Pred __pred, _Assign __assign = _Assign{})
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using _SingleGroupInvoker = __invoke_single_group_copy_if<_CustomName, _Size>;

    // Next power of 2 greater than or equal to __n
    auto __n_uniform = ::oneapi::dpl::__internal::__dpl_bit_ceil(static_cast<std::make_unsigned_t<_Size>>(__n));

    sycl::queue __q_local = __exec.queue();

    // Pessimistically only use half of the memory to take into account memory used by compiled kernel
    const std::size_t __max_slm_size =
        __q_local.get_device().template get_info<sycl::info::device::local_mem_size>() / 2;

    // The kernel stores n integers for the predicate and another n integers for the offsets
    const auto __req_slm_size = sizeof(std::uint16_t) * __n_uniform * 2;

    constexpr std::uint16_t __single_group_upper_limit = 2048;

    std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__q_local);

    if (__n <= __single_group_upper_limit && __max_slm_size >= __req_slm_size &&
        __max_wg_size >= _SingleGroupInvoker::__targeted_wg_size)
    {
        using _SizeBreakpoints = std::integer_sequence<std::uint16_t, 16, 32, 64, 128, 256, 512, 1024, 2048>;

        return __par_backend_hetero::__static_monotonic_dispatcher<_SizeBreakpoints>::__dispatch(
            _SingleGroupInvoker{}, __n, __q_local, __n, std::forward<_InRng>(__in_rng),
            std::forward<_OutRng>(__out_rng), __pred, __assign);
    }
    else if (oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__q_local))
    {
        using _GenMask = oneapi::dpl::__par_backend_hetero::__gen_mask<_Pred>;
        using _WriteOp = oneapi::dpl::__par_backend_hetero::__write_to_id_if<0, _Assign>;

        return __parallel_reduce_then_scan_copy<_CustomName>(__backend_tag, __q_local, std::forward<_InRng>(__in_rng),
                                                             std::forward<_OutRng>(__out_rng), __n,
                                                             _GenMask{__pred, {}}, _WriteOp{__assign},
                                                             /*_IsUniquePattern=*/std::false_type{});
    }
    else
    {
        using _ReduceOp = std::plus<_Size>;
        using _CreateOp = unseq_backend::__create_mask<_Pred, _Size>;
        using _CopyOp = unseq_backend::__copy_by_mask<_ReduceOp, _Assign,
                                                      /*inclusive*/ std::true_type, 1>;

        return __parallel_scan_copy<_CustomName>(__backend_tag, __q_local, std::forward<_InRng>(__in_rng),
                                                 std::forward<_OutRng>(__out_rng), __n, _CreateOp{__pred},
                                                 _CopyOp{_ReduceOp{}, __assign});
    }
}

// This function is currently unused, but may be utilized for small sizes sets at some point in the future.
template <typename _CustomName, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          typename _IsOpDifference>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range3>>>
__parallel_set_reduce_then_scan_set_a_write(oneapi::dpl::__internal::__device_backend_tag __backend_tag,
                                            sycl::queue& __q, _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result,
                                            _Compare __comp, _IsOpDifference)
{
    // fill in reduce then scan impl
    using _GenMaskReduce = oneapi::dpl::__par_backend_hetero::__gen_set_mask<_IsOpDifference, _Compare>;
    using _MaskRangeTransform = oneapi::dpl::__par_backend_hetero::__extract_range_from_zip<2>;
    using _MaskPredicate = oneapi::dpl::__internal::__no_op;
    using _GenMaskScan = oneapi::dpl::__par_backend_hetero::__gen_mask<_MaskPredicate, _MaskRangeTransform>;
    using _WriteOp = oneapi::dpl::__par_backend_hetero::__write_to_id_if<0, oneapi::dpl::__internal::__pstl_assign>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range3>;
    using _ScanRangeTransform = oneapi::dpl::__par_backend_hetero::__extract_range_from_zip<0>;

    using _GenReduceInput = oneapi::dpl::__par_backend_hetero::__gen_count_mask<_GenMaskReduce>;
    using _ReduceOp = std::plus<_Size>;
    using _GenScanInput = oneapi::dpl::__par_backend_hetero::__gen_expand_count_mask<_GenMaskScan, _ScanRangeTransform>;
    using _ScanInputTransform = oneapi::dpl::__par_backend_hetero::__get_zeroth_element;

    oneapi::dpl::__par_backend_hetero::__buffer<std::int32_t> __mask_buf(__rng1.size());

    const std::size_t __n = __rng1.size();
    return __parallel_transform_reduce_then_scan<sizeof(_Size), _CustomName>(
        __backend_tag, __q, __n,
        oneapi::dpl::__ranges::make_zip_view(
            std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            oneapi::dpl::__ranges::all_view<std::int32_t, __par_backend_hetero::access_mode::read_write>(
                __mask_buf.get_buffer())),
        std::forward<_Range3>(__result), _GenReduceInput{_GenMaskReduce{__comp}}, _ReduceOp{},
        _GenScanInput{_GenMaskScan{_MaskPredicate{}, _MaskRangeTransform{}}, _ScanRangeTransform{}},
        _ScanInputTransform{}, _WriteOp{}, oneapi::dpl::unseq_backend::__no_init_value<_Size>{},
        /*_Inclusive=*/std::true_type{}, /*__is_unique_pattern=*/std::false_type{});
}

// balanced path
template <typename _CustomName, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          typename _SetTag>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range3>>>
__parallel_set_reduce_then_scan(oneapi::dpl::__internal::__device_backend_tag __backend_tag, sycl::queue& __q,
                                _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result, _Compare __comp, _SetTag)
{
    constexpr std::uint16_t __diagonal_spacing = 32;

    using _SetOperation = __get_set_operation<_SetTag>;
    using _In1ValueT = oneapi::dpl::__internal::__value_t<_Range1>;
    using _In2ValueT = oneapi::dpl::__internal::__value_t<_Range2>;
    using _OutValueT = oneapi::dpl::__internal::__value_t<_Range3>;
    using _TempData = __temp_data_array<__diagonal_spacing, _OutValueT>;
    using _Size = oneapi::dpl::__internal::__difference_t<_Range3>;
    using _ReduceOp = std::plus<_Size>;

    using _GenReduceInput = oneapi::dpl::__par_backend_hetero::__gen_set_balanced_path<_SetOperation, _Compare>;
    using _GenScanInput =
        oneapi::dpl::__par_backend_hetero::__gen_set_op_from_known_balanced_path<_SetOperation, _TempData, _Compare>;
    using _ScanInputTransform = oneapi::dpl::__par_backend_hetero::__get_zeroth_element;
    using _WriteOp = oneapi::dpl::__par_backend_hetero::__write_multiple_to_id<oneapi::dpl::__internal::__pstl_assign>;

    const std::int32_t __num_diagonals =
        oneapi::dpl::__internal::__dpl_ceiling_div(__rng1.size() + __rng2.size(), __diagonal_spacing);

    // Should be safe to use the type of the range size as the temporary type. Diagonal index will fit in the positive
    // portion of the range so star flag can use sign bit.
    using _TemporaryType = decltype(__rng1.size());
    //TODO: limit to diagonals per block, and only write to a block based index of temporary data
    oneapi::dpl::__par_backend_hetero::__buffer<_TemporaryType> __temp_diags(__num_diagonals);

    constexpr std::uint32_t __bytes_per_work_item_iter =
        ((sizeof(_In1ValueT) + sizeof(_In2ValueT)) / 2) * (__diagonal_spacing + 1) + sizeof(_TemporaryType);
    return __parallel_transform_reduce_then_scan<__bytes_per_work_item_iter, _CustomName>(
        __backend_tag, __q, __num_diagonals,
        oneapi::dpl::__ranges::make_zip_view(
            std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            oneapi::dpl::__ranges::all_view<_TemporaryType, __par_backend_hetero::access_mode::read_write>(
                __temp_diags.get_buffer())),
        std::forward<_Range3>(__result), _GenReduceInput{_SetOperation{}, __diagonal_spacing, __comp}, _ReduceOp{},
        _GenScanInput{_SetOperation{}, __diagonal_spacing, __comp}, _ScanInputTransform{}, _WriteOp{},
        oneapi::dpl::unseq_backend::__no_init_value<_Size>{}, /*_Inclusive=*/std::true_type{},
        /*__is_unique_pattern=*/std::false_type{});
}

template <typename _CustomName, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          typename _IsOpDifference>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range1>>>
__parallel_set_scan(oneapi::dpl::__internal::__device_backend_tag __backend_tag, sycl::queue& __q, _Range1&& __rng1,
                    _Range2&& __rng2, _Range3&& __result, _Compare __comp, _IsOpDifference)
{
    using _Size1 = oneapi::dpl::__internal::__difference_t<_Range1>;
    using _Size2 = oneapi::dpl::__internal::__difference_t<_Range2>;

    _Size1 __n1 = __rng1.size();
    _Size2 __n2 = __rng2.size();

    //Algo is based on the recommended approach of set_intersection algo for GPU: binary search + scan (copying by mask).
    using _ReduceOp = std::plus<_Size1>;
    using _Assigner = unseq_backend::__scan_assigner;
    using _NoAssign = unseq_backend::__scan_no_assign;
    using _MaskAssigner = unseq_backend::__mask_assigner<2>;
    using _InitType = unseq_backend::__no_init_value<_Size1>;
    using _DataAcc = unseq_backend::walk_n<oneapi::dpl::__internal::__no_op>;

    _ReduceOp __reduce_op;
    _Assigner __assign_op;
    _DataAcc __get_data_op;
    unseq_backend::__copy_by_mask<_ReduceOp, oneapi::dpl::__internal::__pstl_assign, /*inclusive*/ std::true_type, 2>
        __copy_by_mask_op;
    unseq_backend::__brick_set_op<_Compare, _Size1, _Size2, _IsOpDifference> __create_mask_op{__comp, __n1, __n2};

    // temporary buffer to store boolean mask
    oneapi::dpl::__par_backend_hetero::__buffer<int32_t> __mask_buf(__n1);

    return __par_backend_hetero::__parallel_transform_scan_base<_CustomName>(
        __backend_tag, __q,
        oneapi::dpl::__ranges::make_zip_view(
            std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2),
            oneapi::dpl::__ranges::all_view<int32_t, __par_backend_hetero::access_mode::read_write>(
                __mask_buf.get_buffer())),
        std::forward<_Range3>(__result), _InitType{},
        // local scan
        unseq_backend::__scan</*inclusive*/ std::true_type, _ReduceOp, _DataAcc, _Assigner, _MaskAssigner,
                              decltype(__create_mask_op), _InitType>{__reduce_op, __get_data_op, __assign_op,
                                                                     _MaskAssigner{}, __create_mask_op},
        // scan between groups
        unseq_backend::__scan</*inclusive=*/std::true_type, _ReduceOp, _DataAcc, _NoAssign, _Assigner, _DataAcc,
                              _InitType>{__reduce_op, __get_data_op, _NoAssign{}, __assign_op, __get_data_op},
        // global scan
        __copy_by_mask_op);
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Compare,
          typename _SetTag>
__future<sycl::event, __result_and_scratch_storage<oneapi::dpl::__internal::__difference_t<_Range3>>>
__parallel_set_op(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                  _Range1&& __rng1, _Range2&& __rng2, _Range3&& __result, _Compare __comp, _SetTag __set_tag)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    if constexpr (_SetTag::__can_write_from_rng2_v)
    {
        return __parallel_set_reduce_then_scan<_CustomName>(__backend_tag, __q_local, std::forward<_Range1>(__rng1),
                                                            std::forward<_Range2>(__rng2),
                                                            std::forward<_Range3>(__result), __comp, __set_tag);
    }
    else
    {
        return __parallel_set_scan<_CustomName>(__backend_tag, __q_local, std::forward<_Range1>(__rng1),
                                                std::forward<_Range2>(__rng2), std::forward<_Range3>(__result), __comp,
                                                __set_tag);
    }
}

template <typename _ExecutionPolicy>
bool
__can_set_op_write_from_set_b(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec)
{
    return oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__exec.queue());
}

//------------------------------------------------------------------------
// find_or tags
//------------------------------------------------------------------------

// Tag for __parallel_find_or to find the first element that satisfies predicate
template <typename... _Ranges>
struct __parallel_find_forward_tag
{
// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = uint32_t;
#else
    using _AtomicType = std::make_unsigned_t<std::common_type_t<oneapi::dpl::__internal::__difference_t<_Ranges>...>>;
#endif

    using _LocalResultsReduceOp = __dpl_sycl::__minimum<_AtomicType>;

    // The template parameter is intended to unify __init_value in tags.
    template <typename _SrcDataSize>
    constexpr static _AtomicType
    __init_value(_SrcDataSize __source_data_size)
    {
        return __source_data_size;
    }

    // As far as we make search from begin to the end of data, we should save the first (minimal) found state
    // in the __save_state_to (local state) / __save_state_to_atomic (global state) methods.

    template <sycl::access::address_space _Space>
    static void
    __save_state_to_atomic(__dpl_sycl::__atomic_ref<_AtomicType, _Space>& __atomic, _AtomicType __new_state)
    {
        __atomic.fetch_min(__new_state);
    }

    template <typename _TFoundState>
    static void
    __save_state_to(_TFoundState& __found, _AtomicType __new_state)
    {
        __found = std::min(__found, __new_state);
    }
};

// Tag for __parallel_find_or to find the last element that satisfies predicate
template <typename _RangeType>
struct __parallel_find_backward_tag
{
// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = int32_t;
#else
    using _AtomicType = oneapi::dpl::__internal::__difference_t<_RangeType>;
#endif

    using _LocalResultsReduceOp = __dpl_sycl::__maximum<_AtomicType>;

    template <typename _SrcDataSize>
    constexpr static _AtomicType
    __init_value(_SrcDataSize /*__source_data_size*/)
    {
        return _AtomicType{-1};
    }

    // As far as we make search from end to the begin of data, we should save the last (maximal) found state
    // in the __save_state_to (local state) / __save_state_to_atomic (global state) methods.

    template <sycl::access::address_space _Space>
    static void
    __save_state_to_atomic(__dpl_sycl::__atomic_ref<_AtomicType, _Space>& __atomic, _AtomicType __new_state)
    {
        __atomic.fetch_max(__new_state);
    }

    template <typename _TFoundState>
    static void
    __save_state_to(_TFoundState& __found, _AtomicType __new_state)
    {
        __found = std::max(__found, __new_state);
    }
};

// Tag for __parallel_find_or for or-semantic
struct __parallel_or_tag
{
    using _AtomicType = int32_t;

    // The template parameter is intended to unify __init_value in tags.
    template <typename _SrcDataSize>
    constexpr static _AtomicType
    __init_value(_SrcDataSize /*__source_data_size*/)
    {
        return 0;
    }

    // Store that a match was found. Its position is not relevant for or semantics
    // in the __save_state_to (local state) / __save_state_to_atomic (global state) methods.
    static constexpr _AtomicType __found_state = 1;

    template <sycl::access::address_space _Space>
    static void
    __save_state_to_atomic(__dpl_sycl::__atomic_ref<_AtomicType, _Space>& __atomic, _AtomicType /*__new_state*/)
    {
        __atomic.store(__found_state);
    }

    template <typename _TFoundState>
    static void
    __save_state_to(_TFoundState& __found, _AtomicType /*__new_state*/)
    {
        __found = __found_state;
    }
};

template <typename _RangeType>
constexpr bool
__is_backward_tag(__parallel_find_backward_tag<_RangeType>)
{
    return true;
}

template <typename _TagType>
constexpr bool
__is_backward_tag(_TagType)
{
    return false;
}

//------------------------------------------------------------------------
// early_exit (find_or)
//------------------------------------------------------------------------

template <typename _Pred>
struct __early_exit_find_or
{
    _Pred __pred;

    template <typename _NDItemId, typename _SrcDataSize, typename _IterationDataSize, typename _LocalFoundState,
              typename _BrickTag, typename... _Ranges>
    void
    operator()(const _NDItemId __item_id, const _SrcDataSize __source_data_size,
               const std::size_t __iters_per_work_item, const _IterationDataSize __iteration_data_size,
               _LocalFoundState& __found_local, _BrickTag __brick_tag, _Ranges&&... __rngs) const
    {
        // Return the index of this item in the kernel's execution range
        const auto __global_id = __item_id.get_global_linear_id();

        bool __something_was_found = false;
        for (_SrcDataSize __i = 0; !__something_was_found && __i < __iters_per_work_item; ++__i)
        {
            auto __local_src_data_idx = __i;
            if constexpr (__is_backward_tag(__brick_tag))
                __local_src_data_idx = __iters_per_work_item - 1 - __i;

            const auto __src_data_idx_current = __global_id + __local_src_data_idx * __iteration_data_size;
            if (__src_data_idx_current < __source_data_size && __pred(__src_data_idx_current, __rngs...))
            {
                // Update local found state
                _BrickTag::__save_state_to(__found_local, __src_data_idx_current);

                // This break is mandatory from the performance point of view.
                // This break is safe for all our cases:
                // 1) __parallel_find_forward_tag : when we search for the first matching data entry, we process data from start to end (forward direction).
                //    This means that after first found entry there is no reason to process data anymore.
                // 2) __parallel_find_backward_tag : when we search for the last matching data entry, we process data from end to start (backward direction).
                //    This means that after the first found entry there is no reason to process data anymore too.
                // 3) __parallel_or_tag : when we search for any matching data entry, we process data from start to end (forward direction).
                //    This means that after the first found entry there is no reason to process data anymore too.
                // But break statement here shows poor perf in some cases.
                // So we use bool variable state check in the for-loop header.
                __something_was_found = true;
            }

            // Share found into state between items in our sub-group to early exit if something was found
            //  - the update of __found_local state isn't required here because it updates later on the caller side
            __something_was_found = __dpl_sycl::__any_of_group(__item_id.get_sub_group(), __something_was_found);
        }
    }
};

//------------------------------------------------------------------------
// parallel_find_or - sync pattern
//------------------------------------------------------------------------

template <typename Tag>
struct __parallel_find_or_nd_range_tuner
{
    // Tune the amount of work-groups and work-group size
    std::tuple<std::size_t, std::size_t>
    operator()(const sycl::queue& __q, const std::size_t __rng_n) const
    {
        // TODO: find a way to generalize getting of reliable work-group size
        // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
        // This value exceeds the current practical limit for GPUs, but may need to be re-evaluated in the future.
        const std::size_t __wgroup_size = oneapi::dpl::__internal::__max_work_group_size(__q, (std::size_t)4096);
        std::size_t __n_groups = 1;
        // If no more than 32 data elements per work item, a single work group will be used
        if (__rng_n > __wgroup_size * 32)
        {
            // Compute the number of groups and limit by the number of compute units
            __n_groups = std::min<std::size_t>(oneapi::dpl::__internal::__dpl_ceiling_div(__rng_n, __wgroup_size),
                                               oneapi::dpl::__internal::__max_compute_units(__q));
        }

        return {__n_groups, __wgroup_size};
    }
};

// No tuning for FPGA_EMU because we are not going to tune here the performance for FPGA emulation.
#if !_ONEDPL_FPGA_EMU
template <>
struct __parallel_find_or_nd_range_tuner<oneapi::dpl::__internal::__device_backend_tag>
{
    // Tune the amount of work-groups and work-group size
    std::tuple<std::size_t, std::size_t>
    operator()(const sycl::queue& __q, const std::size_t __rng_n) const
    {
        // Call common tuning function to get the work-group size
        auto [__n_groups, __wgroup_size] = __parallel_find_or_nd_range_tuner<int>{}(__q, __rng_n);

        if (__n_groups > 1)
        {
            auto __iters_per_work_item =
                oneapi::dpl::__internal::__dpl_ceiling_div(__rng_n, __n_groups * __wgroup_size);

            // If our work capacity is not enough to process all data in one iteration, will tune the number of work-groups
            if (__iters_per_work_item > 1)
            {
                // Empirically found formula for GPU devices.
                // TODO : need to re-evaluate this formula.
                const float __rng_x = (float)__rng_n / 4096.f;
                const float __desired_iters_per_work_item = std::max(std::sqrt(__rng_x), 1.f);

                if (__iters_per_work_item < __desired_iters_per_work_item)
                {
                    // Multiply work per item by a power of 2 to reach the desired number of iterations.
                    // __dpl_bit_ceil rounds the ratio up to the next power of 2.
                    const std::size_t __k = oneapi::dpl::__internal::__dpl_bit_ceil(
                        (std::size_t)std::ceil(__desired_iters_per_work_item / __iters_per_work_item));
                    // Proportionally reduce the number of work groups.
                    __n_groups = oneapi::dpl::__internal::__dpl_ceiling_div(
                        __rng_n, __wgroup_size * __iters_per_work_item * __k);
                }
            }
        }

        return {__n_groups, __wgroup_size};
    }
};
#endif // !_ONEDPL_FPGA_EMU

template <bool __or_tag_check, typename KernelName>
struct __parallel_find_or_impl_one_wg;

// Base pattern for __parallel_or and __parallel_find. The execution depends on tag type _BrickTag.
template <bool __or_tag_check, typename... KernelName>
struct __parallel_find_or_impl_one_wg<__or_tag_check, __internal::__optional_kernel_name<KernelName...>>
{
    template <typename _BrickTag, typename __FoundStateType, typename _Predicate, typename... _Ranges>
    __FoundStateType
    operator()(oneapi::dpl::__internal::__device_backend_tag, sycl::queue& __q, _BrickTag __brick_tag,
               const std::size_t __rng_n, const std::size_t __wgroup_size, const __FoundStateType __init_value,
               _Predicate __pred, _Ranges&&... __rngs)
    {
        using __result_and_scratch_storage_t = __result_and_scratch_storage<__FoundStateType>;
        __result_and_scratch_storage_t __result_storage{__q, 0};

        // Calculate the number of elements to be processed by each work-item.
        const auto __iters_per_work_item = oneapi::dpl::__internal::__dpl_ceiling_div(__rng_n, __wgroup_size);

        // main parallel_for
        auto __event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
            auto __result_acc =
                __result_storage.template __get_result_acc<sycl::access_mode::write>(__cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<KernelName...>(
                sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__wgroup_size), sycl::range</*dim=*/1>(__wgroup_size)),
                [=](sycl::nd_item</*dim=*/1> __item_id) {
                    auto __local_idx = __item_id.get_local_id(0);

                    // 1. Set initial value to local found state
                    __FoundStateType __found_local = __init_value;

                    // 2. Find any element that satisfies pred
                    //  - after this call __found_local may still have initial value:
                    //    1) if no element satisfies pred;
                    //    2) early exit from sub-group occurred: in this case the state of __found_local will updated in the next group operation (3)
                    __pred(__item_id, __rng_n, __iters_per_work_item, __wgroup_size, __found_local, __brick_tag,
                           __rngs...);

                    // 3. Reduce over group: find __dpl_sycl::__minimum (for the __parallel_find_forward_tag),
                    // find __dpl_sycl::__maximum (for the __parallel_find_backward_tag)
                    // or update state with __dpl_sycl::__any_of_group (for the __parallel_or_tag)
                    // inside all our group items
                    if constexpr (__or_tag_check)
                        __found_local = __dpl_sycl::__any_of_group(__item_id.get_group(), __found_local);
                    else
                        __found_local = __dpl_sycl::__reduce_over_group(__item_id.get_group(), __found_local,
                                                                        typename _BrickTag::_LocalResultsReduceOp{});

                    // Set local found state value value to global state to have correct result
                    if (__local_idx == 0)
                    {
                        __result_and_scratch_storage_t::__get_usm_or_buffer_accessor_ptr(__result_acc)[0] =
                            __found_local;
                    }
                });
        });

        // Wait and return result
        return __result_storage.__wait_and_get_value(__event);
    }
};

template <bool __or_tag_check, typename KernelName>
struct __parallel_find_or_impl_multiple_wgs;

// Base pattern for __parallel_or and __parallel_find. The execution depends on tag type _BrickTag.
template <bool __or_tag_check, typename... KernelName>
struct __parallel_find_or_impl_multiple_wgs<__or_tag_check, __internal::__optional_kernel_name<KernelName...>>
{
    template <typename _BrickTag, typename _AtomicType, typename _Predicate, typename... _Ranges>
    _AtomicType
    operator()(oneapi::dpl::__internal::__device_backend_tag, sycl::queue& __q, _BrickTag __brick_tag,
               const std::size_t __rng_n, const std::size_t __n_groups, const std::size_t __wgroup_size,
               const _AtomicType __init_value, _Predicate __pred, _Ranges&&... __rngs)
    {
        auto __result = __init_value;

        // Calculate the number of elements to be processed by each work-item.
        const auto __iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__rng_n, __n_groups * __wgroup_size);

        // scope is to copy data back to __result after destruction of temporary sycl:buffer
        {
            sycl::buffer<_AtomicType, 1> __result_sycl_buf(&__result, 1); // temporary storage for global atomic

            // main parallel_for
            __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __rngs...);
                auto __result_sycl_buf_acc = __result_sycl_buf.template get_access<access_mode::read_write>(__cgh);

                __cgh.parallel_for<KernelName...>(
                    sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                              sycl::range</*dim=*/1>(__wgroup_size)),
                    [=](sycl::nd_item</*dim=*/1> __item_id) {
                        auto __local_idx = __item_id.get_local_id(0);

                        // 1. Set initial value to local found state
                        _AtomicType __found_local = __init_value;

                        // 2. Find any element that satisfies pred
                        //  - after this call __found_local may still have initial value:
                        //    1) if no element satisfies pred;
                        //    2) early exit from sub-group occurred: in this case the state of __found_local will updated in the next group operation (3)
                        __pred(__item_id, __rng_n, __iters_per_work_item, __n_groups * __wgroup_size, __found_local,
                               __brick_tag, __rngs...);

                        // 3. Reduce over group: find __dpl_sycl::__minimum (for the __parallel_find_forward_tag),
                        // find __dpl_sycl::__maximum (for the __parallel_find_backward_tag)
                        // or update state with __dpl_sycl::__any_of_group (for the __parallel_or_tag)
                        // inside all our group items
                        if constexpr (__or_tag_check)
                            __found_local = __dpl_sycl::__any_of_group(__item_id.get_group(), __found_local);
                        else
                            __found_local = __dpl_sycl::__reduce_over_group(
                                __item_id.get_group(), __found_local, typename _BrickTag::_LocalResultsReduceOp{});

                        // Set local found state value value to global atomic
                        if (__local_idx == 0 && __found_local != __init_value)
                        {
                            __dpl_sycl::__atomic_ref<_AtomicType, sycl::access::address_space::global_space> __found(
                                *__dpl_sycl::__get_accessor_ptr(__result_sycl_buf_acc));

                            // Update global (for all groups) atomic state with the found index
                            _BrickTag::__save_state_to_atomic(__found, __found_local);
                        }
                    });
            });
            //The end of the scope  -  a point of synchronization (on temporary sycl buffer destruction)
        }

        return __result;
    }
};

template <typename... _Ranges>
struct __first_size_calc
{
    auto
    operator()(const _Ranges&... __rngs) const
    {
        return oneapi::dpl::__ranges::__get_first_range_size(__rngs...);
    }
};

template <typename... _Ranges>
struct __min_size_calc
{
    auto
    operator()(const _Ranges&... __rngs) const
    {
        using _Size = std::make_unsigned_t<std::common_type_t<oneapi::dpl::__internal::__difference_t<_Ranges>...>>;
        return std::min({_Size(__rngs.size())...});
    }
};

// Base pattern for __parallel_or and __parallel_find. The execution depends on tag type _BrickTag.
template <typename _ExecutionPolicy, typename _Brick, typename _BrickTag, typename _SizeCalc, typename... _Ranges>
std::conditional_t<
    ::std::is_same_v<_BrickTag, __parallel_or_tag>, bool,
    oneapi::dpl::__internal::__difference_t<typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>
__parallel_find_or_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Brick __f,
                        _BrickTag __brick_tag, _SizeCalc __sz_calc, _Ranges&&... __rngs)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    const auto __rng_n = __sz_calc(__rngs...);

    assert(__rng_n > 0);

    // Evaluate the amount of work-groups and work-group size
    const auto [__n_groups, __wgroup_size] =
        __parallel_find_or_nd_range_tuner<oneapi::dpl::__internal::__device_backend_tag>{}(__q_local, __rng_n);

    _PRINT_INFO_IN_DEBUG_MODE(__q_local, __wgroup_size);

    using _AtomicType = typename _BrickTag::_AtomicType;
    const _AtomicType __init_value = _BrickTag::__init_value(__rng_n);
    const auto __pred = oneapi::dpl::__par_backend_hetero::__early_exit_find_or<_Brick>{__f};

    constexpr bool __or_tag_check = std::is_same_v<_BrickTag, __parallel_or_tag>;

    _AtomicType __result;
    if (__n_groups == 1)
    {
        // We shouldn't have any restrictions for _AtomicType type here
        // because we have a single work-group and we don't need to use atomics for inter-work-group communication.

        using __find_or_one_wg_kernel_name =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__find_or_kernel_one_wg<_CustomName>>;

        // Single WG implementation
        __result = __parallel_find_or_impl_one_wg<__or_tag_check, __find_or_one_wg_kernel_name>()(
            oneapi::dpl::__internal::__device_backend_tag{}, __q_local, __brick_tag, __rng_n, __wgroup_size,
            __init_value, __pred, std::forward<_Ranges>(__rngs)...);
    }
    else
    {
        assert("This device does not support 64-bit atomics" &&
               (sizeof(_AtomicType) < 8 || __q_local.get_device().has(sycl::aspect::atomic64)));

        using __find_or_kernel_name =
            oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__find_or_kernel<_CustomName>>;

        // Multiple WG implementation
        __result = __parallel_find_or_impl_multiple_wgs<__or_tag_check, __find_or_kernel_name>()(
            oneapi::dpl::__internal::__device_backend_tag{}, __q_local, __brick_tag, __rng_n, __n_groups, __wgroup_size,
            __init_value, __pred, std::forward<_Ranges>(__rngs)...);
    }

    if constexpr (__or_tag_check)
        return __result != __init_value;
    else
        return __result != __init_value ? __result : __rng_n;
}

template <typename _ExecutionPolicy, typename _Brick, typename _BrickTag, typename... _Ranges>
std::conditional_t<
    ::std::is_same_v<_BrickTag, __parallel_or_tag>, bool,
    oneapi::dpl::__internal::__difference_t<typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>
__parallel_find_or(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Brick __f,
                   _BrickTag __brick_tag, _Ranges&&... __rngs)
{
    return __parallel_find_or_impl(oneapi::dpl::__internal::__device_backend_tag{},
        std::forward<_ExecutionPolicy>(__exec), __f, __brick_tag, __first_size_calc<_Ranges...>{},
        std::forward<_Ranges>(__rngs)...);
}

template <typename _ExecutionPolicy, typename _Brick, typename _BrickTag, typename... _Ranges>
std::conditional_t<
    ::std::is_same_v<_BrickTag, __parallel_or_tag>, bool,
    oneapi::dpl::__internal::__difference_t<typename oneapi::dpl::__ranges::__get_first_range_type<_Ranges...>::type>>
__parallel_find_or_min_size(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Brick __f,
                            _BrickTag __brick_tag, _Ranges&&... __rngs)
{
    return __parallel_find_or_impl(oneapi::dpl::__internal::__device_backend_tag{},
        std::forward<_ExecutionPolicy>(__exec), __f, __brick_tag, __min_size_calc<_Ranges...>{},
        std::forward<_Ranges>(__rngs)...);
}

// parallel_or - sync pattern
//------------------------------------------------------------------------

template <typename Name>
class __or_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick>
bool
__parallel_or(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
              _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first, _Iterator2 __s_last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __backend_tag,
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __f,
        __parallel_or_tag{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick>
bool
__parallel_or(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec, _Iterator __first,
              _Iterator __last, _Brick __f)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        __backend_tag,
        __par_backend_hetero::make_wrapped_policy<__or_policy_wrapper>(::std::forward<_ExecutionPolicy>(__exec)), __f,
        __parallel_or_tag{}, __buf.all_view());
}

//------------------------------------------------------------------------
// parallel_find - sync pattern
//-----------------------------------------------------------------------

template <typename Name>
class __find_policy_wrapper
{
};

template <typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Brick, typename _IsFirst>
_Iterator1
__parallel_find(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                _Iterator1 __first, _Iterator1 __last, _Iterator2 __s_first, _Iterator2 __s_last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf = __keep(__first, __last);
    auto __s_keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __s_buf = __s_keep(__s_first, __s_last);

    using _TagType = ::std::conditional_t<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                          __parallel_find_backward_tag<decltype(__buf.all_view())>>;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __backend_tag,
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view(), __s_buf.all_view());
}

// Special overload for single sequence cases.
// TODO: check if similar pattern may apply to other algorithms. If so, these overloads should be moved out of
// backend code.
template <typename _ExecutionPolicy, typename _Iterator, typename _Brick, typename _IsFirst>
_Iterator
__parallel_find(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                _Iterator __first, _Iterator __last, _Brick __f, _IsFirst)
{
    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    using _TagType = ::std::conditional_t<_IsFirst::value, __parallel_find_forward_tag<decltype(__buf.all_view())>,
                                          __parallel_find_backward_tag<decltype(__buf.all_view())>>;
    return __first + oneapi::dpl::__par_backend_hetero::__parallel_find_or(
                         __backend_tag,
                         __par_backend_hetero::make_wrapped_policy<__find_policy_wrapper>(
                             ::std::forward<_ExecutionPolicy>(__exec)),
                         __f, _TagType{}, __buf.all_view());
}

//------------------------------------------------------------------------
// parallel_merge - async pattern
//-----------------------------------------------------------------------

// Partial merge implementation with O(log(k)) per routine complexity.
// Note: the routine assumes that the 2nd sequence goes after the first one, meaning that end_1 == start_2.
//
// The picture below shows how the merge is performed:
//
// input:
//    start_1     part_end_1   end_1  start_2     part_end_2   end_2
//      |_____________|_________|       |_____________|_________|
//      |______p1_____|___p2____|       |_____p3______|___p4____|
//
// Usual merge is performed on p1 and p3, the result is written to the beginning of the buffer.
// p2 and p4 are just copied to the then of the buffer as pictured below:
//
//    start_3
//      |_____________________________ __________________
//      |______sorted p1 and p3_______|____p2___|___p4___|
//
// Only first k elements from sorted p1 and p3 are guaranteed to be less than(or according to __comp) elements
// from p2 and p4. And these k elements are the only ones we care about.
template <typename _Ksize>
struct __partial_merge_kernel
{
    const _Ksize __k;
    template <typename _Idx, typename _Acc1, typename _Size1, typename _Acc2, typename _Size2, typename _Acc3,
              typename _Size3, typename _Compare>
    void
    operator()(_Idx __global_idx, const _Acc1& __in_acc1, _Size1 __start_1, _Size1 __end_1, const _Acc2& __in_acc2,
               _Size2 __start_2, _Size2 __end_2, const _Acc3& __out_acc, _Size3 __out_shift, _Compare __comp) const
    {
        const auto __part_end_1 = sycl::min(__start_1 + __k, __end_1);
        const auto __part_end_2 = sycl::min(__start_2 + __k, __end_2);

        // Handle elements from p1
        if (__global_idx >= __start_1 && __global_idx < __part_end_1)
        {
            const auto __shift =
                /* index inside p1 */ __global_idx - __start_1 +
                /* relative position in p3 */
                oneapi::dpl::__internal::__pstl_lower_bound(__in_acc2, __start_2, __part_end_2, __in_acc1[__global_idx],
                                                            __comp) -
                __start_2;
            __out_acc[__out_shift + __shift] = __in_acc1[__global_idx];
        }
        // Handle elements from p2
        else if (__global_idx >= __part_end_1 && __global_idx < __end_1)
        {
            const auto __shift =
                /* index inside p2 */ (__global_idx - __part_end_1) +
                /* size of p1 + size of p3 */ (__part_end_1 - __start_1) + (__part_end_2 - __start_2);
            __out_acc[__out_shift + __shift] = __in_acc1[__global_idx];
        }
        // Handle elements from p3
        else if (__global_idx >= __start_2 && __global_idx < __part_end_2)
        {
            const auto __shift =
                /* index inside p3 */ __global_idx - __start_2 +
                /* relative position in p1 */
                oneapi::dpl::__internal::__pstl_upper_bound(__in_acc1, __start_1, __part_end_1, __in_acc2[__global_idx],
                                                            __comp) -
                __start_1;
            __out_acc[__out_shift + __shift] = __in_acc2[__global_idx];
        }
        // Handle elements from p4
        else if (__global_idx >= __part_end_2 && __global_idx < __end_2)
        {
            const auto __shift =
                /* index inside p4 + size of p3 */ __global_idx - __start_2 +
                /* size of p1, p2 */ __end_1 - __start_1;
            __out_acc[__out_shift + __shift] = __in_acc2[__global_idx];
        }
    }
};

// Please see the comment above __parallel_for_small_submitter for optional kernel name explanation
template <typename _GlobalSortName, typename _CopyBackName>
struct __parallel_partial_sort_submitter;

template <typename... _GlobalSortName, typename... _CopyBackName>
struct __parallel_partial_sort_submitter<__internal::__optional_kernel_name<_GlobalSortName...>,
                                         __internal::__optional_kernel_name<_CopyBackName...>>
{
    template <typename _BackendTag, typename _Range, typename _Merge, typename _Compare>
    __future<sycl::event>
    operator()(_BackendTag, sycl::queue& __q, _Range&& __rng, _Merge __merge, _Compare __comp) const
    {
        using _Tp = oneapi::dpl::__internal::__value_t<_Range>;
        using _Size = oneapi::dpl::__internal::__difference_t<_Range>;

        _Size __n = __rng.size();
        assert(__n > 1);

        oneapi::dpl::__par_backend_hetero::__buffer<_Tp> __temp_buf(__n);
        auto __temp = __temp_buf.get_buffer();
        _PRINT_INFO_IN_DEBUG_MODE(__q);

        _Size __k = 1;
        bool __data_in_temp = false;
        sycl::event __event1;
        do
        {
            __event1 = __q.submit([&, __data_in_temp, __k](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read_write>(__cgh);
                __cgh.parallel_for<_GlobalSortName...>(
                    sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                        auto __global_idx = __item_id.get_linear_id();

                        _Size __start = 2 * __k * (__global_idx / (2 * __k));
                        _Size __end_1 = sycl::min(__start + __k, __n);
                        _Size __end_2 = sycl::min(__start + 2 * __k, __n);

                        if (!__data_in_temp)
                        {
                            __merge(__global_idx, __rng, __start, __end_1, __rng, __end_1, __end_2, __temp_acc, __start,
                                    __comp);
                        }
                        else
                        {
                            __merge(__global_idx, __temp_acc, __start, __end_1, __temp_acc, __end_1, __end_2, __rng,
                                    __start, __comp);
                        }
                    });
            });
            __data_in_temp = !__data_in_temp;
            __k *= 2;
        } while (__k < __n);

        // if results are in temporary buffer then copy back those
        if (__data_in_temp)
        {
            __event1 = __q.submit([&](sycl::handler& __cgh) {
                __cgh.depends_on(__event1);
                oneapi::dpl::__ranges::__require_access(__cgh, __rng);
                auto __temp_acc = __temp.template get_access<access_mode::read>(__cgh);
                // we cannot use __cgh.copy here because of zip_iterator usage
                __cgh.parallel_for<_CopyBackName...>(sycl::range</*dim=*/1>(__n), [=](sycl::item</*dim=*/1> __item_id) {
                    __rng[__item_id.get_linear_id()] = __temp_acc[__item_id];
                });
            });
        }
        // return future and extend lifetime of temporary buffer
        return __future{std::move(__event1)};
    }
};

template <typename... _Name>
class __sort_global_kernel;

template <typename _ExecutionPolicy, typename _Range, typename _Merge, typename _Compare>
__future<sycl::event>
__parallel_partial_sort_impl(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __rng,
                             _Merge __merge, _Compare __comp)
{
    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    using _GlobalSortKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_global_kernel<_CustomName>>;
    using _CopyBackKernel =
        oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<__sort_copy_back_kernel<_CustomName>>;

    sycl::queue __q_local = __exec.queue();

    return __parallel_partial_sort_submitter<_GlobalSortKernel, _CopyBackKernel>()(
        oneapi::dpl::__internal::__device_backend_tag{}, __q_local, ::std::forward<_Range>(__rng), __merge, __comp);
}

//------------------------------------------------------------------------
// parallel_stable_sort - async pattern
//-----------------------------------------------------------------------

template <typename _T, typename _Compare>
struct __is_radix_sort_usable_for_type
{
    static constexpr bool value =
#if _ONEDPL_USE_RADIX_SORT
        (::std::is_arithmetic_v<_T> || ::std::is_same_v<sycl::half, _T>) &&
            (__internal::__is_comp_ascending<::std::decay_t<_Compare>>::value ||
            __internal::__is_comp_descending<::std::decay_t<_Compare>>::value);
#else
        false;
#endif // _ONEDPL_USE_RADIX_SORT
};

#if _ONEDPL_USE_RADIX_SORT
template <
    typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    ::std::enable_if_t<
        __is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
__future<sycl::event>
__parallel_stable_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                       _Range&& __rng, _Compare, _Proj __proj)
{
    return __parallel_radix_sort<__internal::__is_comp_ascending<::std::decay_t<_Compare>>::value>(
        __backend_tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng), __proj);
}
#endif // _ONEDPL_USE_RADIX_SORT

template <
    typename _ExecutionPolicy, typename _Range, typename _Compare, typename _Proj,
    ::std::enable_if_t<
        !__is_radix_sort_usable_for_type<oneapi::dpl::__internal::__key_t<_Proj, _Range>, _Compare>::value, int> = 0>
__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__parallel_stable_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                       _Range&& __rng, _Compare __comp, _Proj __proj)
{
    return __parallel_sort_impl(__backend_tag, ::std::forward<_ExecutionPolicy>(__exec), ::std::forward<_Range>(__rng),
                                oneapi::dpl::__internal::__compare<_Compare, _Proj>{__comp, __proj});
}

//------------------------------------------------------------------------
// parallel_partial_sort - async pattern
//-----------------------------------------------------------------------

// TODO: check if it makes sense to move these wrappers out of backend to a common place
// TODO: consider changing __partial_merge_kernel to make it compatible with
//       __full_merge_kernel in order to use __parallel_sort_impl routine
template <typename _ExecutionPolicy, typename _Iterator, typename _Compare>
__future<sycl::event>
__parallel_partial_sort(oneapi::dpl::__internal::__device_backend_tag __backend_tag, _ExecutionPolicy&& __exec,
                        _Iterator __first, _Iterator __mid, _Iterator __last, _Compare __comp)
{
    const auto __mid_idx = __mid - __first;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    return __parallel_partial_sort_impl(__backend_tag, ::std::forward<_ExecutionPolicy>(__exec), __buf.all_view(),
                                        __partial_merge_kernel<decltype(__mid_idx)>{__mid_idx}, __comp);
}

//------------------------------------------------------------------------
// reduce_by_segment - sync pattern
//
// TODO: The non-identity fallback path of reduce-by-segment must currently be implemented synchronously due to the
// inability to create event dependency chains across separate parallel pattern calls. If we ever add support for
// cross parallel pattern dependencies, then we can implement this as an async pattern.
//------------------------------------------------------------------------
template <typename _Name>
struct __reduce1_wrapper;

template <typename _Name>
struct __reduce2_wrapper;

template <typename _Name>
struct __assign_key1_wrapper;

template <typename _Name>
struct __assign_key2_wrapper;

namespace __internal
{
template <typename _BinaryPredicate>
struct __parallel_reduce_by_segment_fallback_fn1
{
    _BinaryPredicate __binary_pred;
    std::size_t __wgroup_size;

    template <typename T>
    bool
    operator()(const T& __a) const
    {
        // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
        // for (i-1), but we still need to get its key value as it is the start of a segment
        const auto index = std::get<0>(__a);
        if (index == 0)
            return true;
        return index % __wgroup_size == 0                             // segment size
               || !__binary_pred(std::get<1>(__a), std::get<2>(__a)); // key comparison
    }
};

template <typename _BinaryPredicate>
struct __parallel_reduce_by_segment_fallback_fn2
{
    _BinaryPredicate __binary_pred;

    template <typename T>
    bool
    operator()(const T& __a) const
    {
        // The size of key range for the (i-1) view is one less, so for the 0th index we do not check the keys
        // for (i-1), but we still need to get its key value as it is the start of a segment
        if (std::get<0>(__a) == 0)
            return true;
        return !__binary_pred(std::get<1>(__a), std::get<2>(__a)); // keys comparison
    }
};
} // namespace __internal

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__parallel_reduce_by_segment_fallback(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec,
                                      _Range1&& __keys, _Range2&& __values, _Range3&& __out_keys,
                                      _Range4&& __out_values, _BinaryPredicate __binary_pred,
                                      _BinaryOperator __binary_op,
                                      /*known_identity=*/std::false_type)
{
    const auto __n = __keys.size();
    assert(__n > 0);

    using __diff_type = oneapi::dpl::__internal::__difference_t<_Range1>;
    using __key_type = oneapi::dpl::__internal::__value_t<_Range1>;
    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;

    sycl::queue __q_local = __exec.queue();

    // Round 1: reduce with extra indices added to avoid long segments
    // TODO: At threshold points check if the key is equal to the key at the previous threshold point, indicating a long sequence.
    // Skip a round of copy_if and reduces if there are none.
    auto __idx = oneapi::dpl::__par_backend_hetero::__buffer<__diff_type>(__n).get_buffer();
    auto __tmp_out_keys = oneapi::dpl::__par_backend_hetero::__buffer<__key_type>(__n).get_buffer();
    auto __tmp_out_values = oneapi::dpl::__par_backend_hetero::__buffer<__val_type>(__n).get_buffer();

    // Replicating first element of keys view to be able to compare (i-1)-th and (i)-th key with aligned sequences,
    //  dropping the last key for the i-1 sequence.
    auto __k1 =
        oneapi::dpl::__ranges::take_view_simple(oneapi::dpl::__ranges::replicate_start_view_simple(__keys, 1), __n);

    // view1 elements are a tuple of the element index and pairs of adjacent keys
    // view2 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view1 = oneapi::dpl::__ranges::zip_view(experimental::ranges::views::iota(0, __n), __k1, __keys);
    auto __view2 = oneapi::dpl::__ranges::zip_view(oneapi::dpl::__ranges::views::all_write(__tmp_out_keys),
                                                   oneapi::dpl::__ranges::views::all_write(__idx));

    // use work group size adjusted to shared local memory as the maximum segment size.
    std::size_t __wgroup_size =
        oneapi::dpl::__internal::__slm_adjusted_work_group_size(__q_local, sizeof(__key_type) + sizeof(__val_type));

    // element is copied if it is the 0th element (marks beginning of first segment), is in an index
    // evenly divisible by wg size (ensures segments are not long), or has a key not equal to the
    // adjacent element (marks end of real segments)
    // TODO: replace wgroup size with segment size based on platform specifics.
    auto __intermediate_result_end =
        oneapi::dpl::__par_backend_hetero::__parallel_copy_if(
            oneapi::dpl::__internal::__device_backend_tag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key1_wrapper>(__exec), __view1, __view2,
            __n, __internal::__parallel_reduce_by_segment_fallback_fn1<_BinaryPredicate>{__binary_pred, __wgroup_size},
            unseq_backend::__brick_assign_key_position{})
            .get();

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__internal::__device_backend_tag{},
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce1_wrapper>(__exec),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__n)>(__binary_op, __n), __intermediate_result_end,
        oneapi::dpl::__ranges::take_view_simple(oneapi::dpl::__ranges::views::all_read(__idx),
                                                __intermediate_result_end),
        std::forward<_Range2>(__values), oneapi::dpl::__ranges::views::all_write(__tmp_out_values))
        .wait();

    // Round 2: final reduction to get result for each segment of equal adjacent keys
    // create views over adjacent keys
    oneapi::dpl::__ranges::all_view<__key_type, __par_backend_hetero::access_mode::read_write> __new_keys(
        __tmp_out_keys);

    // Replicating first element of key views to be able to compare (i-1)-th and (i)-th key,
    //  dropping the last key for the i-1 sequence.  Only taking the appropriate number of keys to start with here.
    auto __clipped_new_keys = oneapi::dpl::__ranges::take_view_simple(__new_keys, __intermediate_result_end);

    auto __k3 = oneapi::dpl::__ranges::take_view_simple(
        oneapi::dpl::__ranges::replicate_start_view_simple(__clipped_new_keys, 1), __intermediate_result_end);

    // view3 elements are a tuple of the element index and pairs of adjacent keys
    // view4 elements are a tuple of the elements where key-index pairs will be written by copy_if
    auto __view3 = oneapi::dpl::__ranges::zip_view(experimental::ranges::views::iota(0, __intermediate_result_end),
                                                   __k3, __clipped_new_keys);
    auto __view4 = oneapi::dpl::__ranges::zip_view(oneapi::dpl::__ranges::views::all_write(__out_keys),
                                                   oneapi::dpl::__ranges::views::all_write(__idx));

    // element is copied if it is the 0th element (marks beginning of first segment), or has a key not equal to
    // the adjacent element (end of a segment). Artificial segments based on wg size are not created.
    auto __result_end =
        oneapi::dpl::__par_backend_hetero::__parallel_copy_if(
            oneapi::dpl::__internal::__device_backend_tag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__assign_key2_wrapper>(__exec), __view3, __view4,
            __view3.size(), __internal::__parallel_reduce_by_segment_fallback_fn2<_BinaryPredicate>{__binary_pred},
            unseq_backend::__brick_assign_key_position{})
            .get();

    //reduce by segment
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        oneapi::dpl::__internal::__device_backend_tag{},
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__reduce2_wrapper>(
            std::forward<_ExecutionPolicy>(__exec)),
        unseq_backend::__brick_reduce_idx<_BinaryOperator, decltype(__intermediate_result_end)>(
            __binary_op, __intermediate_result_end),
        __result_end,
        oneapi::dpl::__ranges::take_view_simple(oneapi::dpl::__ranges::views::all_read(__idx), __result_end),
        oneapi::dpl::__ranges::views::all_read(__tmp_out_values), std::forward<_Range4>(__out_values))
        .__checked_deferrable_wait();
    return __result_end;
}

template <typename _ExecutionPolicy, typename _Range1, typename _Range2, typename _Range3, typename _Range4,
          typename _BinaryPredicate, typename _BinaryOperator>
oneapi::dpl::__internal::__difference_t<_Range3>
__parallel_reduce_by_segment(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range1&& __keys,
                             _Range2&& __values, _Range3&& __out_keys, _Range4&& __out_values,
                             _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    // The algorithm reduces values in __values where the
    // associated keys for the values are equal to the adjacent key.
    //
    // Example: __keys       = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //          __values     = { 1, 2, 3, 4, 1, 1, 3, 3, 1, 1, 3, 3, 0 }
    //
    //          __out_keys   = { 1, 2, 3, 4, 1, 3, 1, 3, 0 }
    //          __out_values = { 1, 2, 3, 4, 2, 6, 2, 6, 0 }

    using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    sycl::queue __q_local = __exec.queue();

    using __val_type = oneapi::dpl::__internal::__value_t<_Range2>;
    // Prior to icpx 2025.0, the reduce-then-scan path performs poorly and should be avoided.
#if !defined(__INTEL_LLVM_COMPILER) || __INTEL_LLVM_COMPILER >= 20250000
    if constexpr (std::is_trivially_copyable_v<__val_type>)
    {
        if (oneapi::dpl::__par_backend_hetero::__is_gpu_with_reduce_then_scan_sg_sz(__q_local))
        {
            auto __res = oneapi::dpl::__par_backend_hetero::__parallel_reduce_by_segment_reduce_then_scan<_CustomName>(
                oneapi::dpl::__internal::__device_backend_tag{}, __q_local, std::forward<_Range1>(__keys),
                std::forward<_Range2>(__values), std::forward<_Range3>(__out_keys), std::forward<_Range4>(__out_values),
                __binary_pred, __binary_op);
            // Because our init type ends up being tuple<std::size_t, ValType>, return the first component which is the write index. Add 1 to return the
            // past-the-end iterator pair of segmented reduction.
            return std::get<0>(__res.get()) + 1;
        }
    }
#endif
    return __parallel_reduce_by_segment_fallback(
        oneapi::dpl::__internal::__device_backend_tag{}, std::forward<_ExecutionPolicy>(__exec),
        std::forward<_Range1>(__keys), std::forward<_Range2>(__values), std::forward<_Range3>(__out_keys),
        std::forward<_Range4>(__out_values), __binary_pred, __binary_op,
        oneapi::dpl::unseq_backend::__has_known_identity<_BinaryOperator, __val_type>{});
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_H
