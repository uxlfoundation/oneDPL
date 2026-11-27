// -*- C++ -*-
//===-- parallel_backend_sycl_utils.h -------------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_UTILS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_UTILS_H

//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL
#include <array>
#include <type_traits>
#include <algorithm> // std::min, std::find
#include <functional>
#include <utility>
#include <cstddef>
#include <cstdint> // std::uint8_t
#include <cassert>
#include <vector>

#include "sycl_defs.h"
#include "execution_sycl_defs.h"
#include "../../utils.h"
#include "utils_storage_sycl.h" // also includes dependencies for __future

#if _ONEDPL_DEBUG_SYCL
#    include <string>
#    include <iostream>
#endif

#define _PRINT_INFO_IN_DEBUG_MODE(...)                                                                                 \
    oneapi::dpl::__par_backend_hetero::__internal::__print_device_debug_info(__VA_ARGS__)

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//-----------------------------------------------------------------------------
// Device run-time information helpers
//-----------------------------------------------------------------------------

#if _ONEDPL_DEBUG_SYCL
inline std::string
__device_info(const sycl::queue& __q)
{
    return __q.get_device().template get_info<sycl::info::device::name>();
}
#endif

inline std::size_t
__max_work_group_size(const sycl::queue& __q, std::size_t __wg_size_limit = 8192)
{
    std::size_t __wg_size = __q.get_device().template get_info<sycl::info::device::max_work_group_size>();
    // Limit the maximum work-group size supported by the device to optimize the throughput or minimize communication
    // costs. This is limited to 8192 which is the highest current limit of the tested hardware (opencl:cpu devices) to
    // prevent huge work-group sizes returned on some devices (e.g., FPGU emulation).
    return std::min(__wg_size, __wg_size_limit);
}

template <typename _Size>
_Size
__slm_adjusted_work_group_size(const sycl::queue& __q, _Size __local_mem_per_wi, _Size __wg_size = 0)
{
    if (__wg_size == 0)
        __wg_size = __max_work_group_size(__q);
    auto __local_mem_size = __q.get_device().template get_info<sycl::info::device::local_mem_size>();
    return std::min<_Size>(__local_mem_size / __local_mem_per_wi, __wg_size);
}

#if _ONEDPL_USE_SUB_GROUPS
inline std::size_t
__max_sub_group_size(const sycl::queue& __q)
{
    auto __supported_sg_sizes = __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    //The result of get_info<sycl::info::device::sub_group_sizes>() can be empty; if so, return 0
    return __supported_sg_sizes.empty() ? 0 : __supported_sg_sizes.back();
}

inline std::size_t
__min_sub_group_size(const sycl::queue& __q)
{
    auto __supported_sg_sizes = __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    //The result of get_info<sycl::info::device::sub_group_sizes>() can be empty; if so, return 1
    return __supported_sg_sizes.empty() ? 1 : __supported_sg_sizes.front();
}
#endif // _ONEDPL_USE_SUB_GROUPS

inline std::uint32_t
__max_compute_units(const sycl::queue& __q)
{
    return __q.get_device().template get_info<sycl::info::device::max_compute_units>();
}

inline bool
__supports_sub_group_size(const sycl::queue& __q, std::size_t __target_size)
{
    const std::vector<std::size_t> __subgroup_sizes =
        __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    return std::find(__subgroup_sizes.begin(), __subgroup_sizes.end(), __target_size) != __subgroup_sizes.end();
}

inline bool
__has_device_scope_atomics([[maybe_unused]] sycl::device __device)
{
#if _ONEDPL_SYCL2020_ATOMIC_MEMORY_SCOPE_CAPABILITIES_BROKEN
    // AdaptiveCPP supports global atomics, but the check itself is broken
    // Enable them assuming almost all devices support global atomics
    return true;
#else
    const std::vector<sycl::memory_scope> __scopes =
        __device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
    return std::find(__scopes.begin(), __scopes.end(), sycl::memory_scope::device) != __scopes.end();
#endif
}

//-----------------------------------------------------------------------------
// Kernel run-time information helpers
//-----------------------------------------------------------------------------

inline std::size_t
__kernel_work_group_size(const sycl::queue& __q, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __q.get_device();
#if _ONEDPL_SYCL2020_KERNEL_DEVICE_API_PRESENT
    return __kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(__device);
#else
    return __kernel.template get_work_group_info<sycl::info::kernel_work_group::work_group_size>(__device);
#endif
}

inline std::uint32_t
__kernel_sub_group_size(const sycl::queue& __q, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __q.get_device();
    [[maybe_unused]] const ::std::size_t __wg_size = __kernel_work_group_size(__q, __kernel);
    const ::std::uint32_t __sg_size =
#if _ONEDPL_SYCL2020_KERNEL_DEVICE_API_PRESENT
        __kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
            __device
#    if _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000)
            ,
            sycl::range<3> { __wg_size, 1, 1 }
#    endif
        );
#else
        __kernel.template get_sub_group_info<sycl::info::kernel_sub_group::max_sub_group_size>(
            __device, sycl::range<3>{__wg_size, 1, 1});
#endif
    return __sg_size;
}
//-----------------------------------------------------------------------------

} // namespace __internal

namespace __par_backend_hetero
{

// aliases for faster access to modes
using access_mode = sycl::access_mode;

// function is needed to wrap kernel name into another policy class
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_device_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
{
    return oneapi::dpl::execution::make_device_policy<
        _NewKernelName<oneapi::dpl::__internal::__policy_kernel_name<_Policy>>>(::std::forward<_Policy>(__policy));
}

#if _ONEDPL_FPGA_DEVICE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
{
    return oneapi::dpl::execution::make_fpga_policy<
        oneapi::dpl::__internal::__policy_unroll_factor<_Policy>,
        _NewKernelName<oneapi::dpl::__internal::__policy_kernel_name<_Policy>>>(::std::forward<_Policy>(__policy));
}
#pragma GCC diagnostic pop
#endif

namespace __internal
{

//-----------------------------------------------------------------------
// Kernel name generation helpers
//-----------------------------------------------------------------------

// extract the deepest kernel name when we have a policy wrapper that might hide the default name
template <typename _CustomName>
struct _HasDefaultName
{
    static constexpr bool value = ::std::is_same_v<_CustomName, oneapi::dpl::execution::DefaultKernelName>
#if _ONEDPL_FPGA_DEVICE
                                  || ::std::is_same_v<_CustomName, oneapi::dpl::execution::DefaultKernelNameFPGA>
#endif
        ;
};

template <template <typename...> class _ExternalName, typename... _InternalName>
struct _HasDefaultName<_ExternalName<_InternalName...>>
{
    static constexpr bool value = (... || _HasDefaultName<_InternalName>::value);
};

template <typename... _Name>
struct __optional_kernel_name;

template <typename _CustomName>
using __kernel_name_provider =
#if __SYCL_UNNAMED_LAMBDA__
    ::std::conditional_t<_HasDefaultName<_CustomName>::value, __optional_kernel_name<>,
                         __optional_kernel_name<_CustomName>>;
#else
    __optional_kernel_name<_CustomName>;
#endif

template <typename _KernelName, char...>
struct __composite
{
};

// Compose kernel name by transforming the constexpr string to the sequence of chars
// and instantiate template with variadic non-type template parameters.
// This approach is required to get reliable work group size when kernel is unnamed
#if _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
template <typename _KernelName, typename _Tp>
class __kernel_name_composer
{
    static constexpr auto __name = __builtin_sycl_unique_stable_name(_Tp);
    static constexpr ::std::size_t __name_size = __builtin_strlen(__name);

    template <::std::size_t... _Is>
    static __composite<_KernelName, __name[_Is]...>
    __compose_kernel_name(::std::index_sequence<_Is...>);

  public:
    using type = decltype(__compose_kernel_name(::std::make_index_sequence<__name_size>{}));
};
#endif // _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT

template <template <typename...> class _BaseName, typename _CustomName, typename... _Args>
using __kernel_name_generator =
#if __SYCL_UNNAMED_LAMBDA__
    ::std::conditional_t<_HasDefaultName<_CustomName>::value,
#    if _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
                         typename __kernel_name_composer<_BaseName<>, _BaseName<_CustomName, _Args...>>::type,
#    else // _ONEDPL_BUILT_IN_STABLE_NAME_PRESENT
                         _BaseName<_CustomName, _Args...>,
#    endif
                         _BaseName<_CustomName>>;
#else // __SYCL_UNNAMED_LAMBDA__
    _BaseName<_CustomName>;
#endif

#if _ONEDPL_COMPILE_KERNEL
template <typename... _KernelNames>
class __kernel_compiler
{
    static constexpr ::std::size_t __kernel_count = sizeof...(_KernelNames);
    using __kernel_array_type = ::std::array<sycl::kernel, __kernel_count>;

    static_assert(__kernel_count > 0, "At least one kernel name should be provided");

  public:
#if _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT
    static auto
    __compile(const sycl::queue& __q)
    {
        std::vector<sycl::kernel_id> __kernel_ids{sycl::get_kernel_id<_KernelNames>()...};

        auto __kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            __q.get_context(), {__q.get_device()}, __kernel_ids);

        if constexpr (__kernel_count > 1)
            return __make_kernels_array(__kernel_bundle, __kernel_ids, ::std::make_index_sequence<__kernel_count>());
        else
            return __kernel_bundle.get_kernel(__kernel_ids[0]);
    }

  private:
    template <typename _KernelBundle, typename _KernelIds, ::std::size_t... _Ip>
    static auto
    __make_kernels_array(_KernelBundle __kernel_bundle, _KernelIds& __kernel_ids, ::std::index_sequence<_Ip...>)
    {
        return __kernel_array_type{__kernel_bundle.get_kernel(__kernel_ids[_Ip])...};
    }
#elif _ONEDPL_LIBSYCL_PROGRAM_PRESENT
    static auto
    __compile(const sycl::queue& __q)
    {
        sycl::program __program(__q.get_context());

        using __return_type = std::conditional_t<(__kernel_count > 1), __kernel_array_type, sycl::kernel>;
        return __return_type{
            (__program.build_with_kernel_type<_KernelNames>(), __program.get_kernel<_KernelNames>())...};
    }
#endif
};
#endif // _ONEDPL_COMPILE_KERNEL

#if _ONEDPL_DEBUG_SYCL
inline void
// Passing policy by value should be enough for debugging
__print_device_debug_info(const sycl::queue& __q, size_t __wg_size = 0, size_t __max_cu = 0)
{
    std::cout << "Device info" << ::std::endl;
    std::cout << " > device name:         " << oneapi::dpl::__internal::__device_info(__q) << ::std::endl;
    std::cout << " > max compute units:   " << (__max_cu ? __max_cu : oneapi::dpl::__internal::__max_compute_units(__q))
              << ::std::endl;
    std::cout << " > max work-group size: "
              << (__wg_size ? __wg_size : oneapi::dpl::__internal::__max_work_group_size(__q)) << ::std::endl;
}
#else
inline void
__print_device_debug_info(const sycl::queue&, size_t = 0, size_t = 0)
{
}
#endif

//-----------------------------------------------------------------------
// type traits for comparators
//-----------------------------------------------------------------------

// traits for ascending functors
template <typename _Comp>
struct __is_comp_ascending
{
    static constexpr bool value = false;
};
template <typename _T>
struct __is_comp_ascending<std::less<_T>>
{
    static constexpr bool value = true;
};

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
template <>
struct __is_comp_ascending<std::ranges::less>
{
    static constexpr bool value = true;
};
#endif

// traits for descending functors
template <typename _Comp>
struct __is_comp_descending
{
    static constexpr bool value = false;
};
template <typename _T>
struct __is_comp_descending<std::greater<_T>>
{
    static constexpr bool value = true;
};

#if defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L
template <>
struct __is_comp_descending<std::ranges::greater>
{
    static constexpr bool value = true;
};
#endif

} // namespace __internal

// Tag __async_mode describe a pattern call mode which should be executed asynchronously
struct __async_mode
{
};
// Tag __sync_mode describe a pattern call mode which should be executed synchronously
struct __sync_mode
{
};
// Tag __deferrable_mode describe a pattern call mode which should be executed
// synchronously/asynchronously : it's depends on ONEDPL_ALLOW_DEFERRED_WAITING macro state
struct __deferrable_mode
{
};

// A contract for future class: <sycl::event, a result value, keep-alive storages...>
// Uses inheritance instead of aggregation for enabling the empty base optimization.
template <typename _Event, typename... _Args>
class __future : private std::tuple<_Args...>
{
    _Event __my_event;

    template <typename _T>
    _T
    __wait_and_get_value(const sycl::buffer<_T>& __buf)
    {
        // according to the contract, returned value is one-element sycl::buffer
        return __buf.get_host_access(sycl::read_only)[0];
    }

    template <typename _T>
    _T
    __wait_and_get_value(const __internal::__copyable_storage_state<_T>& __st)
    {
        if (__st.__kind == sycl::usm::alloc::unknown)
        {
            assert(__st.__sycl_buf.has_value());
            return __st.__sycl_buf->get_host_access(sycl::read_only)[__st.__offset];
        }

        __my_event.wait_and_throw();
        if (__st.__kind == sycl::usm::alloc::host)
            return *__st.__usm_buf;
        else
        {
            assert(__st.__kind == sycl::usm::alloc::device);
            auto __q_proxy = std::get_deleter<__internal::__usm_deleter>(__st.__usm_buf);
            assert(__q_proxy != nullptr && __q_proxy->__q.has_value());
            // Avoid default constructor for _T. Since _T is device copyable, copy construction
            // is equivalent to a bitwise copy and we may treat __space.__v as constructed by memcpy.
            // There is no need to destroy it afterwards, as the destructor must have no effect.
            oneapi::dpl::__internal::__lazy_ctor_storage<_T> __space;
            __q_proxy->__q->memcpy(&__space.__v, __st.__usm_buf.get() + __st.__offset, sizeof(_T)).wait();
            return __space.__v;
        }
    }

    template <typename _T>
    _T
    __wait_and_get_value(const _T& __val)
    {
        wait();
        return __val;
    }

  public:
    __future(_Event __e, _Args... __args) : std::tuple<_Args...>(__args...), __my_event(__e) {}
    __future(_Event __e, std::tuple<_Args...> __t) : std::tuple<_Args...>(std::move(__t)), __my_event(__e) {}

    auto event() const { return __my_event; }
    operator _Event() const { return event(); }

    void
    wait()
    {
        __my_event.wait_and_throw();
    }

    template <typename _WaitModeTag>
    void
    wait(_WaitModeTag)
    {
        if constexpr (std::is_same_v<_WaitModeTag, __sync_mode>)
            wait();
        else if constexpr (std::is_same_v<_WaitModeTag, __deferrable_mode>)
            __checked_deferrable_wait();
    }

    void
    __checked_deferrable_wait()
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        wait();
#else
        if constexpr (sizeof...(_Args) > 0)
        {
            // Ensure that the temporary data is not destroyed before the kernel code finishes
            wait();
        }
#endif
    }

    auto
    get()
    {
        if constexpr (sizeof...(_Args) > 0)
        {
            auto& __val = std::get<0>(*this);
            return __wait_and_get_value(__val);
        }
        else
            wait();
    }

    // Internal API for cases where the return value is set higher up the stack after the future is created.
    template <typename _T>
    __future<_Event, _T, _Args...>
    __make_future(_T __t) const
    {
        auto new_val = std::tuple<_T>(__t);
        auto new_tuple = std::tuple_cat(new_val, (std::tuple<_Args...>)*this);
        return __future<_Event, _T, _Args...>(__my_event, new_tuple);
    }
};

struct __scalar_load_op
{
    oneapi::dpl::__internal::__pstl_assign __assigner;
    template <typename _IdxType1, typename _IdxType2, typename _SourceAcc, typename _DestAcc>
    void
    operator()(_IdxType1 __idx_source, _IdxType2 __idx_dest, _SourceAcc __source_acc, _DestAcc __dest_acc) const
    {
        __assigner(__source_acc[__idx_source], __dest_acc[__idx_dest]);
    }
};

template <std::uint8_t __vec_size>
struct __vector_load
{
    static_assert(__vec_size <= 4, "Only vector sizes of 4 or less are supported");
    std::size_t __full_range_size;
    template <typename _IdxType, typename _LoadOp, typename... _Rngs>
    void
    operator()(/*__is_full*/ std::true_type, _IdxType __start_idx, _LoadOp __load_op, _Rngs&&... __rngs) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __vec_size; ++__i)
            __load_op(__start_idx + __i, __i, __rngs...);
    }

    template <typename _IdxType, typename _LoadOp, typename... _Rngs>
    void
    operator()(/*__is_full*/ std::false_type, _IdxType __start_idx, _LoadOp __load_op, _Rngs&&... __rngs) const
    {
        std::uint8_t __elements = std::min(std::size_t{__vec_size}, std::size_t{__full_range_size - __start_idx});
        for (std::uint8_t __i = 0; __i < __elements; ++__i)
            __load_op(__start_idx + __i, __i, __rngs...);
    }
};

template <typename _TransformOp>
struct __scalar_store_transform_op
{
    _TransformOp __transform;
    // Unary transformations into an output buffer
    template <typename _IdxType1, typename _IdxType2, typename _SourceAcc, typename _DestAcc>
    void
    operator()(_IdxType1 __idx_source, _IdxType2 __idx_dest, _SourceAcc&& __source_acc, _DestAcc&& __dest_acc) const
    {
        __transform(__source_acc[__idx_source], __dest_acc[__idx_dest]);
    }
    // Binary transformations into an output buffer
    template <typename _IdxType1, typename _IdxType2, typename _Source1Acc, typename _Source2Acc, typename _DestAcc>
    void
    operator()(_IdxType1 __idx_source, _IdxType2 __idx_dest, _Source1Acc&& __source1_acc, _Source2Acc&& __source2_acc,
               _DestAcc&& __dest_acc) const
    {
        __transform(__source1_acc[__idx_source], __source2_acc[__idx_source], __dest_acc[__idx_dest]);
    }
};

// TODO: Consider unifying the implementations of __vector_walk, __vector_load, __vector_store, and potentially
// __strided_loop with some common, generic utility
template <std::uint8_t __vec_size>
struct __vector_walk
{
    static_assert(__vec_size <= 4, "Only vector sizes of 4 or less are supported");
    std::size_t __full_range_size;

    template <typename _IdxType, typename _WalkFunction, typename... _Rngs>
    void
    operator()(std::true_type, _IdxType __idx, _WalkFunction __f, _Rngs&&... __rngs) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __vec_size; ++__i)
        {
            __f(__rngs[__idx + __i]...);
        }
    }
    // For a non-full vector path, process it sequentially. This will always be the last sub or work group
    // if it does not evenly divide into input
    template <typename _IdxType, typename _WalkFunction, typename... _Rngs>
    void
    operator()(std::false_type, _IdxType __idx, _WalkFunction __f, _Rngs&&... __rngs) const
    {
        std::uint8_t __elements = std::min(std::size_t{__vec_size}, std::size_t{__full_range_size - __idx});
        for (std::uint8_t __i = 0; __i < __elements; ++__i)
        {
            __f(__rngs[__idx + __i]...);
        }
    }
};

template <std::uint8_t __vec_size>
struct __vector_store
{
    static_assert(__vec_size <= 4, "Only vector sizes of 4 or less are supported");
    std::size_t __full_range_size;

    template <typename _IdxType, typename _StoreOp, typename... _Rngs>
    void
    operator()(std::true_type, _IdxType __start_idx, _StoreOp __store_op, _Rngs&&... __rngs) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __vec_size; ++__i)
            __store_op(__i, __start_idx + __i, __rngs...);
    }
    template <typename _IdxType, typename _StoreOp, typename... _Rngs>
    void
    operator()(std::false_type, _IdxType __start_idx, _StoreOp __store_op, _Rngs&&... __rngs) const
    {
        std::uint8_t __elements = std::min(std::size_t{__vec_size}, std::size_t{__full_range_size - __start_idx});
        for (std::uint8_t __i = 0; __i < __elements; ++__i)
            __store_op(__i, __start_idx + __i, __rngs...);
    }
};

template <std::uint8_t __vec_size>
struct __vector_reverse
{
    static_assert(__vec_size <= 4, "Only vector sizes of 4 or less are supported");
    template <typename _Idx, typename _Array>
    void
    operator()(/*__is_full*/ std::true_type, const _Idx /*__elements_to_process*/, _Array&& __array) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __vec_size / 2; ++__i)
        {
            using std::swap;
            swap(__array[__i], __array[__vec_size - __i - 1]);
        }
    }
    template <typename _Idx, typename _Array>
    void
    operator()(/*__is_full*/ std::false_type, const _Idx __elements_to_process, _Array&& __array) const
    {
        for (std::uint8_t __i = 0; __i < __elements_to_process / 2; ++__i)
        {
            using std::swap;
            swap(__array[__i], __array[__elements_to_process - __i - 1]);
        }
    }
};

// Processes a loop with a given stride. Intended to be used with sub-group / work-group strides
// for good memory access patterns (potentially with vectorization)
template <std::uint8_t __num_strides>
struct __strided_loop
{
    std::size_t __full_range_size = 0;
    template <typename _LoopBodyOp, typename... _Args>
    void
    operator()(/*__is_full*/ std::true_type, std::size_t __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Args&&... __args) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __num_strides; ++__i)
        {
            __loop_body_op(std::true_type{}, __idx, __args...);
            __idx += __stride;
        }
    }
    template <typename _LoopBodyOp, typename... _Args>
    void
    operator()(/*__is_full*/ std::false_type, std::size_t __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Args&&... __args) const
    {
        std::size_t __limit = std::min(__full_range_size, __idx + __num_strides * __stride);
        for (; __idx < __limit; __idx += __stride)
        {
            __loop_body_op(std::false_type{}, __idx, __args...);
        }
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PARALLEL_BACKEND_SYCL_UTILS_H
