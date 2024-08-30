// -*- C++ -*-
//===-- sycl_defs.h ---------------------------------------------===//
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

#ifndef _ONEDPL_SYCL_DEFS_H
#define _ONEDPL_SYCL_DEFS_H

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif
#include <memory>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define _ONEDPL_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#else
#    define _ONEDPL_LIBSYCL_VERSION 0
#endif

#if _ONEDPL_FPGA_DEVICE
#    if _ONEDPL_LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

// Macros to check the new SYCL features
#define _ONEDPL_NO_INIT_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50300)
#define _ONEDPL_KERNEL_BUNDLE_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50300)
#define _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50500)
#define _ONEDPL_SYCL_SUB_GROUP_MASK_PRESENT (SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 1) && (_ONEDPL_LIBSYCL_VERSION >= 50700)
#define _ONEDPL_SYCL_PLACEHOLDER_HOST_ACCESSOR_DEPRECATED (_ONEDPL_LIBSYCL_VERSION >= 60200)
#define _ONEDPL_SYCL_DEVICE_COPYABLE_SPECIALIZATION_BROKEN                                                             \
    (_ONEDPL_LIBSYCL_VERSION < 70100) && (_ONEDPL_LIBSYCL_VERSION != 0)

// TODO: determine which compiler configurations provide subgroup load/store
#define _ONEDPL_SYCL_SUB_GROUP_LOAD_STORE_PRESENT false

#define _ONEDPL_SYCL_SUB_GROUP_PRESENT (_ONEDPL_LIBSYCL_VERSION >= 50700)

// Macro to check if we are compiling for SPIR-V devices. This macro must only be used within
// SYCL kernels for determining SPIR-V compilation. Using this macro on the host may lead to incorrect behavior.
#ifndef _ONEDPL_DETECT_SPIRV_COMPILATION // Check if overridden for testing
#    if (defined(__SPIR__) || defined(__SPIRV__)) && defined(__SYCL_DEVICE_ONLY__)
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 1
#    else
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 0
#    endif
#endif // _ONEDPL_DETECT_SPIRV_COMPILATION

#if _ONEDPL_LIBSYCL_VERSION >= 50300
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) sycl::reqd_sub_group_size(SIZE)
#else
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) intel::reqd_sub_group_size(SIZE)
#endif

// This macro is intended to be used for specifying a subgroup size as a SYCL kernel attribute for SPIR-V targets
// only. For non-SPIR-V targets, it will be empty. This macro should only be used in device code and may lead
// to incorrect behavior if used on the host.
#if _ONEDPL_DETECT_SPIRV_COMPILATION
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE) _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE)
#else
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE)
#endif

// The unified future supporting USM memory and buffers is only supported after DPCPP 2023.1
// but not by 2023.2.
#if (_ONEDPL_LIBSYCL_VERSION >= 60100 && _ONEDPL_LIBSYCL_VERSION != 60200)
#    define _ONEDPL_SYCL_UNIFIED_USM_BUFFER_PRESENT 1
#else
#    define _ONEDPL_SYCL_UNIFIED_USM_BUFFER_PRESENT 0
#endif

namespace __dpl_sycl
{

using __no_init =
#if _ONEDPL_NO_INIT_PRESENT
    sycl::property::no_init;
#else
    sycl::property::noinit;
#endif

#if _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::has_known_identity<_BinaryOp, _T>;

#elif _ONEDPL_LIBSYCL_VERSION == 50200
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::ONEAPI::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::ONEAPI::has_known_identity<_BinaryOp, _T>;
#endif // _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT

template <typename _BinaryOp, typename _T>
inline constexpr auto __known_identity_v = __known_identity<_BinaryOp, _T>::value;

#if _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
template <typename _T = void>
using __plus = sycl::plus<_T>;

template <typename _T = void>
using __maximum = sycl::maximum<_T>;

template <typename _T = void>
using __minimum = sycl::minimum<_T>;
#else  // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT
template <typename _T>
using __plus = sycl::ONEAPI::plus<_T>;

template <typename _T>
using __maximum = sycl::ONEAPI::maximum<_T>;

template <typename _T>
using __minimum = sycl::ONEAPI::minimum<_T>;
#endif // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT

#if _ONEDPL_SYCL_SUB_GROUP_PRESENT
using __sub_group = sycl::sub_group;
#else
using __sub_group = sycl::ONEAPI::sub_group;
#endif

template <typename _Buffer>
constexpr auto
__get_buffer_size(const _Buffer& __buffer)
{
#if _ONEDPL_LIBSYCL_VERSION >= 50300
    return __buffer.size();
#else
    return __buffer.get_count();
#endif
}

template <typename _Accessor>
constexpr auto
__get_accessor_size(const _Accessor& __accessor)
{
#if _ONEDPL_LIBSYCL_VERSION >= 50300
    return __accessor.size();
#else
    return __accessor.get_count();
#endif
}

template <typename _Item>
constexpr void
__group_barrier(_Item __item)
{
#if 0 //_ONEDPL_LIBSYCL_VERSION >= 50300
    //TODO: usage of sycl::group_barrier: probably, we have to revise SYCL parallel patterns which use a group_barrier.
    // 1) sycl::group_barrier() implementation is not ready
    // 2) sycl::group_barrier and sycl::item::group_barrier are not quite equivalent
    sycl::group_barrier(__item.get_group(), sycl::memory_scope::work_group);
#else
    __item.barrier(sycl::access::fence_space::local_space);
#endif
}

template <typename... _Args>
constexpr auto
__group_broadcast(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::group_broadcast(__args...);
#else
    return sycl::ONEAPI::broadcast(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__exclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::exclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::exclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__inclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::inclusive_scan_over_group(__args...);
#else
    return sycl::ONEAPI::inclusive_scan(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__reduce_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::reduce_over_group(__args...);
#else
    return sycl::ONEAPI::reduce(__args...);
#endif
}

template <typename... _Args>
constexpr auto
__any_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::any_of_group(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__all_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::all_of_group(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__none_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::none_of_group(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_exclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_exclusive_scan(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::exclusive_scan(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_inclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_inclusive_scan(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::inclusive_scan(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_reduce(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_reduce(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::reduce(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_any_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_any_of(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_all_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_all_of(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#endif
}

template <typename... _Args>
constexpr auto
__joint_none_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_none_of(::std::forward<_Args>(__args)...);
#else
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#endif
}

#if _ONEDPL_FPGA_DEVICE
#    if _ONEDPL_LIBSYCL_VERSION >= 60100
inline auto __fpga_emulator_selector()
{
    return sycl::ext::intel::fpga_emulator_selector_v;
}
inline auto __fpga_selector()
{
    return sycl::ext::intel::fpga_selector_v;
}

#    elif _ONEDPL_LIBSYCL_VERSION >= 50300
inline auto __fpga_emulator_selector()
{
    return sycl::ext::intel::fpga_emulator_selector{};
}
inline auto __fpga_selector()
{
    return sycl::ext::intel::fpga_selector{};
}
#    else
inline auto __fpga_emulator_selector()
{
    return sycl::INTEL::fpga_emulator_selector{};
}
inline auto __fpga_selector()
{
    return sycl::INTEL::fpga_selector{};
}
#    endif
#endif // _ONEDPL_FPGA_DEVICE

using __target =
#if _ONEDPL_LIBSYCL_VERSION >= 50400
    sycl::target;
#else
    sycl::access::target;
#endif

constexpr __target __target_device =
#if _ONEDPL_LIBSYCL_VERSION >= 50400
    __target::device;
#else
    __target::global_buffer;
#endif

constexpr __target __host_target =
#if _ONEDPL_LIBSYCL_VERSION >= 60200
    __target::host_task;
#else
    __target::host_buffer;
#endif

template <typename _DataT>
using __buffer_allocator =
#if _ONEDPL_LIBSYCL_VERSION >= 60000
    sycl::buffer_allocator<_DataT>;
#else
    sycl::buffer_allocator;
#endif

template <typename _AtomicType, sycl::access::address_space _Space>
#if _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT
using __atomic_ref = sycl::atomic_ref<_AtomicType, sycl::memory_order::relaxed, sycl::memory_scope::work_group, _Space>;
#else
struct __atomic_ref : sycl::atomic<_AtomicType, _Space>
{
    explicit __atomic_ref(_AtomicType& ref)
        : sycl::atomic<_AtomicType, _Space>(sycl::multi_ptr<_AtomicType, _Space>(&ref)){};
};
#endif // _ONEDPL_SYCL2023_ATOMIC_REF_PRESENT

template <typename _DataT, int _Dimensions = 1>
using __local_accessor =
#if _ONEDPL_LIBSYCL_VERSION >= 60000
    sycl::local_accessor<_DataT, _Dimensions>;
#else
    sycl::accessor<_DataT, _Dimensions, sycl::access::mode::read_write, __dpl_sycl::__target::local>;
#endif

template <typename _Buf>
auto
__get_host_access(_Buf&& __buf)
{
#if _ONEDPL_LIBSYCL_VERSION >= 60200
    return ::std::forward<_Buf>(__buf).get_host_access(sycl::read_only);
#else
    return ::std::forward<_Buf>(__buf).template get_access<sycl::access::mode::read>();
#endif
}

template <typename _Acc>
auto
__get_accessor_ptr(const _Acc& __acc)
{
#if _ONEDPL_LIBSYCL_VERSION >= 70000
    return __acc.template get_multi_ptr<sycl::access::decorated::no>().get();
#else
    return __acc.get_pointer();
#endif
}

} // namespace __dpl_sycl

#endif // _ONEDPL_SYCL_DEFS_H
