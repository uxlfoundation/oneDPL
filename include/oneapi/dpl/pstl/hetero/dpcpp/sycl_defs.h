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

// If SYCL_LANGUAGE_VERSION is still not set after including the SYCL header, issue an error
#if !(SYCL_LANGUAGE_VERSION || CL_SYCL_LANGUAGE_VERSION)
#    error "Device execution policies are enabled, \
        but SYCL_LANGUAGE_VERSION/CL_SYCL_LANGUAGE_VERSION macros are not defined"
#endif

#include <memory>

// Combine SYCL runtime library version
#if defined(__LIBSYCL_MAJOR_VERSION) && defined(__LIBSYCL_MINOR_VERSION) && defined(__LIBSYCL_PATCH_VERSION)
#    define _ONEDPL_LIBSYCL_VERSION                                                                                    \
        (__LIBSYCL_MAJOR_VERSION * 10000 + __LIBSYCL_MINOR_VERSION * 100 + __LIBSYCL_PATCH_VERSION)
#endif
#define _ONEDPL_LIBSYCL_VERSION_LESS_THAN(_VERSION) (_ONEDPL_LIBSYCL_VERSION && _ONEDPL_LIBSYCL_VERSION < _VERSION)

#if defined(ACPP_VERSION_MAJOR) && defined(ACPP_VERSION_MINOR) && defined(ACPP_VERSION_PATCH)
#    define _ONEDPL_ACPP_VERSION (ACPP_VERSION_MAJOR * 10000 + ACPP_VERSION_MINOR * 100 + ACPP_VERSION_PATCH)
#endif

#if _ONEDPL_FPGA_DEVICE
#    if _ONEDPL_LIBSYCL_VERSION >= 50400
#        include <sycl/ext/intel/fpga_extensions.hpp>
#    else
#        include <CL/sycl/INTEL/fpga_extensions.hpp>
#    endif
#endif

// SYCL 2020 feature macros. They are enabled by default:
// A SYCL implementation is assumed to support the features unless specified otherwise.
// This is controlled by extendable logic: !(A && A < SUPPORTED_VER_A) && !(B && B < SUPPORTED_VER_B) && ...,
// where A, B, etc., are macros representing the version of a specific SYCL implementation.
#define _ONEDPL_SYCL2020_BITCAST_PRESENT                      (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_NO_INIT_PRESENT                      (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_COLLECTIVES_PRESENT                  (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_BUFFER_SIZE_PRESENT                  (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_ACCESSOR_SIZE_PRESENT                (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
// Kernel bundle support is not expected in ACPP, see https://github.com/AdaptiveCpp/AdaptiveCpp/issues/1296.
#define _ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT                                                                        \
    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300) && !_ONEDPL_ACPP_VERSION)
#define _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT               (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT           (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_REQD_SUB_GROUP_SIZE_PRESENT          (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_GROUP_BARRIER_PRESENT                (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_SYCL2020_TARGET_PRESENT                       (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400))
#define _ONEDPL_SYCL2020_TARGET_DEVICE_PRESENT                (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400))
#define _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT                   (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50500))
#define _ONEDPL_SYCL2020_SUB_GROUP_PRESENT                    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700))
#define _ONEDPL_SYCL2020_SUBGROUP_BARRIER_PRESENT             (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700))
// 20201214 value corresponds to DPC++ 2021.1.2
#define _ONEDPL_SYCL2020_KERNEL_DEVICE_API_PRESENT                                                                     \
    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700) || __SYCL_COMPILER_VERSION > 20201214)
#define _ONEDPL_SYCL2020_BUFFER_ALLOCATOR_PRESENT             (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000))
#define _ONEDPL_SYCL2020_LOCAL_ACCESSOR_PRESENT               (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000))
// The unified future supporting USM memory and buffers is only supported after DPC++ 2023.1 but not by 2023.2.
#define _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT                                                          \
    (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60100) && _ONEDPL_LIBSYCL_VERSION != 60200)
#define _ONEDPL_SYCL2020_HOST_TARGET_PRESENT                  (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_HOST_ACCESSOR_PRESENT                (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_GET_HOST_ACCESS_PRESENT              (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200))
#define _ONEDPL_SYCL2020_LOCAL_ACC_GET_MULTI_PTR_PRESENT      (!_ONEDPL_LIBSYCL_VERSION_LESS_THAN(70000))

// Feature macros for DPC++ SYCL runtime library alternatives to non-supported SYCL 2020 features
#define _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT                   (_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_LIBSYCL_PROGRAM_PRESENT                       (_ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300))
#define _ONEDPL_LIBSYCL_KNOWN_IDENTITY_PRESENT                (_ONEDPL_LIBSYCL_VERSION == 50200)
#define _ONEDPL_LIBSYCL_SUB_GROUP_MASK_PRESENT                                                                         \
    (SYCL_EXT_ONEAPI_SUB_GROUP_MASK >= 1 && _ONEDPL_LIBSYCL_VERSION >= 50700)

// Compilation of a kernel is requiried to obtain valid work_group_size
// when target devices are CPU or FPGA emulator. Since CPU and GPU devices
// cannot be distinguished during compilation, the macro is enabled by default.
#define _ONEDPL_CAN_COMPILE_KERNEL (_ONEDPL_SYCL2020_KERNEL_BUNDLE_PRESENT || _ONEDPL_LIBSYCL_PROGRAM_PRESENT)
#if !defined(_ONEDPL_COMPILE_KERNEL)
#    define _ONEDPL_COMPILE_KERNEL _ONEDPL_CAN_COMPILE_KERNEL
#else
#    if _ONEDPL_COMPILE_KERNEL && !_ONEDPL_CAN_COMPILE_KERNEL
#        error "No SYCL kernel compilation method available (neither SYCL 2020 kernel bundle nor other alternatives)."
#    endif
#endif

#define _ONEDPL_SYCL_DEVICE_COPYABLE_SPECIALIZATION_BROKEN (_ONEDPL_LIBSYCL_VERSION_LESS_THAN(70100))

// Macro to check if we are compiling for SPIR-V devices. This macro must only be used within
// SYCL kernels for determining SPIR-V compilation. Using this macro on the host may lead to incorrect behavior.
#ifndef _ONEDPL_DETECT_SPIRV_COMPILATION // Check if overridden for testing
#    if (defined(__SPIR__) || defined(__SPIRV__)) && defined(__SYCL_DEVICE_ONLY__)
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 1
#    else
#        define _ONEDPL_DETECT_SPIRV_COMPILATION 0
#    endif
#endif // _ONEDPL_DETECT_SPIRV_COMPILATION

#if _ONEDPL_SYCL2020_REQD_SUB_GROUP_SIZE_PRESENT
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) sycl::reqd_sub_group_size(SIZE)
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE) intel::reqd_sub_group_size(SIZE)
#else
#    error "sycl::reqd_sub_group_size is not supported, and no alternative is available"
#endif

// This macro is intended to be used for specifying a subgroup size as a SYCL kernel attribute for SPIR-V targets
// only. For non-SPIR-V targets, it will be empty. This macro should only be used in device code and may lead
// to incorrect behavior if used on the host.
#if _ONEDPL_DETECT_SPIRV_COMPILATION
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE) _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(SIZE)
#else
#    define _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(SIZE)
#endif

namespace __dpl_sycl
{

using __no_init =
#if _ONEDPL_SYCL2020_NO_INIT_PRESENT
    sycl::property::no_init;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    sycl::property::noinit;
#else
#    error "sycl::property::no_init is not supported, and no alternative is available"
#endif

#if _ONEDPL_SYCL2020_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::has_known_identity<_BinaryOp, _T>;

#elif _ONEDPL_LIBSYCL_KNOWN_IDENTITY_PRESENT
template <typename _BinaryOp, typename _T>
using __known_identity = sycl::ONEAPI::known_identity<_BinaryOp, _T>;

template <typename _BinaryOp, typename _T>
using __has_known_identity = sycl::ONEAPI::has_known_identity<_BinaryOp, _T>;

#else
#    error "sycl::__known_identity is not supported, and no alternative is available"
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
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
template <typename _T>
using __plus = sycl::ONEAPI::plus<_T>;

template <typename _T>
using __maximum = sycl::ONEAPI::maximum<_T>;

template <typename _T>
using __minimum = sycl::ONEAPI::minimum<_T>;

#else
#    error "sycl::plus, sycl::maximum, sycl::minimum are not supported, and no alternative is available"
#endif // _ONEDPL_SYCL2020_FUNCTIONAL_OBJECTS_PRESENT

#if _ONEDPL_SYCL2020_SUB_GROUP_PRESENT
using __sub_group = sycl::sub_group;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50700)
using __sub_group = sycl::ONEAPI::sub_group;
#else
#    error "sycl::group is not supported, and no alternative is available"
#endif

template <typename _Buffer>
constexpr auto
__get_buffer_size(const _Buffer& __buffer)
{
#if _ONEDPL_SYCL2020_BUFFER_SIZE_PRESENT
    return __buffer.size();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    return __buffer.get_count();
#else
#    error "buffer::size is not supported, and no alternative is available"
#endif
}

template <typename _Accessor>
constexpr auto
__get_accessor_size(const _Accessor& __accessor)
{
#if _ONEDPL_SYCL2020_ACCESSOR_SIZE_PRESENT
    return __accessor.size();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50300)
    return __accessor.get_count();
#else
#    error "accessor::size is not supported, and no alternative is available"
#endif
}

// TODO: switch to SYCL 2020 with DPC++ compiler.
// SYCL 1.2.1 version is used due to having an API with a local memory fence,
// which gives better performance on Intel GPUs.
// The performance gap is negligible since
// https://github.com/intel/intel-graphics-compiler/commit/ed639f68d142bc963a7b626badc207a42fb281cb (Aug 20, 2024)
// But the fix is not a part of the LTS GPU drivers (Linux) yet.
//
// This macro may also serve as a temporary workaround to strengthen the barriers
// if there are cases where the memory ordering is not strong enough.
#if !defined(_ONEDPL_SYCL121_GROUP_BARRIER)
#    if _ONEDPL_LIBSYCL_VERSION
#        define _ONEDPL_SYCL121_GROUP_BARRIER 1
#    else
// For safety, assume that other SYCL implementations comply with SYCL 2020, which is a oneDPL requirement.
#        define _ONEDPL_SYCL121_GROUP_BARRIER 0
#    endif
#endif

#if _ONEDPL_SYCL121_GROUP_BARRIER
using __fence_space_t = sycl::access::fence_space;
inline constexpr __fence_space_t __fence_space_local = sycl::access::fence_space::local_space;
inline constexpr __fence_space_t __fence_space_global = sycl::access::fence_space::global_space;
#else
struct __fence_space_t{}; // No-op dummy type since SYCL 2020 does not specify memory fence spaces in group barriers
inline constexpr __fence_space_t __fence_space_local{};
inline constexpr __fence_space_t __fence_space_global{};
#endif // _ONEDPL_SYCL121_GROUP_BARRIER

template <typename _Item>
void
__group_barrier(_Item __item, [[maybe_unused]] __dpl_sycl::__fence_space_t __space = __fence_space_local)
{
#if _ONEDPL_SYCL121_GROUP_BARRIER
    __item.barrier(__space);
#elif _ONEDPL_SYCL2020_GROUP_BARRIER_PRESENT
    sycl::group_barrier(__item.get_group(), sycl::memory_scope::work_group);
#else
#    error "sycl::group_barrier is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__group_broadcast(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::group_broadcast(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::broadcast(__args...);
#else
#    error "sycl::group_broadcast is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__exclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::exclusive_scan_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::exclusive_scan(__args...);
#else
#    error "sycl::exclusive_scan_over_group is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__inclusive_scan_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::inclusive_scan_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::inclusive_scan(__args...);
#else
#    error "sycl::inclusive_scan_over_group is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__reduce_over_group(_Args... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::reduce_over_group(__args...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::reduce(__args...);
#else
#    error "sycl::reduce_over_group is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__any_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::any_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::any_of_group is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__all_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::all_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::all_of_group is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__none_of_group(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::none_of_group(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::none_of is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_exclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_exclusive_scan(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::exclusive_scan(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_exclusive_scan is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_inclusive_scan(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_inclusive_scan(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::inclusive_scan(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_inclusive_scan is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_reduce(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_reduce(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::reduce(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_reduce is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_any_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_any_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::any_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_any_of is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_all_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_all_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::all_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_all_of is not supported, and no alternative is available"
#endif
}

template <typename... _Args>
constexpr auto
__joint_none_of(_Args&&... __args)
{
#if _ONEDPL_SYCL2020_COLLECTIVES_PRESENT
    return sycl::joint_none_of(::std::forward<_Args>(__args)...);
#elif _ONEDPL_LIBSYCL_COLLECTIVES_PRESENT
    return sycl::ONEAPI::none_of(::std::forward<_Args>(__args)...);
#else
#    error "sycl::joint_none_of is not supported, and no alternative is available"
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
#if _ONEDPL_SYCL2020_TARGET_PRESENT
    sycl::target;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400)
    sycl::access::target;
#else
#    error "sycl::target is not supported, and no alternative is available"
#endif

constexpr __target __target_device =
#if _ONEDPL_SYCL2020_TARGET_DEVICE_PRESENT
    __target::device;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50400)
    __target::global_buffer;
#else
#    error "sycl::target::device is not supported, and no alternative is available"
#endif

constexpr __target __host_target =
#if _ONEDPL_SYCL2020_HOST_TARGET_PRESENT
    __target::host_task;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200)
    __target::host_buffer;
#else
#    error "sycl::target::host_task is not supported, and no alternative is available"
#endif

template <typename _DataT>
using __buffer_allocator =
#if _ONEDPL_SYCL2020_BUFFER_ALLOCATOR_PRESENT
    sycl::buffer_allocator<_DataT>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000)
    sycl::buffer_allocator;
#else
#    error "sycl::buffer_allocator is not supported, and no alternative is available"
#endif

template <typename _AtomicType, sycl::access::address_space _Space>
#if _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT
using __atomic_ref = sycl::atomic_ref<_AtomicType, sycl::memory_order::relaxed, sycl::memory_scope::work_group, _Space>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(50500)
struct __atomic_ref : sycl::atomic<_AtomicType, _Space>
{
    explicit __atomic_ref(_AtomicType& ref)
        : sycl::atomic<_AtomicType, _Space>(sycl::multi_ptr<_AtomicType, _Space>(&ref)){};
};
#else
#    error "sycl::atomic_ref is not supported, and no alternative is available"
#endif // _ONEDPL_SYCL2020_ATOMIC_REF_PRESENT

template <typename _DataT, int _Dimensions = 1>
using __local_accessor =
#if _ONEDPL_SYCL2020_LOCAL_ACCESSOR_PRESENT
    sycl::local_accessor<_DataT, _Dimensions>;
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60000)
    sycl::accessor<_DataT, _Dimensions, sycl::access::mode::read_write, __dpl_sycl::__target::local>;
#else
#    error "sycl::local_accessor is not supported, and no alternative is available"
#endif

template <typename _Buf>
auto
__get_host_access(_Buf&& __buf)
{
#if _ONEDPL_SYCL2020_GET_HOST_ACCESS_PRESENT
    return ::std::forward<_Buf>(__buf).get_host_access(sycl::read_only);
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(60200)
    return ::std::forward<_Buf>(__buf).template get_access<sycl::access::mode::read>();
#else
#    error "sycl::buffer::get_host_access is not supported, and no alternative is available"
#endif
}

template <typename _Acc>
auto
__get_accessor_ptr(const _Acc& __acc)
{
#if _ONEDPL_SYCL2020_LOCAL_ACC_GET_MULTI_PTR_PRESENT
    return __acc.template get_multi_ptr<sycl::access::decorated::no>().get();
#elif _ONEDPL_LIBSYCL_VERSION_LESS_THAN(70000)
    return __acc.get_pointer();
#else
#    error "sycl::accessor::get_multi_ptr is not supported, and no alternative is available"
#endif
}

#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO) || defined(SYCL_EXT_ACPP_BACKEND_LEVEL_ZERO)
#    define _ONEDPL_SYCL_L0_EXT_PRESENT 1
#endif
#if _ONEDPL_SYCL_L0_EXT_PRESENT
#    if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
inline constexpr auto __level_zero_backend = sycl::backend::ext_oneapi_level_zero;
#    elif defined(SYCL_EXT_ACPP_BACKEND_LEVEL_ZERO)
inline constexpr auto __level_zero_backend = sycl::backend::level_zero;
#    endif
#endif

} // namespace __dpl_sycl

#endif // _ONEDPL_SYCL_DEFS_H
