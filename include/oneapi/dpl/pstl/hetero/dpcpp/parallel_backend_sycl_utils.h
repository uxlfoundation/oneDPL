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
#include <memory>
#include <type_traits>
#include <tuple>
#include <algorithm>
#include <optional>
#include <cassert>

#include "../../iterator_impl.h"

#include "sycl_defs.h"
#include "execution_sycl_defs.h"
#include "sycl_iterator.h"
#include "../../utils.h"

#if _ONEDPL_DEBUG_SYCL
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

// function to simplify zip_iterator creation
template <typename... T>
oneapi::dpl::zip_iterator<T...>
zip(T... args)
{
    return oneapi::dpl::zip_iterator<T...>(args...);
}

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
template <template <typename> class _NewKernelName, typename _Policy,
          oneapi::dpl::__internal::__enable_if_fpga_execution_policy<_Policy, int> = 0>
auto
make_wrapped_policy(_Policy&& __policy)
{
    return oneapi::dpl::execution::make_fpga_policy<
        oneapi::dpl::__internal::__policy_unroll_factor<_Policy>,
        _NewKernelName<oneapi::dpl::__internal::__policy_kernel_name<_Policy>>>(::std::forward<_Policy>(__policy));
}
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
struct __is_comp_ascending<::std::less<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_ascending<oneapi::dpl::__internal::__pstl_less>
{
    static constexpr bool value = true;
};

// traits for descending functors
template <typename _Comp>
struct __is_comp_descending
{
    static constexpr bool value = false;
};
template <typename _T>
struct __is_comp_descending<::std::greater<_T>>
{
    static constexpr bool value = true;
};
template <>
struct __is_comp_descending<oneapi::dpl::__internal::__pstl_greater>
{
    static constexpr bool value = true;
};

//-----------------------------------------------------------------------
// temporary "buffer" constructed over specified container type
//-----------------------------------------------------------------------

template <typename _Unknown>
struct __local_buffer;

template <int __dim, typename _AllocT, typename _T>
struct __local_buffer<sycl::buffer<_T, __dim, _AllocT>>
{
    using type = sycl::buffer<_T, __dim, _AllocT>;
};

//if we take ::std::tuple as a type for buffer we should convert to internal::tuple
template <int __dim, typename _AllocT, typename... _T>
struct __local_buffer<sycl::buffer<::std::tuple<_T...>, __dim, _AllocT>>
{
    using type = sycl::buffer<oneapi::dpl::__internal::tuple<_T...>, __dim, _AllocT>;
};

// impl for sycl::buffer<...>
template <typename _T>
class __buffer_impl
{
  private:
    using __container_t = typename __local_buffer<sycl::buffer<_T>>::type;

    __container_t __container;

  public:
    __buffer_impl(std::size_t __n_elements) : __container{sycl::range<1>(__n_elements)} {}

    auto
    get() -> decltype(oneapi::dpl::begin(__container)) const
    {
        return oneapi::dpl::begin(__container);
    }

    __container_t
    get_buffer() const
    {
        return __container;
    }
};

struct __sycl_usm_free
{
    std::optional<sycl::queue> __q;

    void
    operator()(void* __memory) const
    {
        assert(__q.has_value());
        sycl::free(__memory, *__q);
    }
};

// TODO: remove this function once it is no more used in __result_and_scratch_storage
template <typename _T, sycl::usm::alloc __alloc_t>
_T*
__sycl_usm_alloc(const sycl::queue& __q, std::size_t __elements)
{
    if (_T* __buf = sycl::malloc<_T>(__elements, __q, __alloc_t))
        return __buf;

    throw std::bad_alloc();
}

template <typename _T, sycl::usm::alloc __alloc_t>
_T*
__allocate_usm(const sycl::queue& __q, std::size_t __elements)
{
    static_assert(__alloc_t == sycl::usm::alloc::host || __alloc_t == sycl::usm::alloc::device);
    _T* __result = nullptr;
    if constexpr (__alloc_t == sycl::usm::alloc::host)
    {
#if _ONEDPL_SYCL_L0_EXT_PRESENT
        // Only use host USM on L0 GPUs. Other devices should use device USM instead to avoid notable slowdown.
        sycl::device __device = __q.get_device();
        if (__device.is_gpu() && __device.has(sycl::aspect::usm_host_allocations) &&
            __device.get_backend() == __dpl_sycl::__level_zero_backend)
        {
            __result = sycl::malloc<_T>(__elements, __q, __alloc_t);
        }
#endif
    }
    else
    {
        if (__q.get_device().has(sycl::aspect::usm_device_allocations))
            __result = sycl::malloc<_T>(__elements, __q, __alloc_t);
    }
    return __result;
}

//-----------------------------------------------------------------------
// type traits for objects granting access to some value objects
//-----------------------------------------------------------------------

template <typename _ContainerOrIterator>
struct __memobj_traits
{
    using value_type = typename _ContainerOrIterator::value_type;
};

template <typename _T>
struct __memobj_traits<_T*>
{
    using value_type = _T;
};

} // namespace __internal

template <typename _T>
using __buffer = __internal::__buffer_impl<_T>;

template <typename T>
struct __repacked_tuple
{
    using type = T;
};

template <typename... Args>
struct __repacked_tuple<::std::tuple<Args...>>
{
    using type = oneapi::dpl::__internal::tuple<Args...>;
};

template <typename T>
using __repacked_tuple_t = typename __repacked_tuple<T>::type;

template <typename _ContainerOrIterable>
using __value_t = typename __internal::__memobj_traits<_ContainerOrIterable>::value_type;

//-----------------------------------------------------------------------
// types to create and use data on a device and return those to the host
//-----------------------------------------------------------------------

template <typename _T, sycl::access_mode _AccessMode>
struct __combi_accessor
{
  private:
    using __acc_t = sycl::accessor<_T, 1, _AccessMode, __dpl_sycl::__target_device, sycl::access::placeholder::false_t>;
    _T* __ptr = nullptr;
    __acc_t __acc;

    template <bool __with_offset>
    __acc_t
    __make_accessor(bool __fake, sycl::buffer<_T, 1>& __sycl_buf, sycl::handler& __cgh,
                    const sycl::property_list& __prop_list, std::size_t __sz = 0, std::size_t __offset = 0)
    {
        if (__fake)
        {
            return __acc_t(
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
                __sycl_buf, __cgh, __prop_list
#endif
            );
        }
        if constexpr (__with_offset)
            return __acc_t(__sycl_buf, __cgh, sycl::range{__sz}, sycl::id{__offset}, __prop_list);
        else
            return __acc_t(__sycl_buf, __cgh, __prop_list);
    }

  public:
    __combi_accessor(sycl::handler& __cgh, sycl::buffer<_T, 1>& __sycl_buf, _T* __usm_buf,
                     const sycl::property_list& __prop_list)
        : __ptr(__usm_buf), __acc(__make_accessor<false>(__usm_buf != nullptr, __sycl_buf, __cgh, __prop_list))
        {}

    __combi_accessor(sycl::handler& __cgh, sycl::buffer<_T, 1>& __sycl_buf, _T* __usm_buf, std::size_t __offset,
                     std::size_t __sz, const sycl::property_list& __prop_list)
        : __ptr(__usm_buf ? __usm_buf + __offset : nullptr),
          __acc(__make_accessor<true>(__usm_buf != nullptr, __sycl_buf, __cgh, __prop_list, __sz, __offset))
        {}

    auto // [const] _T*, with constness depending on _AccessMode
    __data() const // the result should be cached within a kernel
    {
        return __ptr ? __ptr : &__acc[0];
    }
};

// The type to exchange information between storage types.
// Useful for the interoperability during the transition period
template <typename _T>
struct __copyable_storage_state
{
    std::shared_ptr<_T> __result_buf;
    std::shared_ptr<_T> __scratch_buf;
    sycl::buffer<_T, 1> __sycl_buf;
    std::size_t         __scratch_sz = 0;
    sycl::usm::alloc    __kind = sycl::usm::alloc::unknown;
};

// This base class is provided to allow same-typed shared pointer return values from kernels in
// a `__future` for keeping alive temporary data, while allowing run-time branches to lead to
// differently typed temporary storage for kernels. Virtual destructor is required to call
// derived class destructor when leaving scope.
struct __result_and_scratch_storage_base
{
    virtual ~__result_and_scratch_storage_base() = default;
    virtual std::size_t
    __get_data(sycl::event, std::size_t* __p_buf) const = 0;
};

template <typename _T, std::size_t _NResults = 1>
struct __result_and_scratch_storage : __result_and_scratch_storage_base
{
    static_assert(sycl::is_device_copyable_v<_T>,
                  "The type _T must be device copyable to use __result_and_scratch_storage.");

  private:
    using __sycl_buffer_t = sycl::buffer<_T, 1>;

    template <sycl::access_mode _AccessMode>
    using __accessor_t =
        sycl::accessor<_T, 1, _AccessMode, __dpl_sycl::__target_device, sycl::access::placeholder::false_t>;

    std::shared_ptr<_T> __scratch_buf;
    std::shared_ptr<_T> __result_buf;
    mutable __sycl_buffer_t __sycl_buf;

    std::size_t __scratch_n;
    bool __use_USM_host;
    bool __supports_USM_device;

    // Only use USM host allocations on L0 GPUs. Other devices show significant slowdowns and will use a device allocation instead.
    bool
    __use_USM_host_allocations([[maybe_unused]] const sycl::queue& __q) const
    {
        bool __result = false;
#if _ONEDPL_SYCL_L0_EXT_PRESENT
        auto __device = __q.get_device();
        __result = __device.is_gpu() && __device.has(sycl::aspect::usm_host_allocations) &&
                   __device.get_backend() == __dpl_sycl::__level_zero_backend;
#endif
        return __result;
    }

    bool
    __use_USM_allocations(const sycl::queue& __q) const
    {
        return __q.get_device().has(sycl::aspect::usm_device_allocations);
    }

  public:
    __result_and_scratch_storage(sycl::queue __q, std::size_t __scratch_n)
        : __sycl_buf{nullptr, sycl::range{0}}, __scratch_n{__scratch_n},
          __use_USM_host{__use_USM_host_allocations(__q)}, __supports_USM_device{__use_USM_allocations(__q)}
    {
        const std::size_t __total_n = _NResults + __scratch_n;
        // Skip in case this is a dummy container
        if (__total_n > 0)
        {
            if (__use_USM_host && __supports_USM_device)
            {
                // Separate scratch (device) and result (host) allocations on performant backends (i.e. L0)
                if (__scratch_n > 0)
                {
                    __scratch_buf = std::shared_ptr<_T>(
                        __internal::__sycl_usm_alloc<_T, sycl::usm::alloc::device>(__q, __scratch_n),
                        __internal::__sycl_usm_free{__q});
                }
                if constexpr (_NResults > 0)
                {
                    __result_buf =
                        std::shared_ptr<_T>(__internal::__sycl_usm_alloc<_T, sycl::usm::alloc::host>(__q, _NResults),
                                            __internal::__sycl_usm_free{__q});
                }
            }
            else if (__supports_USM_device)
            {
                // If we don't use host memory, malloc only a single unified device allocation
                __scratch_buf =
                    std::shared_ptr<_T>(__internal::__sycl_usm_alloc<_T, sycl::usm::alloc::device>(__q, __total_n),
                                        __internal::__sycl_usm_free{__q});
            }
            else
            {
                // If we don't have USM support allocate memory here
                __sycl_buf = __sycl_buffer_t(__total_n);
            }
        }
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
        // A fake buffer to work around problems with accessor construction
        if (__supports_USM_device)
            __sycl_buf = __sycl_buffer_t(sycl::range{1});
#endif
    }

    __result_and_scratch_storage(__copyable_storage_state<_T>&& __transfer)
        : __scratch_buf(std::move(__transfer.__scratch_buf)), __result_buf(std::move(__transfer.__result_buf)),
          __sycl_buf(std::move(__transfer.__sycl_buf)), __scratch_n(__transfer.__scratch_sz),
          __use_USM_host(__transfer.__kind == sycl::usm::alloc::host),
          __supports_USM_device(__transfer.__kind != sycl::usm::alloc::unknown)
        {}

    template <typename _Acc>
    static auto
    __get_usm_or_buffer_accessor_ptr(const _Acc& __acc, std::size_t = 0)
    {
        return __acc.__data();
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_result_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
        if (__use_USM_host && __supports_USM_device)
            return __combi_accessor<_T, _AccessMode>(__cgh, __sycl_buf, __result_buf.get(), __prop_list);
        return __combi_accessor<_T, _AccessMode>(__cgh, __sycl_buf, __scratch_buf.get(), __scratch_n, _NResults,
                                                 __prop_list);
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_scratch_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
        return __combi_accessor<_T, _AccessMode>(__cgh, __sycl_buf, __scratch_buf.get(), __prop_list);
    }

    _T
    __wait_and_get_value(sycl::event __event) const
    {
        static_assert(_NResults == 1);

        if (is_USM())
            __event.wait_and_throw();

        return __get_value();
    }

    // Note: this member function assumes the result is *ready*, since the __future has already
    // waited on the relevant event.
    template <std::size_t _Idx = 0>
    _T
    __get_value() const
    {
        static_assert(0 <= _Idx && _Idx < _NResults);

        if (__use_USM_host && __supports_USM_device)
        {
            return *(__result_buf.get() + _Idx);
        }
        else if (__supports_USM_device)
        {
            auto __q_proxy = std::get_deleter<__internal::__sycl_usm_free>(__scratch_buf);
            assert(__q_proxy != nullptr && __q_proxy->__q.has_value());
            // Avoid default constructor for _T. Since _T is device copyable, copy construction
            // is equivalent to a bitwise copy and we may treat __space.__v as constructed after the memcpy.
            // There is no need to destroy it afterwards, as the destructor must have no effect.
            oneapi::dpl::__internal::__lazy_ctor_storage<_T> __space;
            __q_proxy->__q->memcpy(&__space.__v, __scratch_buf.get() + __scratch_n + _Idx, sizeof(_T)).wait();
            return __space.__v;
        }
        else
        {
            return __sycl_buf.get_host_access(sycl::read_only)[__scratch_n + _Idx];
        }
    }

  private:
    bool
    is_USM() const
    {
        return __supports_USM_device;
    }

    template <typename _Type>
    std::size_t
    __fill_data(std::pair<_Type, _Type>&& __p, std::size_t* __p_buf) const
    {
        __p_buf[0] = __p.first;
        __p_buf[1] = __p.second;
        return 2;
    }

    template <typename _Args>
    std::size_t
    __fill_data(_Args&&...) const
    {
        assert(!"Unsupported return type");
        return 0;
    }

    virtual std::size_t
    __get_data(sycl::event __event, std::size_t* __p_buf) const override
    {
        static_assert(_NResults == 0 || _NResults == 1);

        if (is_USM())
            __event.wait_and_throw();

        if constexpr (_NResults == 1)
            return __fill_data(__get_value(), __p_buf);
        else
            return 0;
    }
};

template <typename _T>
struct __device_storage
{
    std::unique_ptr<_T, __internal::__sycl_usm_free> __usm_buf = nullptr;
    sycl::buffer<_T, 1> __sycl_buf =
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_BROKEN
        {sycl::range{1}}; // A non-empty buffer to avoid problems with accessor construction
#else
        {nullptr, sycl::range{0}};
#endif

    __device_storage() = default;

    __device_storage(const sycl::queue& __q, std::size_t __n) { __initialize(__q, __n); }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_accessor(sycl::handler& __cgh, const sycl::property_list& __prop_list = {})
    {
        return __combi_accessor<_T, _AccessMode>(__cgh, __sycl_buf, __usm_buf.get(), __prop_list);
    }

  protected:
    void
    __initialize(const sycl::queue& __q, std::size_t __n)
    {
        assert(__n > 0);
        _T* __ptr = __internal::__allocate_usm<_T, sycl::usm::alloc::device>(__q, __n);
        if (__ptr)
            __usm_buf = std::unique_ptr<_T, __internal::__sycl_usm_free>(__ptr, __internal::__sycl_usm_free{__q});
        else
            __sycl_buf = sycl::buffer<_T, 1>(__n);
    }

    void
    __copy_n(_T* __dst, _T* __src, std::size_t __n, std::size_t __offset)
    {
        // Derived classes are responsible for bound checking
        if (__src)
        {
            std::copy_n(__src, __n, __dst);
        }
        else if (__usm_buf)
        {
            auto& __q_proxy = __usm_buf.get_deleter();
            assert(__q_proxy.__q.has_value());
            __q_proxy.__q->memcpy(__dst, __usm_buf.get() + __offset, __n * sizeof(_T)).wait();
        }
        else
        {
            std::copy_n(__sycl_buf.get_host_access(sycl::read_only).begin() + __offset, __n, __dst);
        }
    }
};

// A pack of device storages
template <typename... _TPack>
struct __device_storage_tuple_pack : std::tuple<__device_storage<_TPack>...>
{
    static constexpr std::size_t __TPackSize = sizeof...(_TPack);
    static_assert(__TPackSize > 2, "The device storage pack must contain at least two types.");

    __device_storage_tuple_pack() = default;
    __device_storage_tuple_pack(const sycl::queue& __q, const std::array<std::size_t, __TPackSize>& __n_array)
    {
        __initialize(__q, __n_array);
    }

protected:

    void
    __initialize(const sycl::queue& __q, const std::array<std::size_t, __TPackSize>& __n_array)
    {
        __tuple_for_each(
            [&](auto& __device_storage, std::size_t __idx)
            {
                __device_storage.__initialize(__q, __n_array[__idx]);
            },
            std::make_index_sequence<__TPackSize>{});
    }

    template <typename _Callable, std::size_t... _Is>
    void
    __tuple_for_each(_Callable&& __op, std::index_sequence<_Is...>)
    {
        (__op(std::get<_Is>(*this), _Is), ...);
    }
};

using oneapi::dpl::__internal::__access_mode_resolver_v;

template <typename _ModeTagT, typename _T>
auto
__get_accessor(_ModeTagT, __device_storage<_T>& __st, sycl::handler& __cgh, const sycl::property_list& __prop_list = {})
{
    return __st.template __get_accessor<__access_mode_resolver_v<_ModeTagT>>(__cgh, __prop_list);
}

template <typename _T>
struct __result_storage : public __device_storage<_T>
{
    static_assert(sycl::is_device_copyable_v<_T>, "The type _T must be device copyable to use __result_storage.");

    std::size_t __result_sz = 0;
    sycl::usm::alloc __kind = sycl::usm::alloc::unknown;

    __result_storage(const sycl::queue& __q, std::size_t __n) : __result_sz(__n)
    {
        assert(__result_sz > 0);
        _T* __ptr = __internal::__allocate_usm<_T, sycl::usm::alloc::host>(__q, __result_sz);
        if (__ptr)
        {
            this->__usm_buf = std::unique_ptr<_T, __internal::__sycl_usm_free>(__ptr, __internal::__sycl_usm_free{__q});
            __kind = sycl::usm::alloc::host;
        }
        else
        {
            this->__initialize(__q, __n);
            __kind = (this->__usm_buf) ? sycl::usm::alloc::device : sycl::usm::alloc::unknown;
        }
    }

    // Note: this function assumes a kernel has completed and the result can be transferred to host
    void
    __copy_result(_T* __dst, std::size_t __n)
    {
        this->__copy_n(__dst, __kind == sycl::usm::alloc::host ? this->__usm_buf.get() : nullptr,
                       __result_sz < __n ? __result_sz : __n, /*offset*/ 0);
    }
};

template <typename _TResult, typename _TScratch>
using __combined_storage_base_t = std::conditional_t<std::is_same_v<_TResult, _TScratch>, __device_storage<_TResult>,
                                                     __device_storage_tuple_pack<_TScratch, _TResult>>; 

template <typename _TResult, typename _TScratch = _TResult>
struct __combined_storage : public __combined_storage_base_t<_TResult, _TScratch>
{
    static_assert(sycl::is_device_copyable_v<_TResult>, "The type _TResult must be device copyable to use __combined_storage.");
    static_assert(sycl::is_device_copyable_v<_TScratch>, "The type _TScratch must be device copyable to use __combined_storage.");

    std::unique_ptr<_TResult, __internal::__sycl_usm_free> __result_buf = nullptr;
    std::size_t __sz = 0;
    std::size_t __result_sz = 0;
    sycl::usm::alloc __kind = sycl::usm::alloc::unknown;

    __combined_storage(const sycl::queue& __q, std::size_t __scratch_n, std::size_t __result_n)
        : __sz(__scratch_n), __result_sz(__result_n)
    {
        assert(__sz > 0 && __result_sz > 0);
        _TResult* __ptr = __internal::__allocate_usm<_TResult, sycl::usm::alloc::host>(__q, __result_sz);
        if (__ptr)
        {
            __result_buf = std::unique_ptr<_TResult, __internal::__sycl_usm_free>(__ptr, __internal::__sycl_usm_free{__q});
            this->__initialize(__q, __sz); // a separate scratch buffer
            __kind = sycl::usm::alloc::host;
        }
        else
        {
            if constexpr (std::is_same_v<_TResult, _TScratch>)
            {
                // a combined buffer, starting with scratch
                this->__initialize(__q, __sz + __result_sz);
            }
            else
            {
                // a combined buffer, starting with scratch
                this->__initialize(__q, std::array<std::size_t, 2>{/*scratch size*/ __result_sz, /*result size*/ __sz});
            }
            __kind = (this->__usm_buf) ? sycl::usm::alloc::device : sycl::usm::alloc::unknown;
        }
    }

    // Note: this function assumes a kernel has completed and the result can be transferred to host
    void
    __copy_result(_TResult* __dst, std::size_t __n)
    {
        this->__copy_n(__dst, __kind == sycl::usm::alloc::host ? __result_buf.get() : nullptr,
                       __result_sz < __n ? __result_sz : __n, /*offset*/ __sz);
    }

    template <typename _ModeTagT>
    friend auto
    __get_result_accessor(_ModeTagT, __combined_storage& __st, sycl::handler& __cgh,
                          const sycl::property_list& __prop_list = {})
    {
        if (__st.__kind == sycl::usm::alloc::host)
        {
            return __combi_accessor<_TResult, __access_mode_resolver_v<_ModeTagT>>(
                __cgh, __st.__sycl_buf, __st.__result_buf.get(), __prop_list);
        }
        else
        {
            if constexpr (std::is_same_v<_TResult, _TScratch>)
            {
                return __combi_accessor<_TResult, __access_mode_resolver_v<_ModeTagT>>(
                    __cgh, __st.__sycl_buf, __st.__usm_buf.get(), /*offset*/ __st.__sz, __st.__result_sz, __prop_list);
            }
            else
            {
                auto&& __res_st = std::get<1>(__st);

                // Offset is 0 because if _TResult and _TScratch are different types, we save result in the separate device storage without scratch data
                return __combi_accessor<_TResult, __access_mode_resolver_v<_ModeTagT>>(
                    __cgh, __res_st.__sycl_buf, __res_st.__usm_buf.get(),
                    /*offset*/ 0, __st.__result_sz, __prop_list);
            }
        }
    }

    template <typename _Forwarding>
    friend
    std::enable_if_t<std::is_same_v<std::decay_t<_Forwarding>, __combined_storage<_TResult, _TScratch>>, __copyable_storage_state<_TResult>>
    __move_state_from(_Forwarding&& __src)
    {
        return {std::move(__src.__result_buf), std::move(__src.__usm_buf), std::move(__src.__sycl_buf),
                __src.__sz, __src.__kind};
    }
};

template <typename _ModeTagT, typename _TResult, typename _TScratch>
auto
__get_accessor(_ModeTagT, __combined_storage<_TResult, _TScratch>& __st, sycl::handler& __cgh, const sycl::property_list& __prop_list = {})
{
    if constexpr (std::is_same_v<_TResult, _TScratch>)
    {
        __device_storage<_TResult>& __res_st = __st;
        return __get_accessor(_ModeTagT{}, __res_st, __cgh, __prop_list);
    }
    else
    {
        __device_storage<_TResult>& __res_st = std::get<0>(__st);
        return __get_accessor(_ModeTagT{}, __res_st, __cgh, __prop_list);
    }
}

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

//A contract for future class: <sycl::event or other event, a value, sycl::buffers..., or __usm_host_or_buffer_storage>
//Impl details: inheritance (private) instead of aggregation for enabling the empty base optimization.
template <typename _Event, typename... _Args>
class __future : private std::tuple<_Args...>
{
    _Event __my_event;

    template <typename _T>
    _T
    __wait_and_get_value(const sycl::buffer<_T>& __buf)
    {
        //according to a contract, returned value is one-element sycl::buffer
        return __buf.get_host_access(sycl::read_only)[0];
    }

    template <typename _T, std::size_t _NResults>
    _T
    __wait_and_get_value(const __result_and_scratch_storage<_T, _NResults>& __storage)
    {
        return __storage.__wait_and_get_value(__my_event);
    }

    std::pair<std::size_t, std::size_t>
    __wait_and_get_value(const std::shared_ptr<__result_and_scratch_storage_base>& __p_storage)
    {
        std::size_t __buf[2] = {0, 0};
        [[maybe_unused]] auto __n = __p_storage->__get_data(__my_event, __buf);
        assert(__n == 2);

        return {__buf[0], __buf[1]};
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
    __future(_Event __e, std::tuple<_Args...> __t) : std::tuple<_Args...>(__t), __my_event(__e) {}

    auto
    event() const
    {
        return __my_event;
    }
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
            // We should have this wait() call to ensure that the temporary data is not destroyed before the kernel code finished
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

    //The internal API. There are cases where the implementation specifies return value  "higher" than SYCL backend,
    //where a future is created.
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

// Processes a loop with a given stride. Intended to be used with sub-group / work-group strides for good memory access patterns
// (potentially with vectorization)
template <std::uint8_t __num_strides>
struct __strided_loop
{
    std::size_t __full_range_size;
    template <typename _IdxType, typename _LoopBodyOp, typename... _Args>
    void
    operator()(/*__is_full*/ std::true_type, _IdxType __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Args&&... __args) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __num_strides; ++__i)
        {
            __loop_body_op(std::true_type{}, __idx, __args...);
            __idx += __stride;
        }
    }
    template <typename _IdxType, typename _LoopBodyOp, typename... _Args>
    void
    operator()(/*__is_full*/ std::false_type, _IdxType __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Args&&... __args) const
    {
        // This operation improves safety by preventing underflow for unsigned types which would otherwise require a
        // check outside of the __strided_loop body.
        __idx = std::min<std::size_t>(__idx, __full_range_size);
        // Constrain the number of iterations as much as possible and then pass the knowledge that we are not a full loop to the body operation
        const std::uint8_t __adjusted_iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__full_range_size - __idx, __stride);
        for (std::uint8_t __i = 0; __i < __adjusted_iters_per_work_item; ++__i)
        {
            __loop_body_op(std::false_type{}, __idx, __args...);
            __idx += __stride;
        }
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PARALLEL_BACKEND_SYCL_UTILS_H
