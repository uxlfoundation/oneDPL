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
#include <memory>
#include <type_traits>
#include <tuple>
#include <algorithm>

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
template <typename _ExecutionPolicy>
::std::string
__device_info(const _ExecutionPolicy& __policy)
{
    return __policy.queue().get_device().template get_info<sycl::info::device::name>();
}
#endif

template <typename _ExecutionPolicy>
std::size_t
__max_work_group_size(const _ExecutionPolicy& __policy, std::size_t __wg_size_limit = 8192)
{
    std::size_t __wg_size = __policy.queue().get_device().template get_info<sycl::info::device::max_work_group_size>();
    // Limit the maximum work-group size supported by the device to optimize the throughput or minimize communication
    // costs. This is limited to 8192 which is the highest current limit of the tested hardware (opencl:cpu devices) to
    // prevent huge work-group sizes returned on some devices (e.g., FPGU emulation).
    return std::min(__wg_size, __wg_size_limit);
}

template <typename _ExecutionPolicy, typename _Size>
_Size
__slm_adjusted_work_group_size(const _ExecutionPolicy& __policy, _Size __local_mem_per_wi, _Size __wg_size = 0)
{
    if (__wg_size == 0)
        __wg_size = __max_work_group_size(__policy);
    auto __local_mem_size = __policy.queue().get_device().template get_info<sycl::info::device::local_mem_size>();
    return std::min<_Size>(__local_mem_size / __local_mem_per_wi, __wg_size);
}

#if _ONEDPL_USE_SUB_GROUPS
template <typename _ExecutionPolicy>
::std::size_t
__max_sub_group_size(const _ExecutionPolicy& __policy)
{
    auto __supported_sg_sizes = __policy.queue().get_device().template get_info<sycl::info::device::sub_group_sizes>();
    //The result of get_info<sycl::info::device::sub_group_sizes>() can be empty; if so, return 0
    return __supported_sg_sizes.empty() ? 0 : __supported_sg_sizes.back();
}
#endif // _ONEDPL_USE_SUB_GROUPS

template <typename _ExecutionPolicy>
::std::uint32_t
__max_compute_units(const _ExecutionPolicy& __policy)
{
    return __policy.queue().get_device().template get_info<sycl::info::device::max_compute_units>();
}

template <typename _ExecutionPolicy>
bool
__supports_sub_group_size(const _ExecutionPolicy& __exec, std::size_t __target_size)
{
    const std::vector<std::size_t> __subgroup_sizes =
        __exec.queue().get_device().template get_info<sycl::info::device::sub_group_sizes>();
    return std::find(__subgroup_sizes.begin(), __subgroup_sizes.end(), __target_size) != __subgroup_sizes.end();
}

//-----------------------------------------------------------------------------
// Kernel run-time information helpers
//-----------------------------------------------------------------------------

template <typename _ExecutionPolicy>
::std::size_t
__kernel_work_group_size(const _ExecutionPolicy& __policy, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __policy.queue().get_device();
#if _ONEDPL_SYCL2020_KERNEL_DEVICE_API_PRESENT
    return __kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(__device);
#else
    return __kernel.template get_work_group_info<sycl::info::kernel_work_group::work_group_size>(__device);
#endif
}

template <typename _ExecutionPolicy>
::std::uint32_t
__kernel_sub_group_size(const _ExecutionPolicy& __policy, const sycl::kernel& __kernel)
{
    const sycl::device& __device = __policy.queue().get_device();
    [[maybe_unused]] const ::std::size_t __wg_size = __kernel_work_group_size(__policy, __kernel);
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
    template <typename _Exec>
    static auto
    __compile(_Exec&& __exec)
    {
        ::std::vector<sycl::kernel_id> __kernel_ids{sycl::get_kernel_id<_KernelNames>()...};

        auto __kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            __exec.queue().get_context(), {__exec.queue().get_device()}, __kernel_ids);

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
    template <typename _Exec>
    static auto
    __compile(_Exec&& __exec)
    {
        sycl::program __program(__exec.queue().get_context());

        using __return_type = std::conditional_t<(__kernel_count > 1), __kernel_array_type, sycl::kernel>;
        return __return_type{
            (__program.build_with_kernel_type<_KernelNames>(), __program.get_kernel<_KernelNames>())...};
    }
#endif
};
#endif // _ONEDPL_COMPILE_KERNEL

#if _ONEDPL_DEBUG_SYCL
template <typename _Policy>
inline void
// Passing policy by value should be enough for debugging
__print_device_debug_info(const _Policy& __policy, size_t __wg_size = 0, size_t __max_cu = 0)
{
    ::std::cout << "Device info" << ::std::endl;
    ::std::cout << " > device name:         " << oneapi::dpl::__internal::__device_info(__policy) << ::std::endl;
    ::std::cout << " > max compute units:   "
                << (__max_cu ? __max_cu : oneapi::dpl::__internal::__max_compute_units(__policy)) << ::std::endl;
    ::std::cout << " > max work-group size: "
                << (__wg_size ? __wg_size : oneapi::dpl::__internal::__max_work_group_size(__policy)) << ::std::endl;
}
#else
template <typename _Policy>
inline void
__print_device_debug_info(const _Policy&, size_t = 0, size_t = 0)
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
template <typename _ExecutionPolicy, typename _T>
class __buffer_impl
{
  private:
    using __container_t = typename __local_buffer<sycl::buffer<_T>>::type;

    __container_t __container;

  public:
    __buffer_impl(_ExecutionPolicy /*__exec*/, ::std::size_t __n_elements) : __container{sycl::range<1>(__n_elements)}
    {
    }

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

template <typename _ExecutionPolicy, typename _T>
struct __sycl_usm_free
{
    _ExecutionPolicy __exec;

    void
    operator()(_T* __memory) const
    {
        sycl::free(__memory, __exec.queue().get_context());
    }
};

template <typename _ExecutionPolicy, typename _T, sycl::usm::alloc __alloc_t>
struct __sycl_usm_alloc
{
    _ExecutionPolicy __exec;

    _T*
    operator()(::std::size_t __elements) const
    {
        const auto& __queue = __exec.queue();
        if (auto __buf = static_cast<_T*>(
                sycl::malloc(sizeof(_T) * __elements, __queue.get_device(), __queue.get_context(), __alloc_t)))
            return __buf;

        throw std::bad_alloc();
    }
};

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

template <typename _ExecutionPolicy, typename _T>
using __buffer = __internal::__buffer_impl<::std::decay_t<_ExecutionPolicy>, _T>;

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

template <typename _Accessor>
struct __usm_or_buffer_accessor
{
  private:
    using _T = std::decay_t<typename _Accessor::value_type>;
    _Accessor __acc;
    _T* __ptr = nullptr;
    bool __usm = false;
    size_t __offset = 0;

  public:
    // Buffer accessor
    __usm_or_buffer_accessor(sycl::handler& __cgh, sycl::buffer<_T, 1>* __sycl_buf,
                             const sycl::property_list& __prop_list)
        : __acc(*__sycl_buf, __cgh, __prop_list)
    {
    }
    __usm_or_buffer_accessor(sycl::handler& __cgh, sycl::buffer<_T, 1>* __sycl_buf, size_t __acc_offset,
                             const sycl::property_list& __prop_list)
        : __acc(*__sycl_buf, __cgh, __prop_list), __offset(__acc_offset)
    {
    }

    // USM pointer
    __usm_or_buffer_accessor(sycl::handler&, _T* __usm_buf, const sycl::property_list&)
        : __ptr(__usm_buf), __usm(true)
    {
    }
    __usm_or_buffer_accessor(sycl::handler&, _T* __usm_buf, size_t __ptr_offset, const sycl::property_list&)
        : __ptr(__usm_buf), __usm(true), __offset(__ptr_offset)
    {
    }

    auto
    __get_pointer() const // should be cached within a kernel
    {
        return __usm ? __ptr + __offset : &__acc[__offset];
    }
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

template <typename _ExecutionPolicy, typename _T>
struct __result_and_scratch_storage_impl : __result_and_scratch_storage_base
{
  private:
    using __sycl_buffer_t = sycl::buffer<_T, 1>;

    template <sycl::access_mode _AccessMode>
    using __accessor_t =
        sycl::accessor<_T, 1, _AccessMode, __dpl_sycl::__target_device, sycl::access::placeholder::false_t>;

    _ExecutionPolicy __exec;
    std::shared_ptr<_T> __scratch_buf;
    std::shared_ptr<_T> __result_buf;
    std::shared_ptr<__sycl_buffer_t> __sycl_buf;

    std::size_t __result_n;
    std::size_t __scratch_n;
    bool __use_USM_host;
    bool __supports_USM_device;

    // Only use USM host allocations on L0 GPUs. Other devices show significant slowdowns and will use a device allocation instead.
    inline bool
    __use_USM_host_allocations(sycl::queue __queue)
    {
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT && _ONEDPL_SYCL_L0_EXT_PRESENT
        auto __device = __queue.get_device();
        if (!__device.is_gpu())
            return false;
        if (!__device.has(sycl::aspect::usm_host_allocations))
            return false;
        if (__device.get_backend() != __dpl_sycl::__level_zero_backend)
            return false;
        return true;
#else
        return false;
#endif
    }

    inline bool
    __use_USM_allocations(sycl::queue __queue)
    {
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT
        return __queue.get_device().has(sycl::aspect::usm_device_allocations);
#else
        return false;
#endif
    }

  public:
    __result_and_scratch_storage_impl(const _ExecutionPolicy& __exec_, std::size_t __result_n, std::size_t __scratch_n)
        : __exec{__exec_}, __result_n{__result_n}, __scratch_n{__scratch_n},
          __use_USM_host{__use_USM_host_allocations(__exec.queue())},
          __supports_USM_device{__use_USM_allocations(__exec.queue())}
    {
        const std::size_t __total_n = __scratch_n + __result_n;
        // Skip in case this is a dummy container
        if (__total_n > 0)
        {
            if (__use_USM_host && __supports_USM_device)
            {
                // Separate scratch (device) and result (host) allocations on performant backends (i.e. L0)
                if (__scratch_n > 0)
                {
                    __scratch_buf = std::shared_ptr<_T>(
                        __internal::__sycl_usm_alloc<_ExecutionPolicy, _T, sycl::usm::alloc::device>{__exec}(
                            __scratch_n),
                        __internal::__sycl_usm_free<_ExecutionPolicy, _T>{__exec});
                }
                if (__result_n > 0)
                {
                    __result_buf = std::shared_ptr<_T>(
                        __internal::__sycl_usm_alloc<_ExecutionPolicy, _T, sycl::usm::alloc::host>{__exec}(__result_n),
                        __internal::__sycl_usm_free<_ExecutionPolicy, _T>{__exec});
                }
            }
            else if (__supports_USM_device)
            {
                // If we don't use host memory, malloc only a single unified device allocation
                __scratch_buf = std::shared_ptr<_T>(
                    __internal::__sycl_usm_alloc<_ExecutionPolicy, _T, sycl::usm::alloc::device>{__exec}(__total_n),
                    __internal::__sycl_usm_free<_ExecutionPolicy, _T>{__exec});
            }
            else
            {
                // If we don't have USM support allocate memory here
                __sycl_buf = std::make_shared<__sycl_buffer_t>(__sycl_buffer_t(__total_n));
            }
        }
    }

    template <typename _Acc>
    static auto
    __get_usm_or_buffer_accessor_ptr(const _Acc& __acc, [[maybe_unused]] std::size_t __scratch_n = 0)
    {
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT
        return __acc.__get_pointer();
#else
        return &__acc[__scratch_n];
#endif
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_result_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT
        if (__use_USM_host && __supports_USM_device)
            return __usm_or_buffer_accessor<__accessor_t<_AccessMode>>(__cgh, __result_buf.get(), __prop_list);
        else if (__supports_USM_device)
            return __usm_or_buffer_accessor<__accessor_t<_AccessMode>>(__cgh, __scratch_buf.get(), __scratch_n,
                                                                       __prop_list);
        return __usm_or_buffer_accessor<__accessor_t<_AccessMode>>(__cgh, __sycl_buf.get(), __scratch_n, __prop_list);
#else
        return __accessor_t<_AccessMode>(*__sycl_buf.get(), __cgh, __prop_list);
#endif
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_scratch_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
#if _ONEDPL_SYCL2020_DEFAULT_ACCESSOR_CONSTRUCTOR_PRESENT
        if (__use_USM_host || __supports_USM_device)
            return __usm_or_buffer_accessor<__accessor_t<_AccessMode>>(__cgh, __scratch_buf.get(), __prop_list);
        return __usm_or_buffer_accessor<__accessor_t<_AccessMode>>(__cgh, __sycl_buf.get(), __prop_list);
#else
        return __accessor_t<_AccessMode>(*__sycl_buf.get(), __cgh, __prop_list);
#endif
    }

    _T
    __wait_and_get_value(sycl::event __event) const
    {
        if (is_USM())
            __event.wait_and_throw();

        return __get_value();
    }

    // Note: this member function assumes the result is *ready*, since the __future has already
    // waited on the relevant event.
    _T
    __get_value(size_t idx = 0) const
    {
        assert(idx < __result_n);
        if (__use_USM_host && __supports_USM_device)
        {
            return *(__result_buf.get() + idx);
        }
        else if (__supports_USM_device)
        {
            _T __tmp;
            __exec.queue().memcpy(&__tmp, __scratch_buf.get() + __scratch_n + idx, 1 * sizeof(_T)).wait();
            return __tmp;
        }
        else
        {
            return __sycl_buf->get_host_access(sycl::read_only)[__scratch_n + idx];
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
        if (is_USM())
            __event.wait_and_throw();

        return __fill_data(__get_value(), __p_buf);
    }
};

template <typename _ExecutionPolicy, typename _T>
using __result_and_scratch_storage = __result_and_scratch_storage_impl<std::decay_t<_ExecutionPolicy>, _T>;

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
    constexpr _T
    __wait_and_get_value(const sycl::buffer<_T>& __buf)
    {
        //according to a contract, returned value is one-element sycl::buffer
        return __buf.get_host_access(sycl::read_only)[0];
    }

    // Here we use __result_and_scratch_storage_impl rather than __result_and_scratch_storage because we need to
    // match the type with the overload and are deducing the policy type. If we used __result_and_scratch_storage,
    // it would cause issues in type deduction due to decay of the policy in that using statement.
    template <typename _DecayedExecutionPolicy, typename _T>
    constexpr auto
    __wait_and_get_value(const __result_and_scratch_storage_impl<_DecayedExecutionPolicy, _T>& __storage)
    {
        return __storage.__wait_and_get_value(__my_event);
    }

    constexpr std::pair<std::size_t, std::size_t>
    __wait_and_get_value(const std::shared_ptr<__result_and_scratch_storage_base>& __p_storage)
    {
        std::size_t __buf[2] = {0, 0};
        [[maybe_unused]] auto __n = __p_storage->__get_data(__my_event, __buf);
        assert(__n == 2);

        return {__buf[0], __buf[1]};
    }

    template <typename _T>
    constexpr _T
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
            __deferrable_wait();
    }

    void
    __deferrable_wait()
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        wait();
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
    auto
    __make_future(_T __t) const
    {
        auto new_val = std::tuple<_T>(__t);
        auto new_tuple = std::tuple_cat(new_val, (std::tuple<_Args...>)*this);
        return __future<_Event, _T, _Args...>(__my_event, new_tuple);
    }
};

// Invoke a callable and pass a compile-time integer based on a provided run-time integer.
// The compile-time integer that will be provided to the callable is defined as the smallest
// value in the integer_sequence not less than the run-time integer. For example:
//
//   __static_monotonic_dispatcher<::std::integer_sequence<::std::uint16_t, 2, 4, 8, 16>::__dispatch(f, 3);
//
// will call f<4>(), since 4 is the smallest value in the sequence not less than 3.
//
// If there are no values in the sequence less than the run-time integer, the last value in
// the sequence will be used.
//
// Note that the integers provided in the integer_sequence must be monotonically increasing
template <typename>
class __static_monotonic_dispatcher;

template <::std::uint16_t _X, ::std::uint16_t... _Xs>
class __static_monotonic_dispatcher<::std::integer_sequence<::std::uint16_t, _X, _Xs...>>
{
    template <::std::uint16_t... _Vals>
    using _Head = typename ::std::conditional_t<
        sizeof...(_Vals) != 0,
        ::std::tuple_element<0, ::std::tuple<::std::integral_constant<::std::uint32_t, _Vals>...>>,
        ::std::integral_constant<::std::uint32_t, ::std::numeric_limits<::std::uint32_t>::max()>>::type;

    static_assert(_X < _Head<_Xs...>::value, "Sequence must be monotonically increasing");

  public:
    template <typename _F, typename... _Args>
    static auto
    __dispatch(_F&& __f, ::std::uint16_t __x, _Args&&... args)
    {
        if constexpr (sizeof...(_Xs) == 0)
        {
            return ::std::forward<_F>(__f).template operator()<_X>(::std::forward<_Args>(args)...);
        }
        else
        {
            if (__x <= _X)
                return ::std::forward<_F>(__f).template operator()<_X>(::std::forward<_Args>(args)...);
            else
                return __static_monotonic_dispatcher<::std::integer_sequence<::std::uint16_t, _Xs...>>::__dispatch(
                    ::std::forward<_F>(__f), __x, ::std::forward<_Args>(args)...);
        }
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
    operator()(_IdxType1 __idx_source, _IdxType2 __idx_dest, _SourceAcc __source_acc, _DestAcc __dest_acc) const
    {
        __transform(__source_acc[__idx_source], __dest_acc[__idx_dest]);
    }
    // Binary transformations into an output buffer
    template <typename _IdxType1, typename _IdxType2, typename _Source1Acc, typename _Source2Acc, typename _DestAcc>
    void
    operator()(_IdxType1 __idx_source, _IdxType2 __idx_dest, _Source1Acc __source1_acc, _Source2Acc __source2_acc,
               _DestAcc __dest_acc) const
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
    operator()(/*__is_full*/ std::true_type, const _Idx /*__elements_to_process*/, _Array __array) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __vec_size / 2; ++__i)
            std::swap(__array[__i], __array[__vec_size - __i - 1]);
    }
    template <typename _Idx, typename _Array>
    void
    operator()(/*__is_full*/ std::false_type, const _Idx __elements_to_process, _Array __array) const
    {
        for (std::uint8_t __i = 0; __i < __elements_to_process / 2; ++__i)
            std::swap(__array[__i], __array[__elements_to_process - __i - 1]);
    }
};

// Processes a loop with a given stride. Intended to be used with sub-group / work-group strides for good memory access patterns
// (potentially with vectorization)
template <std::uint8_t __num_strides>
struct __strided_loop
{
    std::size_t __full_range_size;
    template <typename _IdxType, typename _LoopBodyOp, typename... _Ranges>
    void
    operator()(/*__is_full*/ std::true_type, _IdxType __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Ranges&&... __rngs) const
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint8_t __i = 0; __i < __num_strides; ++__i)
        {
            __loop_body_op(std::true_type{}, __idx, __rngs...);
            __idx += __stride;
        }
    }
    template <typename _IdxType, typename _LoopBodyOp, typename... _Ranges>
    void
    operator()(/*__is_full*/ std::false_type, _IdxType __idx, std::uint16_t __stride, _LoopBodyOp __loop_body_op,
               _Ranges&&... __rngs) const
    {
        // Constrain the number of iterations as much as possible and then pass the knowledge that we are not a full loop to the body operation
        const std::uint8_t __adjusted_iters_per_work_item =
            oneapi::dpl::__internal::__dpl_ceiling_div(__full_range_size - __idx, __stride);
        for (std::uint8_t __i = 0; __i < __adjusted_iters_per_work_item; ++__i)
        {
            __loop_body_op(std::false_type{}, __idx, __rngs...);
            __idx += __stride;
        }
    }
};

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif //_ONEDPL_PARALLEL_BACKEND_SYCL_UTILS_H
