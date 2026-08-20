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
#include <functional>
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

template <>
struct __is_comp_ascending<oneapi::dpl::__internal::__pstl_less>
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
template <>
struct __is_comp_descending<oneapi::dpl::__internal::__pstl_greater>
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
    using type = sycl::buffer<
        oneapi::dpl::__internal::tuple<_T...>, __dim,
        typename std::allocator_traits<_AllocT>::template rebind_alloc<oneapi::dpl::__internal::tuple<_T...>>>;
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

template <typename _T>
struct __device_storage
{
    using type = _T;

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

template <typename _T>
struct __combined_storage : public __device_storage<_T>
{
    static_assert(sycl::is_device_copyable_v<_T>, "The type _T must be device copyable to use __combined_storage.");

    std::unique_ptr<_T, __internal::__sycl_usm_free> __result_buf = nullptr;
    std::size_t __sz = 0;
    std::size_t __result_sz = 0;
    sycl::usm::alloc __kind = sycl::usm::alloc::unknown;

    __combined_storage() = default;
    __combined_storage(const sycl::queue& __q, std::size_t __scratch_n, std::size_t __result_n)
        : __sz(__scratch_n), __result_sz(__result_n)
    {
        assert(__sz > 0 && __result_sz > 0);
        _T* __ptr = __internal::__allocate_usm<_T, sycl::usm::alloc::host>(__q, __result_sz);
        if (__ptr)
        {
            __result_buf = std::unique_ptr<_T, __internal::__sycl_usm_free>(__ptr, __internal::__sycl_usm_free{__q});
            this->__initialize(__q, __sz); // a separate scratch buffer
            __kind = sycl::usm::alloc::host;
        }
        else
        {
            this->__initialize(__q, __sz + __result_sz); // a combined buffer, starting with scratch
            __kind = (this->__usm_buf) ? sycl::usm::alloc::device : sycl::usm::alloc::unknown;
        }
    }

    // Note: this function assumes a kernel has completed and the result can be transferred to host
    void
    __copy_result(_T* __dst, std::size_t __n)
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
            return __combi_accessor<_T, __access_mode_resolver_v<_ModeTagT>>(
                __cgh, __st.__sycl_buf, __st.__result_buf.get(), __prop_list);
        }
        else
        {
            return __combi_accessor<_T, __access_mode_resolver_v<_ModeTagT>>(
                __cgh, __st.__sycl_buf, __st.__usm_buf.get(), /*offset*/ __st.__sz, __st.__result_sz, __prop_list);
        }
    }
};

// A trait to detect __device_storage and the storages derived from it
template <typename _T, typename = void>
struct __is_device_storage : std::false_type
{
};

template <typename _T>
struct __is_device_storage<_T, std::void_t<typename _T::type>>
    : std::bool_constant<std::is_base_of_v<__device_storage<typename _T::type>, _T>>
{
};

template <typename _T>
constexpr bool __is_device_storage_v = __is_device_storage<std::decay_t<_T>>::value;

// A device storage is filled by a kernel, so waiting is required before its data may be used or released
template <typename _T>
using __wait_required_of_finalize_sycl_call = __is_device_storage<_T>;

// A storage carries
template <typename _T, typename = void>
struct __has_copy_result_method : std::false_type
{
};

template <typename _T>
struct __has_copy_result_method<_T, std::void_t<decltype(std::declval<_T&>().__copy_result(
                                        std::declval<typename _T::type*>(), std::size_t{}))>> : std::true_type
{
};

template <typename _T>
constexpr bool __has_copy_result_method_v = __has_copy_result_method<std::decay_t<_T>>::value;

// Load a single result value from the storage.
template <typename _T, template <typename> typename _Storage>
_T
__load_result(_Storage<_T>& __storage)
{
    oneapi::dpl::__internal::__lazy_ctor_storage<typename _Storage<_T>::type> __space;
    __storage.__copy_result(&__space.__v, 1);

    return __space.__v;
}

template <typename _BackendTag>
struct __hetero_event;

template <>
struct __hetero_event<oneapi::dpl::__internal::__device_backend_tag>
{
    using __type = sycl::event;

    sycl::event __event;

    __hetero_event() = default;
    __hetero_event(sycl::event&& __event) : __event(std::move(__event)) {}

    void
    wait()
    {
        __event.wait();
    }

    void
    wait_and_throw()
    {
        __event.wait_and_throw();
    }

    operator sycl::event() const { return __event; }
};

template <typename>
struct __is_hetero_event : std::false_type
{
};

template <>
struct __is_hetero_event<sycl::event> : std::true_type
{
};

template <>
struct __is_hetero_event<__hetero_event<oneapi::dpl::__internal::__device_backend_tag>> : std::true_type
{
};

#if _ONEDPL_FPGA_DEVICE
template <>
struct __hetero_event<oneapi::dpl::__internal::__fpga_backend_tag>
    : public __hetero_event<oneapi::dpl::__internal::__device_backend_tag>
{
    using __base = __hetero_event<oneapi::dpl::__internal::__device_backend_tag>;

    using __base::__base;
    using __base::operator=;
};

template <>
struct __is_hetero_event<__hetero_event<oneapi::dpl::__internal::__fpga_backend_tag>> : std::true_type
{
};
#endif // _ONEDPL_FPGA_DEVICE

template <typename _TEvent>
constexpr bool __is_hetero_event_v = __is_hetero_event<std::decay_t<_TEvent>>::value;

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

template <typename _WaitModeTag = __sync_mode, typename _TEvent>
std::enable_if_t<__is_hetero_event_v<_TEvent>>
__finalize_call(_TEvent&& __event)
{
    if constexpr (std::is_same_v<_WaitModeTag, __async_mode>)
    {
        // no op
    }
    else if constexpr (std::is_same_v<_WaitModeTag, __sync_mode>)
    {
        __event.wait_and_throw();
    }
    else if constexpr (std::is_same_v<_WaitModeTag, __deferrable_mode>)
    {
#if !ONEDPL_ALLOW_DEFERRED_WAITING
        __event.wait_and_throw();
#endif
    }
    else
    {
        static_assert(sizeof(_WaitModeTag) == 0, "Unknown _WaitModeTag");
    }
}

template <typename _WaitModeTag, typename... _Args>
using __resolve_wait_mode =
    std::conditional_t<(__wait_required_of_finalize_sycl_call<std::decay_t<_Args>>::value || ...), __sync_mode,
                       _WaitModeTag>;

// The tuple is taken by a non-const lvalue reference on purpose:
// sycl::event::wait_and_throw() is non-const, and the payload of the tuple must outlive the waiting,
// so passing a temporary tuple here is prohibited.
template <typename _WaitModeTag = __sync_mode, template <typename...> typename _Tuple, typename... _Args>
std::enable_if_t<!__is_hetero_event_v<_Tuple<_Args...>>>
__finalize_call(_Tuple<_Args...>& __tuple)
{
    __finalize_call<__resolve_wait_mode<_WaitModeTag, _Args...>>(std::get<0>(__tuple));
}

// A copyable wrapper for a move-only payload which has to be kept alive until the kernel completes.
// The payload may additionally carry an algorithm result, see __is_result_payload below.
template <typename _T>
struct __lifetime_payload
{
    std::shared_ptr<_T> __data;
};

template <typename>
struct __is_lifetime_payload : std::false_type
{
};

template <typename _T>
struct __is_lifetime_payload<__lifetime_payload<_T>> : std::true_type
{
};

template <typename _T>
constexpr bool __is_lifetime_payload_v = __is_lifetime_payload<std::decay_t<_T>>::value;

// A payload which keeps the data alive and, additionally, carries an algorithm result readable on the host
template <typename>
struct __is_result_payload : std::false_type
{
};

template <typename _T>
struct __is_result_payload<__lifetime_payload<_T>> : std::bool_constant<__has_copy_result_method_v<_T>>
{
};

template <typename _T>
constexpr bool __is_result_payload_v = __is_result_payload<std::decay_t<_T>>::value;

// Returns the index of the first true value in the pack, or the pack size if there is no such value
template <bool... _Vals>
constexpr std::size_t
__find_first_true()
{
    constexpr bool __vals[] = {_Vals..., true};
    std::size_t __idx = 0;
    while (!__vals[__idx])
        ++__idx;

    return __idx;
}

//A contract for future class: <sycl::event or other event, payload items: a value or __lifetime_payload>
//Impl details: inheritance (private) instead of aggregation for enabling the empty base optimization.
template <typename _BackendTag, typename... _Args>
class __future : private std::tuple<_Args...>
{
    __hetero_event<_BackendTag> __my_event;

    // The index of the first payload item which is a plain value, i.e. not a payload kept alive for a kernel
    static constexpr std::size_t __value_index = __find_first_true<!__is_lifetime_payload_v<_Args>...>();

    // The index of the first payload item which carries an algorithm result readable on the host
    static constexpr std::size_t __result_index = __find_first_true<__is_result_payload_v<_Args>...>();

  public:
    __future(__hetero_event<_BackendTag> __e, _Args... __args)
        : std::tuple<_Args...>(__args...), __my_event(std::move(__e))
    {
    }

    auto
    event() const
    {
        return __my_event;
    }

    using __native_event_t = typename __hetero_event<_BackendTag>::__type;
    operator __native_event_t() const { return event(); }

    void
    wait()
    {
        __my_event.wait_and_throw();
    }

    auto
    get()
    {
        wait();

        // Return the first plain value if there is one, otherwise the result carried by a payload.
        // If there is neither of them, the return type is void.
        if constexpr (__value_index < sizeof...(_Args))
            return std::get<__value_index>(*this);
        else if constexpr (__result_index < sizeof...(_Args))
            return __load_result(*std::get<__result_index>(*this).__data);
    }
};

template <typename _SrcDataT>
std::enable_if_t<!__is_device_storage_v<_SrcDataT>, std::decay_t<_SrcDataT>>
__to_future_payload(_SrcDataT&& __data)
{
    return std::forward<_SrcDataT>(__data);
}

// A device storage is a move-only payload which is required to keep the data alive until the kernel completes.
// It may also carry an algorithm result, but __future must stay copyable, so such a payload is kept
// by a shared ownership.
template <typename _Storage>
std::enable_if_t<__is_device_storage_v<_Storage>, __lifetime_payload<std::decay_t<_Storage>>>
__to_future_payload(_Storage&& __storage)
{
    using __storage_t = std::decay_t<_Storage>;
    return __lifetime_payload<__storage_t>{std::make_shared<__storage_t>(std::move(__storage))};
}

// Additional payload items (__extra) are placed before the items of __res: the first payload item
// is the one returned by __future::get()
template <typename _BackendTag, typename... _Args, typename... _ExtraArgs>
auto
__create_future(std::tuple<__hetero_event<_BackendTag>, _Args...> __res, _ExtraArgs&&... __extra)
{
    static_assert(sizeof...(_ExtraArgs) <= 1, "At most one additional payload item is expected");

    return std::apply(
        [&](auto&& __event, auto&&... __args) {
            return __future(std::forward<decltype(__event)>(__event), std::forward<_ExtraArgs>(__extra)...,
                            __to_future_payload(std::forward<decltype(__args)>(__args))...);
        },
        std::move(__res));
}

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
