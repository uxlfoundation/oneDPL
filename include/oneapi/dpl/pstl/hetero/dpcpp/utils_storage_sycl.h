// -*- C++ -*-
//===-- parallel_backend_sycl_utils.h -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_UTILS_STORAGE_SYCL_H
#define _ONEDPL_UTILS_STORAGE_SYCL_H
//!!! NOTE: This file should be included under the macro _ONEDPL_BACKEND_SYCL

#include <memory>
#include <type_traits>
#include <tuple>
#include <optional>
#include <algorithm> // std::copy_n
#include <cstddef>
#include <cassert>

#include <new> // std::bad_alloc - only used in __sycl_usm_alloc
#include <utility> // std::move - only used in __move_state

#include "sycl_defs.h"
#include "sycl_iterator.h"
#include "../../tuple_impl.h"

namespace oneapi::dpl
{
namespace __par_backend_hetero
{

namespace __internal
{

template <typename _Unknown>
struct __local_buffer;

template <int __dim, typename _AllocT, typename _T>
struct __local_buffer<sycl::buffer<_T, __dim, _AllocT>>
{
    using type = sycl::buffer<_T, __dim, _AllocT>;
};

//if we take std::tuple as a type for buffer we should convert to internal::tuple
template <int __dim, typename _AllocT, typename... _T>
struct __local_buffer<sycl::buffer<std::tuple<_T...>, __dim, _AllocT>>
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

} // namespace __internal

template <typename _T>
using __buffer = __internal::__buffer_impl<_T>;

//-----------------------------------------------------------------------
// types to create and use data on a device and return those to the host
//-----------------------------------------------------------------------

// The type to exchange information between storage types.
// Useful for the interoperability during the transition period
// TODO: afterwards, remove together with __combined_storage::__move_state
template <typename _T>
struct __copyable_storage_state
{
    std::shared_ptr<_T> __result_buf;
    std::shared_ptr<_T> __scratch_buf;
    sycl::buffer<_T, 1> __sycl_buf;
    std::size_t         __scratch_sz = 0;
    sycl::usm::alloc    __kind = sycl::usm::alloc::unknown;
};

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
    using type = _T;

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
    using type = _T;

    static_assert(sycl::is_device_copyable_v<_T>, "The type _T must be device copyable to use __combined_storage.");

    std::unique_ptr<_T, __internal::__sycl_usm_free> __result_buf = nullptr;
    std::size_t __sz = 0;
    std::size_t __result_sz = 0;
    sycl::usm::alloc __kind = sycl::usm::alloc::unknown;

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

    __copyable_storage_state<_T>
    __move_state() &&
    {
        return {std::move(__result_buf), std::move(this->__usm_buf), std::move(this->__sycl_buf), __sz, __kind};
    }
};

template <typename _T, template <typename> typename _Storage>
std::enable_if_t<std::is_default_constructible_v<_T>, _T>
__load_result(_Storage<_T>& __storage)
{
    _T __result{};
    __storage.__copy_result(&__result, 1);
    return __result;
}

} // namespace __par_backend_hetero
} // namespace oneapi::dpl

#endif //_ONEDPL_UTILS_STORAGE_SYCL_H
