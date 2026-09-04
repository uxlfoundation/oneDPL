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
#include <algorithm> // std::copy_n, std::min
#include <utility>
#include <cstddef>
#include <cassert>

#include <new> // std::bad_alloc - only used in __sycl_usm_alloc

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

// avoids a runtime call if there is nothing to free
inline void
__free_usm(const sycl::queue& __q, void* __memory)
{
    if (__memory)
        sycl::free(__memory, __q);
}

struct __sycl_usm_free
{
    std::optional<sycl::queue> __q;

    void
    operator()(void* __memory) const
    {
        assert(__q.has_value());
        __free_usm(*__q, __memory);
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

// __scratch_keepalive and __result_keepalive are deliberately simple structs with raw pointers
// and no ownership semantics. They should only be used as an implementation detail of storage
// ownership and transfer utilities, with memory and lifetime safety ensured at that level.

// type-erased lifetime keeper for temporary storage (either USM or sycl::buffer)
struct __scratch_keepalive
{
    void* __usm_ptr = nullptr;
    std::optional<sycl::buffer<std::byte, 1>> __sycl_buf;
};

// struct to keep the result data in either USM or sycl::buffer
// If __kind == sycl::usm::alloc::host, __usm_ptr points directly to the result.
// If __kind == sycl::usm::alloc::device, the result is at __usm_ptr + __offset in device memory.
// If __kind == sycl::usm::alloc::unknown, the result is in __sycl_buf at __offset.
template <typename _T>
struct __result_keepalive
{
    _T* __usm_ptr = nullptr;
    std::optional<sycl::buffer<_T, 1>> __sycl_buf;
    std::size_t __result_sz = 0;
    std::size_t __offset = 0;
    sycl::usm::alloc __kind = sycl::usm::alloc::unknown;
};

// Extracts data to the given destination array
template <typename _T>
void
__copy_n(_T* __dst, std::size_t __n, const __result_keepalive<_T>& __ka, sycl::queue& __q)
{
    const std::size_t __count = std::min(__n, __ka.__result_sz);
    if (__ka.__kind == sycl::usm::alloc::host)
    {
        std::copy_n(__ka.__usm_ptr, __count, __dst);
    }
    else if (__ka.__kind == sycl::usm::alloc::device)
    {
        assert(__ka.__usm_ptr);
        __q.memcpy(__dst, __ka.__usm_ptr + __ka.__offset, __count * sizeof(_T)).wait();
    }
    else
    {
        assert(__ka.__kind == sycl::usm::alloc::unknown && __ka.__sycl_buf.has_value());
        std::copy_n(__ka.__sycl_buf->get_host_access(sycl::read_only).begin() + __ka.__offset, __count, __dst);
    }
}

// Sentinel type used as a compile-time conditional stand-in for result storage or accessor
struct __no_result_needed_tag
{
    using type = std::size_t; // a safe default
};

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

// A function-style "trait" to apply to a result of __get_accessor
template <typename _T>
constexpr bool
__is_real_accessor(const _T&)
{
    return false;
}
template <typename _T, sycl::access_mode _AccessMode>
constexpr bool
__is_real_accessor(const __combi_accessor<_T, _AccessMode>&)
{
    return true;
}

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

    void
    __move_state_to(__internal::__scratch_keepalive& __ka) &&
    {
        __move_base_state_to(__ka);
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

    // Similar in logic to __internal::__copy_n but optimized for use with __*_storage
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

    template <typename _Keepalive>
    void
    __move_base_state_to(_Keepalive& __ka) &&
    {
        if (__usm_buf)
            __ka.__usm_ptr = __usm_buf.release();
        else
        {
            if constexpr (std::is_same_v<_Keepalive, __internal::__scratch_keepalive>)
                __ka.__sycl_buf = __sycl_buf.template reinterpret<std::byte>();
            else
                __ka.__sycl_buf = std::move(__sycl_buf);
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

template <typename _ModeTagT>
auto
__get_accessor(_ModeTagT, __internal::__no_result_needed_tag&, sycl::handler&, const sycl::property_list& = {})
{
    return __internal::__no_result_needed_tag{};
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

    void
    __move_state_to(__internal::__result_keepalive<_T>& __ka) &&
    {
        __ka.__kind = __kind;
        __ka.__result_sz = __result_sz;
        __ka.__offset = 0;
        __move_base_state_to(__ka);
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

    void*
    __move_state_to(__internal::__result_keepalive<_T>& __ka) &&
    {
        void* __scratch_ptr = nullptr;
        __ka.__kind = __kind;
        __ka.__result_sz = __result_sz;
        __ka.__offset = __sz;
        __move_base_state_to(__ka);
        if (__kind == sycl::usm::alloc::host)
        {
            __scratch_ptr = __ka.__usm_ptr;
            __ka.__usm_ptr = __result_buf.release();
        }
        return __scratch_ptr;
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

template <std::size_t _NScratch, typename... _ResultTypes>
class __storage_holder
{
    sycl::queue __q;
    std::tuple<__internal::__result_keepalive<_ResultTypes>...> __result_slots = {};
    std::array<__internal::__scratch_keepalive, _NScratch> __scratch_slots = {};
    std::size_t __scratch_count = 0;

  public:
    explicit __storage_holder(const sycl::queue& __q) : __q(__q) {}

    __storage_holder(const __storage_holder&) = delete;
    __storage_holder& operator=(const __storage_holder&) = delete;

    __storage_holder(__storage_holder&& __other)
        : __q(std::move(__other.__q)), __result_slots(std::move(__other.__result_slots)),
          __scratch_slots(std::move(__other.__scratch_slots)), __scratch_count(__other.__scratch_count)
    {
        __other.__scratch_count = 0;
        for (auto& __ka : __other.__scratch_slots)
            __ka.__usm_ptr = nullptr;
        std::apply([](auto&... __ka)
        {
            ((__ka.__usm_ptr = nullptr), ...);
        }, __other.__result_slots);
    }

    __storage_holder&
    operator=(__storage_holder&& __other)
    {
        assert(this != &__other);
        using std::swap;
        swap(__q, __other.__q);
        swap(__scratch_count, __other.__scratch_count);
        swap(__scratch_slots, __other.__scratch_slots);
        swap(__result_slots, __other.__result_slots);
        return *this;
    }

    ~__storage_holder()
    {
        for (auto& __ka : __scratch_slots)
            __free_usm(__q, __ka.__usm_ptr);
        std::apply([this](auto&... __ka)
        {
            ((__free_usm(__q, __ka.__usm_ptr)), ...);
        }, __result_slots);
    }

    template <typename _T>
    void
    __deposit(__device_storage<_T>&& __st)
    {
        assert(__scratch_count < _NScratch);
        std::move(__st).__move_state_to(__scratch_slots[__scratch_count++]);
    }

    template <std::size_t _I, typename _T>
    void
    __deposit(__result_storage<_T>&& __st)
    {
        static_assert(_I < sizeof...(_ResultTypes), "Result slot index out of range");
        static_assert(std::is_same_v<_T, std::tuple_element_t<_I, std::tuple<_ResultTypes...>>>);
        auto& __ka = std::get<_I>(__result_slots);
        assert(__ka.__usm_ptr == nullptr && !__ka.__sycl_buf.has_value());
        std::move(__st).__move_state_to(__ka);
    }

    template <std::size_t _I, typename _T>
    void
    __deposit(__combined_storage<_T>&& __st)
    {
        static_assert(_I < sizeof...(_ResultTypes), "Result index out of range");
        static_assert(std::is_same_v<_T, std::tuple_element_t<_I, std::tuple<_ResultTypes...>>>);
        auto& __ka = std::get<_I>(__result_slots);
        assert(__ka.__usm_ptr == nullptr && !__ka.__sycl_buf.has_value());
        void* __scratch_ptr = std::move(__st).__move_state_to(__ka);
        if (__scratch_ptr)
        {
            assert(__scratch_count < _NScratch);
            __scratch_slots[__scratch_count++].__usm_ptr = __scratch_ptr;
        }
    }

    template <std::size_t _I>
    void
    __copy_result(std::tuple_element_t<_I, std::tuple<_ResultTypes...>>* __dst, std::size_t __n)
    {
        __internal::__copy_n(__dst, __n, std::get<_I>(__result_slots), __q);
    }
};

template <bool _Condition, typename _T>
auto
__create_result_storage_opt(sycl::queue& __q, std::size_t __n)
{
    if constexpr (_Condition)
        return __result_storage<_T>(__q, __n);
    else
        return __internal::__no_result_needed_tag{};
}

} // namespace __par_backend_hetero
} // namespace oneapi::dpl

#endif //_ONEDPL_UTILS_STORAGE_SYCL_H
