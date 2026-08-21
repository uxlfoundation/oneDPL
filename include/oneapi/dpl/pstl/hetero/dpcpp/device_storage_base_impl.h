// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H
#define _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H

#include "../../../internal/common_config.h"
#include "../../onedpl_config.h"

#if _ONEDPL_BACKEND_SYCL

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "sycl_defs.h"
#include "../../utils.h"
#include "device_allocator_impl.h"

namespace oneapi::dpl::__internal
{

// Shared storage layer for the oneDPL device containers.
//
// Members are all protected, so a privately-inheriting derived class can re-export selected names with `using`.
template <typename _Tp, typename _Allocator>
class __device_storage_base
{
    static_assert(sycl::is_device_copyable_v<_Tp>,
                  "The element type of a oneDPL device container must be device copyable.");
    static_assert(!std::is_const_v<_Tp> && !std::is_reference_v<_Tp> && !std::is_void_v<_Tp>,
                  "oneDPL device containers require a non-const, non-reference object type.");

  protected:
    using value_type = _Tp;
    using size_type = std::size_t;
    using allocator_type = _Allocator;

    __device_storage_base(size_type __count, sycl::context __ctx, sycl::device __dev, allocator_type __alloc_arg)
        : __context(__ctx), __device(__dev), __alloc(std::move(__alloc_arg))
    {
        __data = __alloc.allocate(__count);
        __size = __count;
    }

    __device_storage_base(const __device_storage_base&) = delete;
    __device_storage_base&
    operator=(const __device_storage_base&) = delete;

    // sycl specification does not guarantee noexcept move for context and device
    __device_storage_base(__device_storage_base&& __other) noexcept(
        std::is_nothrow_move_constructible_v<sycl::context> && std::is_nothrow_move_constructible_v<sycl::device> &&
        std::is_nothrow_move_constructible_v<allocator_type>)
        : __data(__other.__data), __size(__other.__size), __context(std::move(__other.__context)),
          __device(std::move(__other.__device)), __alloc(std::move(__other.__alloc))
    {
        __other.__data = nullptr;
        __other.__size = 0;
    }

    __device_storage_base&
    operator=(__device_storage_base&& __other) noexcept(std::is_nothrow_move_assignable_v<sycl::context> &&
                                                        std::is_nothrow_move_assignable_v<sycl::device> &&
                                                        std::is_nothrow_move_assignable_v<allocator_type>)
    {
        if (this != &__other)
        {
            // deallocation explicitly catches and silences exceptions, so this will not throw
            __deallocate();

            __data = __other.__data;
            __size = __other.__size;
            __context = std::move(__other.__context);
            __device = std::move(__other.__device);
            __alloc = std::move(__other.__alloc);

            __other.__data = nullptr;
            __other.__size = 0;
        }
        return *this;
    }

    ~__device_storage_base() { __deallocate(); }

    size_type
    size() const noexcept
    {
        return __size;
    }

    bool
    empty() const noexcept
    {
        return __size == 0;
    }

    value_type*
    data() noexcept
    {
        return __data;
    }

    const value_type*
    data() const noexcept
    {
        return __data;
    }

    sycl::context
    get_context() const
    {
        return __context;
    }

    sycl::device
    get_device() const
    {
        return __device;
    }

    // -- Blocking transfer helpers --
    //
    // __count arrives already clamped by the caller. A count of zero skips the submission entirely.
    //
    // __copy_to/from use queue::memcpy internally and have the same requirements.
    // SYCL 2020 [4.9.4.3] requires each pointer passed to a memcpy to be either a host pointer or
    // a pointer within a USM allocation that is both accessible on the queue's device and created from the
    // queue's context. Since __data is device USM in __context on __device, that permits:
    //   - a host pointer,
    //   - host or shared USM created from __context,
    //   - device USM on __device created from __context, which is accessible from __device.

    void
    __copy_to(value_type* __dst, size_type __count, size_type __src_offset, sycl::queue __q,
              const sycl::event& __depends_on) const
    {
        if (__count > 0)
            __q.memcpy(__dst, __data + __src_offset, __count * sizeof(value_type), __depends_on).wait_and_throw();
    }

    void
    __copy_from(const value_type* __src, size_type __count, size_type __dst_offset, sycl::queue __q,
                const sycl::event& __depends_on)
    {
        if (__count > 0)
            __q.memcpy(__data + __dst_offset, __src, __count * sizeof(value_type), __depends_on).wait_and_throw();
    }

    void
    __fill_n(const value_type& __value, size_type __count, size_type __offset, sycl::queue __q,
             const sycl::event& __depends_on)
    {
        if (__count > 0)
            __q.fill(__data + __offset, __value, __count, __depends_on).wait_and_throw();
    }

    // Precondition: __pos < size(), must be checked by __check_element_pos() in the caller.
    value_type
    __read_at(size_type __pos, sycl::queue __q, const sycl::event& __depends_on) const
    {
        // Lazy storage to avoid requiring value_type to be default constructible. value_type is device copyable, so
        // copy construction is a bitwise copy and __space.__v may be treated as constructed after the
        // memcpy; its destructor must have no effect, so there is nothing to destroy.
        oneapi::dpl::__internal::__lazy_ctor_storage<value_type> __space;
        __q.memcpy(&__space.__v, __data + __pos, sizeof(value_type), __depends_on).wait_and_throw();
        oneapi::dpl::__internal::__scoped_destroyer<value_type> __destroy_when_leaving_scope{__space};
        return __space.__v;
    }

    sycl::queue
    __make_queue() const
    {
        return sycl::queue{__context, __device};
    }

    // The element count a bulk transfer of __requested elements starting at __offset performs.
    // __offset is a precondition and throws if violated; the count is not, and truncates.
    size_type
    __checked_count(size_type __requested, size_type __offset) const
    {
        if (__offset > __size)
            throw std::out_of_range("oneDPL device container: transfer offset is past the end of the container");
        return std::min(__requested, __size - __offset);
    }

    // For the single-element operations, which address one element and so cannot accept __pos == size().
    void
    __check_element_pos(size_type __pos) const
    {
        if (__pos >= __size)
            throw std::out_of_range("oneDPL device container: element position is out of range");
    }

    void
    __swap(__device_storage_base& __other)
    {
        using std::swap;
        swap(__data, __other.__data);
        swap(__size, __other.__size);
        swap(__context, __other.__context);
        swap(__device, __other.__device);
        swap(__alloc, __other.__alloc);
    }

    // Releasing the storage never throws. An implementation of sycl::free may report a failed release as a
    // synchronous exception. However, here we silence exceptions, so deallocation can be used within the destructor
    // and noexcept move assignment operator. This may result in a silent resource leak if deallocation fails.
    void
    __deallocate() noexcept
    {
        if (__data != nullptr)
        {
            // The ownership is dropped before the allocator call, so that a failed release will still reset size and
            // pointer to nullptr
            value_type* __to_free = std::exchange(__data, nullptr);
            size_type __count = std::exchange(__size, size_type{0});
            try
            {
                __alloc.deallocate(__to_free, __count);
            }
            catch (...)
            {
            }
        }
    }

    value_type* __data = nullptr;
    size_type __size = 0;
    sycl::context __context;
    sycl::device __device;
    allocator_type __alloc;
};

} // namespace oneapi::dpl::__internal

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H
