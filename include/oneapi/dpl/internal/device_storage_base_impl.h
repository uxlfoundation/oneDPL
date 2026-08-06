// -*- C++ -*-
//===-- device_storage_base_impl.h ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H
#define _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "common_config.h"
#include "../pstl/onedpl_config.h"

// The storage layer owns a USM device allocation, so it exists only with the SYCL backend. Without it
// this header declares nothing, following the convention of the other internal headers.
#if _ONEDPL_BACKEND_SYCL

#include "../pstl/hetero/dpcpp/sycl_defs.h"
#include "../pstl/utils.h"
#include "device_allocator_impl.h"

namespace oneapi::dpl::__internal
{

// Shared storage layer for the oneDPL device containers.
//
// It owns a USM device allocation, its element count, the sycl::context and sycl::device it lives
// on, and the allocator instance. Everything except the element type static_asserts is protected:
// a privately-inheriting derived class can call it and re-export selected names with `using`, while
// nothing leaks to users.
//
// The transfer helpers are deliberately element-count based, take a *pre-clamped* count, and do no
// span unwrapping or queue creation. That is exactly the layer a future compat::device_vector can
// reuse unchanged, since its public surface (iterators, proxy references, resize copy-forward) has
// different argument types but identical underlying operations.
//
// Deliberately absent, arriving with compat::device_vector: resize, reserve, capacity,
// shrink_to_fit, clear, get_allocator.
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

    // The context, device and allocator are established in the mem-initializer list; the allocation
    // happens in the body and _M_size is set only after allocate() returns. If allocate() throws,
    // no destructor for this base runs and _M_data is still nullptr, so nothing leaks.
    //
    // The allocator is stateful and carries its own context and device, following
    // sycl::usm_allocator, so it has no default and is always supplied by the caller. The context and
    // device are stored here as well rather than read back from the allocator on demand: they are
    // part of the container's own observable interface (get_context() / get_device() / __make_queue())
    // and must not depend on an arbitrary user-supplied allocator exposing accessors for them.
    __device_storage_base(size_type __count, sycl::context __ctx, sycl::device __dev, _Allocator __alloc)
        : _M_context(std::move(__ctx)), _M_device(std::move(__dev)), _M_alloc(std::move(__alloc))
    {
        // A zero-element container allocates nothing and holds nullptr. sycl::malloc_device(0) is
        // unspecified: some backends return a non-null pointer that cannot be freed. The current
        // device_allocator already short-circuits count == 0, but the base must not depend on that
        // for an arbitrary user-supplied allocator, so the call is skipped here as well.
        if (__count != 0)
        {
            _M_data = _M_alloc.allocate(__count);
            _M_size = __count;
        }
    }

    __device_storage_base(const __device_storage_base&) = delete;
    __device_storage_base&
    operator=(const __device_storage_base&) = delete;

    // The context, device and allocator are *copied* rather than moved, so that a moved-from object
    // retains them. sycl::context has no cheap default constructor (its default path builds a real
    // platform context), both SYCL handles are shared_ptr wrappers so retaining them is free, and it
    // keeps get_context()/get_device()/size()/empty() well-defined and the object a legal
    // move-assignment target.
    __device_storage_base(__device_storage_base&& __other) noexcept
        : _M_data(__other._M_data), _M_size(__other._M_size), _M_context(__other._M_context),
          _M_device(__other._M_device), _M_alloc(__other._M_alloc)
    {
        __other._M_data = nullptr;
        __other._M_size = 0;
    }

    __device_storage_base&
    operator=(__device_storage_base&& __other) noexcept
    {
        if (this == &__other)
            return *this;

        __deallocate();

        _M_data = __other._M_data;
        _M_size = __other._M_size;
        // Overwriting the context, device and allocator is mandatory, not an optimization: the source
        // may live on a *different* device, and keeping our old context or allocator would later free
        // the stolen allocation against the wrong context. This is also what
        // device_allocator::propagate_on_container_move_assignment asks for.
        _M_context = __other._M_context;
        _M_device = __other._M_device;
        _M_alloc = __other._M_alloc;

        __other._M_data = nullptr;
        __other._M_size = 0;

        return *this;
    }

    // Must not throw. __deallocate() is a no-op on a moved-from object.
    ~__device_storage_base() { __deallocate(); }

    size_type
    size() const noexcept
    {
        return _M_size;
    }

    bool
    empty() const noexcept
    {
        return _M_size == 0;
    }

    _Tp*
    data() noexcept
    {
        return _M_data;
    }

    const _Tp*
    data() const noexcept
    {
        return _M_data;
    }

    sycl::context
    get_context() const
    {
        return _M_context;
    }

    sycl::device
    get_device() const
    {
        return _M_device;
    }

    // -- Blocking transfer helpers --
    //
    // __count arrives already clamped by the caller. A count of zero performs no submission at all,
    // which also avoids the SYCL rule that memcpy throws when handed a null pointer -- relevant for
    // the size-0 case, where _M_data is nullptr.
    //
    // __depends_on is forwarded unconditionally, including a default-constructed event: sycl::event()
    // is a ready event, so depending on it is a no-op. One code path, no branch.
    //
    // wait_and_throw() rather than wait(), so asynchronous errors on the user's queue surface at the
    // transfer call site.
    //
    // The queue is copied into a local because sycl::queue's memcpy/fill shortcuts are not
    // const-qualified. A queue is a shared_ptr-like handle, so the copy is cheap.
    void
    __copy_to_host(_Tp* __dst, size_type __count, size_type __src_offset, const sycl::queue& __q,
                   const sycl::event& __depends_on) const
    {
        if (__count == 0)
            return;

        sycl::queue __queue = __q;
        __queue.memcpy(__dst, _M_data + __src_offset, __count * sizeof(_Tp), __depends_on).wait_and_throw();
    }

    void
    __copy_from_host(const _Tp* __src, size_type __count, size_type __dst_offset, const sycl::queue& __q,
                     const sycl::event& __depends_on)
    {
        if (__count == 0)
            return;

        sycl::queue __queue = __q;
        __queue.memcpy(_M_data + __dst_offset, __src, __count * sizeof(_Tp), __depends_on).wait_and_throw();
    }

    void
    __fill_n(const _Tp& __value, size_type __count, size_type __offset, const sycl::queue& __q,
             const sycl::event& __depends_on)
    {
        if (__count == 0)
            return;

        sycl::queue __queue = __q;
        __queue.fill(_M_data + __offset, __value, __count, __depends_on).wait_and_throw();
    }

    // Precondition: __pos < size(), checked by __check_element_pos() in the caller. Unlike the bulk
    // transfers this cannot clamp, since it must return a value.
    _Tp
    __read_at(size_type __pos, const sycl::queue& __q, const sycl::event& __depends_on) const
    {
        // Avoid requiring _Tp to be default constructible. Since _Tp is device copyable, copy
        // construction is equivalent to a bitwise copy, so __space.__v may be treated as constructed
        // after the memcpy. There is no need to destroy it afterwards, as the destructor must have no
        // effect.
        oneapi::dpl::__internal::__lazy_ctor_storage<_Tp> __space;
        sycl::queue __queue = __q;
        __queue.memcpy(&__space.__v, _M_data + __pos, sizeof(_Tp), __depends_on).wait_and_throw();
        return __space.__v;
    }

    // -- Plumbing --

    // Constructing a queue per call is measurable overhead; the queue-taking public overloads are the
    // performance path. This is the documented cost of storing a context rather than a queue.
    sycl::queue
    __make_queue() const
    {
        return sycl::queue{_M_context, _M_device};
    }

    // The element count a bulk transfer of __requested elements starting at __offset performs.
    //
    // __offset is a precondition: it must be <= size(), where an offset of exactly size() names the
    // end of the range and yields an empty transfer. A violation is a programming error and throws
    // std::out_of_range, rather than silently transferring nothing, which would hide the mistake.
    //
    // The count itself is *not* a precondition: a mismatched host side truncates to
    // min(__requested, size() - __offset), which is the count returned to the caller.
    size_type
    __checked_count(size_type __requested, size_type __offset) const
    {
        __check_offset(__offset);
        return std::min(__requested, _M_size - __offset);
    }

    // For the bulk transfers. See __checked_count().
    void
    __check_offset(size_type __offset) const
    {
        if (__offset > _M_size)
            throw std::out_of_range("oneDPL device container: transfer offset is past the end of the container");
    }

    // For the single-element operations, which address one element rather than a range and so cannot
    // accept an offset of size().
    void
    __check_element_pos(size_type __pos) const
    {
        if (__pos >= _M_size)
            throw std::out_of_range("oneDPL device container: element position is out of range");
    }

    // Swapping the allocator along with the memory is what
    // device_allocator::propagate_on_container_swap asks for.
    //
    // Not noexcept: _Allocator is a template parameter, and an arbitrary DeviceAllocator's swap may
    // throw. For the default device_allocator it cannot -- it holds nothing but SYCL handles, which
    // move-assign over a shared_ptr -- but a noexcept here would turn a user allocator's throw into a
    // std::terminate.
    void
    __swap(__device_storage_base& __other)
    {
        std::swap(_M_data, __other._M_data);
        std::swap(_M_size, __other._M_size);
        std::swap(_M_context, __other._M_context);
        std::swap(_M_device, __other._M_device);
        std::swap(_M_alloc, __other._M_alloc);
    }

    void
    __deallocate() noexcept
    {
        if (_M_data != nullptr)
        {
            _M_alloc.deallocate(_M_data, _M_size);
        }
        _M_data = nullptr;
        _M_size = 0;
    }

    _Tp* _M_data = nullptr;
    size_type _M_size = 0;
    sycl::context _M_context;
    sycl::device _M_device;
    _Allocator _M_alloc;
};

} // namespace oneapi::dpl::__internal

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_STORAGE_BASE_IMPL_H
