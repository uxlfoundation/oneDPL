// -*- C++ -*-
//===-- device_array_impl.h -----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_ARRAY_IMPL_H
#define _ONEDPL_DEVICE_ARRAY_IMPL_H

#include <cstddef>
#include <vector>
#include <utility>
#include <type_traits>

#include "common_config.h"
#include "../pstl/onedpl_config.h"

// device_array is a container over USM device memory, so it exists only with the SYCL backend.
// Without it this header declares nothing, following the convention of the other internal headers;
// the public <oneapi/dpl/experimental/device_array> guards its include of this file the same way.
#if _ONEDPL_BACKEND_SYCL

#include "../pstl/hetero/dpcpp/sycl_defs.h"
#include "span_impl.h"
#include "device_allocator_impl.h"
#include "device_storage_base_impl.h"

namespace oneapi::dpl::experimental
{

// A fixed-size container over a USM device allocation.
//
// The element count is established at construction and never changes. There are no element proxies
// and no implicit transfers: every host-visible read or write is an explicit copy_to / copy_from /
// read_at / to_vector call, so the cost of touching device memory is always visible in the source.
//
// The container stores a sycl::context and a sycl::device rather than a sycl::queue, so it is not
// tied to any single queue. Each transfer operation comes in overloads taking the queue (and an
// optional sycl::event to depend on) to use, plus a queue-less convenience overload that builds a
// temporary queue from the stored context and device. All of them block until the transfer
// completes.
//
// The storage layer is inherited privately so that none of it leaks; the observers that are safe to
// publish are re-exported with using-declarations below.
template <typename _Tp>
class device_array : private oneapi::dpl::__internal::__device_storage_base<_Tp, device_allocator<_Tp>>
{
    using _Allocator = device_allocator<_Tp>;
    using _Base = oneapi::dpl::__internal::__device_storage_base<_Tp, _Allocator>;

  public:
    using value_type = _Tp;
    using size_type = std::size_t;

    // -- Construction --
    //
    // The base's constructors are not inherited. `using _Base::_Base;` would make them private,
    // because the base is a private base, and the signatures differ anyway: the base takes
    // (count, context, device, allocator) while device_array publishes (count, queue) and friends.
    // Each constructor therefore delegates explicitly in its mem-initializer list.
    //
    // The queue-taking constructors use the queue's context and device; the queue itself is not
    // retained.
    //
    // device_allocator is stateful, following sycl::usm_allocator, so each constructor builds one
    // from the same context and device it passes to the base. The allocator type is fixed and is not
    // part of device_array's interface; pluggable allocation arrives with compat::device_vector.

    // Allocates without initializing. No memset, no fill, no kernel launch.
    device_array(size_type __count, sycl::queue __q)
        : _Base(__count, __q.get_context(), __q.get_device(), _Allocator(__q))
    {
    }

    device_array(size_type __count, sycl::context __ctx, sycl::device __dev)
        : _Base(__count, __ctx, __dev, _Allocator(__ctx, __dev))
    {
    }

    // Allocates and fills every element with __value.
    device_array(size_type __count, const _Tp& __value, sycl::queue __q)
        : _Base(__count, __q.get_context(), __q.get_device(), _Allocator(__q))
    {
        _Base::__fill_n(__value, __count, 0, __q, sycl::event{});
    }

    device_array(size_type __count, const _Tp& __value, sycl::context __ctx, sycl::device __dev)
        : _Base(__count, __ctx, __dev, _Allocator(__ctx, __dev))
    {
        _Base::__fill_n(__value, __count, 0, _Base::__make_queue(), sycl::event{});
    }

    // Allocates __src.size() elements and copies __src into them. __src may be host memory or USM
    // accessible on this context.
    device_array(oneapi::dpl::span<const _Tp> __src, sycl::queue __q, sycl::event __depends_on = {})
        : _Base(__src.size(), __q.get_context(), __q.get_device(), _Allocator(__q))
    {
        _Base::__copy_from_host(__src.data(), __src.size(), 0, __q, __depends_on);
    }

    device_array(oneapi::dpl::span<const _Tp> __src, sycl::context __ctx, sycl::device __dev)
        : _Base(__src.size(), __ctx, __dev, _Allocator(__ctx, __dev))
    {
        _Base::__copy_from_host(__src.data(), __src.size(), 0, _Base::__make_queue(), sycl::event{});
    }

    device_array(const device_array&) = delete;
    device_array&
    operator=(const device_array&) = delete;

    // A moved-from device_array is empty; its context and device remain queryable, so it is a valid
    // move-assignment target. Move assignment is self-move safe: the base guards on this == &other.
    device_array(device_array&&) noexcept = default;
    device_array&
    operator=(device_array&&) noexcept = default;

    ~device_array() = default;

    // -- Host-device transfer --
    //
    // Argument order is uniform across all of these: what is being transferred first, then where in
    // the container, then the queue and the event to depend on. Each operation comes in three forms:
    //
    //   (data, offset)                 -- queue-less, uses a queue built from the stored context
    //   (data, queue, depends_on)      -- offset defaults to 0; the common case, spelled without a 0
    //   (data, offset, queue, depends_on)
    //
    // The offset is a precondition, not something to be clamped: for the bulk operations it must be
    // <= size(), where exactly size() names the end of the range and transfers zero elements, and for
    // the single-element operations it must be < size(). A violation throws std::out_of_range instead
    // of reading or writing out of bounds, or silently doing nothing.
    //
    // The element count is not a precondition. A mismatched host side truncates to
    // min(other.size(), size() - offset), and the bulk operations return that count, which may be
    // less than requested.

    // -- Device to host --

    size_type
    copy_to(oneapi::dpl::span<_Tp> __dst, size_type __src_offset, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        const size_type __n = _Base::__checked_count(__dst.size(), __src_offset);
        _Base::__copy_to_host(__dst.data(), __n, __src_offset, __q, __depends_on);
        return __n;
    }

    size_type
    copy_to(oneapi::dpl::span<_Tp> __dst, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        return copy_to(__dst, 0, __q, __depends_on);
    }

    _Tp
    read_at(size_type __pos, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        _Base::__check_element_pos(__pos);
        return _Base::__read_at(__pos, __q, __depends_on);
    }

    // Requires _Tp to be default constructible. As a member of a class template this is instantiated
    // only when called, so device_array<NonDefaultConstructible> remains usable minus this one
    // convenience; that is also why there is no static_assert here.
    std::vector<_Tp>
    to_vector(sycl::queue __q, sycl::event __depends_on = {}) const
    {
        std::vector<_Tp> __out(_Base::size());
        copy_to(oneapi::dpl::span<_Tp>{__out.data(), __out.size()}, 0, __q, __depends_on);
        return __out;
    }

    // -- Host to device --

    size_type
    copy_from(oneapi::dpl::span<const _Tp> __src, size_type __dst_offset, sycl::queue __q,
              sycl::event __depends_on = {})
    {
        const size_type __n = _Base::__checked_count(__src.size(), __dst_offset);
        _Base::__copy_from_host(__src.data(), __n, __dst_offset, __q, __depends_on);
        return __n;
    }

    size_type
    copy_from(oneapi::dpl::span<const _Tp> __src, sycl::queue __q, sycl::event __depends_on = {})
    {
        return copy_from(__src, 0, __q, __depends_on);
    }

    // Writes a single element. Unlike the bulk overloads there is nothing to truncate, so the
    // position is checked as an element position: __dst_offset == size() throws as well.
    void
    copy_from(const _Tp& __value, size_type __dst_offset, sycl::queue __q, sycl::event __depends_on = {})
    {
        _Base::__check_element_pos(__dst_offset);
        _Base::__fill_n(__value, 1, __dst_offset, __q, __depends_on);
    }

    void
    copy_from(const _Tp& __value, sycl::queue __q, sycl::event __depends_on = {})
    {
        copy_from(__value, 0, __q, __depends_on);
    }

    // -- Queue-less convenience overloads --
    //
    // Each forwards to its queue-taking sibling with a temporary queue built from the stored context
    // and device, so span unwrapping and range checking happen in exactly one place. Constructing a
    // queue per call is measurable overhead, and the queue-taking overloads are the performance path;
    // this is the documented cost of storing a context rather than a queue.

    size_type
    copy_to(oneapi::dpl::span<_Tp> __dst, size_type __src_offset = 0) const
    {
        return copy_to(__dst, __src_offset, _Base::__make_queue());
    }

    _Tp
    read_at(size_type __pos) const
    {
        return read_at(__pos, _Base::__make_queue());
    }

    std::vector<_Tp>
    to_vector() const
    {
        return to_vector(_Base::__make_queue());
    }

    size_type
    copy_from(oneapi::dpl::span<const _Tp> __src, size_type __dst_offset = 0)
    {
        return copy_from(__src, __dst_offset, _Base::__make_queue());
    }

    void
    copy_from(const _Tp& __value, size_type __dst_offset = 0)
    {
        copy_from(__value, __dst_offset, _Base::__make_queue());
    }

    // -- Observers --
    //
    // data() is deliberately *not* re-exposed; it stays private through the private base.
    // sycl::span's container constructor is unconstrained in C++17, so any class with public data()
    // and size() implicitly converts to sycl::span<const _Tp>. That would make an expression such as
    // d2.copy_from(d) compile under C++17 and fail under C++20, where std::span's corresponding
    // constructor is constrained. Users obtain the raw pointer from oneapi::dpl::begin(d) or
    // d.span().data().
    using _Base::empty;
    using _Base::get_context;
    using _Base::get_device;
    using _Base::size;

    // Not noexcept, following the base: the allocator is swapped along with the memory, and although
    // the fixed device_allocator cannot throw, the base is shared with compat::device_vector, whose
    // allocator is a user-supplied template parameter.
    void
    swap(device_array& __other)
    {
        _Base::__swap(__other);
    }

    // A span over the whole allocation, for use in kernels. When the container is empty this is an
    // empty span over a null pointer, which both std::span and sycl::span support.
    oneapi::dpl::span<_Tp>
    span()
    {
        return oneapi::dpl::span<_Tp>{_Base::data(), _Base::size()};
    }

    oneapi::dpl::span<const _Tp>
    span() const
    {
        return oneapi::dpl::span<const _Tp>{_Base::data(), _Base::size()};
    }
};

template <typename _Tp>
void
swap(device_array<_Tp>& __a, device_array<_Tp>& __b)
{
    __a.swap(__b);
}

} // namespace oneapi::dpl::experimental

namespace oneapi
{
namespace dpl
{

// begin / end for device_array, following the sycl::buffer overloads in
// pstl/hetero/dpcpp/sycl_iterator.h, but returning raw pointers and taking their argument by
// reference.
//
// Raw pointers rather than span iterators: std::span<_Tp>::iterator is __gnu_cxx::__normal_iterator
// (libstdc++), __wrap_iter / __bounded_iter (libc++) or _Span_iterator (MSVC) -- never a raw
// pointer. Such a type does not satisfy oneapi::dpl::is_indirectly_device_accessible (the
// customization point in pstl/iterator_impl.h), so a oneDPL algorithm would silently take the
// host-sycl::buffer path over what is device memory: wrong results, not a compile error. A raw
// pointer satisfies that trait through its std::is_pointer term and works everywhere. Note that
// sycl::span<_Tp>::iterator *is* _Tp*, so handing out span iterators would appear to work under
// C++17 and break only under C++20, which would make the bug harder still to find.
//
// By reference rather than by value, unlike the sycl::buffer overloads: device_array is
// non-copyable, and a sycl::buffer is a cheap handle where a device_array is not.
//
// Because device_array lives in oneapi::dpl::experimental, ADL does not find these overloads. Calls
// must be qualified as oneapi::dpl::begin(d) / oneapi::dpl::end(d), exactly as for the existing
// sycl::buffer overloads.
template <typename _Tp>
_Tp*
begin(experimental::device_array<_Tp>& __d)
{
    return __d.span().data();
}

template <typename _Tp>
_Tp*
end(experimental::device_array<_Tp>& __d)
{
    return __d.span().data() + __d.size();
}

template <typename _Tp>
const _Tp*
begin(const experimental::device_array<_Tp>& __d)
{
    return __d.span().data();
}

template <typename _Tp>
const _Tp*
end(const experimental::device_array<_Tp>& __d)
{
    return __d.span().data() + __d.size();
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_ARRAY_IMPL_H
