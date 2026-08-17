// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_ARRAY_IMPL_H
#define _ONEDPL_DEVICE_ARRAY_IMPL_H


#include "../../../internal/common_config.h"
#include "../../onedpl_config.h"

#if _ONEDPL_BACKEND_SYCL

#include <cstddef>
#include <vector>
#include <utility>
#include <type_traits>

#include "sycl_defs.h"
#include "span_impl.h"
#include "device_allocator_impl.h"
#include "device_storage_base_impl.h"

namespace oneapi::dpl::experimental
{

// A fixed-size container over a USM device allocation.

// The container stores a sycl::context and a sycl::device rather than a sycl::queue, so it is not tied
// to any single queue. Each transfer operation has overloads taking the queue (and an optional event
// to depend on) for synchronization, plus a queue-less one that builds a temporary queue from the stored context and
// device. All of them block until the transfer completes.
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
    // The queue-taking constructors use the queue's context and device; the queue itself is not
    // retained. The allocator is stateful, so each constructor builds one from the same context and
    // device it passes to the base.

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
    device_array(size_type __count, const value_type& __value, sycl::queue __q)
        : _Base(__count, __q.get_context(), __q.get_device(), _Allocator(__q))
    {
        _Base::__fill_n(__value, __count, 0, __q, sycl::event{});
    }

    device_array(size_type __count, const value_type& __value, sycl::context __ctx, sycl::device __dev)
        : _Base(__count, __ctx, __dev, _Allocator(__ctx, __dev))
    {
        _Base::__fill_n(__value, __count, 0, _Base::__make_queue(), sycl::event{});
    }

    // Allocates __src.size() elements and copies __src into them. __src may be host memory or USM
    // accessible on this context.
    device_array(oneapi::dpl::span<const value_type> __src, sycl::queue __q, sycl::event __depends_on = {})
        : _Base(__src.size(), __q.get_context(), __q.get_device(), _Allocator(__q))
    {
        _Base::__copy_from_host(__src.data(), __src.size(), 0, __q, __depends_on);
    }

    device_array(oneapi::dpl::span<const value_type> __src, sycl::context __ctx, sycl::device __dev)
        : _Base(__src.size(), __ctx, __dev, _Allocator(__ctx, __dev))
    {
        _Base::__copy_from_host(__src.data(), __src.size(), 0, _Base::__make_queue(), sycl::event{});
    }

    device_array(const device_array&) = delete;
    device_array&
    operator=(const device_array&) = delete;

    device_array(device_array&&) = default;
    device_array&
    operator=(device_array&&) = default;

    ~device_array() = default;

    // -- device transfers--
    //
    // Argument order is uniform:
    // - what is being transferred
    // - optional offset into the container
    // - optional queue
    // - optional event to depend on.
    //
    //Each operation comes in three forms:
    //
    //   (data, offset)                 -- queue-less, uses a queue built from the stored context
    //   (data, queue, depends_on)      -- offset defaults to 0
    //   (data, offset, queue, depends_on)
    //
    // The offset is a precondition and throws std::out_of_range if violated: <= size() for the bulk
    // operations, where offset == size() transfers zero elements without throwing, and < size() for the
    // single-element transfers.
    //
    // The element count between input and remaining container elements may be mismatched.
    // min(other.size(), size() - offset) elements are transferred, the count is returned for the bulk
    // operations.

    // -- Device transfer out --
    size_type
    copy_to(oneapi::dpl::span<value_type> __dst, size_type __src_offset, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        const size_type __n = _Base::__checked_count(__dst.size(), __src_offset);
        _Base::__copy_to_host(__dst.data(), __n, __src_offset, __q, __depends_on);
        return __n;
    }

    size_type
    copy_to(oneapi::dpl::span<value_type> __dst, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        return copy_to(__dst, 0, __q, __depends_on);
    }

    size_type
    copy_to(oneapi::dpl::span<value_type> __dst, size_type __src_offset = 0) const
    {
        return copy_to(__dst, __src_offset, _Base::__make_queue());
    }

    value_type
    read_at(size_type __pos, sycl::queue __q, sycl::event __depends_on = {}) const
    {
        _Base::__check_element_pos(__pos);
        return _Base::__read_at(__pos, __q, __depends_on);
    }

    value_type
    read_at(size_type __pos) const
    {
        return read_at(__pos, _Base::__make_queue());
    }

    // Requires value_type to be default constructible, but only when called, so
    // device_array<NonDefaultConstructible> remains usable minus this one convenience.
    std::vector<value_type>
    to_vector(sycl::queue __q, sycl::event __depends_on = {}) const
    {
        std::vector<value_type> __host_out(_Base::size());
        copy_to(oneapi::dpl::span<value_type>{__host_out.data(), __host_out.size()}, 0, __q, __depends_on);
        return __host_out;
    }

    std::vector<value_type>
    to_vector() const
    {
        return to_vector(_Base::__make_queue());
    }

    // -- Device transfer in --
    size_type
    copy_from(oneapi::dpl::span<const value_type> __src, size_type __dst_offset, sycl::queue __q,
              sycl::event __depends_on = {})
    {
        const size_type __n = _Base::__checked_count(__src.size(), __dst_offset);
        _Base::__copy_from_host(__src.data(), __n, __dst_offset, __q, __depends_on);
        return __n;
    }

    size_type
    copy_from(oneapi::dpl::span<const value_type> __src, sycl::queue __q, sycl::event __depends_on = {})
    {
        return copy_from(__src, 0, __q, __depends_on);
    }

    size_type
    copy_from(oneapi::dpl::span<const value_type> __src, size_type __dst_offset = 0)
    {
        return copy_from(__src, __dst_offset, _Base::__make_queue());
    }

    // Writes a single element, so unlike the bulk overloads __dst_offset == size() throws.
    void
    copy_from(const value_type& __value, size_type __dst_offset, sycl::queue __q, sycl::event __depends_on = {})
    {
        _Base::__check_element_pos(__dst_offset);
        _Base::__fill_n(__value, 1, __dst_offset, __q, __depends_on);
    }

    void
    copy_from(const value_type& __value, sycl::queue __q, sycl::event __depends_on = {})
    {
        copy_from(__value, 0, __q, __depends_on);
    }

    void
    copy_from(const value_type& __value, size_type __dst_offset = 0)
    {
        copy_from(__value, __dst_offset, _Base::__make_queue());
    }

    // -- Observers --
    //
    // data() is deliberately not re-exposed. sycl::span's container constructor is unconstrained in
    // C++17, so any class with public data() and size() converts implicitly to sycl::span<const value_type>,
    // which would make an expression such as d2.copy_from(d) compile under C++17 and fail under C++20.
    // Users may obtain a raw pointer from oneapi::dpl::begin(d) or d.span().data().
    using _Base::empty;
    using _Base::get_context;
    using _Base::get_device;
    using _Base::size;

    void
    swap(device_array& __other)
    {
        _Base::__swap(__other);
    }

    oneapi::dpl::span<value_type>
    span()
    {
        return oneapi::dpl::span<value_type>{_Base::data(), _Base::size()};
    }

    oneapi::dpl::span<const value_type>
    span() const
    {
        return oneapi::dpl::span<const value_type>{_Base::data(), _Base::size()};
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
// reference (device_array is non-copyable).
template <typename _Tp>
_Tp*
begin(oneapi::dpl::experimental::device_array<_Tp>& __d)
{
    return __d.span().data();
}

template <typename _Tp>
_Tp*
end(oneapi::dpl::experimental::device_array<_Tp>& __d)
{
    return __d.span().data() + __d.size();
}

template <typename _Tp>
const _Tp*
begin(const oneapi::dpl::experimental::device_array<_Tp>& __d)
{
    return __d.span().data();
}

template <typename _Tp>
const _Tp*
end(const oneapi::dpl::experimental::device_array<_Tp>& __d)
{
    return __d.span().data() + __d.size();
}

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_ARRAY_IMPL_H
