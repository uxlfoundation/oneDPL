// -*- C++ -*-
//===-- device_allocator_impl.h -------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_DEVICE_ALLOCATOR_IMPL_H
#define _ONEDPL_DEVICE_ALLOCATOR_IMPL_H

#include <cstddef>
#include <type_traits>
#include <utility>

#include "common_config.h"
#include "../pstl/onedpl_config.h"

// device_allocator allocates USM device memory, so it exists only with the SYCL backend. Without it
// this header declares nothing, following the convention of the other internal headers.
#if _ONEDPL_BACKEND_SYCL

#    include "../pstl/hetero/dpcpp/sycl_defs.h"

namespace oneapi::dpl::experimental
{

// The default allocator for the oneDPL device containers.
//
// The API deliberately mirrors sycl::usm_allocator: the allocator is stateful, carrying the
// sycl::context, sycl::device and sycl::property_list to allocate against, so that allocate() takes
// only an element count. It also matches usm_allocator's alignment template parameter, its rebind
// and propagate_on_container_* members, its converting constructor, and its equality operators.
//
// It exists as a separate type because sycl::usm_allocator cannot serve this role at all: it
// static_asserts that AllocKind != sycl::usm::alloc::device, since device memory is not
// host-accessible and therefore cannot satisfy the std::allocator named requirements that
// usm_allocator is built to satisfy. device_allocator provides exactly allocate() and deallocate()
// and imposes none of those requirements; construction and destruction of elements is the
// container's business, done through kernels or memcpy.
//
// Like usm_allocator, this is not default constructible: an allocation needs a context and a device,
// and there is no meaningful default for either.
//
// Allocation failure surfaces as the sycl::exception that the underlying USM entry point throws. It
// is not translated to std::bad_alloc, which would discard the backend diagnostics, and
// sycl::aspect::usm_device_allocations is not queried up front: a device without that aspect simply
// cannot host a device container, and there is no fallback to select.
template <typename _Tp, std::size_t _Alignment = 0>
class device_allocator
{
  public:
    using value_type = _Tp;

    // Device memory is never host-accessible, so a container can never relocate elements by copying
    // them on the host; it always has to go through the device. Propagating on all three operations
    // keeps a container's allocator consistent with the memory it holds, matching usm_allocator.
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    template <typename _Up>
    struct rebind
    {
        using other = device_allocator<_Up, _Alignment>;
    };

    device_allocator() = delete;

    explicit device_allocator(const sycl::context& __ctx, const sycl::device& __dev,
                              const sycl::property_list& __prop_list = {})
        : _M_context(__ctx), _M_device(__dev), _M_prop_list(__prop_list)
    {
    }

    explicit device_allocator(const sycl::queue& __q, const sycl::property_list& __prop_list = {})
        : _M_context(__q.get_context()), _M_device(__q.get_device()), _M_prop_list(__prop_list)
    {
    }

    device_allocator(const device_allocator&) = default;
    device_allocator(device_allocator&&) noexcept = default;
    device_allocator&
    operator=(const device_allocator&) = default;
    device_allocator&
    operator=(device_allocator&&) noexcept = default;
    ~device_allocator() = default;

    // Rebinding conversion, as on usm_allocator. Only the allocation target is carried over; the
    // element type and therefore the alignment requirement come from the destination type.
    template <typename _Up>
    device_allocator(const device_allocator<_Up, _Alignment>& __other) noexcept
        : _M_context(__other._M_context), _M_device(__other._M_device), _M_prop_list(__other._M_prop_list)
    {
    }

    // Allocates uninitialized device memory for __count objects of type _Tp. A count of zero
    // allocates nothing and returns nullptr: sycl::malloc_device(0) is unspecified, and some
    // backends return a non-null pointer that cannot be freed.
    _Tp*
    allocate(std::size_t __count) const
    {
        if (__count == 0)
            return nullptr;

        if constexpr (_Alignment == 0)
        {
            return sycl::malloc_device<_Tp>(__count, _M_device, _M_context, _M_prop_list);
        }
        else
        {
            // aligned_alloc_device already raises the alignment to max(_Alignment, alignof(_Tp)).
            return sycl::aligned_alloc_device<_Tp>(_Alignment, __count, _M_device, _M_context, _M_prop_list);
        }
    }

    // __count is accepted, and ignored, to match the allocator convention; USM deallocation needs
    // only the pointer and the context it was allocated against.
    void
    deallocate(_Tp* __ptr, std::size_t /*__count*/) const
    {
        if (__ptr != nullptr)
            sycl::free(__ptr, _M_context);
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

    template <typename _Property>
    bool
    has_property() const noexcept
    {
        return _M_prop_list.has_property<_Property>();
    }

    template <typename _Property>
    _Property
    get_property() const
    {
        return _M_prop_list.get_property<_Property>();
    }

    // Two allocators are interchangeable when memory from one can be freed by the other, which for
    // USM device allocations depends on the context and device alone. The comparison spans element
    // types and alignments, as it does for usm_allocator.
    template <typename _Up, std::size_t _AlignmentU>
    friend bool
    operator==(const device_allocator& __lhs, const device_allocator<_Up, _AlignmentU>& __rhs)
    {
        return __lhs._M_context == __rhs._M_context && __lhs._M_device == __rhs._M_device;
    }

    template <typename _Up, std::size_t _AlignmentU>
    friend bool
    operator!=(const device_allocator& __lhs, const device_allocator<_Up, _AlignmentU>& __rhs)
    {
        return !(__lhs == __rhs);
    }

  private:
    template <typename _Up, std::size_t _AlignmentU>
    friend class device_allocator;

    sycl::context _M_context;
    sycl::device _M_device;
    sycl::property_list _M_prop_list;
};

} // namespace oneapi::dpl::experimental

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_ALLOCATOR_IMPL_H
