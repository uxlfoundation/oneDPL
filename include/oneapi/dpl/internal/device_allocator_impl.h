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

#if _ONEDPL_BACKEND_SYCL

#    include "../pstl/hetero/dpcpp/sycl_defs.h"

namespace oneapi::dpl::experimental
{

// The default allocator for the oneDPL device containers.
//
// The API mirrors sycl::usm_allocator, but provides USM device memory. For this
// reason it does not satisfy std::allocator requirements and cannot be used with
// standard containers. It is intended for use only with the oneDPL device containers.
// Allocation failure surfaces as the sycl::exception thrown by the underlying USM,
// sycl::malloc_device() or sycl::aligned_alloc_device().
template <typename _Tp, std::size_t _Alignment = 0>
class device_allocator
{
  public:
    using value_type = _Tp;

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

    template <typename _Up>
    device_allocator(const device_allocator<_Up, _Alignment>& __other) noexcept
        : _M_context(__other._M_context), _M_device(__other._M_device), _M_prop_list(__other._M_prop_list)
    {
    }

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

    // __count is accepted, and ignored, to match the allocator convention; USM deallocation needs only
    // the pointer and the context.
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
