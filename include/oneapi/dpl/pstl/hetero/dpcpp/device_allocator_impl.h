// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "../../../internal/common_config.h"
#include "../../onedpl_config.h"

#if _ONEDPL_BACKEND_SYCL

#include "sycl_defs.h"

namespace oneapi::dpl::experimental
{

// The default allocator for the oneDPL device containers.
//
// The API mirrors sycl::usm_allocator, but provides USM device memory. For this
// reason it does not satisfy std::allocator requirements and cannot be used with
// standard containers. It is intended for use only with the oneDPL device containers.
// Allocation failure surfaces as a sycl::exception with errc::memory_allocation.
template <typename _Tp, std::size_t _Alignment = 0>
class device_allocator
{
  public:
    using value_type = _Tp;

    explicit device_allocator(sycl::context __ctx, sycl::device __dev, const sycl::property_list& __prop_list = {})
        : _M_context(__ctx), _M_device(__dev), _M_prop_list(__prop_list)
    {
    }

    explicit device_allocator(sycl::queue __q, const sycl::property_list& __prop_list = {})
        : _M_context(__q.get_context()), _M_device(__q.get_device()), _M_prop_list(__prop_list)
    {
    }

    device_allocator(const device_allocator&) = default;
    device_allocator(device_allocator&&) = default;
    device_allocator&
    operator=(const device_allocator&) = default;
    device_allocator&
    operator=(device_allocator&&) = default;
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

        _Tp* __ptr = nullptr;
        if constexpr (_Alignment == 0)
        {
            __ptr = sycl::malloc_device<_Tp>(__count, _M_device, _M_context, _M_prop_list);
        }
        else
        {
            // aligned_alloc_device already raises the alignment to max(_Alignment, alignof(_Tp)).
            __ptr = sycl::aligned_alloc_device<_Tp>(_Alignment, __count, _M_device, _M_context, _M_prop_list);
        }

        // The USM allocation functions return nullptr on failure rather than throwing, both when there
        // are insufficient resources and when _Alignment is unsupported.
        if (__ptr == nullptr)
            throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                  "oneDPL device container: USM device allocation failed");

        return __ptr;
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

// Two device allocators compare equal if they share an alignment, a context and a device, following the
// requirement SYCL 2020 section 4.8.3.1 places on sycl::usm_allocator. As with sycl::usm_allocator, the value type
// and the property list do not participate in the comparison.
template <typename _Tp, std::size_t _AlignmentT, typename _Up, std::size_t _AlignmentU>
bool
operator==(const device_allocator<_Tp, _AlignmentT>& __lhs, const device_allocator<_Up, _AlignmentU>& __rhs) noexcept
{
    return _AlignmentT == _AlignmentU && __lhs.get_context() == __rhs.get_context() &&
           __lhs.get_device() == __rhs.get_device();
}

template <typename _Tp, std::size_t _AlignmentT, typename _Up, std::size_t _AlignmentU>
bool
operator!=(const device_allocator<_Tp, _AlignmentT>& __lhs, const device_allocator<_Up, _AlignmentU>& __rhs) noexcept
{
    return !(__lhs == __rhs);
}

} // namespace oneapi::dpl::experimental

#endif // _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_DEVICE_ALLOCATOR_IMPL_H
