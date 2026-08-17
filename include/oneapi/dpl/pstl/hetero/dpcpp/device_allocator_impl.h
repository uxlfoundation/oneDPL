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

#include "../../../internal/common_config.h"
#include "../../onedpl_config.h"

#if _ONEDPL_BACKEND_SYCL

#include <cstddef>
#include <type_traits>
#include <utility>

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
    using size_type = std::size_t;

    explicit device_allocator(sycl::context __ctx, sycl::device __dev, const sycl::property_list& __prop_list = {})
        : __context(__ctx), __device(__dev), __prop_list(__prop_list)
    {
    }

    explicit device_allocator(sycl::queue __q, const sycl::property_list& __prop_list = {})
        : __context(__q.get_context()), __device(__q.get_device()), __prop_list(__prop_list)
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
        : __context(__other.__context), __device(__other.__device), __prop_list(__other.__prop_list)
    {
    }

    _Tp*
    allocate(size_type __count) const
    {
        if (__count == 0)
            return nullptr;

        _Tp* __ptr = nullptr;
        if constexpr (_Alignment == 0)
        {
            __ptr = sycl::malloc_device<_Tp>(__count, __device, __context, __prop_list);
        }
        else
        {
            // aligned_alloc_device already raises the alignment to max(_Alignment, alignof(_Tp)).
            __ptr = sycl::aligned_alloc_device<_Tp>(_Alignment, __count, __device, __context, __prop_list);
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
    deallocate(_Tp* __ptr, size_type /*__count*/) const
    {
        if (__ptr != nullptr)
            sycl::free(__ptr, __context);
    }

  private:
    template <typename _Up, std::size_t _AlignmentU>
    friend class device_allocator;

    template <typename _Tp2, std::size_t _AlignmentT2, typename _Up2, std::size_t _AlignmentU2>
    friend bool
    operator==(const device_allocator<_Tp2, _AlignmentT2>& __lhs,
               const device_allocator<_Up2, _AlignmentU2>& __rhs) noexcept;

    sycl::context __context;
    sycl::device __device;
    sycl::property_list __prop_list;
};

// Two device allocators compare equal if they share an alignment, a context and a device, following the
// requirement SYCL 2020 section 4.8.3.1 places on sycl::usm_allocator. As with sycl::usm_allocator, the value type
// and the property list do not participate in the comparison.
template <typename _Tp, std::size_t _AlignmentT, typename _Up, std::size_t _AlignmentU>
bool
operator==(const device_allocator<_Tp, _AlignmentT>& __lhs, const device_allocator<_Up, _AlignmentU>& __rhs) noexcept
{
    return _AlignmentT == _AlignmentU && __lhs.__context == __rhs.__context && __lhs.__device == __rhs.__device;
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
