// -*- C++ -*-
//===-- span_impl.h -------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_SPAN_IMPL_H
#define _ONEDPL_SPAN_IMPL_H

#include <cstddef>

#include "common_config.h"
#include "../pstl/onedpl_config.h"

#if _ONEDPL_CPP20_SPAN_PRESENT
#    include <span>
#elif _ONEDPL_BACKEND_SYCL
#    include "../pstl/hetero/dpcpp/sycl_defs.h"
#endif

// oneapi::dpl::span is an alias for std::span where the standard library provides it, and for
// sycl::span otherwise. The rationale for the choice, the properties the two alternatives share,
// and the one way in which they differ -- sycl::span::iterator is a raw pointer while
// std::span::iterator is not, so span iterators must never be exposed in oneDPL interfaces or
// passed to oneDPL algorithms -- are documented in <oneapi/dpl/experimental/device_array>, the
// public header through which this alias reaches users.
#if _ONEDPL_CPP20_SPAN_PRESENT || _ONEDPL_BACKEND_SYCL

namespace oneapi
{
namespace dpl
{

#    if _ONEDPL_CPP20_SPAN_PRESENT

inline constexpr std::size_t dynamic_extent = std::dynamic_extent;

template <typename _Tp, std::size_t _Extent = dynamic_extent>
using span = std::span<_Tp, _Extent>;

#    else

inline constexpr std::size_t dynamic_extent = sycl::dynamic_extent;

template <typename _Tp, std::size_t _Extent = dynamic_extent>
using span = sycl::span<_Tp, _Extent>;

#    endif // _ONEDPL_CPP20_SPAN_PRESENT

} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_CPP20_SPAN_PRESENT || _ONEDPL_BACKEND_SYCL

#endif // _ONEDPL_SPAN_IMPL_H
