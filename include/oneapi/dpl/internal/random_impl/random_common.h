// -*- C++ -*-
//===-- random_common.h ---------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Public header file provides common utils for random implementation

#ifndef _ONEDPL_RANDOM_COMMON_H
#define _ONEDPL_RANDOM_COMMON_H

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename _T>
struct type_traits_t
{
    using element_type = _T;
    static constexpr int num_elems = 0;
};

template <typename _T, int _N>
struct type_traits_t<sycl::vec<_T, _N>>
{
    using element_type = _T;
    static constexpr int num_elems = _N;
};

template <typename _T>
using element_type_t = typename type_traits_t<_T>::element_type;

typedef union
{
    uint32_t hex[2];
} dp_union_t;

typedef union
{
    uint32_t hex[1];
} sp_union_t;

template <class _CharT, class _Traits>
class save_stream_flags
{
    typedef ::std::basic_ios<_CharT, _Traits> __stream_type;

  public:
    save_stream_flags(const save_stream_flags&) = delete;
    save_stream_flags&
    operator=(const save_stream_flags&) = delete;

    explicit save_stream_flags(__stream_type& __stream)
        : __fmtflags_(__stream.flags()), __stream_(__stream), __fill_(__stream.fill())
    {
    }
    ~save_stream_flags()
    {
        __stream_.flags(__fmtflags_);
        __stream_.fill(__fill_);
    }

  private:
    typename __stream_type::fmtflags __fmtflags_;
    __stream_type& __stream_;
    _CharT __fill_;
};

} // namespace internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_RANDOM_COMMON_H
