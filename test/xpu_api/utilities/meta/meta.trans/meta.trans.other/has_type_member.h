// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _HAS_TYPE_MEMBER_H
#define _HAS_TYPE_MEMBER_H

#include <oneapi/dpl/type_traits>

template <class, class = dpl::void_t<>>
struct has_type_member : dpl::false_type
{
};

template <class T>
struct has_type_member<T, dpl::void_t<typename T::type>> : dpl::true_type
{
};

#endif // _HAS_TYPE_MEMBER_H

