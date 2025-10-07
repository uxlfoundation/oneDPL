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

#ifndef _ASSIGNABLE_FROM_H
#define _ASSIGNABLE_FROM_H

template <class T>
struct AssignableFrom
{
    int type_constructed = 0;
    int type_assigned = 0;
    int int_constructed = 0;
    int int_assigned = 0;

    AssignableFrom() = default;

    explicit AssignableFrom(T) { ++type_constructed; }
    AssignableFrom& operator=(T)
    {
        ++type_assigned;
        return *this;
    }

    AssignableFrom(int) { ++int_constructed; }
    AssignableFrom&
    operator=(int)
    {
        ++int_assigned;
        return *this;
    }

    AssignableFrom(AssignableFrom const&) = delete;
    AssignableFrom&
    operator=(AssignableFrom const&) = delete;
};

#endif // _ASSIGNABLE_FROM_H
