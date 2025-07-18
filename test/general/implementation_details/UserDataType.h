// -*- C++ -*-
//===-- UserDataType.h -----------------------------------------------------===//
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

#ifndef __USERDATA_TYPE_H
#define __USERDATA_TYPE_H

#include <cstddef> // for std::size_t

struct UserDataType
{
    int a;
    int b;
};

namespace std
{
    template <std::size_t Idx>
    int
    get(const UserDataType& data)
    {
        if constexpr (Idx == 0)
            return data.a;
        else if constexpr (Idx == 1)
            return data.b;
        else
            static_assert(Idx < 2, "Index out of bounds for UserDataType");
    }

    template <std::size_t Idx>
    int&
    get(UserDataType& data)
    {
        if constexpr (Idx == 0)
            return data.a;
        else if constexpr (Idx == 1)
            return data.b;
        else
            static_assert(Idx < 2, "Index out of bounds for UserDataType");
    }
}

#endif // __USERDATA_TYPE_H
