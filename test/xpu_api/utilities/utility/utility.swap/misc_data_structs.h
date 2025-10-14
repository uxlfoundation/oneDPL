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

#ifndef _MISC_DATA_STRUCTS_H
#define _MISC_DATA_STRUCTS_H

struct CopyOnly
{
    CopyOnly() = default;
    CopyOnly(CopyOnly const&) noexcept = default;
    CopyOnly&
    operator=(CopyOnly const&)
    {
        return *this;
    }
};

struct NoexceptMoveOnly
{
    NoexceptMoveOnly() = default;
    NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept = default;
    NoexceptMoveOnly&
    operator=(NoexceptMoveOnly&&) noexcept
    {
        return *this;
    }
};

struct NotMoveConstructible
{
    NotMoveConstructible&
    operator=(NotMoveConstructible&&)
    {
        return *this;
    }

    NotMoveConstructible(NotMoveConstructible&&) = delete;
};

struct NotMoveAssignable
{
    NotMoveAssignable(NotMoveAssignable&&) = delete;

    NotMoveAssignable&
    operator=(NotMoveAssignable&&) = delete;
};

#endif // _MISC_DATA_STRUCTS_H
