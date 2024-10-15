// -*- C++ -*-
//===-- zip_iterator_funcs.h ---------------------------------------------===//
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

#ifndef _ZIP_ITERATOR_FUNCS_H
#define _ZIP_ITERATOR_FUNCS_H

#include <oneapi/dpl/iterator>

struct TupleNoOp
{
    template <typename T>
    T
    operator()(const T& val) const
    {
        return val;
    }

    template <typename T1, typename T2>
    T2
    operator()(const T1&, const T2& t2) const
    {
        return t2;
    }
};

template <typename Predicate, int KeyIndex>
struct TuplePredicate
{
    Predicate pred;

    template <typename... Args>
    auto
    operator()(const Args&... args) const -> decltype(pred(std::get<KeyIndex>(args)...))
    {
        return pred(std::get<KeyIndex>(args)...);
    }
};

#endif
