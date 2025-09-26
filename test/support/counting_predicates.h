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

#ifndef TEST_SUPPORT_COUNTING_PREDICATES_H
#define TEST_SUPPORT_COUNTING_PREDICATES_H

#include <oneapi/dpl/cstddef>

template <typename Predicate, typename Arg>
struct unary_counting_predicate {
public:
    using argument_type = Arg;
    using result_type = bool;

    unary_counting_predicate(Predicate p) : p_(p), count_(0) {}
    ~unary_counting_predicate() {}

    bool operator () (const Arg &a) const { ++count_; return p_(a); }
    dpl::size_t count() const { return count_; }
    void reset() { count_ = 0; }

private:
    Predicate p_;
    mutable dpl::size_t count_;
};

#endif // TEST_SUPPORT_COUNTING_PREDICATES_H
