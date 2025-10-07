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

#include "support/test_config.h"

#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

// A type T where move ctor is noexcept and copy ctor is default : result should be T&&
struct MoveNoexceptCopy
{
    MoveNoexceptCopy() = default;

    MoveNoexceptCopy(MoveNoexceptCopy&&) noexcept {};
    MoveNoexceptCopy(const MoveNoexceptCopy&) = default;
};

// A type T where move ctor is noexcept and copy ctor is deleted : result should be T&&
struct MoveNoexceptNoCopy
{

    MoveNoexceptNoCopy() = default;

    MoveNoexceptNoCopy(MoveNoexceptNoCopy&&) noexcept {};
    MoveNoexceptNoCopy(const MoveNoexceptNoCopy&) = delete;
};

// A type T where move ctor is not noexcept and copy ctor is default : result is const T&
struct MoveNotNoexceptCopy
{
    MoveNotNoexceptCopy() = default;

    MoveNotNoexceptCopy(MoveNotNoexceptCopy&&) noexcept(false) {};
    MoveNotNoexceptCopy(const MoveNotNoexceptCopy&) = default;
};

// A type T where move ctor is not noexcept and copy ctor is deleted : result is T&&
struct MoveNotNoexceptNoCopy
{
    MoveNotNoexceptNoCopy() = default;

    MoveNotNoexceptNoCopy(MoveNotNoexceptNoCopy&&) noexcept(false){};
    MoveNotNoexceptNoCopy(const MoveNotNoexceptNoCopy&) = delete;
};

// A type T where move ctor is deleted and copy ctor is deleted : result is T&&
struct NoMoveNoCopy
{
    NoMoveNoCopy() = default;

    NoMoveNoCopy(NoMoveNoCopy&&) = delete;
    NoMoveNoCopy(const NoMoveNoCopy&) = delete;
};

void
kernel_test()
{
    int i = 0;
    const int ci = 0;

    static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(i)), int&&>);
    static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(ci)), const int&&>);

    constexpr int i1 = 23;
    constexpr int i2 = dpl::move_if_noexcept(i1);
    static_assert(i2 == 23);

    {
        MoveNoexceptCopy data;
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(data)), MoveNoexceptCopy&&>);
    }

    {
        MoveNoexceptNoCopy data;
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(data)), MoveNoexceptNoCopy&&>);
    }

    {
        MoveNotNoexceptCopy data;
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(data)), const MoveNotNoexceptCopy&>);
    }

    {
        MoveNotNoexceptNoCopy data;
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(data)), MoveNotNoexceptNoCopy&&>);
    }

    {
        NoMoveNoCopy data;
        static_assert(dpl::is_same_v<decltype(dpl::move_if_noexcept(data)), NoMoveNoCopy&&>);
    }
}

class KernelTest;

int
main()
{
    sycl::queue deviceQueue = TestUtils::get_test_queue();
    sycl::range<1> numOfItems{1};
    {
        deviceQueue.submit([&](sycl::handler& cgh) { cgh.single_task<class KernelTest>([=]() { kernel_test(); }); });
    }

    return TestUtils::done();
}
