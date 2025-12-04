// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_UTILS_DYNAMIC_SELECTION_H
#define _ONEDPL_UTILS_DYNAMIC_SELECTION_H

#include <type_traits>

namespace TestUtils
{

// Helper to check if backend defines a wait_type
template <typename T, typename = void> //assumes wait_type does not exist
struct get_wait_type
{
    using type = int; //defaults to int
};

template <typename T> //specialization if wait_type exists
struct get_wait_type<T, std::void_t<typename T::wait_type>>
{
    using type = typename T::wait_type;
};
} // namespace TestUtils

//resource providing a wait functionality
struct DummyResource
{
    int value;

    DummyResource(int v) : value(v) {}
    bool
    operator==(const DummyResource& other) const
    {
        return value == other.value;
    }

    bool
    operator!=(const DummyResource& other) const
    {
        return !(*this == other);
    }

    void
    wait()
    {
    }
};

#endif /* _ONEDPL_UTILS_DYNAMIC_SELECTION_H */
