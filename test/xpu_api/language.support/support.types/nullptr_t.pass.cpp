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

//
// typedef decltype(nullptr) nullptr_t;
//

#include "support/test_config.h"

#include <oneapi/dpl/cstddef>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>

#include "support/test_macros.h"
#include "support/utils.h"

#include <cstdint>

struct A
{
    A(dpl::nullptr_t) {}
};

template <class T>
void
test_conversions(std::int32_t& i)
{
    {
        T p = 0;
        i += (p == nullptr);
    }
    {
        T p = nullptr;
        i += (p == nullptr);
        i += (nullptr == p);
        i += (!(p != nullptr));
        i += (!(nullptr != p));
    }
}

template <class T>
struct Voider
{
    typedef void type;
};
template <class T, class = void>
struct has_less : std::false_type
{
};

template <class T>
struct has_less<T, typename Voider<decltype(std::declval<T>() < nullptr)>::type> : std::true_type
{
};

template <class T>
void
test_comparisons(std::int32_t& i)
{
    T p = nullptr;
    i += (p == nullptr);
    i += (!(p != nullptr));
    i += (nullptr == p);
    i += (!(nullptr != p));
}

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnull-conversion"
#endif
void
test_nullptr_conversions(std::int32_t& i)
{
    // GCC does not accept this due to CWG Defect #1423
    // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1423
    {
        bool b(nullptr);
        i += (!b);
    }
}
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

int
main()
{
    const dpl::size_t N = 1;
    bool ret = true;

    {
        sycl::buffer<bool, 1> buf(&ret, sycl::range<1>{N});
        sycl::queue q = TestUtils::get_test_queue();
        q.submit([&](sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<class KernelTest1>([=]() {
                static_assert(sizeof(dpl::nullptr_t) == sizeof(void*), "sizeof(dpl::nullptr_t) == sizeof(void*)");

                std::int32_t i = 0;
                {
                    test_conversions<dpl::nullptr_t>(i);
                    test_conversions<void*>(i);
                    test_conversions<A*>(i);
                    test_conversions<int A::*>(i);
                }
                {
#ifdef _LIBCPP_HAS_NO_NULLPTR
                    static_assert(!has_less<dpl::nullptr_t>::value);
#endif
                    test_comparisons<dpl::nullptr_t>(i);
                    test_comparisons<void*>(i);
                    test_comparisons<A*>(i);
                }
                test_nullptr_conversions(i);

                acc[0] &= (i == 33);
            });
        });
    }

    EXPECT_TRUE(ret, "Wrong result of work with null_ptr in Kernel");

    return TestUtils::done();
}
