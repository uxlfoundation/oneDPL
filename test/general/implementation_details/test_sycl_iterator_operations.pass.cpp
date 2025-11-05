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

#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(numeric) // for __iterators_possibly_equal(const sycl_iterator<_Mode1, _T, _Allocator>& __it1, const sycl_iterator<_Mode2, _T, _Allocator>& __it2)

#include <vector>
#include <type_traits>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#endif

#if TEST_DPCPP_BACKEND_PRESENT

namespace oneapi::dpl::__internal
{
void check_is_equality_comparable_with()
{
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, int*>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, float*>);
    static_assert(!__is_equality_comparable_with_v<int*,   std::move_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<float*, std::move_iterator<int*>>);

    static_assert( __is_equality_comparable_with_v<std::move_iterator<int*>, std::move_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, std::reverse_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, std::reverse_iterator<move_iterator<int*>>>);

    static_assert(!__is_equality_comparable_with_v<int*, int       >);
    static_assert( __is_equality_comparable_with_v<int*, int*      >);
    static_assert(!__is_equality_comparable_with_v<int*, float*    >);
    static_assert( __is_equality_comparable_with_v<int*, const int*>);
    static_assert( __is_equality_comparable_with_v<decltype(std::vector<int>().begin()), 
                                                   decltype(std::vector<int>().cbegin())>);
    static_assert(!__is_equality_comparable_with_v<decltype(std::vector<int>().begin()), 
                                                   decltype(std::vector<float>().cbegin())>);

    ////////////////////////////////////////////////////////////////////////////
    // Check if move_iterator and reverse_iterator work as expected
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, std::move_iterator<bool*>>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, int*>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, float*>);
    static_assert(!__is_equality_comparable_with_v<int*,   std::move_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<float*, std::move_iterator<int*>>);
    static_assert( __is_equality_comparable_with_v<std::move_iterator<int*>, std::move_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, std::reverse_iterator<int*>>);
    static_assert(!__is_equality_comparable_with_v<std::move_iterator<int*>, std::reverse_iterator<move_iterator<int*>>>);
    static_assert( __is_equality_comparable_with_v<reverse_iterator<move_iterator<int*>>, reverse_iterator<move_iterator<int*>>>);
    static_assert( __is_equality_comparable_with_v<reverse_iterator<double*>, reverse_iterator<double*>>);
    static_assert(!__is_equality_comparable_with_v<reverse_iterator<int*>, reverse_iterator<bool*>>);

    using __zip_iterator_1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned long long>,
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;
    using __zip_iterator_2 =
        oneapi::dpl::zip_iterator<unsigned long long*,
                                  oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    static_assert(!__is_equality_comparable_with_v<__zip_iterator_1, __zip_iterator_2>);

    using __zip_iterator_1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned long long>,
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    using __zip_iterator_2 =
        oneapi::dpl::zip_iterator<unsigned long long*,
                                  oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    static_assert(!__is_equality_comparable_with_v<__zip_iterator_1, __zip_iterator_2>);
}

// Additional checks that iterator types with const and/or different value categories work
void check_is_equality_comparable_with_for_different_types()
{
    std::vector<int> container(10);

    std::vector<int>::iterator it1 = container.begin();
    const auto&                it1_const_ref = it1;
    decltype(auto)             it1_moved     = std::move(it1);

    using _t1 = decltype(it1);
    using _t2 = decltype(it1_const_ref);
    using _t3 = decltype(it1_moved);

    static_assert(__is_equality_comparable_with_v<_t1, _t1>);
    static_assert(__is_equality_comparable_with_v<_t1, _t2>);
    static_assert(__is_equality_comparable_with_v<_t1, _t3>);

    static_assert(__is_equality_comparable_with_v<_t2, _t1>);
    static_assert(__is_equality_comparable_with_v<_t2, _t2>);
    static_assert(__is_equality_comparable_with_v<_t2, _t3>);

    static_assert(__is_equality_comparable_with_v<_t3, _t1>);
    static_assert(__is_equality_comparable_with_v<_t3, _t2>);
    static_assert(__is_equality_comparable_with_v<_t3, _t3>);
}

#if !_ONEDPL_CPP20_CONCEPTS_PRESENT
void
test_iterators_possibly_equal_internals()
{
    ////////////////////////////////////////////////////////////////////////////
    // Check if the iterators are equality comparable

    static_assert(!__has_equality_op<int*, int >::value);
    static_assert( __has_equality_op<int*, int*>::value);
    static_assert( __has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<float>().cbegin())>::value);

    static_assert(!__has_equality_op<int*, int >::value);
    static_assert( __has_equality_op<int*, int*>::value);
    static_assert( __has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<float>().cbegin())>::value);

    ////////////////////////////////////////////////////////////////////////////
    using __zip_iterator_1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned long long>,
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    using __zip_iterator_2 =
        oneapi::dpl::zip_iterator<unsigned long long*,
                                  oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    static_assert(!std::is_same_v<__zip_iterator_1, __zip_iterator_2>);

    static_assert( __has_equality_op<__zip_iterator_1, __zip_iterator_1>::value);
    static_assert( __has_equality_op<__zip_iterator_2, __zip_iterator_2>::value);
    static_assert(!__has_equality_op<__zip_iterator_1, __zip_iterator_2>::value);
}
#endif // _ONEDPL_CPP20_CONCEPTS_PRESENT

// Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
void
test_iterators_possibly_equal()
{
    // Check some internals from oneapi::dpl::__internal
    namespace __internal = oneapi::dpl::__internal;

    constexpr std::size_t count = 0;
    sycl::buffer<int> buf1(count);
    sycl::buffer<int> buf2(count);

    auto it1 = oneapi::dpl::begin(buf1);
    auto it2 = oneapi::dpl::begin(buf2);
    auto& it1Ref = it1;
    auto& it2Ref = it2;

    EXPECT_TRUE(__internal::__iterators_possibly_equal(it1, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__internal::__iterators_possibly_equal(it1, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__internal::__iterators_possibly_equal(it1Ref, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__internal::__iterators_possibly_equal(it1Ref, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1Ref, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1Ref, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2Ref),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(oneapi::dpl::begin(buf1), oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1, oneapi::dpl::begin(buf2)), "wrong __iterators_possibly_equal result");

    EXPECT_FALSE(__internal::__iterators_possibly_equal(oneapi::dpl::begin(buf1), nullptr),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__internal::__iterators_possibly_equal(nullptr, oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");

    // sub - buffer vs it's "root" buffer (expect true)
    sycl::buffer<int, 1> buf11{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__internal::__iterators_possibly_equal(oneapi::dpl::end(buf1), oneapi::dpl::begin(buf11)),
                "wrong __iterators_possibly_equal result");

    // sub - buffer vs sub - buffer which share a "root" buffer(expect true)
    sycl::buffer<int, 1> buf12{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__internal::__iterators_possibly_equal(oneapi::dpl::begin(buf11), oneapi::dpl::end(buf12)),
                "wrong __iterators_possibly_equal result");

    // two sycl_iterators pointing to different elements in the same "root" buffer(expect false)
    auto it1next = it1 + 1;
    EXPECT_FALSE(__internal::__iterators_possibly_equal(it1, it1next), "wrong __iterators_possibly_equal result");

    {
        float floatData = .0;

        std::vector<int> dataVec{1, 2, 3};
        std::vector<int>::const_iterator intConstData = dataVec.cbegin();
        std::vector<int>::iterator intData = dataVec.begin();

        // check pointer + pointer
        EXPECT_TRUE(__internal::__iterators_possibly_equal(intData, intData), "wrong __iterators_possibly_equal result");
        // check const pointer + pointer
        EXPECT_TRUE(__internal::__iterators_possibly_equal(intConstData, intData), "wrong __iterators_possibly_equal result");
        // check pointer + const pointer
        EXPECT_TRUE(__internal::__iterators_possibly_equal(intData, intConstData), "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other type
        EXPECT_FALSE(__internal::__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
    }

#ifndef _PSTL_TEST_ITERATORS_POSSIBLY_EQUAL_BROKEN
    {
        std::vector<int> dataVec1{1, 2, 3};
        std::vector<int> dataVec2{4, 5, 6};
        EXPECT_FALSE(__internal::__iterators_possibly_equal(dataVec1.begin(), dataVec2.begin()), "wrong __iterators_possibly_equal result");
    }
#endif

    {
        int srcIntData = 0;
        const auto& intConstData = srcIntData;
        auto& intData = srcIntData;
        const float floatData = .0;

        // Check pointer to const data + pointer to data
        EXPECT_TRUE(__internal::__iterators_possibly_equal(&intConstData, &intData), "wrong __iterators_possibly_equal result");
        // Check pointer to data + pointer to const data
        EXPECT_TRUE(__internal::__iterators_possibly_equal(&intData, &intConstData), "wrong __iterators_possibly_equal result");
        // Check pointer to const data + pointer to const data
        EXPECT_TRUE(__internal::__iterators_possibly_equal(&intConstData, &intConstData),
                    "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other const type
        EXPECT_FALSE(__internal::__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
    }

    {
        std::vector<int>   dataVecInt;
        std::vector<float> dataVecFloat;

        auto itRBeginInt = dataVecInt.rbegin();
        auto itRBeginFloat = dataVecFloat.rbegin();

        static_assert(!__is_equality_comparable_with_v<decltype(itRBeginInt), decltype(itRBeginFloat)>);
        EXPECT_FALSE(__internal::__iterators_possibly_equal(itRBeginInt, itRBeginFloat), "wrong __iterators_possibly_equal result");
    }

    // For now we are not going to support comparison of iterators with raw pointers
    {
        std::vector<int> dataVec{1, 2, 3};
        std::vector<int>::iterator itBegin = dataVec.begin();
        int* rawData = dataVec.data();

        static_assert(!__is_equality_comparable_with_v<decltype(itBegin), decltype(rawData)>);
        EXPECT_FALSE(__internal::__iterators_possibly_equal(itBegin, rawData), "wrong __iterators_possibly_equal result");
    }
}

class CustomIterator
{
public:

    struct Tag { };

    CustomIterator(Tag) {}

    bool
    operator==(const CustomIterator&) const
    {
        return true;
    }
};

void
test_custom_iterators_possibly_equal()
{
    CustomIterator it1(CustomIterator::Tag{});
    CustomIterator it2(CustomIterator::Tag{});

    EXPECT_TRUE(__iterators_possibly_equal(it1, it2),
                "wrong __iterators_possibly_equal result for custom iterator which is not default constructible");
}

}  // namespace oneapi::dpl::__internal

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    oneapi::dpl::__internal::check_is_equality_comparable_with();
    oneapi::dpl::__internal::check_is_equality_comparable_with_for_different_types();

#if !_ONEDPL_CPP20_CONCEPTS_PRESENT
    oneapi::dpl::__internal::test_iterators_possibly_equal_internals();
#endif

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
    oneapi::dpl::__internal::test_iterators_possibly_equal();

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal for custom iterators
    oneapi::dpl::__internal::test_custom_iterators_possibly_equal();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
