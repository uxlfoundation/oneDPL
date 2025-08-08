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

#include "sycl_iterator_test.h"

#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT

namespace oneapi::dpl::__internal
{
void
test_is_iterator_type()
{
    static_assert(__is_iterator_type<int*>::value);
    static_assert(__is_iterator_type<const int*>::value);
    static_assert(__is_iterator_type<decltype(std::vector<int>().begin())>::value);

    static_assert(!__is_iterator_type<std::nullptr_t>::value);
    static_assert(!__is_iterator_type<int>::value);
}

#if !_ONEDPL_CPP20_CONCEPTS_PRESENT
void
test_iterators_possibly_equal_internals()
{
    using namespace __iterators_possibly_equal_impl;

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of base iterator types
    static_assert(std::is_same_v<int*, typename __base_iterator_type<int*>::__type>);
    static_assert(std::is_same_v<int,  typename __base_iterator_type<int >::__type>);

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of iterator value_type
    static_assert(std::is_same_v<int,  typename __iterator_value_type<int*>::__type>);
    static_assert(std::is_same_v<void, typename __iterator_value_type<int >::__type>);
    static_assert(std::is_same_v<int,  typename __iterator_value_type<decltype(std::vector<int>().begin())>::__type>);

    ////////////////////////////////////////////////////////////////////////////
    // Check that the iterators iterate over the same types
    static_assert(!__has_same_value_types<int*, int >::value);
    static_assert( __has_same_value_types<int*, int*>::value);
    static_assert(!__has_same_value_types<decltype(std::vector<int>().begin()), 
                                          decltype(std::vector<float>().cbegin())>::value);
    static_assert( __has_same_value_types<decltype(std::vector<int>().begin()), 
                                          decltype(std::vector<int>().crbegin())>::value);

    ////////////////////////////////////////////////////////////////////////////
    // Check if the iterators are equality comparable

    static_assert(!__has_equality_op<int*, int >::value);
    static_assert( __has_equality_op<int*, int*>::value);
    static_assert( __has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__has_equality_op<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<float>().cbegin())>::value);

    static_assert(!__is_equality_self_comparable<int*, int >::value);
    static_assert( __is_equality_self_comparable<int*, int*>::value);
    static_assert( __is_equality_self_comparable<decltype(std::vector<int>().begin()), 
                                                 decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__is_equality_self_comparable<decltype(std::vector<int>().begin()), 
                                                 decltype(std::vector<float>().cbegin())>::value);

    static_assert(!__is_equality_comparable<int*, int       >::value);
    static_assert( __is_equality_comparable<int*, int*      >::value);
    static_assert(!__is_equality_comparable<int*, float*    >::value);
    static_assert( __is_equality_comparable<int*, const int*>::value);
    static_assert( __is_equality_comparable<decltype(std::vector<int>().begin()), 
                                            decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__is_equality_comparable<decltype(std::vector<int>().begin()), 
                                            decltype(std::vector<float>().cbegin())>::value);

    ////////////////////////////////////////////////////////////////////////////
    using __IteratorType1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<
            sycl::access::mode::read_write,
            unsigned long long
        >,
        oneapi::dpl::__internal::sycl_iterator<
            sycl::access::mode::read_write,
            unsigned int
        >
    >;
    using __IteratorType2 = oneapi::dpl::zip_iterator<
        unsigned long long *,
        oneapi::dpl::__internal::sycl_iterator<
            sycl::access::mode::read_write,
            unsigned int
        >
    >;

    static_assert(!std::is_same_v<__IteratorType1, __IteratorType2>);

    static_assert(!std::is_same_v<typename __iterator_value_type<__IteratorType1>::__type, void>);
    static_assert(!std::is_same_v<typename __iterator_value_type<__IteratorType2>::__type, void>);

    static_assert(__has_same_value_types<__IteratorType1, __IteratorType2>::value);

    static_assert( __has_equality_op<__IteratorType1, __IteratorType1>::value);
    static_assert( __has_equality_op<__IteratorType2, __IteratorType2>::value);
    static_assert(!__has_equality_op<__IteratorType1, __IteratorType2>::value);

    static_assert(!
        oneapi::dpl::__internal::__iterators_possibly_equal_impl::__is_equality_self_comparable<
        oneapi::dpl::zip_iterator<
            oneapi::dpl::__internal::sycl_iterator<
                sycl::access::mode::read_write,
                unsigned long long
            >,
            oneapi::dpl::__internal::sycl_iterator<
                sycl::access::mode::read_write,
                unsigned int
            >
        >,
        oneapi::dpl::zip_iterator<
            unsigned long long *,
            oneapi::dpl::__internal::sycl_iterator<
                sycl::access::mode::read_write,
                unsigned int
            >
        >
    >::value);
    static_assert(!__is_equality_self_comparable<__IteratorType1, __IteratorType2>::value);
    static_assert(!__is_equality_comparable<__IteratorType1, __IteratorType2>::value);
}
#endif

// Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
void
test_iterators_possibly_equal()
{
    // Check some internals from oneapi::dpl::__internal
    using namespace oneapi::dpl::__internal;

    constexpr size_t count = 0;
    sycl::buffer<int> buf1(count);
    sycl::buffer<int> buf2(count);

    auto it1 = oneapi::dpl::begin(buf1);
    auto it2 = oneapi::dpl::begin(buf2);
    auto& it1Ref = it1;
    auto& it2Ref = it2;

    EXPECT_TRUE(__iterators_possibly_equal(it1, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1Ref, it1), "wrong __iterators_possibly_equal result");
    EXPECT_TRUE(__iterators_possibly_equal(it1Ref, it1Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1Ref, it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1Ref, it2Ref), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2), "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), it2Ref),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(it1, oneapi::dpl::begin(buf2)), "wrong __iterators_possibly_equal result");

    EXPECT_FALSE(__iterators_possibly_equal(oneapi::dpl::begin(buf1), nullptr),
                 "wrong __iterators_possibly_equal result");
    EXPECT_FALSE(__iterators_possibly_equal(nullptr, oneapi::dpl::begin(buf2)),
                 "wrong __iterators_possibly_equal result");

    // sub - buffer vs it's "root" buffer (expect true)
    sycl::buffer<int, 1> buf11{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__iterators_possibly_equal(oneapi::dpl::end(buf1), oneapi::dpl::begin(buf11)),
                "wrong __iterators_possibly_equal result");

    // sub - buffer vs sub - buffer which share a "root" buffer(expect true)
    sycl::buffer<int, 1> buf12{buf1, sycl::range<1>{0}, sycl::range<1>{0}};
    EXPECT_TRUE(__iterators_possibly_equal(oneapi::dpl::begin(buf11), oneapi::dpl::end(buf12)),
                "wrong __iterators_possibly_equal result");

    // two sycl_iterators pointing to different elements in the same "root" buffer(expect false)
    auto it1next = it1 + 1;
    EXPECT_FALSE(__iterators_possibly_equal(it1, it1next), "wrong __iterators_possibly_equal result");

    {
        float floatData = .0;

        ::std::vector<int> dataVec{1, 2, 3};
        const auto intConstData = dataVec.data();
        auto intData = dataVec.data();

        // check pointer + pointer
        EXPECT_TRUE(__iterators_possibly_equal(intData, intData), "wrong __iterators_possibly_equal result");
        // check const pointer + pointer
        EXPECT_TRUE(__iterators_possibly_equal(intConstData, intData), "wrong __iterators_possibly_equal result");
        // check pointer + const pointer
        EXPECT_TRUE(__iterators_possibly_equal(intData, intConstData), "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other type
        EXPECT_FALSE(__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
    }

    {
        int srcIntData = 0;
        const auto& intConstData = srcIntData;
        auto& intData = srcIntData;
        const float floatData = .0;

        // Check pointer to const data + pointer to data
        EXPECT_TRUE(__iterators_possibly_equal(&intConstData, &intData), "wrong __iterators_possibly_equal result");
        // Check pointer to data + pointer to const data
        EXPECT_TRUE(__iterators_possibly_equal(&intData, &intConstData), "wrong __iterators_possibly_equal result");
        // Check pointer to const data + pointer to const data
        EXPECT_TRUE(__iterators_possibly_equal(&intConstData, &intConstData),
                    "wrong __iterators_possibly_equal result");
        // check pointer + pointer to other const type
        EXPECT_FALSE(__iterators_possibly_equal(intData, &floatData), "wrong __iterators_possibly_equal result");
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

};  // namespace oneapi::dpl::__internal

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    oneapi::dpl::__internal::test_is_iterator_type();

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
