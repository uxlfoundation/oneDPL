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

void
test_iterators_possibly_equal_internals()
{
    using namespace __iterators_possibly_equal_impl;

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of base iterator types
    static_assert(std::is_same_v<int*, typename __base_iterator<int*>::__type>);
    static_assert(std::is_same_v<int,  typename __base_iterator<int >::__type>);

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of iterator value_type
    static_assert(std::is_same_v<int,  __iterator_value_type_t<int*>>);
    static_assert(std::is_same_v<void, __iterator_value_type_t<int >>);
    static_assert(std::is_same_v<int,  __iterator_value_type_t<decltype(std::vector<int>().begin())>>);

    ////////////////////////////////////////////////////////////////////////////
    // Check that the iterators iterate over the same types
    static_assert(!__is_the_same_types_iterated<int*, int>::value);
    static_assert(__is_the_same_types_iterated<int*, int*>::value);
    static_assert(!__is_the_same_types_iterated<decltype(std::vector<int>().begin()), 
                                                 decltype(std::vector<float>().cbegin())>::value);
    static_assert(__is_the_same_types_iterated<decltype(std::vector<int>().begin()), 
                                                decltype(std::vector<int>().crbegin())>::value);

    ////////////////////////////////////////////////////////////////////////////
    // Check if the iterators are equality comparable

    static_assert(!__is_eq_op_exists<int*, int>::value);
    static_assert(__is_eq_op_exists<int*, int*>::value);
    static_assert(__is_eq_op_exists<decltype(std::vector<int>().begin()), 
                                    decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__is_eq_op_exists<decltype(std::vector<int>().begin()), 
                                     decltype(std::vector<float>().cbegin())>::value);

    static_assert(!__is_eq_op_may_be_called<int*, int>::value);
    static_assert(__is_eq_op_may_be_called<int*, int*>::value);
    static_assert(__is_eq_op_may_be_called<decltype(std::vector<int>().begin()), 
                                           decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__is_eq_op_may_be_called<decltype(std::vector<int>().begin()), 
                                            decltype(std::vector<float>().cbegin())>::value);

    static_assert(!__is_eq_op_may_be_called_through_base<int*, int>::value);
    static_assert(__is_eq_op_may_be_called_through_base<int*, int*>::value);
    static_assert(!__is_eq_op_may_be_called_through_base<int*, float*>::value);
    static_assert(__is_eq_op_may_be_called_through_base<int*, const int*>::value);
    static_assert(__is_eq_op_may_be_called_through_base<decltype(std::vector<int>().begin()), 
                                                        decltype(std::vector<int>().cbegin())>::value);
    static_assert(!__is_eq_op_may_be_called_through_base<decltype(std::vector<int>().begin()), 
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

    // oneapi::dpl::__internal::tuple<unsigned long long, unsigned int>
    //__iterator_value_type_t<__IteratorType1>::dummy;
    // oneapi::dpl::__internal::tuple<unsigned long long, unsigned int>
    //__iterator_value_type_t<__IteratorType2>::dummy;

    static_assert(!std::is_same_v<__iterator_value_type_t<__IteratorType1>, void>);
    static_assert(!std::is_same_v<__iterator_value_type_t<__IteratorType2>, void>);

    static_assert(__is_the_same_types_iterated<__IteratorType1, __IteratorType2>::value);

    static_assert( __is_eq_op_exists<__IteratorType1, __IteratorType1>::value);
    static_assert( __is_eq_op_exists<__IteratorType2, __IteratorType2>::value);
    static_assert(!__is_eq_op_exists<__IteratorType1, __IteratorType2>::value);

    static_assert(!
        oneapi::dpl::__internal::__iterators_possibly_equal_impl::__is_eq_op_may_be_called<
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
    static_assert(!__is_eq_op_may_be_called<__IteratorType1, __IteratorType2>::value);
    static_assert(!__is_eq_op_may_be_called_through_base<__IteratorType1, __IteratorType2>::value);
}

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

    oneapi::dpl::__internal::test_iterators_possibly_equal_internals();

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal
    oneapi::dpl::__internal::test_iterators_possibly_equal();

    // Check the correctness of oneapi::dpl::__internal::__iterators_possibly_equal for custom iterators
    oneapi::dpl::__internal::test_custom_iterators_possibly_equal();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
