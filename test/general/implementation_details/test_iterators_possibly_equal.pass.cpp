// -*- C++ -*-
//===-- test_iterators_possibly_equal.pass.cpp ----------------------------===//
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

#include "support/utils.h"

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

#if _ONEDPL_CPP20_CONCEPTS_PRESENT
void
test_iterators_possibly_equal_internals_on_concepts()
{
    using __zip_iterator_1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned long long>,
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;
    using __zip_iterator_2 =
        oneapi::dpl::zip_iterator<unsigned long long*,
                                  oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    static_assert(!__is_equality_comparable_with_v<__zip_iterator_1, __zip_iterator_2>);
}
#else
void
test_iterators_possibly_equal_internals()
{
    using namespace __is_equality_comparable_impl;

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of base iterator types
    static_assert(!__has_base_iterator<int*>::value);
    static_assert(!__has_base_iterator<int >::value);
    static_assert( __has_base_iterator<decltype(std::vector<float>().rbegin())>::value);

    static_assert(std::is_same_v<int*, typename __base_iterator_type<int*>::__type>);
    static_assert(std::is_same_v<int,  typename __base_iterator_type<int >::__type>);

    ////////////////////////////////////////////////////////////////////////////
    // The definitions of iterator value_type
    static_assert(std::is_same_v<int,  typename __iterator_value_type<int*>::__type>);
    static_assert(std::is_same_v<void, typename __iterator_value_type<int >::__type>);
    static_assert(std::is_same_v<int,  typename __iterator_value_type<decltype(std::vector<int>().begin())>::__type>);

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

    static_assert(!__is_equality_comparable_with_v<int*, int       >);
    static_assert( __is_equality_comparable_with_v<int*, int*      >);
    static_assert(!__is_equality_comparable_with_v<int*, float*    >);
    static_assert( __is_equality_comparable_with_v<int*, const int*>);
    static_assert( __is_equality_comparable_with_v<decltype(std::vector<int>().begin()), 
                                                   decltype(std::vector<int>().cbegin())>);
    static_assert(!__is_equality_comparable_with_v<decltype(std::vector<int>().begin()), 
                                                   decltype(std::vector<float>().cbegin())>);

    ////////////////////////////////////////////////////////////////////////////
    using __zip_iterator_1 = oneapi::dpl::zip_iterator<
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned long long>,
        oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    using __zip_iterator_2 =
        oneapi::dpl::zip_iterator<unsigned long long*,
                                  oneapi::dpl::__internal::sycl_iterator<sycl::access::mode::read_write, unsigned int>>;

    using __zip_iterator_1_base = decltype(declval<__zip_iterator_1>().base());
    using __zip_iterator_2_base = decltype(declval<__zip_iterator_2>().base());

    static_assert(!std::is_same_v<__zip_iterator_1, __zip_iterator_2>);

    static_assert(!std::is_same_v<typename __iterator_value_type<__zip_iterator_1>::__type, void>);
    static_assert(!std::is_same_v<typename __iterator_value_type<__zip_iterator_2>::__type, void>);

    static_assert( __has_equality_op<__zip_iterator_1, __zip_iterator_1>::value);
    static_assert( __has_equality_op<__zip_iterator_2, __zip_iterator_2>::value);
    static_assert(!__has_equality_op<__zip_iterator_1, __zip_iterator_2>::value);

    static_assert(__has_base_iterator<__zip_iterator_1>::value == __is_iterator_type<__zip_iterator_1_base>::value);
    static_assert(__has_base_iterator<__zip_iterator_2>::value == __is_iterator_type<__zip_iterator_2_base>::value);

    static_assert(std::is_same_v<typename __base_iterator_type<__zip_iterator_1>::__type, __zip_iterator_1_base>);
    static_assert(std::is_same_v<typename __base_iterator_type<__zip_iterator_2>::__type, __zip_iterator_2_base>);

    static_assert(!__has_equality_op<__zip_iterator_1, __zip_iterator_2>::value);
    static_assert(!__is_equality_comparable_with_v<__zip_iterator_1, __zip_iterator_2>);
}
#endif // _ONEDPL_CPP20_CONCEPTS_PRESENT
} // namespace oneapi::dpl::__internal
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    oneapi::dpl::__internal::test_is_iterator_type();

#if _ONEDPL_CPP20_CONCEPTS_PRESENT
    oneapi::dpl::__internal::test_iterators_possibly_equal_internals_on_concepts();
#else
    oneapi::dpl::__internal::test_iterators_possibly_equal_internals();
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
