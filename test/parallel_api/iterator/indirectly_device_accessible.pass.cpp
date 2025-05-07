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
#include <cstdint>
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(iterator)

#include "support/utils_device_copyable.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

struct simple_iterator
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    using is_passed_directly = std::false_type;

    simple_iterator(int start = 0) : value(start) {}

    int
    operator*() const
    {
        return value;
    }

    simple_iterator&
    operator++()
    {
        ++value;
        return *this;
    }

    simple_iterator
    operator++(int)
    {
        simple_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    simple_iterator&
    operator--()
    {
        --value;
        return *this;
    }

    simple_iterator
    operator--(int)
    {
        simple_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    simple_iterator
    operator+(int i) const
    {
        return simple_iterator(value + i);
    }

    simple_iterator
    operator-(int i) const
    {
        return simple_iterator(value - i);
    }

    difference_type
    operator-(const simple_iterator& other) const
    {
        return value - other.value;
    }

    simple_iterator&
    operator+=(int i)
    {
        value += i;
        return *this;
    }

    simple_iterator&
    operator-=(int i)
    {
        value -= i;
        return *this;
    }

    int
    operator[](int i) const
    {
        return value + i;
    }

    friend bool
    operator==(const simple_iterator& a, const simple_iterator& b)
    {
        return a.value == b.value;
    }

    friend bool
    operator!=(const simple_iterator& a, const simple_iterator& b)
    {
        return !(a == b);
    }

    friend bool
    operator<(const simple_iterator& a, const simple_iterator& b)
    {
        return a.value < b.value;
    }

    friend bool
    operator<=(const simple_iterator& a, const simple_iterator& b)
    {
        return a.value <= b.value;
    }

    friend bool
    operator>(const simple_iterator& a, const simple_iterator& b)
    {
        return a.value > b.value;
    }

    friend bool
    operator>=(const simple_iterator& a, const simple_iterator& b)
    {
        return a.value >= b.value;
    }

  private:
    int value;
};

//IDA= indirectly_device_accessible
using implicit_non_IDA_iter = simple_iterator;

//IDA= indirectly_device_accessible
struct IDA_iter : public simple_iterator
{
    using is_passed_directly = std::true_type;

    IDA_iter(int start = 0) : simple_iterator(start) {}
};

//IDA= indirectly_device_accessible
struct explicit_non_IDA_iterator : public simple_iterator
{
    using is_passed_directly = std::false_type;

    explicit_non_IDA_iterator(int start = 0) : simple_iterator(start) {}
};

namespace custom_user
{
template <typename BaseIter>
struct base_strided_iterator
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::iterator_traits<BaseIter>::value_type;
    using difference_type = typename std::iterator_traits<BaseIter>::difference_type;
    using pointer = typename std::iterator_traits<BaseIter>::pointer;
    using reference = typename std::iterator_traits<BaseIter>::reference;

    base_strided_iterator(BaseIter base, int stride) : base(base), stride(stride) {}

    reference
    operator*() const
    {
        return *base;
    }

    base_strided_iterator&
    operator++()
    {
        std::advance(base, stride);
        return *this;
    }

    base_strided_iterator
    operator++(int)
    {
        base_strided_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    base_strided_iterator&
    operator--()
    {
        std::advance(base, -stride);
        return *this;
    }

    base_strided_iterator
    operator--(int)
    {
        base_strided_iterator tmp = *this;
        --(*this);
        return tmp;
    }

    base_strided_iterator
    operator+(difference_type n) const
    {
        return base_strided_iterator(std::next(base, n * stride), stride);
    }

    base_strided_iterator
    operator-(difference_type n) const
    {
        return base_strided_iterator(std::prev(base, n * stride), stride);
    }

    difference_type
    operator-(const base_strided_iterator& other) const
    {
        return (base - other.base) / stride;
    }

    base_strided_iterator&
    operator+=(difference_type n)
    {
        std::advance(base, n * stride);
        return *this;
    }

    base_strided_iterator&
    operator-=(difference_type n)
    {
        std::advance(base, -n * stride);
        return *this;
    }

    reference
    operator[](difference_type n) const
    {
        return *(base + n * stride);
    }

    friend bool
    operator==(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return a.base == b.base;
    }

    friend bool
    operator!=(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return !(a == b);
    }

    friend bool
    operator<(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return a.base < b.base;
    }

    friend bool
    operator<=(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return a.base <= b.base;
    }

    friend bool
    operator>(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return a.base > b.base;
    }

    friend bool
    operator>=(const base_strided_iterator& a, const base_strided_iterator& b)
    {
        return a.base >= b.base;
    }

  private:
    BaseIter base;
    int stride;
};

template <typename BaseIter>
struct first_strided_iterator : public base_strided_iterator<BaseIter>
{
    first_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
};

template <typename BaseIter>
auto
is_onedpl_indirectly_device_accessible(first_strided_iterator<BaseIter>)
{
    return oneapi::dpl::is_indirectly_device_accessible<BaseIter>{};
}

template <typename BaseIter>
struct second_strided_iterator : public base_strided_iterator<BaseIter>
{
    second_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
};

template <typename BaseIter>
auto is_onedpl_indirectly_device_accessible(second_strided_iterator<BaseIter>)
    -> decltype(oneapi::dpl::is_indirectly_device_accessible<BaseIter>{});

template <typename BaseIter>
struct third_strided_iterator : public base_strided_iterator<BaseIter>
{
    third_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
    friend auto
    is_onedpl_indirectly_device_accessible(third_strided_iterator<BaseIter>)
    {
        return oneapi::dpl::is_indirectly_device_accessible<BaseIter>{};
    }
};

template <typename BaseIter>
struct fourth_strided_iterator : public base_strided_iterator<BaseIter>
{
    fourth_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
    friend auto is_onedpl_indirectly_device_accessible(fourth_strided_iterator<BaseIter>)
        -> oneapi::dpl::is_indirectly_device_accessible<BaseIter>;
};

} // namespace custom_user

template <bool base_indirectly_device_accessible, typename BaseIter>
void
test_with_base_iterator()
{
    //test assumption about base iterator device accessible content iterator
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<BaseIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for base iterator");

    // test wrapping base in transform_iterator
    using TransformIter = oneapi::dpl::transform_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<TransformIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::transform_iterator");

    // test wrapping base in permutation_iterator with counting iter
    using PermutationIter = oneapi::dpl::permutation_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<PermutationIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::permutation_iterator");

    // test wrapping base in permutation_iter with functor
    using PermutationIterFunctor = oneapi::dpl::permutation_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<PermutationIterFunctor> ==
                      base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::permutation_iterator "
                  "with functor");

    // test wrapping base in zip_iterator
    using ZipIter = oneapi::dpl::zip_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<ZipIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::zip_iterator");

    // test wrapping base in zip_iterator with counting_iterator first
    using ZipIterCounting = oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<std::int32_t>, BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<ZipIterCounting> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::zip_iterator with "
                  "oneapi::dpl::counting_iterator as first element");

    // test wrapping base in zip_iterator with counting_iterator second
    using ZipIterCounting2 = oneapi::dpl::zip_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<ZipIterCounting2> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::zip_iterator with "
                  "oneapi::dpl::counting_iterator as second element");

    // test custom user first strided iterator with normal ADL function
    using FirstStridedIter = custom_user::first_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<FirstStridedIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for custom user strided iterator");

    // test custom user second strided iterator (no body for is_onedpl_indirectly_device_accessible)
    using SecondStridedIter = custom_user::second_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<SecondStridedIter> ==
                      base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for custom user strided "
                  "iterator with no body in ADL function definition");

    // test custom user first strided iterator with hidden friend ADL function
    using ThirdStridedIter = custom_user::third_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<ThirdStridedIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for custom user strided "
                  "iterator with hidden friend ADL function");

    // test custom user first strided iterator with hidden friend ADL function without body
    using FourthStridedIter = custom_user::fourth_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<FourthStridedIter> ==
                      base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for custom user strided "
                  "iterator with hidden friend ADL function without a body");
}

template <bool base_indirectly_device_accessible, typename BaseIter>
void
test_base_with_reverse_iter()
{
    // test wrapping base in reverse_iterator
    using ReverseIter = std::reverse_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<ReverseIter> == base_indirectly_device_accessible,
                  "is_indirectly_device_accessible is not working correctly for std::reverse_iterator");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT
    // counting_iterator
    test_with_base_iterator<true, oneapi::dpl::counting_iterator<std::int32_t>>();
    test_base_with_reverse_iter<true, oneapi::dpl::counting_iterator<std::int32_t>>();

    // pointer (USM assumed)
    test_with_base_iterator<true, int*>();
    test_base_with_reverse_iter<true, int*>();

    // create a usm allocated vector
    sycl::queue q;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(q);
    std::vector<int, sycl::usm_allocator<int, sycl::usm::alloc::shared>> vec(alloc);
    test_with_base_iterator<TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(vec.begin())>,
                            decltype(vec.begin())>();
    test_base_with_reverse_iter<
        TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(vec.begin())>,
        decltype(vec.begin())>();

    // custom iter type with legacy is_passed_directly trait defined
    test_with_base_iterator<true, IDA_iter>();
    test_base_with_reverse_iter<true, IDA_iter>();

    // custom iter type with explicit is_passed_directly trait defined as false
    test_with_base_iterator<false, explicit_non_IDA_iterator>();
    test_base_with_reverse_iter<false, explicit_non_IDA_iterator>();

    // custom iter type implicitly not device accessible content iterator
    test_with_base_iterator<false, implicit_non_IDA_iter>();
    test_base_with_reverse_iter<false, implicit_non_IDA_iter>();

    // std vector with normal allocator
    std::vector<int> vec2(10);
    test_with_base_iterator<false, decltype(vec2.begin())>();
    test_base_with_reverse_iter<false, decltype(vec2.begin())>();

    // test discard_iterator
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<oneapi::dpl::discard_iterator> == true,
                  "is_indirectly_device_accessible is not working correctly for oneapi::dpl::discard_iterator");

    // test buffer_wrapper
    sycl::buffer<int, 1> buf(10);
    auto buffer_wrapper = oneapi::dpl::begin(buf);
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<decltype(buffer_wrapper)> == true,
                  "is_indirectly_device_accessible is not working correctly for return type of oneapi::dpl::begin()");
    test_with_base_iterator<true, decltype(buffer_wrapper)>();
    // Do not test with reverse_iterator, because buffer_wrapper is not a random access iterator

#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
