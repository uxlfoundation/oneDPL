// -*- C++ -*-
//===-- passed_directly.pass.cpp -----------------------------------------------===//
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

struct simple_passed_directly_iterator
{
    using iterator_category = std::input_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    using is_passed_directly = std::true_type;

    simple_passed_directly_iterator(int start = 0) : value(start) {}

    int operator*() const { return value; }

    simple_passed_directly_iterator& operator++() {
        ++value;
        return *this;
    }

    simple_passed_directly_iterator operator++(int) {
        simple_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const simple_passed_directly_iterator& a, const simple_passed_directly_iterator& b) {
        return a.value == b.value;
    }

    friend bool operator!=(const simple_passed_directly_iterator& a, const simple_passed_directly_iterator& b) {
        return !(a == b);
    }

private:
    int value;
};

struct simple_explicitly_not_passed_directly_iterator
{
    using iterator_category = std::input_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    using is_passed_directly = std::false_type;

    simple_explicitly_not_passed_directly_iterator(int start = 0) : value(start) {}

    int operator*() const { return value; }

    simple_explicitly_not_passed_directly_iterator& operator++() {
        ++value;
        return *this;
    }

    simple_explicitly_not_passed_directly_iterator operator++(int) {
        simple_explicitly_not_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const simple_explicitly_not_passed_directly_iterator& a, const simple_explicitly_not_passed_directly_iterator& b) {
        return a.value == b.value;
    }

    friend bool operator!=(const simple_explicitly_not_passed_directly_iterator& a, const simple_explicitly_not_passed_directly_iterator& b) {
        return !(a == b);
    }

private:
    int value;
};

struct simple_implicitly_not_passed_directly_iterator
{
    using iterator_category = std::input_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    using is_passed_directly = std::false_type;

    simple_implicitly_not_passed_directly_iterator(int start = 0) : value(start) {}

    int operator*() const { return value; }

    simple_implicitly_not_passed_directly_iterator& operator++() {
        ++value;
        return *this;
    }

    simple_implicitly_not_passed_directly_iterator operator++(int) {
        simple_implicitly_not_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const simple_implicitly_not_passed_directly_iterator& a, const simple_implicitly_not_passed_directly_iterator& b) {
        return a.value == b.value;
    }

    friend bool operator!=(const simple_implicitly_not_passed_directly_iterator& a, const simple_implicitly_not_passed_directly_iterator& b) {
        return !(a == b);
    }

private:
    int value;
};

namespace custom_user
{
template <typename BaseIter>
struct strided_iterator
{
    using iterator_category = std::input_iterator_tag;
    using value_type = typename std::iterator_traits<BaseIter>::value_type;

    strided_iterator(BaseIter base, int stride) : base(base), stride(stride) {}

    int operator*() const { return *base; }

    strided_iterator& operator++() {
        std::advance(base, stride);
        return *this;
    }

    strided_iterator operator++(int) {
        strided_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool operator==(const strided_iterator& a, const strided_iterator& b) {
        return a.base == b.base;
    }

    friend bool operator!=(const strided_iterator& a, const strided_iterator& b) {
        return !(a == b);
    }

private:
    BaseIter base;
    int stride;
};

template <typename BaseIter>
auto
is_passed_directly_in_onedpl_device_policies(const strided_iterator<BaseIter>&)
{
    if constexpr (oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<BaseIter>)
        return std::true_type{};
    else
        return std::false_type{};
}

} // namespace custom_user


template <bool base_passed_directly, typename BaseIter>
void
test_with_base_iterator()
{
    //test assumption about base iterator passed directly
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<BaseIter> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for base iterator");

    // test wrapping base in transform_iterator
    using TransformIter = oneapi::dpl::transform_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<TransformIter> == base_passed_directly,
        "is_passed_directly_in_onedpl_device_policies is not working correctly for transform iterator");
    
    // test wrapping base in permutation_iterator with counting iter
    using PermutationIter = oneapi::dpl::permutation_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<PermutationIter> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for permutation iterator");

    // test wrapping base in permutation_iter with functor
    using PermutationIterFunctor = oneapi::dpl::permutation_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<PermutationIterFunctor> == base_passed_directly,
        "is_passed_directly_in_onedpl_device_policies is not working correctly for permutation iterator with functor");
                        
    // test wrapping base in zip_iterator
    using ZipIter = oneapi::dpl::zip_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<ZipIter> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator");

                    // test wrapping base in zip_iterator with counting_iterator first
    using ZipIterCounting = oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<std::int32_t>, BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<ZipIterCounting> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator with counting iterator first");

    // test wrapping base in zip_iterator with counting_iterator second
    using ZipIterCounting2 = oneapi::dpl::zip_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<ZipIterCounting2> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator with counting iterator first");

    // test wrapping base in reverse_iterator
    using ReverseIter = std::reverse_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<ReverseIter> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for reverse iterator");

    // test custom user strided iterator
    using StridedIter = custom_user::strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<StridedIter> == base_passed_directly,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for custom user strided iterator");
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{

#if TEST_DPCPP_BACKEND_PRESENT
    // counting_iterator
    test_with_base_iterator<true, oneapi::dpl::counting_iterator<std::int32_t>>();

    // pointer (USM assumed)
    test_with_base_iterator<true, int*>();

    // create a usm allocated vector
    sycl::queue q;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(q);
    std::vector<int, sycl::usm_allocator<int, sycl::usm::alloc::shared>> vec(alloc);
    test_with_base_iterator<TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(vec.begin())>, decltype(vec.begin())>();

    // custom iter type with legacy is_passed_directly trait defined
    test_with_base_iterator<true, simple_passed_directly_iterator>();

    // custom iter type with explicit is_passed_directly trait defined as false
    test_with_base_iterator<false, simple_explicitly_not_passed_directly_iterator>();

    // custom iter type implicitly not passed directly
    test_with_base_iterator<false, simple_implicitly_not_passed_directly_iterator>();

    // std vector with normal allocator
    std::vector<int> vec2(10);
    test_with_base_iterator<false, decltype(vec2.begin())>();

    // test discard_iterator
    static_assert(oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<oneapi::dpl::discard_iterator> == true,
                    "is_passed_directly_in_onedpl_device_policies is not working correctly for discard iterator");

#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
