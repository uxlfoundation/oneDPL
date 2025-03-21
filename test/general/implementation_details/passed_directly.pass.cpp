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

    int
    operator*() const
    {
        return value;
    }

    simple_passed_directly_iterator&
    operator++()
    {
        ++value;
        return *this;
    }

    simple_passed_directly_iterator
    operator++(int)
    {
        simple_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool
    operator==(const simple_passed_directly_iterator& a, const simple_passed_directly_iterator& b)
    {
        return a.value == b.value;
    }

    friend bool
    operator!=(const simple_passed_directly_iterator& a, const simple_passed_directly_iterator& b)
    {
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

    int
    operator*() const
    {
        return value;
    }

    simple_explicitly_not_passed_directly_iterator&
    operator++()
    {
        ++value;
        return *this;
    }

    simple_explicitly_not_passed_directly_iterator
    operator++(int)
    {
        simple_explicitly_not_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool
    operator==(const simple_explicitly_not_passed_directly_iterator& a,
               const simple_explicitly_not_passed_directly_iterator& b)
    {
        return a.value == b.value;
    }

    friend bool
    operator!=(const simple_explicitly_not_passed_directly_iterator& a,
               const simple_explicitly_not_passed_directly_iterator& b)
    {
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

    int
    operator*() const
    {
        return value;
    }

    simple_implicitly_not_passed_directly_iterator&
    operator++()
    {
        ++value;
        return *this;
    }

    simple_implicitly_not_passed_directly_iterator
    operator++(int)
    {
        simple_implicitly_not_passed_directly_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    friend bool
    operator==(const simple_implicitly_not_passed_directly_iterator& a,
               const simple_implicitly_not_passed_directly_iterator& b)
    {
        return a.value == b.value;
    }

    friend bool
    operator!=(const simple_implicitly_not_passed_directly_iterator& a,
               const simple_implicitly_not_passed_directly_iterator& b)
    {
        return !(a == b);
    }

  private:
    int value;
};

namespace custom_user
{
template <typename BaseIter>
struct base_strided_iterator
{
    using iterator_category = std::input_iterator_tag;
    using value_type = typename std::iterator_traits<BaseIter>::value_type;

    base_strided_iterator(BaseIter base, int stride) : base(base), stride(stride) {}

    int
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
is_passed_directly_in_onedpl_device_policies(const first_strided_iterator<BaseIter>&)
{
    return oneapi::dpl::is_passed_directly_to_device<BaseIter>{};
}

template <typename BaseIter>
struct second_strided_iterator : public base_strided_iterator<BaseIter>
{
    second_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
};

template <typename BaseIter>
auto
is_passed_directly_in_onedpl_device_policies(const second_strided_iterator<BaseIter>&)
    -> decltype(oneapi::dpl::is_passed_directly_to_device<BaseIter>{});

template <typename BaseIter>
struct third_strided_iterator : public base_strided_iterator<BaseIter>
{
    third_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
    friend auto
    is_passed_directly_in_onedpl_device_policies(const third_strided_iterator<BaseIter>&)
    {
        return oneapi::dpl::is_passed_directly_to_device<BaseIter>{};
    }
};

template <typename BaseIter>
struct fourth_strided_iterator : public base_strided_iterator<BaseIter>
{
    fourth_strided_iterator(BaseIter base, int stride) : base_strided_iterator<BaseIter>(base, stride) {}
    friend auto
    is_passed_directly_in_onedpl_device_policies(const fourth_strided_iterator<BaseIter>&)
        -> oneapi::dpl::is_passed_directly_to_device<BaseIter>;
};

} // namespace custom_user

template <bool base_passed_directly, typename BaseIter>
void
test_with_base_iterator()
{
    //test assumption about base iterator passed directly
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<BaseIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for base iterator");

    // test wrapping base in transform_iterator
    using TransformIter = oneapi::dpl::transform_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<TransformIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for transform iterator");

    // test wrapping base in permutation_iterator with counting iter
    using PermutationIter = oneapi::dpl::permutation_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<PermutationIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for permutation iterator");

    // test wrapping base in permutation_iter with functor
    using PermutationIterFunctor = oneapi::dpl::permutation_iterator<BaseIter, TestUtils::noop_device_copyable>;
    static_assert(
        oneapi::dpl::is_passed_directly_to_device_v<PermutationIterFunctor> == base_passed_directly,
        "is_passed_directly_in_onedpl_device_policies is not working correctly for permutation iterator with functor");

    // test wrapping base in zip_iterator
    using ZipIter = oneapi::dpl::zip_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<ZipIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator");

    // test wrapping base in zip_iterator with counting_iterator first
    using ZipIterCounting = oneapi::dpl::zip_iterator<oneapi::dpl::counting_iterator<std::int32_t>, BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<ZipIterCounting> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator with "
                  "counting iterator first");

    // test wrapping base in zip_iterator with counting_iterator second
    using ZipIterCounting2 = oneapi::dpl::zip_iterator<BaseIter, oneapi::dpl::counting_iterator<std::int32_t>>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<ZipIterCounting2> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for zip iterator with "
                  "counting iterator first");

    // test wrapping base in reverse_iterator
    using ReverseIter = std::reverse_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<ReverseIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for reverse iterator");

    // test custom user first strided iterator with normal ADL function
    using FirstStridedIter = custom_user::first_strided_iterator<BaseIter>;
    static_assert(
        oneapi::dpl::is_passed_directly_to_device_v<FirstStridedIter> == base_passed_directly,
        "is_passed_directly_in_onedpl_device_policies is not working correctly for custom user strided iterator");

    // test custom user second strided iterator (no body for is_passed_directly_in_onedpl_device_policies)
    using SecondStridedIter = custom_user::second_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<SecondStridedIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for custom user strided "
                  "iterator with no body in ADL function definiton");

    // test custom user first strided iterator with hidden friend ADL function
    using ThirdStridedIter = custom_user::third_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<ThirdStridedIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for custom user strided "
                  "iterator with hidden friend ADL function");

    // test custom user first strided iterator with hidden friend ADL function without body
    using FourthStridedIter = custom_user::fourth_strided_iterator<BaseIter>;
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<FourthStridedIter> == base_passed_directly,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for custom user strided "
                  "iterator with hidden friend ADL function without a body");
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
    test_with_base_iterator<TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(vec.begin())>,
                            decltype(vec.begin())>();

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
    static_assert(oneapi::dpl::is_passed_directly_to_device_v<oneapi::dpl::discard_iterator> == true,
                  "is_passed_directly_in_onedpl_device_policies is not working correctly for discard iterator");

#endif // TEST_DPCPP_BACKEND_PRESENT
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
