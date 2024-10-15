// -*- C++ -*-
//===-- zip_iterator_scan.pass.cpp ---------------------------------------------===//
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

#include "zip_iterator_funcs.h"
#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#   include "support/utils_sycl.h"
#endif

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(iterator)

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;

DEFINE_TEST(test_transform_inclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_inclusive_scan, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (inclusive_scan1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>, "zip_iterator (inclusive_scan2) not properly copyable");
        }

        auto value = T1(333);

        auto res = std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1,
                                                 tuple_last1, tuple_first2, TupleNoOp{}, TupleNoOp{},
                                                 std::make_tuple(value, value));
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        EXPECT_TRUE(res == tuple_last2, "wrong effect from inclusive_scan(tuple)");
    }
};

DEFINE_TEST(test_unique)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using Iterator1ValueType = typename std::iterator_traits<Iterator1>::value_type;

        int index = 0;
        std::for_each(host_keys.get(), host_keys.get() + n,
                      [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        host_keys.update_data();

        const std::int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (unique) not properly copyable");
        }

        auto tuple_lastnew =
            std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                        TuplePredicate<std::equal_to<Iterator1ValueType>, 0>{std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_lastnew - tuple_first1) == expected_size;
        host_keys.retrieve_data();
        for (int i = 0; i < std::min(tuple_lastnew - tuple_first1, expected_size) && is_correct; ++i)
            if ((*host_keys.get() + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique(tuple)");
    }
};

DEFINE_TEST(test_unique_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename std::iterator_traits<Iterator1>::value_type;

        int index = 0;
        std::for_each(host_keys.get(), host_keys.get() + n,
                      [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        std::fill(host_vals.get(), host_vals.get() + n, Iterator1ValueType{-1});
        update_data(host_keys, host_vals);

        const std::int64_t expected_size = (n - 1) / 4 + 1;

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (unique_copy1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>, "zip_iterator (unique_copy2) not properly copyable");
        }

        auto tuple_last2 = std::unique_copy(
            make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1, tuple_first2,
            TuplePredicate<std::equal_to<Iterator1ValueType>, 0>{std::equal_to<Iterator1ValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        bool is_correct = (tuple_last2 - tuple_first2) == expected_size;
        host_vals.retrieve_data();
        for (int i = 0; i < std::min(tuple_last2 - tuple_first2, expected_size) && is_correct; ++i)
            if ((*host_vals.get() + i) != i + 1)
                is_correct = false;

        EXPECT_TRUE(is_correct, "wrong effect from unique_copy(tuple)");
    }
};

struct Assigner {
    template<typename T>
    bool operator()(T x) const {
        using std::get;
        return get<1>(x) != 0;
    }
};

// Make sure that it's possible to use counting iterator inside zip iterator
DEFINE_TEST(test_counting_zip_transform)
{
    DEFINE_TEST_CONSTRUCTOR(test_counting_zip_transform, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        if (n < 6)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename std::iterator_traits<Iterator2>::value_type;

        std::fill(host_keys.get(), host_keys.get() + n, ValueType{0});
        std::fill(host_vals.get(), host_vals.get() + n, ValueType{0});
        *(host_keys.get() + (n / 3)) = 10;
        *(host_keys.get() + (n / 3 * 2)) = 100;
        update_data(host_keys, host_vals);

        auto idx = oneapi::dpl::counting_iterator<ValueType>(0);
        auto start = oneapi::dpl::make_zip_iterator(idx, first1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(start)>, "zip_iterator (counting_iterator1) not properly copyable");
        }

        // This usage pattern can be rewritten equivalently and more simply using zip_iterator and discard_iterator,
        // see test_counting_zip_discard
        auto res =
            std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), start, start + n,
                         oneapi::dpl::make_transform_iterator(first2,
                                                              [](ValueType& x1)
                                                              {
                                                                  // It's required to use forward_as_tuple instead of make_tuple
                                                                  // as the latter do not propagate references.
                                                                  return std::forward_as_tuple(x1, std::ignore);
                                                              }),
                         Assigner{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(res.base() - first2 == 2, "Incorrect number of elements");
        EXPECT_TRUE(*host_vals.get() == n / 3, "Incorrect 1st element");
        EXPECT_TRUE(*(host_vals.get() + 1) == (n / 3 * 2), "Incorrect 2nd element");
    }
};

//make sure its possible to use a discard iterator in a zip iterator
DEFINE_TEST(test_counting_zip_discard)
{
    DEFINE_TEST_CONSTRUCTOR(test_counting_zip_discard, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        if (n < 6)
            return;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename std::iterator_traits<Iterator2>::value_type;

        std::fill(host_keys.get(), host_keys.get() + n, ValueType{0});
        std::fill(host_vals.get(), host_vals.get() + n, ValueType{0});
        *(host_keys.get() + (n / 3)) = 10;
        *(host_keys.get() + (n / 3 * 2)) = 100;
        update_data(host_keys, host_vals);

        auto idx = oneapi::dpl::counting_iterator<ValueType>(0);
        auto start = oneapi::dpl::make_zip_iterator(idx, first1);
        auto discard = oneapi::dpl::discard_iterator();
        auto out = oneapi::dpl::make_zip_iterator(first2, discard);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(start)>, "zip_iterator (discard_iterator1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(out)>, "zip_iterator (discard_iterator2) not properly copyable");
        }

        auto res = std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), start, start + n, out, Assigner{});

#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        host_vals.retrieve_data();
        EXPECT_TRUE(res - out == 2, "Incorrect number of elements");
        EXPECT_TRUE(*host_vals.get() == n / 3, "Incorrect 1st element");
        EXPECT_TRUE(*(host_vals.get() + 1) == (n / 3 * 2), "Incorrect 2nd element");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_inclusive_scan");
    test2buffers<alloc_type, test_transform_inclusive_scan<ValueType>>();
    PRINT_DEBUG("test_unique");
    test1buffer<alloc_type, test_unique<ValueType>>();
    PRINT_DEBUG("test_unique_copy");
    test2buffers<alloc_type, test_unique_copy<ValueType>>();
    PRINT_DEBUG("test_counting_zip_transform");
    test2buffers<alloc_type, test_counting_zip_transform<ValueType>>();
    PRINT_DEBUG("test_counting_zip_discard");
    test2buffers<alloc_type, test_counting_zip_discard<ValueType>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
    //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

    // Run tests for USM shared memory
    test_usm_and_buffer<sycl::usm::alloc::shared>();
    // Run tests for USM device memory
    test_usm_and_buffer<sycl::usm::alloc::device>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return done(TEST_DPCPP_BACKEND_PRESENT);
}
