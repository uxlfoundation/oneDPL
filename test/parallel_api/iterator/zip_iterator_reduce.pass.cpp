// -*- C++ -*-
//===-- zip_iterator_reduce.pass.cpp ---------------------------------------------===//
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
#include _PSTL_TEST_HEADER(numeric)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
using namespace oneapi::dpl::execution;

DEFINE_TEST(test_transform_reduce_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_unary, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(1);
        std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (reduce_unary) not properly copyable");
        }

        std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                              std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
    }
};

DEFINE_TEST(test_transform_reduce_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_binary, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* first2 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename std::iterator_traits<Iterator1>::value_type T1;

        auto value = T1(1);
        std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>, "zip_iterator (reduce_binary) not properly copyable");
        }

        std::transform_reduce(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1,
                              tuple_last1, tuple_first1, std::make_tuple(T1{42}, T1{42}), TupleNoOp{}, TupleNoOp{});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
    }
};

DEFINE_TEST(test_min_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_min_element, 1.0f, 1.0f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{static_cast<IteratorValueType>(n)};
        std::for_each(host_keys.get(), host_keys.get() + n,
                      [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        auto min_dis = n;
        if (min_dis)
        {
            auto min_it = host_keys.get() + /*min_idx*/ min_dis / 2;
            *(min_it) = IteratorValueType{/*min_val*/ 0};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd min*/ 0};
        }
        host_keys.update_data();

        auto tuple_first = oneapi::dpl::make_zip_iterator(first, first);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last, last);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first)>, "zip_iterator (min_element) not properly copyable");
        }

        auto tuple_result =
            std::min_element(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                             TuplePredicate<std::less<IteratorValueType>, 0>{std::less<IteratorValueType>{}});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif
        auto expected_min = std::min_element(host_keys.get(), host_keys.get() + n);

        EXPECT_TRUE((tuple_result - tuple_first) == (expected_min - host_keys.get()),
                    "wrong effect from min_element(tuple)");
    }
};

DEFINE_TEST(test_count_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_count_if, 1.0f, 1.0f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        auto tuple_first = oneapi::dpl::make_zip_iterator(first, first);
        auto tuple_last = oneapi::dpl::make_zip_iterator(last, last);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first)>, "zip_iterator (count_if) not properly copyable");
        }

        auto comp = [](ValueType const& value) { return value % 10 == 0; };
        ReturnType expected = (n - 1) / 10 + 1;

        auto result = std::count_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first, tuple_last,
                                    TuplePredicate<decltype(comp), 0>{comp});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        EXPECT_TRUE(result == expected, "wrong effect from count_if(tuple)");
    }
};

DEFINE_TEST(test_lexicographical_compare)
{
    DEFINE_TEST_CONSTRUCTOR(test_lexicographical_compare, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename std::iterator_traits<Iterator1>::value_type;

        // init
        ValueType fill_value1{0};
        std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
        ValueType fill_value2{0};
        std::for_each(host_vals.get(), host_vals.get() + n,
                        [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
        if (n > 1)
            *(host_vals.get() + n - 2) = 222;
        update_data(host_keys, host_vals);

        auto tuple_first1 = oneapi::dpl::make_zip_iterator(first1, first1);
        auto tuple_last1 = oneapi::dpl::make_zip_iterator(last1, last1);
        auto tuple_first2 = oneapi::dpl::make_zip_iterator(first2, first2);
        auto tuple_last2 = oneapi::dpl::make_zip_iterator(last2, last2);

        //check device copyable only for usm iterator based data, it is not required or expected for sycl buffer data
        if (!this->host_buffering_required())
        {
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first1)>,
                        "zip_iterator (lexicographical_compare1) not properly copyable");
            EXPECT_TRUE(sycl::is_device_copyable_v<decltype(tuple_first2)>,
                        "zip_iterator (lexicographical_compare2) not properly copyable");
        }

        auto comp = [](ValueType const& first, ValueType const& second) { return first < second; };

        bool is_less_exp = n > 1 ? 1 : 0;
        bool is_less_res =
            std::lexicographical_compare(make_new_policy<new_kernel_name<Policy, 0>>(exec), tuple_first1, tuple_last1,
                                           tuple_first2, tuple_last2, TuplePredicate<decltype(comp), 0>{comp});
#if _PSTL_SYCL_TEST_USM
        exec.queue().wait_and_throw();
#endif

        if (is_less_res != is_less_exp)
            std::cout << "N=" << n << ": got " << is_less_res << ", expected " << is_less_exp << std::endl;
        EXPECT_TRUE(is_less_res == is_less_exp, "wrong effect from lex_compare (tuple)");
    }
};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = std::int32_t;
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<alloc_type, test_transform_reduce_unary<ValueType>>();
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<alloc_type, test_transform_reduce_binary<ValueType>>();
    PRINT_DEBUG("test_min_element");
    test1buffer<alloc_type, test_min_element<ValueType>>();
    PRINT_DEBUG("test_count_if");
    test1buffer<alloc_type, test_count_if<ValueType>>();
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<alloc_type, test_lexicographical_compare<ValueType>>();
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

