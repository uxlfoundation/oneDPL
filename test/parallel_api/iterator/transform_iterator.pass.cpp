// -*- C++ -*-
//===-- transform_iterator.pass.cpp ---------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils.h"

#include <tuple>

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/utils_device_copyable.h"

DEFINE_TEST(test_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy, 1.0f, 1.0f)

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Size, typename TExpectedValue>
    void operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Size n, TExpectedValue expected_value)
    {
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        ::std::copy(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);

        host_vals.retrieve_data();
        auto res_begin = host_vals.get();
        for(int i = 0; i != n; ++i) {
            EXPECT_EQ(expected_value, *res_begin, "Wrong result from copy with transform_iterator");
            ++res_begin;
        }
    }
}; // struct test_copy

void test_simple_copy(size_t buffer_size)
{
    // 1. create buffers
    using TestBaseData = test_base_data_buffer<int>;
    TestBaseData test_base_data({ { buffer_size, 0 },
                                  { buffer_size, 0 } });

    // 2. create iterators over buffers
    auto sycl_source_begin = test_base_data.get_start_from(UDTKind::eKeys);
    auto sycl_result_begin = test_base_data.get_start_from(UDTKind::eVals);

    // 3. run algorithms
    auto transformation = [](int item) { return item + 1; };

    auto tr_sycl_source_begin = oneapi::dpl::make_transform_iterator(sycl_source_begin, transformation);
    auto tr_sycl_source_end = tr_sycl_source_begin + buffer_size;

    int identity = 0;
    auto& sycl_src_buf = test_base_data.get_buffer(UDTKind::eKeys);
    auto host_source_begin = sycl_src_buf.get_host_access(sycl::write_only).get_pointer();
    ::std::fill_n(host_source_begin, buffer_size, identity);

    test_copy<int> test(test_base_data);
    invoke_on_all_hetero_policies<0>()(test, tr_sycl_source_begin, tr_sycl_source_end, sycl_result_begin, buffer_size, identity + 1);
}

void test_ignore_copy(size_t buffer_size)
{
    // 1. create buffers
    using TestBaseData = test_base_data_buffer<int>;
    TestBaseData test_base_data({ { buffer_size, 0 },
                                  { buffer_size, 0 } });

    // 2. create iterators over buffers
    auto& source_buf = test_base_data.get_buffer(UDTKind::eKeys);
    auto& result_buf = test_base_data.get_buffer(UDTKind::eVals);

    auto host_source_begin = source_buf.get_host_access(sycl::write_only).get_pointer();
    auto host_result_begin = result_buf.get_host_access(sycl::write_only).get_pointer();

    auto sycl_source_begin = oneapi::dpl::begin(source_buf);
    auto sycl_source_end = oneapi::dpl::end(source_buf);
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    // 3. run algorithms

    auto transformation = [](int) { return ::std::ignore; };

    auto tr_sycl_result_begin = oneapi::dpl::make_transform_iterator(sycl_result_begin, transformation);

    int ignored = -100;
    ::std::fill_n(host_source_begin, buffer_size, 1);
    ::std::fill_n(host_result_begin, buffer_size, ignored);

    test_copy<int> test(test_base_data);
    invoke_on_all_hetero_policies<1>()(test, sycl_source_begin, sycl_source_end, tr_sycl_result_begin, buffer_size, ignored);
}

void test_multi_transform_copy(size_t buffer_size)
{
    // 1. create buffers
    using TestBaseData = test_base_data_buffer<int>;
    TestBaseData test_base_data({ { buffer_size, 0 },
                                  { buffer_size, 0 } });

    // 2. create iterators over buffers
    sycl::buffer<int>& source_buf = test_base_data.get_buffer(UDTKind::eKeys);
    sycl::buffer<int>& result_buf = test_base_data.get_buffer(UDTKind::eVals);

    auto host_source_begin = source_buf.get_host_access(sycl::write_only).get_pointer();

    auto sycl_source_begin = oneapi::dpl::begin(source_buf);
    auto sycl_source_end = sycl_source_begin + buffer_size;
    auto sycl_result_begin = oneapi::dpl::begin(result_buf);

    // 3. run algorithms

    auto transformation = [](int item) { return item + 1; };

    auto tr_sycl_source_begin = oneapi::dpl::make_transform_iterator(sycl_source_begin, transformation);
    auto tr2_sycl_source_begin = oneapi::dpl::make_transform_iterator(tr_sycl_source_begin, transformation);
    auto tr3_sycl_source_begin = oneapi::dpl::make_transform_iterator(tr2_sycl_source_begin, transformation);
    auto tr3_sycl_source_end = tr3_sycl_source_begin + buffer_size;

    int identity = 0;
    ::std::fill_n(host_source_begin, buffer_size, identity);

    test_copy<int> test(test_base_data);
    invoke_on_all_hetero_policies<2>()(test, tr3_sycl_source_begin, tr3_sycl_source_end, sycl_result_begin, buffer_size, identity + 3);
}

void
test_copyable()
{
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::counting_iterator<int>>,
                  "counting_iterator is not device copyable");

    auto trans_count = ::dpl::make_transform_iterator(oneapi::dpl::counting_iterator<int>(0), [](auto i) { return i; });
    static_assert(sycl::is_device_copyable_v<decltype(trans_count)>,
                  "transform_iterator is not device copyable with a counting iterator and noop lambda");

    sycl::queue q;
    int* ptr = sycl::malloc_shared<int>(1, q);
    auto trans_array = ::dpl::make_transform_iterator(ptr, [](auto i) { return i; });
    static_assert(sycl::is_device_copyable_v<decltype(trans_array)>,
                  "transform_iterator is not device copyable with a pointer and noop lambda");
    sycl::free(ptr, q);

    static_assert(sycl::is_device_copyable_v<
                      oneapi::dpl::transform_iterator<constant_iterator_device_copyable, noop_device_copyable>>,
                  "transform_iterator is not device copyable with device copyable types");

    static_assert(!sycl::is_device_copyable_v<oneapi::dpl::transform_iterator<int*, noop_non_device_copyable>>,
                  "transform_iterator is device copyable with non device copyable types");

    static_assert(!sycl::is_device_copyable_v<
                      oneapi::dpl::transform_iterator<constant_iterator_non_device_copyable, noop_device_copyable>>,
                  "transform_iterator is device copyable with non device copyable types");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_copyable();

    size_t max_n = 10000;
    for (size_t n = 1; n <= max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_simple_copy(n);
        test_ignore_copy(n);
        test_multi_transform_copy(n);
    }
#endif

    return done(TEST_DPCPP_BACKEND_PRESENT);
}
