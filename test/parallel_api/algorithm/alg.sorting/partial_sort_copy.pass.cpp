// -*- C++ -*-
//===-- partial_sort_copy.pass.cpp ----------------------------------------===//
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

// Tests for partial_sort_copy

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <cmath>

using namespace TestUtils;

template <typename T>
struct Num
{
    T val;

    Num() : val(0) {}
    Num(T v) : val(v) {}
    Num(const Num<T>& v) : val(v.val) {}
    Num(Num<T>&& v) : val(v.val) {}
    Num<T>&
    operator=(const Num<T>& v)
    {
        val = v.val;
        return *this;
    }
    operator T() const { return val; }
    bool
    operator<(const Num<T>& v) const
    {
        return val < v.val;
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template <typename _T>
struct sycl::is_device_copyable<Num<_T>> : sycl::is_device_copyable<_T>
{
};
#endif //TEST_DPCPP_BACKEND_PRESENT

template <typename Type>
struct test_one_policy
{
    // Entities defined in ::std:: are prohibited to be inside a device kernel name,
    // thus avoid passing the iterator type as a template parameter to test_one_policy
    using RandomAccessIterator = typename Sequence<Type>::iterator;

    RandomAccessIterator d_first;
    RandomAccessIterator d_last;
    RandomAccessIterator exp_first;
    RandomAccessIterator exp_last;
    // This ctor is needed because output shouldn't be transformed to any iterator type (only random access iterators are allowed)
    test_one_policy(RandomAccessIterator b1, RandomAccessIterator e1, RandomAccessIterator b2, RandomAccessIterator e2)
        : d_first(b1), d_last(e1), exp_first(b2), exp_last(e2)
    {
    }

    template <typename Policy, typename InputIterator, typename Size, typename T, typename Compare>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash,
               Compare compare)
    {
        prepare_data(first, last, n1, trash);
        RandomAccessIterator exp = ::std::partial_sort_copy(first, last, exp_first, exp_last, compare);
        RandomAccessIterator res = ::std::partial_sort_copy(exec, first, last, d_first, d_last, compare);

        EXPECT_TRUE((exp - exp_first) == (res - d_first), "wrong result from partial_sort_copy with predicate");
        EXPECT_EQ_N(exp_first, d_first, n2, "wrong effect from partial_sort_copy with predicate");
    }

    template <typename Policy, typename InputIterator, typename Size, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, Size n1, Size n2, const T& trash)
    {
        prepare_data(first, last, n1, trash);
        RandomAccessIterator exp = ::std::partial_sort_copy(first, last, exp_first, exp_last);
        RandomAccessIterator res = ::std::partial_sort_copy(exec, first, last, d_first, d_last);

        EXPECT_TRUE((exp - exp_first) == (res - d_first), "wrong result from partial_sort_copy without predicate");
        EXPECT_EQ_N(exp_first, d_first, n2, "wrong effect from partial_sort_copy without predicate");
    }

  private:
    template <typename InputIterator, typename Size, typename T>
    void
    prepare_data(InputIterator first, InputIterator last, Size n1, const T& trash)
    {
        // The rand()%(2*n+1) encourages generation of some duplicates.
        ::std::srand(42);
        ::std::generate(first, last, [n1]() { return T(rand() % (2 * n1 + 1)); });

        ::std::fill(exp_first, exp_last, trash);
        ::std::fill(d_first, d_last, trash);
    }
};

template <typename T, typename Compare>
void
test_partial_sort_copy(Compare compare)
{
    const ::std::size_t n_max = 100000;
    Sequence<T> in(n_max);
    Sequence<T> out(2 * n_max);
    Sequence<T> exp(2 * n_max);
    ::std::size_t n1 = 0;
    ::std::size_t n2;
    T trash = T(-666);
    for (; n1 < n_max; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
#if !ONEDPL_FPGA_DEVICE
        // If both sequences are equal
        n2 = n1;
        invoke_on_all_policies<0>()(
            test_one_policy<T>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
                                              in.begin(), in.begin() + n1, n1, n2, trash, compare);
#endif

        // If first sequence is greater than second
        n2 = n1 / 3;
        invoke_on_all_policies<1>()(
            test_one_policy<T>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
                                              in.begin(), in.begin() + n1, n1, n2, trash, compare);

#if !ONEDPL_FPGA_DEVICE
        // If first sequence is less than second
        n2 = 2 * n1;
        invoke_on_all_policies<2>()(
            test_one_policy<T>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2),
                                              in.begin(), in.begin() + n1, n1, n2, trash, compare);
#endif
    }
    // Test partial_sort_copy without predicate
#if !ONEDPL_FPGA_DEVICE
    n1 = n_max;
    n2 = 2 * n1;
    invoke_on_all_policies<3>()(
        test_one_policy<T>(out.begin(), out.begin() + n2, exp.begin(), exp.begin() + n2), in.begin(),
                                          in.begin() + n1, n1, n2, trash);
#endif
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        partial_sort_copy(exec, input_iter, input_iter, out_iter, out_iter, non_const(::std::less<T>()));
    }
};

int
main()
{
#if !ONEDPL_FPGA_DEVICE
    test_partial_sort_copy<Num<float32_t>>([](Num<float32_t> x, Num<float32_t> y) { return x < y; });
    test_algo_basic_double<std::int32_t>(run_for_rnd<test_non_const<std::int32_t>>());
#endif
    test_partial_sort_copy<std::int32_t>([](std::int32_t x, std::int32_t y) { return x > y; });

#if !TEST_DPCPP_BACKEND_PRESENT
    test_partial_sort_copy<MemoryChecker>(
        [](const MemoryChecker& val1, const MemoryChecker& val2){ return val1.value() < val2.value(); });
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from partial_sort_copy: number of ctor and dtor calls is not equal");
#endif

    return done();
}
