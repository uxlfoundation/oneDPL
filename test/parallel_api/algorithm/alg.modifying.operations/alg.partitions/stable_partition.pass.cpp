// -*- C++ -*-
//===-- stable_partition.pass.cpp -----------------------------------------===//
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

// Tests for stable_partition
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iterator>
#include <type_traits>

using namespace TestUtils;

template <typename T>
struct DataType
{
    explicit DataType(std::int32_t k) : my_val(k) {}
    DataType(DataType&& input) { my_val = ::std::move(input.my_val); }
    DataType&
    operator=(DataType&& input)
    {
        my_val = ::std::move(input.my_val);
        return *this;
    }
    T
    get_val() const
    {
        return my_val;
    }

    friend ::std::ostream&
    operator<<(::std::ostream& stream, const DataType<T>& input)
    {
        return stream << input.my_val;
    }

  private:
    T my_val;
};

template <typename Iterator>
::std::enable_if_t<::std::is_trivial_v<typename ::std::iterator_traits<Iterator>::value_type>, bool>
is_equal(Iterator first, Iterator last, Iterator d_first)
{
    return ::std::equal(first, last, d_first);
}

template <typename Iterator>
::std::enable_if_t<!::std::is_trivial_v<typename ::std::iterator_traits<Iterator>::value_type>, bool>
is_equal(Iterator /* first */, Iterator /* last */, Iterator /* d_first */)
{
    return true;
}

template <typename T>
struct test_stable_partition
{
    template <typename Policy, typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt>>
    operator()(Policy&& exec, BiDirIt first, BiDirIt last, BiDirIt exp_first, BiDirIt exp_last, Size /* n */,
               UnaryOp unary_op, Generator generator)
    {
        fill_data(exp_first, exp_last, generator);
        BiDirIt exp_ret = ::std::stable_partition(exp_first, exp_last, unary_op);
        fill_data(first, last, generator);
        BiDirIt actual_ret = ::std::stable_partition(exec, first, last, unary_op);

        EXPECT_TRUE(::std::distance(first, actual_ret) == ::std::distance(exp_first, exp_ret),
                    "wrong result from stable_partition");
        EXPECT_TRUE((is_equal<BiDirIt>(exp_first, exp_last, first)), "wrong effect from stable_partition");
    }

    template <typename Policy, typename BiDirIt, typename Size, typename UnaryOp, typename Generator>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt>>
    operator()(Policy&& /* exec */, BiDirIt /* first */, BiDirIt /* last */, BiDirIt /* exp_first */, BiDirIt /* exp_last */, Size /* n */,
               UnaryOp /* unary_op */, Generator /* generator */)
    {
    }
};

template <typename T, typename Generator, typename UnaryPred>
void
test_by_type(Generator generator, UnaryPred pred)
{

    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<1>()(test_stable_partition<T>(), in.begin(), in.begin() + n, exp.begin(),
                                    exp.begin() + n, n, pred, generator);
    }
}

struct test_non_const_stable_partition
{
    template <typename Policy, typename Iterator>
    void
        operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };

        stable_partition(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
    test_by_type<std::int32_t>([](std::int32_t i) { return i; }, [](std::int32_t) { return true; });
    test_by_type<float64_t>([](std::int32_t i) { return -i; }, [](const float64_t x) { return x < 0; });
    test_by_type<std::int64_t>([](std::int32_t i) { return i + 1; }, [](std::int64_t x) { return x % 3 == 0; });

#if !TEST_DPCPP_BACKEND_PRESENT
    test_by_type<DataType<float32_t>>([](std::int32_t i) { return DataType<float32_t>(2 * i + 1); },
                                      [](const DataType<float32_t>& x) { return x.get_val() < 0; });
#endif

    test_algo_basic_single<std::int32_t>(run_for_rnd_bi<test_non_const_stable_partition>());

    return done();
}
