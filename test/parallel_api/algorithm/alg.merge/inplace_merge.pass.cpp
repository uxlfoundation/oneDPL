// -*- C++ -*-
//===-- inplace_merge.pass.cpp --------------------------------------------===//
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

#include "support/utils.h"

using namespace TestUtils;

template <typename Type>
struct test_one_policy
{
    // inplace_merge works with bidirectional iterators at least
    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt1>>
    operator()(Policy&& exec, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2, BiDirIt1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2, Compare comp)
    {
        auto mid = init(first1, last1, first2, last2, generator1, generator2, m);
        ::std::inplace_merge(first1, mid.first, last1, comp);
        ::std::inplace_merge(exec, first2, mid.second, last2, comp);
        EXPECT_EQ_N(first1, first2, n, "wrong effect from inplace_merge with predicate");
    }

    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt1>>
    operator()(Policy&& exec, BiDirIt1 first1, BiDirIt1 last1, BiDirIt1 first2, BiDirIt1 last2, Size n, Size m,
               Generator1 generator1, Generator2 generator2)
    {
        auto mid = init(first1, last1, first2, last2, generator1, generator2, m);
        ::std::inplace_merge(first1, mid.first, last1);
        ::std::inplace_merge(exec, first2, mid.second, last2);
        EXPECT_EQ_N(first1, first2, n, "wrong effect from inplace_merge without predicate");
    }

    template<typename BiDirIt, typename Generator1, typename Generator2, typename Size>
    ::std::pair<const BiDirIt, const BiDirIt> init(BiDirIt first1, BiDirIt last1, BiDirIt first2, BiDirIt last2,
                                                    Generator1 generator1, Generator2 generator2, Size m)
    {
        const BiDirIt mid1 = ::std::next(first1, m);
        fill_data(first1, mid1, generator1);
        fill_data(mid1, last1, generator2);

        const BiDirIt mid2 = ::std::next(first2, m);
        fill_data(first2, mid2, generator1);
        fill_data(mid2, last2, generator2);
        return ::std::make_pair(mid1, mid2);
    }

    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt1>>
    operator()(Policy&& /* exec */, BiDirIt1 /* first1 */, BiDirIt1 /* last1 */, BiDirIt1 /* first2 */, BiDirIt1 /* last2 */, Size /* n */, Size /* m */,
               Generator1 /* generator1 */, Generator2 /* generator2 */)
    {
    }
    template <typename Policy, typename BiDirIt1, typename Size, typename Generator1, typename Generator2,
              typename Compare>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, BiDirIt1>>
    operator()(Policy&& /* exec */, BiDirIt1 /* first1 */, BiDirIt1 /* last1 */, BiDirIt1 /* first2 */, BiDirIt1 /* last2 */, Size /* n */, Size /* m */,
               Generator1 /* generator1 */, Generator2 /* generator2 */, Compare /* comp */)
    {
    }
};

template <typename T, typename Generator1, typename Generator2, typename Compare>
void
test_by_type(Generator1 generator1, Generator2 generator2, bool comp_flag, Compare comp)
{
    using namespace std;
    size_t max_size = 100000;
    Sequence<T> in1(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });
    size_t m;

    
    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        m = 0;
        if(comp_flag)
            invoke_on_all_policies<0>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                                   generator1, generator2, comp);
        else
            invoke_on_all_policies<1>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2);
        m = n / 3;
        if(comp_flag)
            invoke_on_all_policies<2>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                                   generator1, generator2, comp);
        else
            invoke_on_all_policies<3>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2);
#if !ONEDPL_FPGA_DEVICE
        m = 2 * n / 3;
        if(comp_flag)
            invoke_on_all_policies<4>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                                   generator1, generator2, comp);
        else
            invoke_on_all_policies<5>()(test_one_policy<T>(), in1.begin(), in1.begin() + n, exp.begin(), exp.begin() + n, n, m,
                               generator1, generator2);
#endif
    }
}

template <typename T>
struct LocalWrapper
{
    explicit LocalWrapper(std::int32_t k) : my_val(k) {}
    LocalWrapper(LocalWrapper&& input) { my_val = ::std::move(input.my_val); }
    LocalWrapper&
    operator=(LocalWrapper&& input)
    {
        my_val = ::std::move(input.my_val);
        return *this;
    }
    bool
    operator<(const LocalWrapper<T>& w) const
    {
        return my_val < w.my_val;
    }
    friend bool
    operator==(const LocalWrapper<T>& x, const LocalWrapper<T>& y)
    {
        return x.my_val == y.my_val;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& stream, const LocalWrapper<T>& input)
    {
        return stream << input.my_val;
    }

  private:
    T my_val;
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        inplace_merge(exec, iter, iter, iter, non_const(::std::less<T>()));
    }
};

int
main()
{
#if !ONEDPL_FPGA_DEVICE
    test_by_type<float64_t>([](std::int32_t i) { return -2 * i; }, [](std::int32_t i) { return -(2 * i + 1); }, true,
                            [](const float64_t x, const float64_t y) { return x > y; });
#endif

    test_by_type<std::int32_t>([](std::int32_t i) { return 10 * i; }, [](std::int32_t i) { return i + 1; }, false, ::std::less<std::int32_t>());

#if !TEST_DPCPP_BACKEND_PRESENT
    test_by_type<LocalWrapper<float32_t>>([](std::int32_t i) { return LocalWrapper<float32_t>(2 * i + 1); },
                                          [](std::int32_t i) { return LocalWrapper<float32_t>(2 * i); }, true,
                                          ::std::less<LocalWrapper<float32_t>>());
    test_by_type<MemoryChecker>(
        [](::std::size_t idx){ return MemoryChecker{::std::int32_t(idx * 2)}; },
        [](::std::size_t idx){ return MemoryChecker{::std::int32_t(idx * 2 + 1)}; }, true,
        [](const MemoryChecker& val1, const MemoryChecker& val2){ return val1.value() < val2.value(); });
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from inplace_merge: number of ctor and dtor calls is not equal");
#endif
    test_algo_basic_single<std::int32_t>(run_for_rnd_bi<test_non_const<std::int32_t>>());

    return done();
}
