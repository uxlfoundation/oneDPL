// -*- C++ -*-
//===-- minmax_element.pass.cpp -------------------------------------------===//
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

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <set>
#include <type_traits>
#include <vector>

#if  !defined(_PSTL_TEST_MIN_ELEMENT) && !defined(_PSTL_TEST_MAX_ELEMENT) &&\
     !defined(_PSTL_TEST_MINMAX_ELEMENT) && !_PSTL_ICPX_TEST_MINMAX_ELEMENT_PASS_BROKEN
#define _PSTL_TEST_MIN_ELEMENT
#define _PSTL_TEST_MAX_ELEMENT
#define _PSTL_TEST_MINMAX_ELEMENT
#endif

using namespace TestUtils;

template <typename Type>
struct check_minelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const Iterator expect = std::min_element(begin, end);
        const Iterator result = std::min_element(std::forward<Policy>(exec), begin, end);
        EXPECT_EQ(expect, result, "wrong return result from min_element");
    }
};

template <typename Type>
struct check_minelement_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        const Iterator expect = std::min_element(begin, end);
        const Iterator result_pred = std::min_element(std::forward<Policy>(exec), begin, end, std::less<T>());
        EXPECT_EQ(expect, result_pred, "wrong return result from min_element with predicate");
    }
};

template <typename Type>
struct check_maxelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const Iterator expect = std::max_element(begin, end);
        const Iterator result = std::max_element(std::forward<Policy>(exec), begin, end);
        EXPECT_EQ(expect, result, "wrong return result from max_element");
    }
};

template <typename Type>
struct check_maxelement_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        const Iterator expect = std::max_element(begin, end);
        const Iterator result_pred = std::max_element(std::forward<Policy>(exec), begin, end, std::less<T>());
        EXPECT_EQ(expect, result_pred, "wrong return result from max_element with predicate");
    }
};

template <typename Type>
struct check_minmaxelement
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const std::pair<Iterator, Iterator> expect = std::minmax_element(begin, end);
        const std::pair<Iterator, Iterator> got = std::minmax_element(std::forward<Policy>(exec), begin, end);
        EXPECT_EQ(expect.first, got.first, "wrong return result from minmax_element (min part)");
        EXPECT_EQ(expect.second, got.second, "wrong return result from minmax_element (max part)");
    }
};

template <typename Type>
struct check_minmaxelement_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        const std::pair<Iterator, Iterator> expect = std::minmax_element(begin, end);
        const std::pair<Iterator, Iterator> got_pred = std::minmax_element(std::forward<Policy>(exec), begin, end, std::less<T>());
        EXPECT_EQ(expect, got_pred, "wrong return result from minmax_element with predicate");
    }
};

// Unary operator& is deleted, so the address of the comparator may only be taken with std::addressof.
struct OverloadedAddressOfLess
{
    void
    operator&() = delete;
    void
    operator&() const = delete;

    bool
    operator()(const std::int32_t& lhs, const std::int32_t& rhs) const
    {
        return lhs < rhs;
    }
};

template <typename Type>
struct check_minelement_overloaded_address_of
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const Iterator expect = std::min_element(begin, end);
        const Iterator result = std::min_element(std::forward<Policy>(exec), begin, end, OverloadedAddressOfLess());
        EXPECT_EQ(expect, result, "wrong return result from min_element with a comparator overloading operator&");
    }
};

template <typename Type>
struct check_maxelement_overloaded_address_of
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const Iterator expect = std::max_element(begin, end);
        const Iterator result = std::max_element(std::forward<Policy>(exec), begin, end, OverloadedAddressOfLess());
        EXPECT_EQ(expect, result, "wrong return result from max_element with a comparator overloading operator&");
    }
};

template <typename Type>
struct check_minmaxelement_overloaded_address_of
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
        const std::pair<Iterator, Iterator> expect = std::minmax_element(begin, end);
        const std::pair<Iterator, Iterator> got =
            std::minmax_element(std::forward<Policy>(exec), begin, end, OverloadedAddressOfLess());
        EXPECT_EQ(expect, got, "wrong return result from minmax_element with a comparator overloading operator&");
    }
};

template <typename T>
struct sequence_wrapper
{
    TestUtils::Sequence<T> seq;
    const T min_value;
    const T max_value;
    static const std::size_t bits = 30; // We assume that T can handle signed 2^bits+1 value

    // TestUtils::HashBits returns value between 0 and (1<<bits)-1,
    // therefore we could threat 1<<bits as maximum and -(1<<bits) as a minimum
    sequence_wrapper(std::size_t n) : seq(n), min_value(-(1 << bits)), max_value(1 << bits) {}

    void
    pattern_fill()
    {
        seq.fill([](std::size_t i) -> T { return T(TestUtils::HashBits(i, bits)); });
    }

    // sets first one at position `at` and bunch of them farther
    void
    set_desired_value(std::size_t at, T value)
    {
        if (seq.size() == 0)
            return;
        seq[at] = value;

        //Producing several red herrings
        for (std::size_t i = at + 1; i < seq.size(); i += 1 + TestUtils::HashBits(i, 5))
            seq[i] = value;
    }
};

template <typename T>
void
test_by_type(std::size_t n)
{
    sequence_wrapper<T> wseq(n);

    // to avoid overtesing we use std::set to leave only unique indexes
    std::set<std::size_t> targets{0};
    if (n > 1)
    {
        targets.insert(1);
        targets.insert(2.718282 * n / 3);
        targets.insert(n / 2);
        targets.insert(n / 7.389056);
        targets.insert(n - 1); // last
    }

    for (std::set<std::size_t>::iterator it = targets.begin(); it != targets.end(); ++it)
    {
        wseq.pattern_fill();
#ifdef _PSTL_TEST_MIN_ELEMENT
        wseq.set_desired_value(*it, wseq.min_value);
        invoke_on_all_policies<0>()(check_minelement<T>(), wseq.seq.begin(), wseq.seq.end());
        invoke_on_all_policies<1>()(check_minelement_predicate<T>(), wseq.seq.begin(), wseq.seq.end());
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<2>()(check_minelement<T>(), wseq.seq.cbegin(), wseq.seq.cend());
        invoke_on_all_policies<3>()(check_minelement_predicate<T>(), wseq.seq.cbegin(), wseq.seq.cend());
#endif
#endif

#ifdef _PSTL_TEST_MAX_ELEMENT
        wseq.set_desired_value(*it, wseq.max_value);
        invoke_on_all_policies<4>()(check_maxelement<T>(), wseq.seq.begin(), wseq.seq.end());
        invoke_on_all_policies<5>()(check_maxelement_predicate<T>(), wseq.seq.begin(), wseq.seq.end());
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<6>()(check_maxelement<T>(), wseq.seq.cbegin(), wseq.seq.cend());
        invoke_on_all_policies<7>()(check_maxelement_predicate<T>(), wseq.seq.cbegin(), wseq.seq.cend());
#endif
#endif

#ifdef _PSTL_TEST_MINMAX_ELEMENT
        if (targets.size() > 1)
        {
            for (std::set<std::size_t>::reverse_iterator rit = targets.rbegin(); rit != targets.rend(); ++rit)
            {
                if (*rit == *it) // we requires at least 2 unique indexes in targets
                    break;
                wseq.pattern_fill();
                wseq.set_desired_value(*it, wseq.min_value);  // setting minimum element
                wseq.set_desired_value(*rit, wseq.max_value); // setting maximum element
                invoke_on_all_policies<8>()(check_minmaxelement<T>(), wseq.seq.begin(), wseq.seq.end());
                invoke_on_all_policies<9>()(check_minmaxelement_predicate<T>(), wseq.seq.begin(), wseq.seq.end());
#if !ONEDPL_FPGA_DEVICE
                invoke_on_all_policies<10>()(check_minmaxelement<T>(), wseq.seq.cbegin(), wseq.seq.cend());
                invoke_on_all_policies<11>()(check_minmaxelement_predicate<T>(), wseq.seq.cbegin(), wseq.seq.cend());
#endif
            }
        }
        else
        { // we must check this corner case; it can not be tested in loop above
            invoke_on_all_policies<12>()(check_minmaxelement<T>(), wseq.seq.begin(), wseq.seq.end());
            invoke_on_all_policies<13>()(check_minmaxelement_predicate<T>(), wseq.seq.begin(), wseq.seq.end());
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<14>()(check_minmaxelement<T>(), wseq.seq.cbegin(), wseq.seq.cend());
            invoke_on_all_policies<15>()(check_minmaxelement_predicate<T>(), wseq.seq.cbegin(), wseq.seq.cend());
#endif
        }
#endif
    }
}

// should provide minimal requirements only
struct OnlyLessCompare
{
    std::int32_t val;
    OnlyLessCompare() : val(0) {}
    OnlyLessCompare(std::int32_t val_) : val(val_) {}
    bool
    operator<(const OnlyLessCompare& other) const
    {
        return val < other.val;
    }
};

// Default-constructible through an explicit default constructor only.
struct ExplicitDefaultCtorCompare
{
    std::int32_t val;
    explicit ExplicitDefaultCtorCompare() : val(0) {}
    ExplicitDefaultCtorCompare(std::int32_t val_) : val(val_) {}
    bool
    operator<(const ExplicitDefaultCtorCompare& other) const
    {
        return val < other.val;
    }
};

// Default-constructible, but not brace-initializable: empty braces copy-list-initialize the member, which its explicit
// default constructor rejects.
struct AggregateOfExplicitDefaultCtorCompare
{
    ExplicitDefaultCtorCompare member;
    bool
    operator<(const AggregateOfExplicitDefaultCtorCompare& other) const
    {
        return member < other.member;
    }
};

// Not default-constructible: neither constructor takes zero arguments. Empty braces select the initializer-list one.
struct BraceInitOnlyCompare
{
    std::int32_t val;
    BraceInitOnlyCompare(std::initializer_list<std::int32_t> init) : val(init.size() == 0 ? 0 : *init.begin()) {}
    BraceInitOnlyCompare(std::int32_t val_) : val(val_) {}
    bool
    operator<(const BraceInitOnlyCompare& other) const
    {
        return val < other.val;
    }
};

// Copyable, but with deleted move operations.
struct CopyOnlyNoMoveCompare
{
    std::int32_t val;
    CopyOnlyNoMoveCompare() : val(0) {}
    CopyOnlyNoMoveCompare(std::int32_t val_) : val(val_) {}
    CopyOnlyNoMoveCompare(const CopyOnlyNoMoveCompare&) = default;
    CopyOnlyNoMoveCompare&
    operator=(const CopyOnlyNoMoveCompare&) = default;
    CopyOnlyNoMoveCompare(CopyOnlyNoMoveCompare&&) = delete;
    CopyOnlyNoMoveCompare&
    operator=(CopyOnlyNoMoveCompare&&) = delete;
    bool
    operator<(const CopyOnlyNoMoveCompare& other) const
    {
        return val < other.val;
    }
};

// The copy assignment returns void instead of VoidAssignCompare&.
struct VoidAssignCompare
{
    std::int32_t val;
    VoidAssignCompare() : val(0) {}
    VoidAssignCompare(std::int32_t val_) : val(val_) {}
    void
    operator=(const VoidAssignCompare& other)
    {
        val = other.val;
    }
    bool
    operator<(const VoidAssignCompare& other) const
    {
        return val < other.val;
    }
};

// Copyable and assignable from a const lvalue only, so it requires const iterators.
struct ConstCopyOnlyCompare
{
    std::int32_t val;
    ConstCopyOnlyCompare() : val(0) {}
    ConstCopyOnlyCompare(std::int32_t val_) : val(val_) {}
    ConstCopyOnlyCompare(const ConstCopyOnlyCompare&) = default;
    ConstCopyOnlyCompare(ConstCopyOnlyCompare&) = delete;
    ConstCopyOnlyCompare&
    operator=(const ConstCopyOnlyCompare&) = default;
    ConstCopyOnlyCompare&
    operator=(ConstCopyOnlyCompare&) = delete;
    bool
    operator<(const ConstCopyOnlyCompare& other) const
    {
        return val < other.val;
    }
};

// A type that is not default-constructible is taken from the test utilities:
// TestUtils::NoDefaultCtorWrapper<std::int32_t>. It compares through its conversion to the underlying type.

// Not copy-assignable.
struct NoCopyAssignCompare
{
    std::int32_t val;
    NoCopyAssignCompare() : val(0) {}
    NoCopyAssignCompare(std::int32_t val_) : val(val_) {}
    NoCopyAssignCompare(const NoCopyAssignCompare&) = default;
    NoCopyAssignCompare&
    operator=(const NoCopyAssignCompare&) = delete;
    bool
    operator<(const NoCopyAssignCompare& other) const
    {
        return val < other.val;
    }
};

// Not copy-constructible.
struct MoveOnlyCompare
{
    std::int32_t val;
    MoveOnlyCompare() : val(0) {}
    MoveOnlyCompare(std::int32_t val_) : val(val_) {}
    MoveOnlyCompare(MoveOnlyCompare&&) = default;
    MoveOnlyCompare&
    operator=(MoveOnlyCompare&&) = default;
    MoveOnlyCompare(const MoveOnlyCompare&) = delete;
    MoveOnlyCompare&
    operator=(const MoveOnlyCompare&) = delete;
    bool
    operator<(const MoveOnlyCompare& other) const
    {
        return val < other.val;
    }
};

template <typename T, typename Iterator>
static void
check_by_type_host_policies(Iterator first, Iterator last)
{
#ifdef _PSTL_TEST_MIN_ELEMENT
    invoke_on_all_host_policies()(check_minelement<T>(), first, last);
    invoke_on_all_host_policies()(check_minelement_predicate<T>(), first, last);
#endif
#ifdef _PSTL_TEST_MAX_ELEMENT
    invoke_on_all_host_policies()(check_maxelement<T>(), first, last);
    invoke_on_all_host_policies()(check_maxelement_predicate<T>(), first, last);
#endif
#ifdef _PSTL_TEST_MINMAX_ELEMENT
    invoke_on_all_host_policies()(check_minmaxelement<T>(), first, last);
    invoke_on_all_host_policies()(check_minmaxelement_predicate<T>(), first, last);
#endif
}

// The value types checked here do not satisfy the requirements of TestUtils::Sequence, so the data is built in place.
template <typename T, bool UseConstIterators = false>
static void
test_by_type_host_policies(std::size_t n)
{
    std::vector<T> data;
    data.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        data.emplace_back(std::int32_t(TestUtils::HashBits(i, 30)));

    using Iterator = std::conditional_t<UseConstIterators, typename std::vector<T>::const_iterator,
                                          typename std::vector<T>::iterator>;
    check_by_type_host_policies<T>(Iterator(data.begin()), Iterator(data.end()));
}

// An aggregate cannot be constructed with parentheses before C++20, so elements are brace-initialized, not emplaced.
template <typename T>
static void
test_by_type_host_policies_brace_init(std::size_t n)
{
    std::vector<T> data;
    data.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        data.push_back(T{std::int32_t(TestUtils::HashBits(i, 30))});

    check_by_type_host_policies<T>(data.begin(), data.end());
}

// A type with deleted move operations cannot be pushed into a std::vector, so the vector is sized up front and its
// elements are assigned. A plain array is not used on purpose: with the bounds known at compile time, GCC reports a
// false out-of-bounds subscript in the parallel reduction.
template <typename T>
static void
test_by_type_host_policies_no_move(std::size_t n)
{
    std::vector<T> data(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const T value(std::int32_t(TestUtils::HashBits(i, 30)));
        data[i] = value;
    }

    check_by_type_host_policies<T>(data.begin(), data.end());
}

static void
test_comparator_with_overloaded_address_of(std::size_t n)
{
    Sequence<std::int32_t> in(n, [](std::size_t i) { return std::int32_t(TestUtils::HashBits(i, 30)); });

#ifdef _PSTL_TEST_MIN_ELEMENT
    invoke_on_all_host_policies()(check_minelement_overloaded_address_of<std::int32_t>(), in.begin(), in.end());
#endif
#ifdef _PSTL_TEST_MAX_ELEMENT
    invoke_on_all_host_policies()(check_maxelement_overloaded_address_of<std::int32_t>(), in.begin(), in.end());
#endif
#ifdef _PSTL_TEST_MINMAX_ELEMENT
    invoke_on_all_host_policies()(check_minmaxelement_overloaded_address_of<std::int32_t>(), in.begin(), in.end());
#endif
}

template <typename T>
struct test_non_const_max_element
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        max_element(std::forward<Policy>(exec), iter, iter, non_const(std::less<T>()));
    }
};

template <typename T>
struct test_non_const_min_element
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        min_element(std::forward<Policy>(exec), iter, iter, non_const(std::less<T>()));
    }
};

template <typename T>
struct test_non_const_minmax_element
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        minmax_element(std::forward<Policy>(exec), iter, iter, non_const(std::less<T>()));
    }
};

int
main()
{
    using TestUtils::float64_t;
    const std::size_t N = 100000;
    const std::size_t NSmall = 10;

    for (std::size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.14159 * n))
    {
#if !ONEDPL_FPGA_DEVICE
        test_by_type<std::int32_t>(n);
#endif
        test_by_type<float64_t>(n);
        test_by_type<OnlyLessCompare>(n);
    }

    // These value types are accepted by the vector code path: it must be instantiated for them. Whether it compiles
    // does not depend on the sequence size, so a single small size is enough for all the checks below.
    test_by_type<ExplicitDefaultCtorCompare>(NSmall);
    test_by_type_host_policies_brace_init<AggregateOfExplicitDefaultCtorCompare>(NSmall);
    test_by_type_host_policies_no_move<CopyOnlyNoMoveCompare>(NSmall);
    test_by_type_host_policies<VoidAssignCompare>(NSmall);
    test_by_type_host_policies<ConstCopyOnlyCompare, /*UseConstIterators*/ true>(NSmall);

    // These value types are rejected by the vector code path: the call must compile and fall back to the serial one.
    test_by_type_host_policies<TestUtils::NoDefaultCtorWrapper<std::int32_t>>(NSmall);
    test_by_type_host_policies<BraceInitOnlyCompare>(NSmall);
    test_by_type_host_policies<NoCopyAssignCompare>(NSmall);
    test_by_type_host_policies<MoveOnlyCompare>(NSmall);

    // The sequence is long enough for the vector code to process several blocks and to combine their results.
    test_comparator_with_overloaded_address_of(1000);

#ifdef _PSTL_TEST_MIN_ELEMENT
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_min_element<std::int32_t>>());
#endif
#ifdef _PSTL_TEST_MAX_ELEMENT
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_max_element<std::int32_t>>());
#endif
#ifdef _PSTL_TEST_MINMAX_ELEMENT
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_minmax_element<std::int32_t>>());
#endif

    return done();
}
