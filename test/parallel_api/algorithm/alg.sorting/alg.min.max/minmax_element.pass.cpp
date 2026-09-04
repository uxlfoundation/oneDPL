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

#include <set>
#include <vector>
#include <cassert>
#include <cmath>

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
        const Iterator expect = ::std::min_element(begin, end);
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
        const Iterator expect = ::std::min_element(begin, end);
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
        const Iterator expect = ::std::max_element(begin, end);
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
        const Iterator expect = ::std::max_element(begin, end);
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
        const ::std::pair<Iterator, Iterator> expect = ::std::minmax_element(begin, end);
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
        const ::std::pair<Iterator, Iterator> expect = ::std::minmax_element(begin, end);
        const std::pair<Iterator, Iterator> got_pred = std::minmax_element(std::forward<Policy>(exec), begin, end, std::less<T>());
        EXPECT_EQ(expect, got_pred, "wrong return result from minmax_element with predicate");
    }
};

template <typename T>
struct sequence_wrapper
{
    TestUtils::Sequence<T> seq;
    const T min_value;
    const T max_value;
    static const ::std::size_t bits = 30; // We assume that T can handle signed 2^bits+1 value

    // TestUtils::HashBits returns value between 0 and (1<<bits)-1,
    // therefore we could threat 1<<bits as maximum and -(1<<bits) as a minimum
    sequence_wrapper(::std::size_t n) : seq(n), min_value(-(1 << bits)), max_value(1 << bits) {}

    void
    pattern_fill()
    {
        seq.fill([](::std::size_t i) -> T { return T(TestUtils::HashBits(i, bits)); });
    }

    // sets first one at position `at` and bunch of them farther
    void
    set_desired_value(::std::size_t at, T value)
    {
        if (seq.size() == 0)
            return;
        seq[at] = value;

        //Producing several red herrings
        for (::std::size_t i = at + 1; i < seq.size(); i += 1 + TestUtils::HashBits(i, 5))
            seq[i] = value;
    }
};

template <typename T>
void
test_by_type(::std::size_t n)
{
    sequence_wrapper<T> wseq(n);

    // to avoid overtesing we use ::std::set to leave only unique indexes
    ::std::set<::std::size_t> targets{0};
    if (n > 1)
    {
        targets.insert(1);
        targets.insert(2.718282 * n / 3);
        targets.insert(n / 2);
        targets.insert(n / 7.389056);
        targets.insert(n - 1); // last
    }

    for (::std::set<::std::size_t>::iterator it = targets.begin(); it != targets.end(); ++it)
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
            for (::std::set<::std::size_t>::reverse_iterator rit = targets.rbegin(); rit != targets.rend(); ++rit)
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

// The value type is default-constructible, but only through an explicit default constructor:
// the vector code path is still applicable for it, because the reduction object initializes its
// members with direct-list-initialization.
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

// The value type is not default-constructible, so it cannot be used in a user-defined reduction
// and the vector code path must not be selected for it.
struct NoDefaultCtorCompare
{
    std::int32_t val;
    explicit NoDefaultCtorCompare(std::int32_t val_) : val(val_) {}
    bool
    operator<(const NoDefaultCtorCompare& other) const
    {
        return val < other.val;
    }
};

// The value type is not copy-assignable, so it cannot be used in a user-defined reduction
// and the vector code path must not be selected for it.
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

// The value type is not copy-constructible, so it cannot be copied into a user-defined reduction object
// and the vector code path must not be selected for it.
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

// The value type is default-constructible, copy-constructible and copy-assignable, but it is not move-assignable and
// therefore neither movable nor swappable: it satisfies the C++17 spelling of the requirements of the vector code
// path, but not std::semiregular, so the vector code path is selected for it in C++17 and is not selected in C++20.
struct NoMoveAssignCompare
{
    std::int32_t val;
    NoMoveAssignCompare() : val(0) {}
    NoMoveAssignCompare(std::int32_t val_) : val(val_) {}
    NoMoveAssignCompare(const NoMoveAssignCompare&) = default;
    NoMoveAssignCompare&
    operator=(const NoMoveAssignCompare&) = default;
    NoMoveAssignCompare&
    operator=(NoMoveAssignCompare&&) = delete;
    bool
    operator<(const NoMoveAssignCompare& other) const
    {
        return val < other.val;
    }
};

// The comparator requires mutable references to the elements: it can be applied to the elements of a mutable
// sequence, which is what the serial code path does, but not to the const copies of them which the vector code path
// keeps in its reduction object.
struct MutableRefLess
{
    bool
    operator()(std::int32_t& x, std::int32_t& y) const
    {
        return x < y;
    }
};

// A reference type an object of the value type cannot be created from.
struct NotConvertibleToValueType
{
};

// The requirements of the vector code path are checked directly, because a run-time test cannot tell the vector code
// path from the serial one: an unnecessarily strict requirement silently disables vectorization, while a requirement
// that is too weak breaks the build of the vector code path rather than a run-time check.
namespace dpl_internal = oneapi::dpl::__internal;

static_assert(dpl_internal::__is_value_storable_v<std::int32_t, std::int32_t&>);
static_assert(dpl_internal::__is_value_storable_v<OnlyLessCompare, OnlyLessCompare&>);
static_assert(dpl_internal::__is_value_storable_v<OnlyLessCompare, const OnlyLessCompare&>);
static_assert(dpl_internal::__is_value_storable_v<ExplicitDefaultCtorCompare, ExplicitDefaultCtorCompare&>);
static_assert(!dpl_internal::__is_value_storable_v<OnlyLessCompare, NotConvertibleToValueType>);
static_assert(!dpl_internal::__is_value_storable_v<NoDefaultCtorCompare, NoDefaultCtorCompare&>);
static_assert(!dpl_internal::__is_value_storable_v<NoCopyAssignCompare, NoCopyAssignCompare&>);
static_assert(!dpl_internal::__is_value_storable_v<MoveOnlyCompare, MoveOnlyCompare&>);
#if _ONEDPL_CPP20_CONCEPTS_PRESENT
static_assert(!dpl_internal::__is_value_storable_v<NoMoveAssignCompare, NoMoveAssignCompare&>);
#else
static_assert(dpl_internal::__is_value_storable_v<NoMoveAssignCompare, NoMoveAssignCompare&>);
#endif // _ONEDPL_CPP20_CONCEPTS_PRESENT

static_assert(dpl_internal::__is_value_type_predicate_v<std::int32_t, std::less<std::int32_t>>);
static_assert(dpl_internal::__is_value_type_predicate_v<OnlyLessCompare, std::less<OnlyLessCompare>>);
static_assert(dpl_internal::__is_value_type_predicate_v<std::int32_t, NonConstAdapter<std::less<std::int32_t>>>);
static_assert(!dpl_internal::__is_value_type_predicate_v<std::int32_t, MutableRefLess>);
// max_element is implemented on top of min_element with the comparator wrapped into __reorder_pred, so the wrapper
// must not hide the requirements of the comparator it wraps.
static_assert(!dpl_internal::__is_value_type_predicate_v<std::int32_t, dpl_internal::__reorder_pred<MutableRefLess>>);

// The sequence is built in place because the value types checked here do not satisfy the requirements of
// TestUtils::Sequence (which default-constructs and assigns its elements).
template <typename T>
static void
test_value_type_in_vector(::std::size_t n)
{
    ::std::vector<T> data;
    data.reserve(n);
    for (::std::size_t i = 0; i < n; ++i)
        data.emplace_back(std::int32_t(TestUtils::HashBits(i, 30)));

#ifdef _PSTL_TEST_MIN_ELEMENT
    invoke_on_all_host_policies()(check_minelement<T>(), data.begin(), data.end());
    invoke_on_all_host_policies()(check_minelement_predicate<T>(), data.begin(), data.end());
#endif
#ifdef _PSTL_TEST_MAX_ELEMENT
    invoke_on_all_host_policies()(check_maxelement<T>(), data.begin(), data.end());
    invoke_on_all_host_policies()(check_maxelement_predicate<T>(), data.begin(), data.end());
#endif
#ifdef _PSTL_TEST_MINMAX_ELEMENT
    invoke_on_all_host_policies()(check_minmaxelement<T>(), data.begin(), data.end());
    invoke_on_all_host_policies()(check_minmaxelement_predicate<T>(), data.begin(), data.end());
#endif
}

// The comparator is applied to the elements of a mutable sequence, which a comparator requiring mutable references to
// them is applicable to: the vector code path must not be selected in this case, but the algorithms must still work.
struct check_min_max_element_mutable_ref_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end)
    {
#ifdef _PSTL_TEST_MIN_ELEMENT
        EXPECT_EQ(::std::min_element(begin, end, MutableRefLess{}),
                  std::min_element(exec, begin, end, MutableRefLess{}),
                  "wrong return result from min_element with a mutable reference predicate");
#endif
#ifdef _PSTL_TEST_MAX_ELEMENT
        EXPECT_EQ(::std::max_element(begin, end, MutableRefLess{}),
                  std::max_element(exec, begin, end, MutableRefLess{}),
                  "wrong return result from max_element with a mutable reference predicate");
#endif
#ifdef _PSTL_TEST_MINMAX_ELEMENT
        EXPECT_EQ(::std::minmax_element(begin, end, MutableRefLess{}),
                  std::minmax_element(exec, begin, end, MutableRefLess{}),
                  "wrong return result from minmax_element with a mutable reference predicate");
#endif
    }
};

static void
test_mutable_ref_predicate(::std::size_t n)
{
    ::std::vector<std::int32_t> data(n);
    for (::std::size_t i = 0; i < n; ++i)
        data[i] = std::int32_t(TestUtils::HashBits(i, 30));

    invoke_on_all_host_policies()(check_min_max_element_mutable_ref_predicate{}, data.begin(), data.end());
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

    for (std::size_t n = 0; n < N; n = n < 16 ? n + 1 : size_t(3.14159 * n))
    {
#if !ONEDPL_FPGA_DEVICE
        test_by_type<std::int32_t>(n);
#endif
        test_by_type<float64_t>(n);
        test_by_type<OnlyLessCompare>(n);
        test_by_type<ExplicitDefaultCtorCompare>(n);
    }

    // The value types and the comparator below are checked against the requirements of the vector code path, which
    // does not depend on the size of the sequence, so a few sizes are enough for them.
    for (std::size_t n : {0, 1, 2, 15, 10000})
    {
        test_value_type_in_vector<NoDefaultCtorCompare>(n);
        test_value_type_in_vector<NoCopyAssignCompare>(n);
        test_value_type_in_vector<MoveOnlyCompare>(n);
        test_value_type_in_vector<NoMoveAssignCompare>(n);
        test_mutable_ref_predicate(n);
    }

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
