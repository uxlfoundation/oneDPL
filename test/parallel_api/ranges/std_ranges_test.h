// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _STD_RANGES_TEST_H
#define _STD_RANGES_TEST_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"
#include "support/utils_invoke.h"       // for CLONE_TEST_POLICY macro

#if _ENABLE_STD_RANGES_TESTING

static_assert(ONEDPL_HAS_RANGE_ALGORITHMS >= 202505L);

#if TEST_CPP20_SPAN_PRESENT
#include <span>
#endif
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <string>
#include <ranges>
#include <algorithm>
#include <memory>
#include <array>

namespace test_std_ranges
{

// To suppress dangling iterator checks for some tests
template <typename>
constexpr bool supress_dangling_iterators_check = false;

// The largest specializations of algorithms with device policies handle 16M+ elements.
inline constexpr int big_size = (1<<24) + 10; //16M

// ~100K is sufficient for parallel policies.
// It also usually results in using multiple-work-group specializations for device policies.
inline constexpr int medium_size = (1<<17) + 10; //128K

// It is a sufficient size for sequential policies.
// It also usually results in using single-work-group specializations for device policies.
inline constexpr int small_size = 2025;

#if TEST_DPCPP_BACKEND_PRESENT
inline constexpr std::array<int, 3> big_sz = {/*serial*/ small_size, /*par*/ medium_size, /*device*/ big_size};
#else
inline constexpr std::array<int, 2> big_sz = {/*serial*/ small_size, /*par*/ medium_size};
#endif

enum TestDataMode
{
    data_in,
    data_in_out,
    data_in_out_lim,
    data_in_in,
    data_in_in_out,
    data_in_in_out_lim
};

auto f_mutuable = [](auto&& val) { return val *= val; };
auto proj_mutuable = [](auto&& val) { return val *= 2; };

auto f = [](auto&& val) { return val * val; };
auto binary_f = [](auto&& val1, auto&& val2) { return val1 * val2; };
auto proj = [](auto&& val){ return val * 2; };
auto pred = [](auto&& val) { return val == 5; };

auto binary_pred = [](auto&& val1, auto&& val2) { return val1 == val2; };
auto binary_pred_const = [](const auto& val1, const auto& val2) { return val1 == val2; };

auto pred1 = [](auto&& val) -> bool { return val > 0; };
auto pred2 = [](auto&& val) -> bool { return val == 4; };
auto pred3 = [](auto&& val) -> bool { return val < 0; };

struct P2
{
    P2() {}
    P2(int v): x(v) {}
    int x = {};
    int y = {};

    int proj() const { return x; }
    friend bool operator==(const P2& a, const P2& b) { return a.x == b.x && a.y == b.y; }
};


// These are copies of __range_size and __range_size_t utilities from oneDPL
// to get a size type of a range be it sized or not
template <typename R>
struct range_size {
    using type = std::uint8_t;
};

template <std::ranges::sized_range R>
struct range_size<R> {
    using type = std::ranges::range_size_t<R>;
};

template <typename R>
using range_size_t = typename range_size<R>::type;

template<typename, typename = void>
static constexpr bool is_iterator{};

template<typename T>
static constexpr
bool is_iterator<T, std::void_t<decltype(++std::declval<T&>()), decltype(*std::declval<T&>())>> = true;

template<typename, typename = void>
static constexpr bool check_in{};

template<typename T>
static constexpr
bool check_in<T, std::void_t<decltype(std::declval<T>().in)>> = true;

template<typename, typename = void>
static constexpr bool check_in1{};

template<typename T>
static constexpr
bool check_in1<T, std::void_t<decltype(std::declval<T>().in1)>> = true;

template<typename, typename = void>
static constexpr bool check_in2{};

template<typename T>
static constexpr
bool check_in2<T, std::void_t<decltype(std::declval<T>().in2)>> = true;

template<typename, typename = void>
static constexpr bool check_out{};

template<typename T>
static constexpr
bool check_out<T, std::void_t<decltype(std::declval<T>().out)>> = true;

template<typename, typename = void>
static constexpr bool is_range{};

template<typename T>
static constexpr
bool is_range<T, std::void_t<decltype(std::declval<T&>().begin())>> = true;

template<typename, typename = void>
static constexpr bool check_minmax{};

template<typename T>
static constexpr
bool check_minmax<T, std::void_t<decltype(std::declval<T>().min, std::declval<T>().max)>> = true;

template<typename>
constexpr int trivial_size{0};

template<typename>
constexpr int calc_res_size(int n, int) { return n; }

auto data_gen2_default = [](auto i) { return i % 5 ? i : 0;};
auto data_gen_zero = [](auto) { return 0;};

// Extension: host execution policy type trait
template <class _T>
struct __is_host_execution_policy : std::false_type
{
};

template <>
struct __is_host_execution_policy<oneapi::dpl::execution::sequenced_policy> : std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_policy> : std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_unsequenced_policy> : std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::unsequenced_policy> : std::true_type
{
};

template <typename _ReturnType>
struct __all_dangling_in_result : std::false_type
{
};

template <>
struct __all_dangling_in_result<std::ranges::dangling> : std::true_type
{
};

template <>
struct __all_dangling_in_result<std::ranges::in_in_result<std::ranges::dangling, std::ranges::dangling>> : std::true_type
{
};

template <>
struct __all_dangling_in_result<std::ranges::in_out_result<std::ranges::dangling, std::ranges::dangling>> : std::true_type
{
};

template <>
struct __all_dangling_in_result<std::ranges::in_in_out_result<std::ranges::dangling, std::ranges::dangling, std::ranges::dangling>> : std::true_type
{
};

template <>
struct __all_dangling_in_result<std::ranges::in_out_out_result<std::ranges::dangling, std::ranges::dangling, std::ranges::dangling>> : std::true_type
{
};

template <typename _ReturnType>
constexpr bool __all_dangling_in_result_v = __all_dangling_in_result<_ReturnType>::value;

template<typename DataType, typename Container, TestDataMode test_mode = data_in, typename DataGen1 = std::identity,
         typename DataGen2 = decltype(data_gen2_default)>
struct test
{
    void
    host_policies(int n_serial, int n_parallel, auto algo, auto& checker, auto... args)
    {
        operator()(n_serial,   oneapi::dpl::execution::seq,       algo, checker, args...);
        operator()(n_serial,   oneapi::dpl::execution::unseq,     algo, checker, args...);
        operator()(n_parallel, oneapi::dpl::execution::par,       algo, checker, args...);
        operator()(n_parallel, oneapi::dpl::execution::par_unseq, algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, TransIn tr_in, TransOut, auto... args)
    {
        process_data_in(max_n, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);

        //test with empty sequence
        process_data_in(trivial_size<std::remove_cvref_t<Algo>>, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);
    }

private:

  template <typename Policy, typename T>
  using TmpContainerType =
#if TEST_DPCPP_BACKEND_PRESENT
      std::conditional_t<__is_host_execution_policy<Policy>::value, std::vector<T>,
                         std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>>;
#else
      std::vector<T>;
#endif

    // Test dangling iterators in return types for call with temporary data
    template <int idx, typename Policy, typename Algo, typename ...Args>
    constexpr void
    test_dangling_pointers_arg_1(Policy&& exec, Algo&& algo, Args&& ...args)
    {
        // Check dangling iterators in return types for call with temporary data
        if constexpr (!supress_dangling_iterators_check<std::remove_cvref_t<decltype(algo)>>)
        {
            if constexpr (!std::is_same_v<decltype(res), bool>)
            {
                using T = typename Container::__value_type;

                // Check dangling with temporary containers in implementation
                using res_ret_t = decltype(algo(CLONE_TEST_POLICY_IDX(exec, idx),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                args...));

                if constexpr (!std::is_fundamental_v<res_ret_t>)
                {
                    if constexpr (!__all_dangling_in_result_v<res_ret_t>)
                        res_ret_t::dummy;
                }
            }
        }        
    }

    // Test dangling iterators in return types for call with temporary data
    template <int idx, typename Policy, typename Algo, typename ...Args>
    constexpr void
    test_dangling_pointers_args_2(Policy&& exec, Algo&& algo, Args&& ...args)
    {
        // Check dangling iterators in return types for call with temporary data
        if constexpr (!supress_dangling_iterators_check<std::remove_cvref_t<decltype(algo)>>)
        {
            if constexpr (!std::is_same_v<decltype(res), bool>)
            {
                using T = typename Container::__value_type;

                // Check dangling with temporary containers in implementation
                using res_ret_t = decltype(algo(CLONE_TEST_POLICY_IDX(exec, idx),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                args...));

                if constexpr (!std::is_fundamental_v<res_ret_t>)
                {
                    if constexpr (!__all_dangling_in_result_v<res_ret_t>)
                        res_ret_t::dummy;
                }
            }
        }        
    }

    // Test dangling iterators in return types for call with temporary data
    template <int idx, typename Policy, typename Algo, typename ...Args>
    constexpr void
    test_dangling_pointers_args_3(Policy&& exec, Algo&& algo, Args&& ...args)
    {
        // Check dangling iterators in return types for call with temporary data
        if constexpr (!supress_dangling_iterators_check<std::remove_cvref_t<decltype(algo)>>)
        {
            if constexpr (!std::is_same_v<decltype(res), bool>)
            {
                using T = typename Container::__value_type;

                // Check dangling with temporary containers in implementation
                using res_ret_t = decltype(algo(CLONE_TEST_POLICY_IDX(exec, idx),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                std::declval<TmpContainerType<decltype(exec), T>>(),
                                                args...));

                if constexpr (!std::is_fundamental_v<res_ret_t>)
                {
                    if constexpr (!__all_dangling_in_result_v<res_ret_t>)
                        res_ret_t::dummy;
                }
            }
        }        
    }

    void
    process_data_in(int max_n, auto&& exec, auto algo, auto& checker, auto tr_in, auto... args)
    {
        Container cont_in(exec, max_n, DataGen1{});
        Container cont_exp(exec, max_n, DataGen1{});

        auto expected_view = tr_in(std::views::all(cont_exp()));
        auto expected_res = checker(expected_view, args...);

        typename Container::type& A = cont_in();
        decltype(auto) r_in = tr_in(A);
        auto res = algo(CLONE_TEST_POLICY(exec), r_in, args...);

        // check result types
        static_assert(std::is_same_v<decltype(res), decltype(expected_res)>, "Wrong return type");

        using Algo = decltype(algo);
        EXPECT_EQ(ret_in_val(expected_res, expected_view.begin()), ret_in_val(res, r_in.begin()),
                  (std::string("wrong return value from algo with ranges: ") + typeid(Algo).name() +
                   typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());

        //check result
        auto n = std::ranges::size(expected_view);
        if constexpr(is_range<std::remove_cvref_t<decltype(res)>>)
            n = calc_res_size<std::remove_cvref_t<decltype(algo)>>(n, std::ranges::size(res));

        EXPECT_EQ_N(cont_exp().begin(), cont_in().begin(), n, (std::string("wrong effect algo with ranges: ")
            + typeid(Algo).name() + typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());

        // Test dangling iterators in return types for call with temporary data
        test_dangling_pointers_arg_1<100>(exec, algo, std::forward<decltype(args)>(args)...);
    }

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    void
    process_data_in_out(int max_n, int n_in, int n_out, Policy&& exec, Algo algo, Checker& checker, TransIn tr_in,
                        TransOut tr_out, auto... args)
    {
        static_assert(mode == data_in_out || mode == data_in_out_lim);

        Container cont_in(exec, n_in, DataGen1{});
        Container cont_out(exec, n_out, data_gen_zero);
        Container cont_exp(exec, n_out, data_gen_zero);

        assert(n_in <= max_n);
        assert(n_out <= max_n);

        auto src_view = tr_in(std::views::all(cont_in()));
        auto exp_view = tr_out(std::views::all(cont_exp()));
        auto expected_res = checker(src_view, exp_view, args...);

        typename Container::type& A = cont_in();
        typename Container::type& B = cont_out();

        auto res = algo(CLONE_TEST_POLICY(exec), tr_in(A), tr_out(B), args...);

        // check result types
        static_assert(std::is_same_v<decltype(res), decltype(expected_res)>, "Wrong return type");

        EXPECT_EQ(ret_in_val(expected_res, src_view.begin()), ret_in_val(res, tr_in(A).begin()),
                  (std::string("wrong return value from algo with input range: ") + typeid(Algo).name()).c_str());

        EXPECT_EQ(ret_out_val(expected_res, exp_view.begin()), ret_out_val(res, tr_out(B).begin()),
                  (std::string("wrong return value from algo with output range: ") + typeid(Algo).name()).c_str());

        //check result
        auto n = std::ranges::size(exp_view);
        EXPECT_EQ_N(cont_exp().begin(), cont_out().begin(), n, (std::string("wrong effect algo with ranges: ") + typeid(Algo).name()).c_str());

        // Test dangling iterators in return types for call with temporary data
        test_dangling_pointers_args_2<200>(exec, algo, std::forward<decltype(args)>(args)...);
    }

public:
    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in_out>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = max_n;
        process_data_in_out(max_n, r_size, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);

        //test cases with empty sequence(s)
	    process_data_in_out(max_n, 0, 0, CLONE_TEST_POLICY(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in_out_lim>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = max_n;
        process_data_in_out(max_n, r_size, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);

        //test case size of input range is less than size of output and vice-versa
        process_data_in_out(max_n, r_size/2, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);
        process_data_in_out(max_n, r_size, r_size/2, CLONE_TEST_POLICY(exec), algo, checker, args...);

        //test cases with empty sequence(s)
        process_data_in_out(max_n, 0, 0, CLONE_TEST_POLICY(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in_in>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, TransIn tr_in, TransOut, auto... args)
    {
        const int r_size = max_n;
        process_data_in_in(max_n, r_size, r_size, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);

        //test case the sizes of input ranges are different
        process_data_in_in(max_n, r_size/2, r_size, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);
        process_data_in_in(max_n, r_size, r_size/2, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);

        //test cases with empty sequence(s)
        process_data_in_in(max_n, 0, 0, CLONE_TEST_POLICY(exec), algo, checker, tr_in, args...);
    }

private:
    void
    process_data_in_in(int max_n, int n_in1, int n_in2, auto&& exec, auto algo, auto& checker, auto tr_in, auto... args)
    {
        assert(n_in1 <= max_n);
        assert(n_in2 <= max_n);

        Container cont_in1(exec, n_in1, DataGen1{});
        Container cont_in2(exec, n_in2, DataGen2{});

        auto src_view1 = tr_in(std::views::all(cont_in1()));
        auto src_view2 = tr_in(std::views::all(cont_in2()));
        auto expected_res = checker(src_view1, src_view2, args...);

        typename Container::type& A = cont_in1();
        typename Container::type& B = cont_in2();

        auto res = algo(CLONE_TEST_POLICY(exec), tr_in(A), tr_in(B), args...);

        // check result types
        static_assert(std::is_same_v<decltype(res), decltype(expected_res)>, "Wrong return type");

        if constexpr (!std::is_same_v<decltype(res), bool>)
        {
            EXPECT_EQ(ret_in_val(expected_res, src_view1.begin()), ret_in_val(res, tr_in(A).begin()),
                      (std::string("wrong return value from algo: ") + typeid(decltype(algo)).name() +
                       typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
        }
        else
        {
            EXPECT_EQ(expected_res, res,
                      (std::string("wrong return value from algo: ") + typeid(decltype(algo)).name() +
                       typeid(decltype(tr_in(std::declval<Container&>()()))).name()).c_str());
        }

        // Test dangling iterators in return types for call with temporary data
        test_dangling_pointers_args_2<300>(exec, algo, std::forward<decltype(args)>(args)...);
    }

    struct TransformOp
    {
        template <typename T>
        auto operator()(T i) const
        {
            return i / 3;
        }
    };

    template<typename Policy, typename Algo, typename Checker, typename TransIn, typename TransOut, TestDataMode mode = test_mode>
    void
    process_data_in_in_out(int max_n, int n_in1, int n_in2, int n_out, Policy&& exec, Algo algo, Checker& checker,
                           TransIn tr_in, TransOut tr_out, auto... args)
    {
        static_assert(mode == data_in_in_out || mode == data_in_in_out_lim);

        Container cont_in1(exec, n_in1, DataGen1{});
        Container cont_in2(exec, n_in2, TransformOp{});

        Container cont_out(exec, n_out, data_gen_zero);
        Container cont_exp(exec, n_out, data_gen_zero);

        assert(n_in1 <= max_n);
        assert(n_in2 <= max_n);

        auto src_view1 = tr_in(std::views::all(cont_in1()));
        auto src_view2 = tr_in(std::views::all(cont_in2()));
        auto expected_view = tr_out(std::views::all(cont_exp()));
        auto expected_res = checker(src_view1, src_view2, expected_view, args...);

        typename Container::type& A = cont_in1();
        typename Container::type& B = cont_in2();
        typename Container::type& C = cont_out();

        auto res = algo(CLONE_TEST_POLICY(exec), tr_in(A), tr_in(B), tr_out(C), args...);

        // check result types
        static_assert(std::is_same_v<decltype(res), decltype(expected_res)>, "Wrong return type");

        EXPECT_EQ(ret_in_val(expected_res, src_view1.begin()), ret_in_val(res, tr_in(A).begin()),
                  (std::string("wrong return value from algo with input range 1: ") + typeid(Algo).name()).c_str());

        EXPECT_EQ(ret_in_val(expected_res, src_view2.begin()), ret_in_val(res, tr_in(B).begin()),
                  (std::string("wrong return value from algo with input range 2: ") + typeid(Algo).name()).c_str());

        EXPECT_EQ(ret_out_val(expected_res, expected_view.begin()), ret_out_val(res, tr_out(C).begin()),
                  (std::string("wrong return value from algo with output range: ") + typeid(Algo).name()).c_str());

        //check result
        auto n = std::ranges::size(expected_view);
        EXPECT_EQ_N(cont_exp().begin(), cont_out().begin(), n, (std::string("wrong effect algo with ranges: ") + typeid(Policy).name()
            + typeid(Algo).name()).c_str());

        // Test dangling iterators in return types for call with temporary data
        test_dangling_pointers_args_3<400>(exec, algo, std::forward<decltype(args)>(args)...);
    }

public:
    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in_in_out>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = max_n;
        process_data_in_in_out(max_n, r_size, r_size, r_size*2, CLONE_TEST_POLICY(exec), algo, checker, args...);

        //test cases with empty sequence(s)
        process_data_in_in_out(max_n, 0, 0, 0, CLONE_TEST_POLICY(exec), algo, checker, args...);
    }

    template<typename Policy, typename Algo, typename Checker, TestDataMode mode = test_mode>
    std::enable_if_t<mode == data_in_in_out_lim>
    operator()(int max_n, Policy&& exec, Algo algo, Checker& checker, auto... args)
    {
        const int r_size = max_n;
        process_data_in_in_out(max_n, r_size, r_size, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);
        process_data_in_in_out(max_n, r_size, r_size, r_size*2, CLONE_TEST_POLICY(exec), algo, checker, args...);
        process_data_in_in_out(max_n, r_size/2, r_size, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);
        process_data_in_in_out(max_n, r_size, r_size/2, r_size, CLONE_TEST_POLICY(exec), algo, checker, args...);
        process_data_in_in_out(max_n, r_size, r_size, r_size/2, CLONE_TEST_POLICY(exec), algo, checker, args...);

	    //test cases with empty sequence(s)
        process_data_in_in_out(max_n, 0, 0, 0, CLONE_TEST_POLICY(exec), algo, checker, args...);
    }
private:

    template<typename Ret, typename Begin>
    auto ret_in_val(Ret&& ret, Begin&& begin)
    {
        if constexpr (check_in<Ret>)
            return std::distance(begin, ret.in);
        else if constexpr (check_in1<Ret>)
            return std::distance(begin, ret.in1);
        else if constexpr (check_in2<Ret>)
            return std::distance(begin, ret.in2);
        else if constexpr (is_iterator<Ret>)
            return std::distance(begin, ret);
        else if constexpr(is_range<Ret>)
            return std::pair{std::distance(begin, ret.begin()), std::ranges::distance(ret.begin(), ret.end())};
        else if constexpr(check_minmax<Ret>)
        {
            const auto& [first, second] = ret;
            if constexpr(std::random_access_iterator<std::remove_cvref_t<decltype(first)>>)
                return std::pair{std::distance(begin, first), std::ranges::distance(begin, second)};
            else
                return std::pair{first, second};
        }
        else
            return ret;
    }

    template<typename Ret, typename Begin>
    auto ret_out_val(Ret&& ret, Begin&& begin)
    {
        if constexpr (check_out<Ret>)
            return std::distance(begin, ret.out);
        else if constexpr (is_iterator<Ret>)
            return std::distance(begin, ret);
        else if constexpr(is_range<Ret>)
            return std::pair{std::distance(begin, ret.begin()), std::ranges::distance(ret.begin(), ret.end())};
        else
            return ret;
    }
};

template<typename T, typename ViewType>
struct host_subrange_impl
{
    using __value_type = T;

    static_assert(std::is_trivially_copyable_v<T>,
        "Memory initialization within the class relies on trivially copyability of the type T");

    using type = ViewType;
    ViewType view;
    T* mem = nullptr;

    std::allocator<T> alloc;

    template<typename Policy>
    host_subrange_impl(Policy&&, T* data, int n): view(data, data + n) {}

    template<typename Policy, typename DataGen>
    host_subrange_impl(Policy&&, int n, DataGen gen)
    {
        mem = alloc.allocate(n);
        view = ViewType(mem, mem + n);
        for(int i = 0; i < n; ++i)
            view[i] = gen(i);
    }
    ViewType& operator()()
    {
        return view;
    }
    ~host_subrange_impl()
    {
        if(mem)
            alloc.deallocate(mem, view.size());
    }
};

template<typename T>
using  host_subrange = host_subrange_impl<T, std::ranges::subrange<T*>>;

#if TEST_CPP20_SPAN_PRESENT
template<typename T>
using  host_span = host_subrange_impl<T, std::span<T>>;
#endif

template<typename T>
struct host_vector
{
    using __value_type = T;

    using type = std::vector<T>;
    type vec;
    T* p = nullptr;

    template<typename Policy>
    host_vector(Policy&&, T* data, int n): vec(data, data + n), p(data) {}

    template<typename Policy, typename DataGen>
    host_vector(Policy&&, int n, DataGen gen): vec(n)
    {
        for(int i = 0; i < n; ++i)
            vec[i] = gen(i);
    }
    type& operator()()
    {
        return vec;
    }
    ~host_vector()
    {
        if(p)
            std::copy_n(vec.begin(), vec.size(), p);
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template<typename T>
struct usm_vector
{
    using __value_type = T;

    using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    using type = std::vector<T, shared_allocator>;

    std::vector<T, shared_allocator> vec;
    T* p = nullptr;

    template<typename Policy>
    usm_vector(Policy&& exec, T* data, int n): vec(data, data + n, shared_allocator(exec.queue())), p(data)
    {
        assert(vec.size() == n);
    }
    template<typename Policy, typename DataGen>
    usm_vector(Policy&& exec, int n, DataGen gen): vec(n, shared_allocator(exec.queue()))
    {
        for(int i = 0; i < n; ++i)
            vec[i] = gen(i);
    }
    type& operator()()
    {
        return vec;
    }
    ~usm_vector()
    {
        if(p)
            std::copy_n(vec.begin(), vec.size(), p);
    }
};

template<typename T, typename ViewType>
struct usm_subrange_impl
{
    using __value_type = T;

    static_assert(std::is_trivially_copyable_v<T>,
        "Memory initialization within the class relies on trivially copyability of the type T");

    using shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
    using type = ViewType;

    shared_allocator alloc;
    T* p = nullptr;
    ViewType view;

    template<typename Policy>
    usm_subrange_impl(Policy&& exec, T* data, int n): alloc(exec.queue()), p(data)
    {
        auto mem = alloc.allocate(n);
        view = ViewType(mem, mem + n);
        std::copy_n(data, n, view.data());
    }
    template<typename Policy, typename DataGen>
    usm_subrange_impl(Policy&& exec, int n, DataGen gen): alloc(exec.queue())
    {
        auto mem = alloc.allocate(n);
        view = ViewType(mem, mem + n);
        for(int i = 0; i < n; ++i)
            view[i] = gen(i);
    }

    ViewType& operator()()
    {
        return view;
    }

    ~usm_subrange_impl()
    {
        if(p)
            std::copy_n(view.data(), view.size(), p);
        alloc.deallocate(view.data(), view.size());
    }
};

template<typename T>
using  usm_subrange = usm_subrange_impl<T, std::ranges::subrange<T*>>;

#if TEST_CPP20_SPAN_PRESENT
template<typename T>
using  usm_span = usm_subrange_impl<T, std::span<T>>;
#endif

#endif // TEST_DPCPP_BACKEND_PRESENT

struct subrange_view_fo
{
    template <typename T>
    auto operator()(T&& v) const
    {
        return std::ranges::subrange(v);
    }
};

#if TEST_CPP20_SPAN_PRESENT
struct span_view_fo
{
    template <typename T>
    auto operator()(T&& v) const
    {
        return std::span(v);
    }
};
#endif

template<int call_id = 0, typename T = int, TestDataMode mode = data_in, typename DataGen1 = std::identity,
         typename DataGen2 = decltype(data_gen2_default)>
struct test_range_algo
{
    const int n_serial = small_size;
    const int n_parallel = small_size;
#if TEST_DPCPP_BACKEND_PRESENT
    const int n_device = small_size;
#endif

    test_range_algo() = default;

    // Mode with a uniform number of elements for each policy type
#if TEST_DPCPP_BACKEND_PRESENT
    test_range_algo(int n) : n_serial(n), n_parallel(n), n_device(n) {}
#else
    test_range_algo(int n) : n_serial(n), n_parallel(n) {}
#endif

    // Mode that tests different policy types with different sizes.
    // Serial (seq/unseq), parallel (par/par_unseq), and device policies
    // specialize algorithms for different number of elements, which this mode covers.
#if TEST_DPCPP_BACKEND_PRESENT
    test_range_algo(std::array<int, 3> sizes) : n_serial(sizes[0]), n_parallel(sizes[1]), n_device(sizes[2]) {}
#else
    test_range_algo(std::array<int, 2> sizes) : n_serial(sizes[0]), n_parallel(sizes[1]) {}
#endif

    void test_view_host(auto view, auto algo, auto& checker, auto... args)
    {
        test<T, host_subrange<T>, mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, view, std::identity{}, args...);
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    void test_view_hetero(Policy&& exec, auto view, auto algo, auto& checker, auto... args)
    {
        test<T, usm_subrange<T>, mode, DataGen1, DataGen2>{}(n_device, CLONE_TEST_POLICY_IDX(exec, call_id), algo, checker, view, std::identity{}, args...);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT

    void
    test_range_algo_impl_host(auto algo, auto& checker, auto... args)
    {
        auto subrange_view = subrange_view_fo{};
#if TEST_CPP20_SPAN_PRESENT
        auto span_view = span_view_fo{};
#endif

        test<T, host_vector<T>,   mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, std::identity{},  std::identity{}, args...);
        test<T, host_vector<T>,   mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, subrange_view,    std::identity{}, args...);
        test<T, host_vector<T>,   mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, std::views::all,  std::identity{}, args...);
        test<T, host_subrange<T>, mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, std::views::all,  std::identity{}, args...);
#if TEST_CPP20_SPAN_PRESENT
        test<T, host_vector<T>,   mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, span_view,        std::identity{}, args...);
        test<T, host_span<T>,     mode, DataGen1, DataGen2>{}.host_policies(n_serial, n_parallel, algo, checker, std::views::all,  std::identity{}, args...);
#endif
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename Policy>
    void test_range_algo_impl_hetero(Policy&& exec, auto algo, auto& checker, auto... args)
    {
        auto subrange_view = subrange_view_fo{};
#if TEST_CPP20_SPAN_PRESENT
        auto span_view = span_view_fo{};
#endif

        //Skip the cases with pointer-to-function and hetero policy because pointer-to-function is not supported within kernel code.
        if constexpr(!std::disjunction_v<std::is_member_function_pointer<decltype(args)>...>)
        {
#if _PSTL_LAMBDA_PTR_TO_MEMBER_WINDOWS_BROKEN
            if constexpr(!std::disjunction_v<std::is_member_pointer<decltype(args)>...>)
#endif
            {
                test<T, usm_vector<T>,   mode, DataGen1, DataGen2>{}(n_device, CLONE_TEST_POLICY_IDX(exec, call_id + 10), algo, checker, subrange_view,   subrange_view,   args...);
                test<T, usm_subrange<T>, mode, DataGen1, DataGen2>{}(n_device, CLONE_TEST_POLICY_IDX(exec, call_id + 30), algo, checker, std::identity{}, std::identity{}, args...);
#if TEST_CPP20_SPAN_PRESENT
                test<T, usm_vector<T>,   mode, DataGen1, DataGen2>{}(n_device, CLONE_TEST_POLICY_IDX(exec, call_id + 20), algo, checker, span_view,       subrange_view,   args...);
                test<T, usm_span<T>,     mode, DataGen1, DataGen2>{}(n_device, CLONE_TEST_POLICY_IDX(exec, call_id + 40), algo, checker, std::identity{}, std::identity{}, args...);
#endif
            }
        }
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    void
    operator()(auto algo, auto& checker, auto... args)
    {
        test_range_algo_impl_host(algo, checker, args...);

#if TEST_DPCPP_BACKEND_PRESENT
        auto policy = TestUtils::get_dpcpp_test_policy();
        test_range_algo_impl_hetero(policy, algo, checker, args...);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
        TestUtils::check_compilation(policy, [&](auto&& policy) { test_range_algo_impl_hetero(policy, algo, checker, args...); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
};

}; //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING

#endif //_STD_RANGES_TEST_H
