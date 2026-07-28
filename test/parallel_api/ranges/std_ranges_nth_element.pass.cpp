// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING

#include <functional>
#include <vector>

using namespace test_std_ranges;
namespace dpl_ranges = oneapi::dpl::ranges;

// Custom comparator to exercise a user-defined function object in addition to std::ranges::less/greater.
struct CustomLess
{
    template <typename T>
    bool
    operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs;
    }
};

// nth_element does not produce a fully deterministic order: only the element at 'nth' is placed
// in its final sorted position, and the range is partitioned around it. The parallel and device
// specializations rearrange the rest of the elements differently from std::ranges::nth_element,
// so we verify the algorithm post-conditions instead of comparing the whole range element-wise.
template <typename Range, typename Comp, typename Proj>
void
check_nth_element_effect(Range&& r, int n, int nth, Comp comp, Proj proj, const char* msg)
{
    if (n == 0 || nth == n) // no element is selected: nothing to check
        return;

    auto ordered = [&](auto&& a, auto&& b) { return std::invoke(comp, std::invoke(proj, a), std::invoke(proj, b)); };

    for (int i = 0; i < nth; ++i)
        EXPECT_TRUE(!ordered(r[nth], r[i]), (std::string("wrong left partition: ") + msg).c_str());
    for (int i = nth + 1; i < n; ++i)
        EXPECT_TRUE(!ordered(r[i], r[nth]), (std::string("wrong right partition: ") + msg).c_str());
}

// The set of 'nth' positions to exercise for a range of size n, including the end position (no-op).
std::vector<int>
nth_positions(int n)
{
    std::vector<int> res;
    for (int p : {0, n / 4, n / 2, 3 * n / 4, n > 0 ? n - 1 : 0, n})
        if (p >= 0 && p <= n && std::find(res.begin(), res.end(), p) == res.end())
            res.push_back(p);
    return res;
}

// Default data generator: values in [-48, 48] with duplicates to exercise partition ties.
inline auto nth_element_gen = [](int i) { return (i * 7 + 3) % 97 - 48; };

// Degenerate distributions that stress the partition/selection paths differently from random data:
//  - all elements equal (every pivot choice ties),
//  - already sorted ascending,
//  - sorted descending.
inline auto nth_element_equal_gen = [](int) { return 42; };
inline auto nth_element_sorted_gen = [](int i) { return i; };
inline auto nth_element_reverse_gen = [](int i) { return -i; };

// A single declarative test case, in the spirit of test_range_algo: one instantiation runs the
// tested algorithm against std::ranges::nth_element for every policy (serial/parallel and, when
// available, device), across a set of sizes and 'nth' positions, for a fixed comparator/projection.
template <int CallId, typename T, typename Gen = decltype(nth_element_gen)>
struct test_nth_element
{
    template <typename Algo, typename Checker, typename Comp = std::ranges::less, typename Proj = std::identity>
    void
    operator()(Algo algo, Checker checker, Comp comp = {}, Proj proj = {}) const
    {
        for (int n : {0, 1, 2, 3, 7, 20, small_size})
            host_case(algo, checker, n, comp, proj);

        // Borrowed views (std::ranges::subrange, std::span): exercise the algorithm on non-container
        // random-access sized ranges, not only on std::vector.
        for (int n : {0, 1, 2, 3, 7, 20, small_size})
            host_view_case(algo, checker, n, comp, proj);

        // A non-borrowed rvalue range must yield std::ranges::dangling as the return type. The niebloid's
        // return type does not depend on the policy, so a single check is sufficient; reuse the harness trait.
        using rvalue_ret_t = decltype(algo(oneapi::dpl::execution::par, std::declval<std::vector<T>>(),
                                           std::declval<std::vector<T>>().begin(), std::declval<Comp>(),
                                           std::declval<Proj>()));
        static_assert(all_dangling_in_result_v<rvalue_ret_t>);
#if TEST_DPCPP_BACKEND_PRESENT


        // Pointer-to-member-function comparators/projections are not supported inside device kernels.
        if constexpr (!std::disjunction_v<std::is_member_function_pointer<Comp>,
                                          std::is_member_function_pointer<Proj>>)
        {
#if _PSTL_LAMBDA_PTR_TO_MEMBER_WINDOWS_BROKEN
            if constexpr (!std::disjunction_v<std::is_member_pointer<Comp>, std::is_member_pointer<Proj>>)
#endif
            {
                auto policy = TestUtils::get_dpcpp_test_policy();

                // Skip device tests for double if the device does not support double precision floating point.
                if (!std::is_same_v<T, double> || policy.queue().get_device().has(sycl::aspect::fp64))
                {
                    for (int n : {0, 1, small_size, medium_size})
                        device_case(policy, algo, checker, n, comp, proj);
                }
            }
        }
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

  private:
    static std::vector<T>
    make_data(int n)
    {
        Gen gen{};
        std::vector<T> data(n);
        for (int i = 0; i < n; ++i)
            data[i] = T(gen(i));
        return data;
    }

    template <typename Algo, typename Checker, typename Comp, typename Proj>
    void
    host_case(Algo algo, Checker checker, int n, Comp comp, Proj proj) const
    {
        const std::string msg = "host, nth_element<" + std::to_string(CallId) + ">";
        for (int nth : nth_positions(n))
        {
            std::vector<T> reference = make_data(n);
            checker(reference, reference.begin() + nth, comp, proj);

            auto run = [&](auto&& policy)
            {
                std::vector<T> data = make_data(n);
                if constexpr (!std::is_same_v<Proj, std::identity>)
                {
                    auto res = algo(policy, data, data.begin() + nth, comp, proj);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value: ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else if constexpr (!std::is_same_v<Comp, std::ranges::less>)
                {
                    auto res = algo(policy, data, data.begin() + nth, comp);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value: ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else
                {
                    auto res = algo(policy, data, data.begin() + nth);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value: ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
            };

            run(oneapi::dpl::execution::seq);
            run(oneapi::dpl::execution::unseq);
            run(oneapi::dpl::execution::par);
            run(oneapi::dpl::execution::par_unseq);
        }
    }

    template <typename Algo, typename Checker, typename Comp, typename Proj>
    void
    host_view_case(Algo algo, Checker checker, int n, Comp comp, Proj proj) const
    {
        const std::string msg = "host view, nth_element<" + std::to_string(CallId) + ">";
        for (int nth : nth_positions(n))
        {
            std::vector<T> reference = make_data(n);
            checker(reference, reference.begin() + nth, comp, proj);

            // std::ranges::subrange over an lvalue vector: a borrowed, random-access, sized view.
            auto run_subrange = [&](auto&& policy)
            {
                std::vector<T> data = make_data(n);
                auto view = std::ranges::subrange(data.begin(), data.end());
                if constexpr (!std::is_same_v<Proj, std::identity>)
                {
                    auto res = algo(policy, view, view.begin() + nth, comp, proj);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value (subrange): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else if constexpr (!std::is_same_v<Comp, std::ranges::less>)
                {
                    auto res = algo(policy, view, view.begin() + nth, comp);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value (subrange): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else
                {
                    auto res = algo(policy, view, view.begin() + nth);
                    EXPECT_TRUE(res == data.end(), (std::string("wrong return value (subrange): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
            };
            run_subrange(oneapi::dpl::execution::seq);
            run_subrange(oneapi::dpl::execution::par);

#if TEST_CPP20_SPAN_PRESENT
            // std::span: a borrowed, non-owning view without an owning container interface.
            auto run_span = [&](auto&& policy)
            {
                std::vector<T> data = make_data(n);
                std::span<T> view(data.data(), data.size());
                if constexpr (!std::is_same_v<Proj, std::identity>)
                {
                    auto res = algo(policy, view, view.begin() + nth, comp, proj);
                    EXPECT_TRUE(res == view.end(), (std::string("wrong return value (span): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else if constexpr (!std::is_same_v<Comp, std::ranges::less>)
                {
                    auto res = algo(policy, view, view.begin() + nth, comp);
                    EXPECT_TRUE(res == view.end(), (std::string("wrong return value (span): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
                else
                {
                    auto res = algo(policy, view, view.begin() + nth);
                    EXPECT_TRUE(res == view.end(), (std::string("wrong return value (span): ") + msg).c_str());
                    check_nth_element_effect(data, n, nth, comp, proj, msg.c_str());
                }
            };
            run_span(oneapi::dpl::execution::seq);
            run_span(oneapi::dpl::execution::par);
#endif // TEST_CPP20_SPAN_PRESENT
        }
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename ExecutionPolicy, typename Algo, typename Checker, typename Comp, typename Proj>
    void
    device_case(ExecutionPolicy&& policy, Algo algo, Checker checker, int n, Comp comp, Proj proj) const
    {
        const std::string msg = "device, nth_element<" + std::to_string(CallId) + ">";

        for (int nth : nth_positions(n))
        {
            std::vector<T> reference = make_data(n);
            checker(reference, reference.begin() + nth, comp, proj);

            std::vector<T> host = make_data(n);
            usm_vector<T> usm(policy, host.data(), n);
            auto& vec = usm();

            auto res = algo(CLONE_TEST_POLICY_IDX(policy, CallId), vec, vec.begin() + nth, comp, proj);
            EXPECT_TRUE(res == vec.end(), (std::string("wrong return value: ") + msg).c_str());
            check_nth_element_effect(vec, n, nth, comp, proj, msg.c_str());
        }
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
};

class A_PM
{
  protected:
    int a = {};

  public:
    A_PM() = default;
    A_PM(int v) : a(v) {}

    int
    get() const
    {
        return a;
    }
};
auto proj_apm = [](const A_PM& a) { return a.get(); };

#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    auto nth_element_checker = TEST_PREPARE_CALLABLE(std::ranges::nth_element);

    // comp = less/greater/CustomLess, proj = identity: plain integer keys.
    test_nth_element<0, int>{}(dpl_ranges::nth_element, nth_element_checker);
    test_nth_element<1, int>{}(dpl_ranges::nth_element, nth_element_checker, std::ranges::greater{});
    test_nth_element<2, int>{}(dpl_ranges::nth_element, nth_element_checker, CustomLess{});

    // Projection applied to integer keys.
    test_nth_element<3, int>{}(dpl_ranges::nth_element, nth_element_checker, std::ranges::less{}, proj);

    // Member-data projection (P2::x): exercised on host and device.
    test_nth_element<4, P2>{}(dpl_ranges::nth_element, nth_element_checker, std::ranges::less{}, &P2::x);
    test_nth_element<5, P2>{}(dpl_ranges::nth_element, nth_element_checker, CustomLess{}, &P2::x);

    // Member-function projection (P2::proj): host only (skipped inside device kernels).
    test_nth_element<6, P2>{}(dpl_ranges::nth_element, nth_element_checker, std::ranges::greater{}, &P2::proj);

    // External projection proj_apm: exercised on host and device.
    test_nth_element<7, A_PM>{}(dpl_ranges::nth_element, nth_element_checker, std::ranges::greater{}, proj_apm);

    // Degenerate data distributions: all-equal, ascending, descending.
    test_nth_element<8, int, decltype(nth_element_equal_gen)>{}(dpl_ranges::nth_element, nth_element_checker);
    test_nth_element<9, int, decltype(nth_element_sorted_gen)>{}(dpl_ranges::nth_element, nth_element_checker);
    test_nth_element<10, int, decltype(nth_element_reverse_gen)>{}(dpl_ranges::nth_element, nth_element_checker);

    // Floating-point keys.
    test_nth_element<11, double>{}(dpl_ranges::nth_element, nth_element_checker);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
