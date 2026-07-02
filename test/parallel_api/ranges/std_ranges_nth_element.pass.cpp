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

// nth_element does not produce a fully deterministic order: only the element at 'nth' is placed
// in its final sorted position, and the range is partitioned around it. The parallel and device
// specializations rearrange the rest of the elements differently from std::ranges::nth_element,
// so we verify the algorithm post-conditions instead of comparing the whole range element-wise.
//   1) the projected key at 'nth' is equivalent (under comp) to the reference one produced by
//      std::ranges::nth_element for the same input;
//   2) nothing before 'nth' is ordered after it and nothing after 'nth' is ordered before it.
template <typename Range, typename Reference, typename Comp, typename Proj>
void
check_nth_element_effect(Range&& r, const Reference& reference, int n, int nth, Comp comp, Proj proj, const char* msg)
{
    if (n == 0 || nth == n) // no element is selected: nothing to check
        return;

    auto ordered = [&](auto&& a, auto&& b) { return std::invoke(comp, std::invoke(proj, a), std::invoke(proj, b)); };

    EXPECT_TRUE(!ordered(r[nth], reference[nth]) && !ordered(reference[nth], r[nth]),
                (std::string("wrong nth element value: ") + msg).c_str());

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

template <typename T, typename Checker, typename Comp, typename Proj, typename Gen>
void
test_nth_element_host(int n, Checker checker, Comp comp, Proj proj, Gen gen, const char* msg)
{
    for (int nth : nth_positions(n))
    {
        std::vector<T> reference(n);
        for (int i = 0; i < n; ++i)
            reference[i] = gen(i);
        checker(reference, reference.begin() + nth, comp, proj);

        auto run = [&](auto&& policy)
        {
            std::vector<T> data(n);
            for (int i = 0; i < n; ++i)
                data[i] = gen(i);

            auto res = dpl_ranges::nth_element(policy, data, data.begin() + nth, comp, proj);
            EXPECT_TRUE(res == data.end(), (std::string("wrong return value: ") + msg).c_str());
            check_nth_element_effect(data, reference, n, nth, comp, proj, msg);
        };

        run(oneapi::dpl::execution::seq);
        run(oneapi::dpl::execution::unseq);
        run(oneapi::dpl::execution::par);
        run(oneapi::dpl::execution::par_unseq);
    }
}

#if TEST_DPCPP_BACKEND_PRESENT
template <int CallId, typename T, typename Checker, typename Comp, typename Proj, typename Gen>
void
test_nth_element_device(int n, Checker checker, Comp comp, Proj proj, Gen gen, const char* msg)
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    for (int nth : nth_positions(n))
    {
        std::vector<T> reference(n);
        for (int i = 0; i < n; ++i)
            reference[i] = gen(i);
        checker(reference, reference.begin() + nth, comp, proj);

        std::vector<T> host(n);
        for (int i = 0; i < n; ++i)
            host[i] = gen(i);

        usm_vector<T> usm(policy, host.data(), n);
        auto& vec = usm();

        auto res =
            dpl_ranges::nth_element(CLONE_TEST_POLICY_IDX(policy, CallId), vec, vec.begin() + nth, comp, proj);
        EXPECT_TRUE(res == vec.end(), (std::string("wrong return value: ") + msg).c_str());
        check_nth_element_effect(vec, reference, n, nth, comp, proj, msg);
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    auto nth_element_checker = TEST_PREPARE_CALLABLE(std::ranges::nth_element);

    auto gen_int = [](int i) { return (i * 7 + 3) % 97 - 48; };
    auto gen_p2 = [](int i) { return P2((i * 7 + 3) % 97 - 48); };

    for (int n : {0, 1, 2, 3, 7, 20, small_size})
    {
        test_nth_element_host<int>(n, nth_element_checker, std::ranges::less{},    std::identity{}, gen_int, "host, less");
        test_nth_element_host<int>(n, nth_element_checker, std::ranges::greater{}, std::identity{}, gen_int, "host, greater");
        test_nth_element_host<int>(n, nth_element_checker, std::ranges::less{},    proj,            gen_int, "host, less, proj");
        test_nth_element_host<P2> (n, nth_element_checker, std::ranges::less{},    &P2::x,           gen_p2, "host, less, &P2::x");
        test_nth_element_host<P2> (n, nth_element_checker, std::ranges::greater{}, &P2::proj,        gen_p2, "host, greater, &P2::proj");
    }

#if TEST_DPCPP_BACKEND_PRESENT
    for (int n : {0, 1, small_size, medium_size})
    {
        test_nth_element_device<0, int>(n, nth_element_checker, std::ranges::less{},    std::identity{}, gen_int, "device, less");
        test_nth_element_device<1, int>(n, nth_element_checker, std::ranges::greater{}, std::identity{}, gen_int, "device, greater");
        test_nth_element_device<2, int>(n, nth_element_checker, std::ranges::less{},    proj,            gen_int, "device, less, proj");
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
