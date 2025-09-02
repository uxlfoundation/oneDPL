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

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
namespace test_std_ranges
{
template<>
inline int out_size_with_empty_in2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int in1_size)
{
    return in1_size;
}
template<>
inline int out_size_with_empty_in1<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>>(int in2_size)
{
    return in2_size;
}
}

struct A
{
    int a;
    operator int() const { return a; }
};

struct B
{
    int b;
    operator int() const { return b; }
};

void test_mixed_types_host()
{
    std::vector<A> r1 = {{1}, {2}, {3}};
    std::vector<B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {0, 1, 2, 2, 3};

    std::vector<int> out_seq(5, 0xCD);
    std::vector<int> out_par(5, 0xCD);
    std::vector<int> out_unseq(5, 0xCD);
    std::vector<int> out_par_unseq(5, 0xCD);

    auto proj_a = [](const A& a) { return a.a; };
    auto proj_b = [](const B& b) { return b.b; };

    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::seq, r1, r2, out_seq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::par, r1, r2, out_par, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::unseq, r1, r2, out_unseq, std::ranges::less{}, proj_a, proj_b);
    oneapi::dpl::ranges::set_union(
        oneapi::dpl::execution::par_unseq, r1, r2, out_par_unseq, std::ranges::less{}, proj_a, proj_b);

    EXPECT_EQ_RANGES(out_expected, out_seq, "wrong result with seq policy");
    EXPECT_EQ_RANGES(out_expected, out_par, "wrong result with par policy");
    EXPECT_EQ_RANGES(out_expected, out_unseq, "wrong result with unseq policy");
    EXPECT_EQ_RANGES(out_expected, out_par_unseq, "wrong result with par_unseq policy");
}

#if TEST_DPCPP_BACKEND_PRESENT
void test_mixed_types_device()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        using r1_alloc_t = sycl::usm_allocator<A, sycl::usm::alloc::shared>;
        using r2_alloc_t = sycl::usm_allocator<B, sycl::usm::alloc::shared>;
        using r_out_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

        std::vector<A, r1_alloc_t> v1({{1}, {2}, {3}}, r1_alloc_t(q));
        std::vector<B, r2_alloc_t> v2({{0}, {2}, {2}, {3}}, r2_alloc_t(q));
        std::vector<int, r_out_alloc_t> out(5, 0xCD, r_out_alloc_t(q));
        std::vector<int> out_expected = {0, 1, 2, 2, 3};

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + 3);
        std::ranges::subrange r2(v2.data(), v2.data() + 4);
        std::ranges::subrange r_out(out.data(), out.data() + 5);

        auto proj_a = [](const A& a) { return a.a; };
        auto proj_b = [](const B& b) { return b.b; };

        oneapi::dpl::ranges::set_union(policy, r1, r2, r_out, std::ranges::less{}, proj_a, proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

// Equivalent to std::ranges::set_union, but accepting the output as a range
// and properly handling the case when the output range does not have enough space to store all the elements
struct set_union_checker_fn
{
    template<std::ranges::random_access_range R1,
             std::ranges::random_access_range R2,
             std::ranges::random_access_range OutR,
             typename Comp = std::ranges::less, typename Proj1 = std::identity, typename Proj2 = std::identity>
    std::ranges::set_union_result<std::ranges::borrowed_iterator_t<R1>,
                                  std::ranges::borrowed_iterator_t<R2>,
                                  std::ranges::borrowed_iterator_t<OutR>>
    operator()(R1&& r1, R2&& r2, OutR&& r_out, Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {}) const
    {
        auto it1 = std::ranges::begin(r1);
        auto end1 = std::ranges::end(r1);
        auto it2 = std::ranges::begin(r2);
        auto end2 = std::ranges::end(r2);
        auto out_it = std::ranges::begin(r_out);
        auto out_end = std::ranges::end(r_out);

        // Do the main set_union operation until either range is exhausted
        while (it1 != end1 && it2 != end2 && out_it != out_end)
        {
            if (std::invoke(comp, std::invoke(proj1, *it1), std::invoke(proj2, *it2)))
            {
                *out_it = *it1;
                ++it1;
            }
            else if (std::invoke(comp, std::invoke(proj2, *it2), std::invoke(proj1, *it1)))
            {
                *out_it = *it2;
                ++it2;
            }
            else
            {
                *out_it = *it1;
                ++it1;
                ++it2;
            }
            ++out_it;
        }
        // Copy the residual elements if one of the input ranges is exhausted
        using size1_t = std::common_type_t<std::ranges::range_size_t<R1>, std::ranges::range_size_t<OutR>>;
        const size1_t copy_n1 = std::min<size1_t>(end1 - it1, out_end - out_it);
        auto copy1 = std::ranges::copy_n(it1, copy_n1, out_it);
        using size2_t = std::common_type_t<std::ranges::range_size_t<R2>, std::ranges::range_size_t<OutR>>;
        const size2_t copy_n2 = std::min<size2_t>(end2 - it2, out_end - copy1.out);
        auto copy2 = std::ranges::copy_n(it2, copy_n2, copy1.out);

        return {copy1.in, copy2.in, copy2.out};
    }
};

inline constexpr set_union_checker_fn set_union_checker{};
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_union, set_union_checker);
    test_range_algo<1, int, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out, mul1_t, div3_t>{100}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif // TEST_DPCPP_BACKEND_PRESENT

    bProcessed = true;

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
