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

#if _ENABLE_STD_RANGES_TESTING && !_PSTL_LIBCPP_RANGE_SET_BROKEN
namespace test_std_ranges
{
template<>
inline int out_size_with_empty_in2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_difference)>>(int in1_size)
{
    return in1_size;
}
}

template <typename RngA, typename RngB, typename RngOutExpected>
void
test_mixed_types_host(RngA&& __rngA, RngB&& __rngB, const RngOutExpected& out_expected)
{
    std::vector<int> out_seq      (out_expected.size(), 0xCD);
    std::vector<int> out_par      (out_expected.size(), 0xCD);
    std::vector<int> out_unseq    (out_expected.size(), 0xCD);
    std::vector<int> out_par_unseq(out_expected.size(), 0xCD);

    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::seq,       __rngA, __rngB, out_seq,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::par,       __rngA, __rngB, out_par,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::unseq,     __rngA, __rngB, out_unseq,     std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::par_unseq, __rngA, __rngB, out_par_unseq, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);

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
        using r1_alloc_t = sycl::usm_allocator<test_std_ranges::A, sycl::usm::alloc::shared>;
        using r2_alloc_t = sycl::usm_allocator<test_std_ranges::B, sycl::usm::alloc::shared>;
        using r_out_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

        std::vector<test_std_ranges::A, r1_alloc_t> v1({{1}, {2}, {5}}, r1_alloc_t(q));
        std::vector<test_std_ranges::B, r2_alloc_t> v2({{0}, {2}, {2}, {3}}, r2_alloc_t(q));
        std::vector<int> out_expected = {1, 5};

        std::vector<int, r_out_alloc_t> out(out_expected.size(), 0xCD, r_out_alloc_t(q));

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + v1.size());
        std::ranges::subrange r2(v2.data(), v2.data() + v2.size());
        std::ranges::subrange r_out(out.data(), out.data() + out.size());

        oneapi::dpl::ranges::set_difference(policy, r1, r2, r_out, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING && !_PSTL_LIBCPP_RANGE_SET_BROKEN

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING && !_PSTL_LIBCPP_RANGE_SET_BROKEN
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // TODO: use data_in_in_out_lim when set_difference supports
    // output range not-sufficiently large to hold all the processed elements
    // this will also require adding a custom serial implementation of the algorithm into the checker

    auto checker = [](std::ranges::random_access_range auto&& r1,
                      std::ranges::random_access_range auto&& r2,
                      std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        auto res = std::ranges::set_difference(std::forward<decltype(r1)>(r1), std::forward<decltype(r2)>(r2),
                                               std::ranges::begin(r_out), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::set_difference_result<std::ranges::borrowed_iterator_t<decltype(r1)>,
                                                            std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{res.in, res.out};
    };

    test_range_algo<0, int, data_in_in_out, div3_t, mul1_t>{big_sz}(dpl_ranges::set_difference, checker);
    test_range_algo<1, int, data_in_in_out, div3_t, mul1_t>{big_sz}(dpl_ranges::set_difference, checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out, div3_t, mul1_t>{100}(dpl_ranges::set_difference, checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out, div3_t, mul1_t>{}(dpl_ranges::set_difference, checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out, div3_t, mul1_t>{}(dpl_ranges::set_difference, checker, std::ranges::less{}, &P2::proj, &P2::proj);

    // Testing no intersection
    auto large_shift = [](auto&& v) { return v + 5000; };
    using ls_t = decltype(large_shift);
    test_range_algo<5, int, data_in_in_out, mul1_t, ls_t>{1000}(dpl_ranges::set_difference, checker);
    test_range_algo<6, int, data_in_in_out, ls_t, mul1_t>{1000}(dpl_ranges::set_difference, checker);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    std::vector<test_std_ranges::A> vec_a_src = {{1}, {2}, {5}};
    std::vector<test_std_ranges::B> vec_b_src = {{0}, {2}, {2}, {3}};
    std::vector<int> out_expected = {1, 5};

    test_mixed_types_host(vec_a_src, vec_b_src, out_expected);

    TestUtils::MinimalisticRange vec_a{vec_a_src.begin(), vec_a_src.end()};
    TestUtils::MinimalisticRange vec_b{vec_b_src.begin(), vec_b_src.end()};
    test_mixed_types_host(vec_a, vec_b, out_expected);

#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif

    bProcessed = true;

#endif //_ENABLE_STD_RANGES_TESTING && !_PSTL_LIBCPP_RANGE_SET_BROKEN

    return TestUtils::done(bProcessed);
}
