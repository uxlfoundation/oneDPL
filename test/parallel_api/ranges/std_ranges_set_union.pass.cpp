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
// TODO remove after implementation range-based set operations for bounded output range with hetero policies
template <>
struct ResolveTestDataModeForHeteroPolicy<TestDataMode::data_in_in_out_lim>
{
    static constexpr TestDataMode res_mode = TestDataMode::data_in_in_out;
};

#if TEST_DPCPP_BACKEND_PRESENT
#if STD_RANGES_SET_OP_BROKEN_FOR_HETERO_POLICY
template <>
inline constexpr bool skip_test_for_hetero_policy<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_union)>> = true;
#endif
#endif

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

void test_mixed_types_host()
{
    std::vector<test_std_ranges::A> r1 = {{1}, {2}, {3}};
    std::vector<test_std_ranges::B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {0, 1, 2, 2, 3};

    std::vector<int> out_seq(5, 0xCD);
    std::vector<int> out_par(5, 0xCD);
    std::vector<int> out_unseq(5, 0xCD);
    std::vector<int> out_par_unseq(5, 0xCD);

    oneapi::dpl::ranges::set_union(oneapi::dpl::execution::seq,       r1, r2, out_seq,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_union(oneapi::dpl::execution::par,       r1, r2, out_par,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_union(oneapi::dpl::execution::unseq,     r1, r2, out_unseq,     std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_union(oneapi::dpl::execution::par_unseq, r1, r2, out_par_unseq, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);

    EXPECT_EQ_RANGES(out_expected, out_seq, "wrong result with seq policy");
    EXPECT_EQ_RANGES(out_expected, out_par, "wrong result with par policy");
    EXPECT_EQ_RANGES(out_expected, out_unseq, "wrong result with unseq policy");
    EXPECT_EQ_RANGES(out_expected, out_par_unseq, "wrong result with par_unseq policy");
}

#if TEST_DPCPP_BACKEND_PRESENT
#if !STD_RANGES_SET_OP_BROKEN_FOR_HETERO_POLICY
void test_mixed_types_device()
{
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        using r1_alloc_t = sycl::usm_allocator<test_std_ranges::A, sycl::usm::alloc::shared>;
        using r2_alloc_t = sycl::usm_allocator<test_std_ranges::B, sycl::usm::alloc::shared>;
        using r_out_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;

        std::vector<test_std_ranges::A, r1_alloc_t> v1({{1}, {2}, {3}}, r1_alloc_t(q));
        std::vector<test_std_ranges::B, r2_alloc_t> v2({{0}, {2}, {2}, {3}}, r2_alloc_t(q));
        std::vector<int, r_out_alloc_t> out(5, 0xCD, r_out_alloc_t(q));
        std::vector<int> out_expected = {0, 1, 2, 2, 3};

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + 3);
        std::ranges::subrange r2(v2.data(), v2.data() + 4);
        std::ranges::subrange r_out(out.data(), out.data() + 5);

        oneapi::dpl::ranges::set_union(policy, r1, r2, r_out, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
}
#endif // !STD_RANGES_SET_OP_BROKEN_FOR_HETERO_POLICY
#endif // TEST_DPCPP_BACKEND_PRESENT

struct
{
    template <std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
              std::ranges::random_access_range _ROut, typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
    std::ranges::set_union_result<std::ranges::borrowed_iterator_t<_R1>,
                                  std::ranges::borrowed_iterator_t<_R2>,
                                  std::ranges::borrowed_iterator_t<_ROut>>
    operator()(_R1&& r_1, _R2&& r_2, _ROut&& r_out, Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
    {
        // r_1 : 0, 1, 2, 3, ..., 131081
        // r_2 : 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, ..., 21845, 21845, 21845, 21846, 21846, 21846
        // r_out : size = 131082
        //     : 0, 0, 0, 1, 1, 1, 2, 2, 2, ...., 87385, 87386, 87387
        auto in1 = std::ranges::begin(r_1);
        auto in2 = std::ranges::begin(r_2);
        auto out = std::ranges::begin(r_out);

        const auto n1 = std::ranges::size(r_1);
        const auto n2 = std::ranges::size(r_2);
        const auto nOut = std::ranges::size(r_out);

        std::size_t idx1 = 0;
        std::size_t idx2 = 0;
        std::size_t idxOut = 0;

        while (idx1 < n1 && idx2 < n2 && idxOut < nOut)
        {
            if (std::invoke(comp, std::invoke(proj1, in1[idx1]), std::invoke(proj2, in2[idx2])))
            {
                out[idxOut++] = in1[idx1++];
            }
            else if (std::invoke(comp, std::invoke(proj2, in2[idx2]), std::invoke(proj1, in1[idx1])))
            {
                out[idxOut++] = in2[idx2++];
            }
            else
            {
                out[idxOut++] = in1[idx1++];
                ++idx2;
            }
        }

        const auto to_copy1 = test_std_ranges::eval_remaining_space_min(nOut, idxOut, n1, idx1);
        std::copy_n(in1 + idx1, to_copy1, out + idxOut);
        idx1 += to_copy1;
        idxOut += to_copy1;

        const auto to_copy2 = test_std_ranges::eval_remaining_space_min(nOut, idxOut, n2, idx2);
        std::copy_n(in2 + idx2, to_copy2, out + idxOut);
        idx2 += to_copy2;
        idxOut += to_copy2;

        return {in1 + idx1, in2 + idx2, out + idxOut};
    }
} set_union_checker;

// Check the correctness of the set_union_checker against the logic of std::ranges::set_union
void
test_set_union_checker()
{
    // range 1 shorter than range2
    {
        // set1: 0, 1,    5, 6,    9, 10
        // set2:       3,    6, 7, 9,     13, 15, 100
        //       ------------------------------------
        // res:  0, 1, 3, 5, 6, 7, 9, 10, 13, 15, 100

        std::vector<int> set1{0, 1, 5, 6, 9, 10};
        std::vector<int> set2{3, 6, 7, 9, 13, 15, 100};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{0, 1, 3, 5, 6, 7, 9, 10, 13, 15, 100};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 shorter than range 1
    {
        // set1:    2, 6, 8, 12, 15, 16
        // set2: 0, 2,    8
        //       ----------------------
        // res:  0, 2, 6, 8, 12, 15, 16

        std::vector<int> set1{2, 6, 8, 12, 15, 16};
        std::vector<int> set2{0, 2, 8};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{0, 2, 6, 8, 12, 15, 16};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 and range 2 have the same length but different elements
    {
        // set1:    2, 6, 8, 12, 15, 16
        // set2: 0, 2,    8,     15,     17, 19
        //       ------------------------------
        // res:  0, 2, 6, 8, 12, 15, 16, 17, 19

        std::vector<int> set1{2, 6, 8, 12, 15, 16};
        std::vector<int> set2{0, 2, 8, 15, 17, 19};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{0, 2, 6, 8, 12, 15, 16, 17, 19};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 == range 2
    {
        // set1: 0, 1, 2
        // set2: 0, 1, 2
        //       -------
        // res:  0, 1, 2

        std::vector<int> set1{0, 1, 2};
        std::vector<int> set2{0, 1, 2};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{0, 1, 2};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 is super set of range 2
    {
        // set1: 8, 8, 10, 12, 13
        // set2: 8,    10
        //       ----------------
        // res:  8, 8, 10, 12, 13

        std::vector<int> set1{8, 8, 10, 12, 13};
        std::vector<int> set2{8, 10};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{8, 8, 10, 12, 13};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 is super set of range 1
    {
        // set1: 0, 1, 1
        // set2: 0, 1, 1, 2, 5
        //       -------------
        // res:  0, 1, 1, 2, 5

        std::vector<int> set1{0, 1, 1};
        std::vector<int> set2{0, 1, 1, 2, 5};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{0, 1, 1, 2, 5};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 and range 2 have no elements in common
    {
        // set1:          7, 7,    9,     12
        // set2: 1, 5, 5,       8,    10
        //       ---------------------------
        // res:  1, 5, 5, 7, 7, 8, 9, 10, 12

        std::vector<int> set1{7, 7, 9, 12};
        std::vector<int> set2{1, 5, 5, 8, 10};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{1, 5, 5, 7, 7, 8, 9, 10, 12};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 and range 2 have duplicated equal elements
    {
        // set1: 7, 7,     9, 12
        // set2: 7, 7, 7,        13
        //       ------------------
        // res:  7, 7, 7, 9, 12, 13

        std::vector<int> set1{7, 7, 9, 12};
        std::vector<int> set2{7, 7, 7, 13};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{7, 7, 7, 9, 12, 13};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 is empty
    {
        // set1: (empty)
        // set2: 3, 4, 5
        //       -------
        // res:  3, 4, 5

        std::vector<int> set1{};
        std::vector<int> set2{3, 4, 5};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{3, 4, 5};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 is empty
    {
        // set1: 3, 4, 5
        // set2: (empty)
        //       -------
        // res:  3, 4, 5

        std::vector<int> set1{3, 4, 5};
        std::vector<int> set2{};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{3, 4, 5};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // both ranges are empty
    {
        // set1: (empty)
        // set2: (empty)
        // res:  (empty)

        std::vector<int> set1{};
        std::vector<int> set2{};
        std::vector<int> set3(0);
        const std::vector<int> resExpected{};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }
    {
        // set1:1, 2, 3
        // set2:         4, 5, 6
        //      ----------------
        // res: 1, 2, 3, 4, 5, 6

        std::vector<int> set1{1, 2, 3};
        std::vector<int> set2{        4, 5, 6};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{1, 2, 3, 4, 5, 6};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
    {
        // set1:                   1, 2, 3
        // set2:                          ^ 4, 5, 6
        //                         -------+---^
        // res:                    1, 2, 3, 4 |
        // final position in set1: -------+   |
        // final position in set2:------------+

        std::vector<int> set1{1, 2, 3};
        std::vector<int> set2{4, 5, 6};
        std::vector<int> set3(4);
        const std::vector<int> resExpected{1, 2, 3, 4};

        auto res = set_union_checker(set1, set2, set3);

        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 5) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

#endif
}
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING

    // Check the correctness of the set_union_checker against the logic of std::ranges::set_union
    test_set_union_checker();

    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_in_out_lim, mul1_t, div3_t>{get_scan_big_sz()}(dpl_ranges::set_union, set_union_checker);
    test_range_algo<1, int, data_in_in_out_lim, div3_t, mul1_t>{get_scan_big_sz()}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out_lim, mul1_t, div3_t>{100}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out_lim, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out_lim, mul1_t, div3_t>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
#if !STD_RANGES_SET_OP_BROKEN_FOR_HETERO_POLICY
    test_mixed_types_device();
#endif
#endif

    bProcessed = true;
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
