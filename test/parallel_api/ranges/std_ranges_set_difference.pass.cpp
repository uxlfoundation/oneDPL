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
inline constexpr bool skip_test_for_hetero_policy<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_difference)>> = true;
#endif
#endif

template <>
inline int out_size_with_empty_in2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_difference)>>(int in1_size)
{
    return in1_size;
}
}

void test_mixed_types_host()
{
    std::vector<test_std_ranges::A> r1 = {{1}, {2}, {5}};
    std::vector<test_std_ranges::B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {1, 5};

    std::vector<int> out_seq(out_expected.size(), 0xCD);
    std::vector<int> out_par(out_expected.size(), 0xCD);
    std::vector<int> out_unseq(out_expected.size(), 0xCD);
    std::vector<int> out_par_unseq(out_expected.size(), 0xCD);

    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::seq,       r1, r2, out_seq,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::par,       r1, r2, out_par,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::unseq,     r1, r2, out_unseq,     std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_difference(oneapi::dpl::execution::par_unseq, r1, r2, out_par_unseq, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);

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
#endif // !STD_RANGES_SET_OP_BROKEN_FOR_HETERO_POLICY
#endif // TEST_DPCPP_BACKEND_PRESENT

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
template <std::ranges::range R1, std::ranges::range R2, std::ranges::range ROut>
using set_difference_result_t =
    std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<R1>, std::ranges::borrowed_iterator_t<R2>,
                                  std::ranges::borrowed_iterator_t<ROut>>;
#else
template <std::ranges::range R1, std::ranges::range R2, std::ranges::range ROut>
using set_difference_result_t =
    std::ranges::in_out_result<std::ranges::borrowed_iterator_t<R1>, std::ranges::borrowed_iterator_t<ROut>>;
#endif

struct
{
    template <std::ranges::random_access_range R1, std::ranges::random_access_range R2,
              std::ranges::random_access_range ROut, typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
    set_difference_result_t<R1, R2, ROut>
    operator()(R1&& r_1, R2&& r_2, ROut&& r_out, Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
    {
        auto in1 = std::ranges::begin(r_1);
        auto in2 = std::ranges::begin(r_2);
        auto out = std::ranges::begin(r_out);

        const auto n1 = std::ranges::size(r_1);
        const auto n2 = std::ranges::size(r_2);
        const auto nOut = std::ranges::size(r_out);

        std::size_t idx1 = 0;
        std::size_t idx2 = 0;
        std::size_t idxOut = 0;

        while (idx1 < n1 && idx2 < n2)
        {
            if (std::invoke(comp, std::invoke(proj1, in1[idx1]), std::invoke(proj2, in2[idx2])))
            {
                if (idxOut < nOut)
                    out[idxOut++] = in1[idx1++];
                else
                    break;
            }
            else if (!std::invoke(comp, std::invoke(proj2, in2[idx2]), std::invoke(proj1, in1[idx1])))
            {
                ++idx1;
                ++idx2;
            }
            else
            {
                ++idx2;
            }
        }

        const auto to_copy = test_std_ranges::eval_remaining_space_min(nOut, idxOut, n1, idx1);
        std::copy_n(in1 + idx1, to_copy, out + idxOut);
        idx1 += to_copy;
        idxOut += to_copy;

        assert(idx1 <= n1);
        assert(idxOut <= nOut);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        return {in1 + idx1, in2 + idx2, out + idxOut};
#else
        return {in1 + idx1, out + idxOut};
#endif
    }
} set_difference_checker;

void
test_set_difference_checker()
{
    // range 1 shorter than range2
    {
        std::vector<int> set1{0, 1, 5, 6,    9, 10};
        std::vector<int> set2{3,       6, 7, 9,     13, 15, 100};
        std::vector<int> set3(set1.size() + set2.size());
        std::vector<int> resExpected{0, 1, 5, 10};
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 13) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 shorter than range 1
    {
        std::vector<int> set1{2, 6, 8, 12, 15, 16};
        std::vector<int> set2{0, 2, 8};
        std::vector<int> set3(set1.size() + set2.size());
        std::vector<int> resExpected{6, 12, 15, 16};
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 and range 2 has the same length but different elements
    {
        std::vector<int> set1{2, 6, 8, 12, 15, 16};
        std::vector<int> set2{0, 2, 8,     15,     17, 19};
        std::vector<int> set3(set1.size() + set2.size());
        std::vector<int> resExpected{6, 12, 16};
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 17) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 == range 2
    {
        std::vector<int> set1{0, 1, 2};
        std::vector<int> set2{0, 1, 2};
        std::vector<int> set3(set1.size() + set2.size());
        std::vector<int> resExpected{};
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 is super set of range 2
    {
        std::vector<int> set1{8, 8, 10, 12, 13};
        std::vector<int> set2{8,    10};
        std::vector<int> set3(set1.size() + set2.size());
        std::vector<int> resExpected{8, 12, 13};
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 is super set of range 1
    {
        std::vector<int> set1{0, 1, 1};
        std::vector<int> set2{0, 1, 1, 2, 5};
        std::vector<int> resExpected{};
        std::vector<int> set3(set1.size() + set2.size());
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 2) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 1 is empty
    {
        std::vector<int> set1{};
        std::vector<int> set2{3, 4, 5};
        std::vector<int> resExpected{};
        std::vector<int> set3(set1.size() + set2.size());
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 3) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // range 2 is empty
    {
        std::vector<int> set1{3, 4, 5};
        std::vector<int> set2{};
        std::vector<int> resExpected{3, 4, 5};
        std::vector<int> set3(set1.size() + set2.size());
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // both ranges are empty
    {
        std::vector<int> set1{};
        std::vector<int> set2{};
        std::vector<int> resExpected{};    
        std::vector<int> set3(set1.size() + set2.size());
        
        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");

        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    {
        // set1:                   1, 2, 3, 4, 5,             10, 11, 12, 13, 14, 15
        // set2:                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,                 20, 21, 22, 23, 24, 25
        //                         -------------------------------------------------^---------------------------------------
        // res:                                                                     |
        // final position in set1: -------------------------------------------------+
        // final position in set2:--------------------------------------------------+

        std::vector<int> set1{1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15};
        std::vector<int> set2{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25};
        std::vector<int> set3(set1.size() + set2.size());

        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 20) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(set3.begin(), res.out, "Wrong 'out' state of result");
    }

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
    {
        // set1:                   1, 2, 3, 4, 5,             10, 11, 12, 13, 14, 15, 16, 17
        // set2:                                  6, 7, 8, 9,     11, 12, ^   14
        //                         -----------------------------          |  ^
        // res:                    1, 2, 3, 4, 5,             10          |  |
        // final position in set1: ---------------------------------------+  |
        // final position in set2:-------------------------------------------+

        std::vector<int> set1{1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17};
        std::vector<int> set2{6, 7, 8, 9, 11, 12, 14};
        std::vector<int> set3(6);
        const std::vector<int> resExpected{1, 2, 3, 4, 5, 10};

        auto res = set_difference_checker(set1, set2, set3);

        EXPECT_EQ(std::find(set1.begin(), set1.end(), 13) - set1.begin(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 14) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }
#endif

    {
        // set1:                   1, 2, 3, 4, 5,             10, 11, 12, 13, 14, 15
        // set2:                   1, 2, 3, 4, 5, 6, 7, 8, 9,                                          20, 21, 22, 23, 24, 25
        //                         --------------------------------------------------^---------------------------------------
        // res:                                               10, 11, 12, 13, 14, 15 |
        // final position in set1: --------------------------------------------------+
        // final position in set2:---------------------------------------------------+

        std::vector<int> set1{1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15};
        std::vector<int> set2{1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{10, 11, 12, 13, 14, 15};

        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 20) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    {
        // set1:                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10,                 15, 16, 17, 18, 19, 20
        // set2:                            4, 5, 6, 7,           11, 12, 13, 15, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
        //                         ---------------------------------------------------------------------^-------------------
        // res:                    1, 2, 3,             8, 9, 10                                        |
        // final position in set1: ---------------------------------------------------------------------+
        // final position in set2:----------------------------------------------------------------------+

        std::vector<int> set1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,             15, 16,     17, 18, 19, 20};
        std::vector<int> set2{         4, 5, 6, 7,           11, 12, 13, 15, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
        std::vector<int> set3(set1.size() + set2.size());
        const std::vector<int> resExpected{1, 2, 3, 8, 9, 10};

        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(std::find(set2.begin(), set2.end(), 21) - set2.begin(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else                                                               
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }

    // Check duplicate handling logic
    {
        // set1:                   1, 2, 3, 4, 5, 5, 5, 5, 6, 7
        // set2:                   1, 2, 3, 4, 5, 5, 5,    6    ^
        //                         -------------------------^---|
        // res:                                         5,  | 7 |
        // final position in set1: -------------------------|---+
        // final position in set2:--------------------------+

        std::vector<int> set1{1, 2, 3, 4, 5, 5, 5, 5, 6, 7};
        std::vector<int> set2{1, 2, 3, 4, 5, 5, 5,    6};
        std::vector<int> set3(set1.size());
        const std::vector<int> resExpected{5, 7};

        auto res = set_difference_checker(set1, set2, set3);

#if ONEDPL_RANGES_SET_ALGORITHMS_CPP26_ALIGNED
        EXPECT_EQ(set1.size(), res.in1 - set1.begin(), "Wrong 'in1' state of result");
        EXPECT_EQ(set2.size(), res.in2 - set2.begin(), "Wrong 'in2' state of result");
#else
        EXPECT_EQ(set1.size(), res.in - set1.begin(), "Wrong 'in' state of result");
#endif
        EXPECT_EQ(resExpected.size(), res.out - set3.begin(), "Wrong 'out' state of result");
        EXPECT_EQ_N(resExpected.begin(), set3.begin(), resExpected.size(), "Wrong output data state");
    }
}
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING

    // Check the correctness of the set_difference_checker against the logic of std::ranges::set_difference
    test_set_difference_checker();

    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_in_out, div3_t, mul1_t>{get_scan_big_sz()}(dpl_ranges::set_difference, set_difference_checker);
    test_range_algo<1, int, data_in_in_out, div3_t, mul1_t>{get_scan_big_sz()}(dpl_ranges::set_difference, set_difference_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out_lim, div3_t, mul1_t>{100}(dpl_ranges::set_difference, set_difference_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out_lim, div3_t, mul1_t>{}(dpl_ranges::set_difference, set_difference_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out_lim, div3_t, mul1_t>{}(dpl_ranges::set_difference, set_difference_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    // Testing no intersection
    auto large_shift = [](auto&& v) { return v + 5000; };
    using ls_t = decltype(large_shift);
    test_range_algo<5, int, data_in_in_out_lim,  mul1_t, ls_t>{1000}(dpl_ranges::set_difference, set_difference_checker);
    test_range_algo<6, int, data_in_in_out_lim, ls_t, mul1_t>{1000}(dpl_ranges::set_difference, set_difference_checker);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
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
