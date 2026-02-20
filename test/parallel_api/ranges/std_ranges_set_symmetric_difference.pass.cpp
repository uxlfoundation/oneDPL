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
int out_size_with_empty_in2<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_symmetric_difference)>>(int in1_size)
{
    return in1_size;
}
template<>
int out_size_with_empty_in1<std::remove_cvref_t<decltype(oneapi::dpl::ranges::set_symmetric_difference)>>(int in2_size)
{
    return in2_size;
}
}

void test_mixed_types_host()
{
    std::vector<test_std_ranges::A> r1 = {{1}, {2}, {5}};
    std::vector<test_std_ranges::B> r2 = {{0}, {2}, {2}, {3}};

    std::vector<int> out_expected = {0, 1, 2, 3, 5};

    std::vector<int> out_seq(out_expected.size(), 0xCD);
    std::vector<int> out_par(out_expected.size(), 0xCD);
    std::vector<int> out_unseq(out_expected.size(), 0xCD);
    std::vector<int> out_par_unseq(out_expected.size(), 0xCD);

    oneapi::dpl::ranges::set_symmetric_difference(oneapi::dpl::execution::seq,       r1, r2, out_seq,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_symmetric_difference(oneapi::dpl::execution::par,       r1, r2, out_par,       std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_symmetric_difference(oneapi::dpl::execution::unseq,     r1, r2, out_unseq,     std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
    oneapi::dpl::ranges::set_symmetric_difference(oneapi::dpl::execution::par_unseq, r1, r2, out_par_unseq, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);

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
        std::vector<int> out_expected = {0, 1, 2, 3, 5};

        std::vector<int, r_out_alloc_t> out(out_expected.size(), 0xCD, r_out_alloc_t(q));

        // Wrap vector with a USM allocator into the subrange because it is not device copyable
        std::ranges::subrange r1(v1.data(), v1.data() + v1.size());
        std::ranges::subrange r2(v2.data(), v2.data() + v2.size());
        std::ranges::subrange r_out(out.data(), out.data() + out.size());

        oneapi::dpl::ranges::set_symmetric_difference(policy, r1, r2, r_out, std::ranges::less{}, test_std_ranges::proj_a, test_std_ranges::proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

struct
{
    template <std::ranges::random_access_range _R1, std::ranges::random_access_range _R2,
              std::ranges::random_access_range _ROut, typename Comp = std::ranges::less, typename Proj1 = std::identity,
              typename Proj2 = std::identity>
    std::ranges::set_intersection_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                         std::ranges::borrowed_iterator_t<_ROut>>
    operator()(_R1&& r_1, _R2&& r_2, _ROut&& r_out, Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
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

        while (idx1 < n1)
        {
            if (idx2 == n2)
            {
                auto remaining_space = std::ranges::size(r_out) - idxOut;
                auto remaining_input = n1 - idx1;
                auto to_copy = std::min(remaining_space, remaining_input);
                std::copy(in1 + idx1, in1 + idx1 + to_copy, out + idxOut);

                idx1 += to_copy;
                idxOut += to_copy;
                break;
            }

            if (std::invoke(comp, std::invoke(proj1, in1[idx1]), std::invoke(proj2, in2[idx2])))
            {
                if (idxOut < nOut)
                {
                    out[idxOut] = in1[idx1];
                    ++idx1;
                    ++idxOut;
                }
                else
                    break;
            }
            else
            {
                if (std::invoke(comp, std::invoke(proj2, in2[idx2]), std::invoke(proj1, in1[idx1])))
                {
                    if (idxOut < nOut)
                    {
                        out[idxOut] = in2[idx2];
                        ++idxOut;
                    }
                    else
                        break;
                }
                else
                    ++idx1;
                ++idx2;
            }
        }

        const auto remaining_space = nOut - idxOut;
        const auto remaining_input = n2 - idx2;
        const auto to_copy = std::min(remaining_space, remaining_input);
        std::copy(in2 + idx2, in2 + idx2 + to_copy, out + idxOut);

        idx2 += to_copy;
        idxOut += to_copy;

        return {in1 + idx1, in2 + idx2, out + idxOut};
    }
} set_symmetric_difference_checker;
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_in_out_lim, data_in_in_out, div3_t, mul1_t>{big_sz}(dpl_ranges::set_symmetric_difference, set_symmetric_difference_checker);
    test_range_algo<1, int, data_in_in_out_lim, data_in_in_out, mul1_t, div3_t>{big_sz}(dpl_ranges::set_symmetric_difference, set_symmetric_difference_checker, std::ranges::less{}, proj);

    // Testing the cut-off with the serial implementation (less than __set_algo_cut_off)
    test_range_algo<2, int, data_in_in_out_lim, data_in_in_out, mul1_t, mul1_t>{100}(dpl_ranges::set_symmetric_difference, set_symmetric_difference_checker, std::ranges::less{}, proj, proj);

    test_range_algo<3,  P2, data_in_in_out_lim, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_symmetric_difference, set_symmetric_difference_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out_lim, data_in_in_out, mul1_t, div3_t>{}(dpl_ranges::set_symmetric_difference, set_symmetric_difference_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    test_mixed_types_host();
#if TEST_DPCPP_BACKEND_PRESENT
    test_mixed_types_device();
#endif

    bProcessed = true;

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
