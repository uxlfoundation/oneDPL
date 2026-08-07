// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
namespace dpl_ranges = oneapi::dpl::ranges;

void test_mixed_types()
{
    using namespace test_std_ranges;

    std::vector<A> r1 = {{1}, {2}, {5}, {0}, {2}, {7}, {3}};

    std::vector<int> out_expected = {0, 1, 2, 2, 3};

    std::vector<B> out_seq(out_expected.size(), B{0xCD});
    std::vector<B> out_par(out_expected.size(), B{0xCD});
    std::vector<B> out_unseq(out_expected.size(), B{0xCD});
    std::vector<B> out_par_unseq(out_expected.size(), B{0xCD});

    dpl_ranges::partial_sort_copy(oneapi::dpl::execution::seq,       r1, out_seq,  std::ranges::less{}, proj_a, proj_b);
    dpl_ranges::partial_sort_copy(oneapi::dpl::execution::par,       r1, out_par,  std::ranges::less{}, proj_a, proj_b);
    dpl_ranges::partial_sort_copy(oneapi::dpl::execution::unseq,     r1, out_unseq,     std::less{}, proj_a, proj_b);
    dpl_ranges::partial_sort_copy(oneapi::dpl::execution::par_unseq, r1, out_par_unseq, std::less{}, proj_a, proj_b);

    EXPECT_EQ_RANGES(out_expected, out_seq, "wrong result with seq policy");
    EXPECT_EQ_RANGES(out_expected, out_par, "wrong result with par policy");
    EXPECT_EQ_RANGES(out_expected, out_unseq, "wrong result with unseq policy");
    EXPECT_EQ_RANGES(out_expected, out_par_unseq, "wrong result with par_unseq policy");
#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        using r1_alloc_t = sycl::usm_allocator<A, sycl::usm::alloc::shared>;
        using out_alloc_t = sycl::usm_allocator<B, sycl::usm::alloc::shared>;
        std::vector<A, r1_alloc_t> v1(r1.begin(), r1.end(), r1_alloc_t(q));
        std::vector<B, out_alloc_t> out(out_expected.size(), B{0xCD}, out_alloc_t(q));

        dpl_ranges::partial_sort_copy(policy, std::ranges::subrange(v1), std::ranges::subrange(out), std::ranges::less{}, proj_a, proj_b);
        EXPECT_EQ_RANGES(out_expected, out, "wrong result with device policy");
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto checker = TEST_PREPARE_CALLABLE(std::ranges::partial_sort_copy);

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::partial_sort_copy, checker);
    test_range_algo<1, int, data_in_out_lim>{}(dpl_ranges::partial_sort_copy, checker, std::greater{}, proj, proj);
    test_range_algo<2, P2, data_in_out_lim>{}(dpl_ranges::partial_sort_copy, checker, std::less{}, &P2::proj, &P2::x);
    test_range_algo<3, P2, data_in_out_lim>{}(dpl_ranges::partial_sort_copy, checker, std::greater{}, &P2::x, &P2::proj);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    test_mixed_types();
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
