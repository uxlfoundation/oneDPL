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

// partial_sort_copy applies its first projection to the input sequence and its second one to the output
// sequence, and the output elements are copies of the input ones. So as long as the two projections
// order the elements the same way, applying the wrong one to the input sequence stays invisible in the
// result, and an order inconsistent pair makes the result depend on the order in which the input
// elements are examined, i.e. on the implementation. The check below therefore compares the policies
// with each other instead of comparing them with std::ranges::partial_sort_copy: whichever elements an
// implementation selects, all of its policies have to select the same ones, which they cannot do while
// some of them ignore the first projection.
void test_projections_consistency()
{
    using namespace test_std_ranges;

    const int n = medium_size;
    const int n_out = n / 8;

    std::vector<P2> in(n);
    for (int i = 0; i < n; ++i)
        in[i] = P2{i, i};

    // The two projections are deliberately inconsistent: the first one orders the input sequence by
    // descending x and the second one orders the output sequence by ascending x.
    auto in_proj = [](const P2& v) { return -v.x; };
    auto out_proj = [](const P2& v) { return v.x; };

    auto call = [&](auto&& exec)
    {
        std::vector<P2> out(n_out, P2{-1, -1});
        dpl_ranges::partial_sort_copy(std::forward<decltype(exec)>(exec), in, out, std::ranges::less{}, in_proj,
                                      out_proj);
        return out;
    };

    // Both seq and unseq forward the two projections to std::ranges::partial_sort_copy, so either of
    // them is the reference the remaining policies are compared with.
    std::vector<P2> out_seq = call(oneapi::dpl::execution::seq);
    EXPECT_EQ_RANGES(out_seq, call(oneapi::dpl::execution::unseq), "unseq policy disagrees with seq policy");

    // KSATODO: the parallel host pattern and the device pattern drop _Proj1 altogether
    // (algorithm_ranges_impl.h:595,608 and hetero/algorithm_ranges_impl_hetero.h:1664,1675 build
    // __binary_op<_Comp, _Proj2, _Proj2>), so they order the input sequence by the projection of the
    // output sequence and select the wrong elements.
#if !_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_PARTIAL_SORT_COPY_PROJ1_HOST
    EXPECT_EQ_RANGES(out_seq, call(oneapi::dpl::execution::par), "par policy disagrees with seq policy");
    EXPECT_EQ_RANGES(out_seq, call(oneapi::dpl::execution::par_unseq), "par_unseq policy disagrees with seq policy");
#endif
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_PARTIAL_SORT_COPY_PROJ1_HETERO
    auto policy = TestUtils::get_dpcpp_test_policy();
    sycl::queue q = policy.queue();
    if (q.get_device().has(sycl::aspect::usm_shared_allocations))
    {
        using alloc_t = sycl::usm_allocator<P2, sycl::usm::alloc::shared>;
        std::vector<P2, alloc_t> v_in(in.begin(), in.end(), alloc_t(q));
        std::vector<P2, alloc_t> v_out(n_out, P2{-1, -1}, alloc_t(q));

        dpl_ranges::partial_sort_copy(policy, std::ranges::subrange(v_in), std::ranges::subrange(v_out),
                                      std::ranges::less{}, in_proj, out_proj);
        EXPECT_EQ_RANGES(out_seq, v_out, "device policy disagrees with seq policy");
    }
#endif
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

    // Check if the first projection is applied to the input sequence, which no case above can see
    test_projections_consistency();
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
