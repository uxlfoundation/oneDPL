// -*- C++ -*-
//===-- unique_ramp_compose.pass.cpp --------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. See unique_ramp_common.h.
// The crashing coordinate rebuilt in user code out of oneapi::dpl::unique_copy and a hand-written one-element copy
// kernel, so that none of __pattern_unique's new code runs. The copy back is a read_write accessor rather than
// oneapi::dpl::copy, whose no_init output would drop the write's dependency on the reads that precede it and so remove
// the very thing under test. A crash here puts the defect below the library change; a pass puts it inside
// __pattern_hetero_walk2 or the loop itself.

#include <cstddef>

namespace segment_size_override
{
std::size_t bytes = 64 * 1024 * 1024;
}
#define _ONEDPL_COMPACTION_SEGMENT_SIZE_BYTES (::segment_size_override::bytes)

#include "unique_ramp_common.h"

#if TEST_DPCPP_BACKEND_PRESENT
struct compose_copy_kernel;

// One element per segment, so the extended input range is two elements and the staging temporary is two.
template <typename Policy>
void
run_compose(Policy&& exec, std::size_t n, std::size_t run_length)
{
    using kernel_name = unique_ramp_kernel<buffer_tag, plain_unique>;

    const std::vector<value_type> input = make_input(n, run_length);
    std::vector<value_type> expected(input);
    const std::size_t expected_n = std::size_t(std::unique(expected.begin(), expected.end()) - expected.begin());

    std::vector<value_type> actual(input);
    std::size_t out_pos = 0;
    {
        sycl::buffer<value_type> in(actual.data(), sycl::range<1>(n));
        sycl::buffer<value_type> stage(sycl::range<1>(2));
        auto in_first = oneapi::dpl::begin(in);
        auto stage_first = oneapi::dpl::begin(stage);

        for (std::size_t in_pos = 0; in_pos < n; ++in_pos)
        {
            const std::size_t stage_off = in_pos > 0 ? 1 : 0;
            auto stage_last = oneapi::dpl::unique_copy(CLONE_TEST_POLICY_NAME(exec, kernel_name),
                                                       in_first + (in_pos - stage_off), in_first + in_pos + 1,
                                                       stage_first);
            const std::size_t survivors = std::size_t(stage_last - stage_first) - stage_off;
            if (survivors > 0)
            {
                const std::size_t dst = out_pos;
                exec.queue()
                    .submit([&](sycl::handler& h) {
                        sycl::accessor src{stage, h, sycl::read_only};
                        sycl::accessor dst_acc{in, h, sycl::read_write};
                        h.parallel_for<compose_copy_kernel>(sycl::range<1>(survivors), [=](sycl::id<1> i) {
                            dst_acc[dst + i] = src[stage_off + i];
                        });
                    })
                    .wait();
            }
            out_pos += survivors;
        }
    }

    EXPECT_EQ(expected_n, out_pos, "wrong number of survivors");
    EXPECT_EQ_N(expected.begin(), actual.begin(), expected_n, "wrong survivors");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    auto exec = TestUtils::get_dpcpp_test_policy();
    for (std::size_t rep = 0; rep < 4; ++rep)
    {
        std::cout << "rep " << rep << std::endl;
        run_compose(exec, 1024, 8);
        std::cout << "    ok" << std::endl;
    }
    std::cout << "=== complete ===" << std::endl;
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
