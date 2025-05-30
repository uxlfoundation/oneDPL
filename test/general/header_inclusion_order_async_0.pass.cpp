// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#if TEST_DPCPP_BACKEND_PRESENT
#    include _PSTL_TEST_HEADER(async)
#endif // TEST_DPCPP_BACKEND_PRESENT

#include <vector>

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::dpcpp_policy();
    sycl::queue q = policy.queue();

    constexpr std::size_t n = 100;

    using T = float;
    using allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

    allocator alloc(q);
    std::vector<T, allocator> data(n, 1, alloc);

    auto f = oneapi::dpl::experimental::reduce_async(policy, data.begin(), data.end());
    f.wait();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
