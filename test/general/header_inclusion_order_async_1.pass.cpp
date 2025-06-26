// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include _PSTL_TEST_HEADER(async)
#endif // TEST_DPCPP_BACKEND_PRESENT
#include _PSTL_TEST_HEADER(execution)

#include <vector>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <typename Policy>
void
test_impl(Policy&& exec)
{
    sycl::queue q = exec.queue();

    constexpr std::size_t n = 100;

    using T = float;
    using allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

    allocator alloc(q);
    std::vector<T, allocator> data(n, 1, alloc);

    auto f = oneapi::dpl::experimental::reduce_async(std::forward<Policy>(exec), data.begin(), data.end());
    f.wait();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
