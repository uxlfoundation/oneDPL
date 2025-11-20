// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "oneapi/dpl/functional"
#include "support/test_dynamic_selection_utils.h"

template <typename Policy, typename Backend, typename ResourceContainer, typename FunctionType, typename ResourceAdapter,
          typename... Args>
int
run_round_robin_policy_tests(const ResourceContainer& resources, const FunctionType& f, ResourceAdapter adapter,
                             Args&&... args)
{
    int result = 0;

    result += test_initialization<Policy, typename ResourceContainer::value_type>(resources, adapter,
                                                                                  std::forward<Args>(args)...);
    result += test_default_universe_initialization<Policy, Backend>(adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<Policy, Backend>(resources, f, adapter, std::forward<Args>(args)...);

    return result;
}

int
main()
{
    bool bProcessed = false;

    try
    {
#if TEST_DYNAMIC_SELECTION_AVAILABLE
        std::vector<sycl::queue> u;
        build_universe(u);
        if (!u.empty())
        {
            auto n = u.size();
            std::cout << "UNIVERSE SIZE " << n << std::endl;

            // Test with direct sycl::queue resources
            using policy_t =
                oneapi::dpl::experimental::round_robin_policy<sycl::queue, oneapi::dpl::identity,
                                                              oneapi::dpl::experimental::default_backend<sycl::queue>>;
            auto f = [u, n](int i) { return u[(i - 1) % n]; };

            std::cout << "\nRunning round robin tests for sycl::queue ...\n";
            EXPECT_EQ(0, (run_round_robin_policy_tests<policy_t, oneapi::dpl::experimental::default_backend<sycl::queue>>(u, f, oneapi::dpl::identity{})), "");

            // Test with sycl::queue* resources and dereference adapter
            auto deref_op = [](auto pointer) { return *pointer; };
            using policy_pointer_t = oneapi::dpl::experimental::round_robin_policy<
                sycl::queue*, decltype(deref_op),
                oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;

            std::vector<sycl::queue*> u_ptrs;
            u_ptrs.reserve(u.size());
            for (auto& e : u)
            {
                u_ptrs.push_back(&e);
            }
            auto f_ptrs = [u_ptrs, n](int i) { return u_ptrs[(i - 1) % n]; };

            std::cout << "\nRunning round robin tests for sycl::queue* ...\n";
            EXPECT_EQ(0, (run_round_robin_policy_tests<policy_pointer_t, oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>(u_ptrs, f_ptrs, deref_op)), "");

            //CTAD tests (testing policy construction without template arguments)
            //Template arguments types are deduced with CTAD
            sycl::queue q1(sycl::default_selector_v);
            sycl::queue q2(sycl::default_selector_v);
            oneapi::dpl::experimental::round_robin_policy p1{{q1, q2}};
            oneapi::dpl::experimental::round_robin_policy p2({q1, q2});

            oneapi::dpl::experimental::round_robin_policy p3({&q1, &q2}, deref_op);
            oneapi::dpl::experimental::round_robin_policy p4{{&q1, &q2}, deref_op};

            bProcessed = true;
        }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
    }
    catch (const std::exception& exc)
    {
        std::stringstream str;
        str << "Exception occurred";
        if (exc.what())
            str << " : " << exc.what();

        TestUtils::issue_error_message(str);
    }

    return TestUtils::done(bProcessed);
}
