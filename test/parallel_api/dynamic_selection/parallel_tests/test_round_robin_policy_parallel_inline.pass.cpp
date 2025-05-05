// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/inline_backend.h"
#include "support/test_round_robin_policy_parallel_utils.h"

#if TEST_DYNAMIC_SELECTION_AVAILABLE
static inline void
build_universe(std::vector<int>& u, std::unordered_map<int, int>& map)
{
    for(int i=0;i<u.size();i++){
        map[u[i]]=i;
    }
}
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::round_robin_policy<TestUtils::int_inline_backend_t>;
    std::vector<int> u{4, 5, 6, 7};
    std::unordered_map<int, int> map;
    build_universe(u, map);
    int constexpr count = 50;
    int actual;
    if (!u.empty())
    {
        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        //Just call submit
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait<just_call_submit, policy_t>(u, map, build_result(u.size(), count));

        bProcessed = true;
    }
#endif //TEST_DYNAMIC_SELECTION_AVAILABLE
    return TestUtils::done(bProcessed);
}
