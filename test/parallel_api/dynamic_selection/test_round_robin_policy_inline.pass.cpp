// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include "oneapi/dpl/dynamic_selection"

#include "support/test_dynamic_selection_utils.h"
#include "support/inline_backend.h"
#include "support/utils.h"

int
main()
{
    {
        using policy_t = oneapi::dpl::experimental::round_robin_policy<int, oneapi::dpl::identity,
                                                                       TestUtils::int_inline_backend_t<>>;
        std::vector<int> u{4, 5, 6, 7};
        auto f = [u](int i) { return u[(i - 1) % 4]; };

        EXPECT_EQ(0, (test_initialization<policy_t, int>(u)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<policy_t>(u, f)), "");
    }
    {
        using policy_t = oneapi::dpl::experimental::round_robin_policy<int, oneapi::dpl::identity,
                                                                       oneapi::dpl::experimental::default_backend<int>>;
        std::vector<int> u{4, 5, 6, 7};
        auto f = [u](int i) { return u[(i - 1) % 4]; };

        EXPECT_EQ(0, (test_initialization<policy_t, int>(u)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<policy_t>(u, f)), "");
    }
    {
        //tests using minimal backend that only provides a wait functionality through the resource
        using policy1_t =
            oneapi::dpl::experimental::round_robin_policy<DummyResource, oneapi::dpl::identity,
                                                          oneapi::dpl::experimental::default_backend<DummyResource>>;
        std::vector<DummyResource> u1;
        for (int i = 0; i < 4; ++i)
        {
            u1.push_back(DummyResource(i));
        }
        auto f1 = [u1](int i) { return u1[(i - 1) % 4]; };

        EXPECT_EQ(0, (test_initialization<policy1_t, DummyResource>(u1)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<policy1_t>(u1, f1)), "");
        EXPECT_EQ(0, (test_submit_and_wait<policy1_t>(u1, f1)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<policy1_t>(u1, f1)), "");
    }
    return TestUtils::done();
}
