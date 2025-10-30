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

#include "support/inline_backend.h"
#include "support/test_offset_utils.h"
#include "support/utils.h"

int
main()
{
    using policy_t =
        oneapi::dpl::experimental::fixed_resource_policy<int, oneapi::dpl::identity, TestUtils::int_inline_backend_t<>>;
    std::vector<int> u{4, 5, 6, 7};
    auto f = [u](size_t, size_t offset = 0) { return u[offset]; };

    test_initialization<policy_t, int>(u);
    test_submit_and_wait_on_event<policy_t>(u, f);
    test_submit_and_wait_on_event<policy_t>(u, f, 1);
    test_submit_and_wait_on_event<policy_t>(u, f, 2);
    test_submit_and_wait_on_event<policy_t>(u, f, 3);
    test_submit_and_wait<policy_t>(u, f);
    test_submit_and_wait<policy_t>(u, f, 1);
    test_submit_and_wait<policy_t>(u, f, 2);
    test_submit_and_wait<policy_t>(u, f, 3);
    test_submit_and_wait_on_group<policy_t>(u, f);
    test_submit_and_wait_on_group<policy_t>(u, f, 1);
    test_submit_and_wait_on_group<policy_t>(u, f, 2);
    test_submit_and_wait_on_group<policy_t>(u, f, 3);

    using policy1_t =
        oneapi::dpl::experimental::fixed_resource_policy<DummyResource, oneapi::dpl::identity,
                                                         oneapi::dpl::experimental::default_backend<DummyResource>>;
    std::vector<DummyResource> u1;
    for (int i = 4; i < 8; ++i)
    {
        u1.push_back(DummyResource(i));
    }
    auto f1 = [u1](size_t, size_t offset = 0) { return u1[offset]; };

    test_initialization<policy1_t, DummyResource>(u1);
    test_submit_and_wait_on_event<policy1_t>(u1, f1);
    test_submit_and_wait_on_event<policy1_t>(u1, f1, 1);
    test_submit_and_wait_on_event<policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_event<policy1_t>(u1, f1, 3);
    test_submit_and_wait<policy1_t>(u1, f1);
    test_submit_and_wait<policy1_t>(u1, f1, 1);
    test_submit_and_wait<policy1_t>(u1, f1, 2);
    test_submit_and_wait<policy1_t>(u1, f1, 3);
    test_submit_and_wait_on_group<policy1_t>(u1, f1);
    test_submit_and_wait_on_group<policy1_t>(u1, f1, 1);
    test_submit_and_wait_on_group<policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_group<policy1_t>(u1, f1, 3);
    return TestUtils::done();
}
