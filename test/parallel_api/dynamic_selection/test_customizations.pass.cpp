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
#include <iostream>
#include "support/test_dynamic_selection_one_policy.h"
#include "support/utils.h"

int
test_no_customizations()
{
    int trace = 0;
    one_with_no_customizations p(trace);

    trace = 0;
    auto s = oneapi::dpl::experimental::select(p);
    EXPECT_EQ((int)t_select, trace, "ERROR: unexpected trace of select");

    trace = 0;
    oneapi::dpl::experimental::submit(s, [](int i) { return i; });
    EXPECT_EQ((int)t_submit_selection, trace, "ERROR: unexpected trace of submit selection");

    trace = 0;
    oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select | t_submit_selection), trace, "ERROR: unexpected trace of submit function");

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(s, [](int i) { return i; });
    EXPECT_EQ((int)(t_submit_selection | t_wait), trace, "ERROR: unexpected trace of submit_and_wait selection");

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select | t_submit_selection | t_wait), trace, "ERROR: unexpected trace of submit_and_wait function");

    std::cout << "test_no_customizations: OK\n";

    return 0;
}

int
test_all_customizations()
{
    int trace = 0;
    one_with_all_customizations p(trace);

    trace = 0;
    auto s = oneapi::dpl::experimental::select(p);
    EXPECT_EQ((int)t_select, trace, "ERROR: unexpected trace of select");

    trace = 0;
    oneapi::dpl::experimental::submit(s, [](int i) { return i; });
    EXPECT_EQ((int)t_submit_selection, trace, "ERROR: unexpected trace of submit selection");

    trace = 0;
    oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)t_submit_function, trace, "ERROR: unexpected trace of submit function");

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(s, [](int i) { return i; });
    EXPECT_EQ((int)t_submit_and_wait_selection, trace, "ERROR: unexpected trace of submit_and_wait selection");

    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)t_submit_and_wait_function, trace, "ERROR: unexpected trace of submit_and_wait function");

    std::cout << "test_all_customizations: OK\n";
    return 0;
}

int
main()
{
    test_no_customizations();
    test_all_customizations();

    return TestUtils::done();
}
