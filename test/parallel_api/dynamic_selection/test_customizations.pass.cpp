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
    std::cout << "Starting test_no_customizations...\n";
    int trace = 0;
    no_customizations_policy_base p(trace);
    p.initialize();
    EXPECT_EQ((int)(t_init), trace, "ERROR: unexpected trace of initialize function");

    std::cout << "  Testing try_submit...\n";
    trace = 0;
    auto opt_sub = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
    EXPECT_TRUE((bool)opt_sub, "ERROR: try_submit should return a value");
    EXPECT_EQ((int)(t_select), trace, "ERROR: unexpected trace of try_submit function");

    std::cout << "  Testing submit...\n";
    trace = 0;
    auto sub = oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select), trace, "ERROR: unexpected trace of submit function");
    sub.wait();

    std::cout << "  Testing submit_and_wait...\n";
    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select), trace, "ERROR: unexpected trace of submit_and_wait function");

    std::cout << "test_no_customizations: OK\n";

    return 0;
}

int
test_all_customizations()
{
    int trace = 0;
    one_with_all_customizations p(trace);

    std::cout << "  Testing try_submit...\n";
    trace = 0;
    auto opt_sub = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
    EXPECT_TRUE((bool)opt_sub, "ERROR: try_submit should return a value");
    EXPECT_EQ((int)(t_select | t_try_submit_function), trace, "ERROR: unexpected trace of try_submit function");

    std::cout << "  Testing submit...\n";
    trace = 0;
    auto sub = oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select | t_try_submit_function | t_submit_function), trace,
              "ERROR: unexpected trace of submit function");
    sub.wait();

    std::cout << "  Testing submit_and_wait...\n";
    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_select | t_try_submit_function | t_submit_function | t_submit_and_wait_function | t_wait), trace,
              "ERROR: unexpected trace of submit_and_wait function");

    std::cout << "test_all_customizations: OK\n";
    return 0;
}

int
test_only_try_submit()
{
    std::cout << "Starting test_only_try_submit...\n";
    int trace = 0;
    one_with_only_try_submit p(trace);

    std::cout << "  Testing try_submit...\n";
    // Test try_submit - should use custom try_submit
    trace = 0;
    auto opt_sub = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
    EXPECT_TRUE((bool)opt_sub, "ERROR: try_submit should return a value");
    EXPECT_EQ((int)(t_try_submit_function), trace, "ERROR: try_submit should use custom try_submit");

    std::cout << "  Testing submit (should use generic based on try_submit)...\n";
    // Test submit - should have generic implementation based on try_submit
    trace = 0;
    auto sub = oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_try_submit_function), trace,
              "ERROR: submit should use generic submit (which loops on try_submit)");
    sub.wait();

    std::cout << "  Testing submit_and_wait (should use generic based on try_submit)...\n";
    // Test submit_and_wait - should use generic implementation based on try_submit
    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_try_submit_function | t_wait), trace,
              "ERROR: submit_and_wait should use generic submit_and_wait (which uses try_submit)");

    std::cout << "test_only_try_submit: OK\n";
    return 0;
}

int
test_only_submit()
{
    std::cout << "Starting test_only_submit...\n";
    int trace = 0;
    one_with_only_submit p(trace);

    // try_submit should NOT work - only select_impl is implemented, not try_select_impl

    std::cout << "  Testing submit...\n";
    // Test submit - should use custom submit
    trace = 0;
    auto sub = oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_submit_function), trace, "ERROR: submit should use custom submit");
    sub.wait();

    std::cout << "  Testing submit_and_wait...\n";
    // Test submit_and_wait - should use custom submit + wait
    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_submit_function | t_wait), trace,
              "ERROR: submit_and_wait should use custom submit + wait when only submit is customized");

    std::cout << "test_only_submit: OK\n";
    return 0;
}

int
test_only_submit_and_wait()
{
    std::cout << "Starting test_only_submit_and_wait...\n";
    int trace = 0;
    one_with_only_submit_and_wait p(trace);

    // try_submit should NOT work - no try_select_impl
    // submit should NOT work - no select_impl or try_select_impl

    std::cout << "  Testing submit_and_wait...\n";
    // Test submit_and_wait - should use custom submit_and_wait
    trace = 0;
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ((int)(t_submit_and_wait_function), trace, "ERROR: submit_and_wait should use custom submit_and_wait");

    std::cout << "test_only_submit_and_wait: OK\n";
    return 0;
}

int
main()
{
    try
    {
        test_no_customizations();
        test_all_customizations();
        test_only_try_submit();
        test_only_submit();
        test_only_submit_and_wait();
    }
    catch (const std::exception& exc)
    {
        std::stringstream str;

        str << "Exception occurred";
        if (exc.what())
            str << " : " << exc.what();

        TestUtils::issue_error_message(str);
    }

    return TestUtils::done();
}
