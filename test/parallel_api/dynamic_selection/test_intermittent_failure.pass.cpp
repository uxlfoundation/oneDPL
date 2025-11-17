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
test_intermittent_failure()
{
    std::cout << "Starting test_intermittent_failure...\n";
    one_with_intermittent_failure p;

    std::cout << "  Testing try_submit with intermittent failures...\n";
    
    // First attempt should fail (count = 0, even)
    p.reset_attempt_count();
    auto opt_sub1 = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
   EXPECT_FALSE((bool)opt_sub1, "ERROR: first try_submit should fail (even attempt)");
   EXPECT_EQ(1, p.get_attempt_count(), "ERROR: should have made 1 attempt");

    // Second attempt should succeed (count = 1, odd)
    auto opt_sub2 = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
   EXPECT_TRUE((bool)opt_sub2, "ERROR: second try_submit should succeed (odd attempt)");
   EXPECT_EQ(2, p.get_attempt_count(), "ERROR: should have made 2 attempts");

    // Third attempt should fail again (count = 2, even)
    auto opt_sub3 = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
   EXPECT_FALSE((bool)opt_sub3, "ERROR: third try_submit should fail (even attempt)");
   EXPECT_EQ(3, p.get_attempt_count(), "ERROR: should have made 3 attempts");

    // Fourth attempt should succeed (count = 3, odd)
    auto opt_sub4 = oneapi::dpl::experimental::try_submit(p, [](int i) { return i; });
   EXPECT_TRUE((bool)opt_sub4, "ERROR: fourth try_submit should succeed (odd attempt)");
   EXPECT_EQ(4, p.get_attempt_count(), "ERROR: should have made 4 attempts");

    std::cout << "  Testing submit with intermittent failures (should retry automatically)...\n";
    
    // Reset and test submit - it should retry internally and eventually succeed
    p.reset_attempt_count();
    auto sub = oneapi::dpl::experimental::submit(p, [](int i) { return i; });
    // Submit should have retried once after the first failure, succeeding on second attempt
    sub.wait();
    EXPECT_EQ(2, p.get_attempt_count(), "ERROR: submit should have made exactly 2 attempts (retry logic)");
    
    std::cout << "  Testing submit_and_wait with intermittent failures...\n";
    p.reset_attempt_count();
    oneapi::dpl::experimental::submit_and_wait(p, [](int i) { return i; });
    EXPECT_EQ(2, p.get_attempt_count(), "ERROR: submit_and_wait should have made exactly 2 attempts");
    std::cout << "test_intermittent_failure: OK\n";

    return 0;
}

int
main()
{
    try
    {
        test_intermittent_failure();
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
