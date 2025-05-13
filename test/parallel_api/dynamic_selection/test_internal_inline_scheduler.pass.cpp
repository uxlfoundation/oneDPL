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
#include "support/utils.h"

#include <atomic>
#include <iostream>
#include <sstream>

class fake_selection_handle_t
{
    int q_;

  public:
    fake_selection_handle_t(int q = 0) : q_(q) {}
    auto
    unwrap()
    {
        return q_;
    }
};

int
test_cout()
{
    TestUtils::int_inline_backend_t s;
    TestUtils::int_inline_backend_t::execution_resource_t e;
    return 0;
}

int
test_submit_and_wait_on_submission_group()
{
    const int N = 100;
    TestUtils::int_inline_backend_t s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        s.submit(h,
                 [&](int, int i) {
                     ecount += i;
                     return 0;
                 },
                 i);
    }
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(N * (N + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler: OK\n";
    return 0;
}

int
test_submit_and_wait_on_submission_group_single_element()
{
    const int N = 1;
    TestUtils::int_inline_backend_t s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        s.submit(h,
                 [&](int, int i) {
                     ecount += i;
                     return 0;
                 },
                 i);
    }
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(1, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler single element: OK\n";
    return 0;
}

int
test_submit_and_wait_on_submission_group_empty()
{
    TestUtils::int_inline_backend_t s;
    std::atomic<int> ecount = 0;
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(0, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler empty list: OK\n";
    return 0;
}

int
test_submit_and_wait_on_submission()
{
    const int N = 100;
    TestUtils::int_inline_backend_t s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto w = s.submit(h,
                          [&](int, int i) {
                              ecount += i;
                              return 0;
                          },
                          i);
        w.wait();
        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    std::cout << "wait_on_sync: OK\n";
    return 0;
}

int
test_submit_and_wait_on_submission_single_element()
{
    const int N = 1;
    TestUtils::int_inline_backend_t s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto w = s.submit(h,
                          [&](int, int i) {
                              ecount += i;
                              return 0;
                          },
                          i);
        w.wait();
        int count = ecount.load();
        EXPECT_EQ(1, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    std::cout << "wait_on_sync single element: OK\n";
    return 0;
}

int
test_properties()
{
    std::vector<int> v = {1, 2};
    TestUtils::int_inline_backend_t s(v);
    auto v2 = s.get_resources();
    auto v2s = v2.size();
    EXPECT_EQ(v2s, v.size(), "ERROR: universe size inconsistent with queried universe\n");

    for (int i = 0; i < v2s; ++i)
    {
        EXPECT_EQ(v[i], oneapi::dpl::experimental::unwrap(v2[i]),
                  "ERROR: reported universe and queried universe are not equal\n");
    }
    std::cout << "properties: OK\n";
    return 0;
}

int
main()
{
    EXPECT_EQ(0, (test_cout()), "test_cout failed");
    EXPECT_EQ(0, (test_submit_and_wait_on_submission_group()), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_submission_group_single_element()), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_submission_group_empty()), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_submission()), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_submission_single_element()), "");
    EXPECT_EQ(0, (test_properties()), "");

    return TestUtils::done();
}
