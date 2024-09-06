// -*- C++ -*-
//===-- philox_test.pass.cpp ----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test for Philox random number generation engine - comparison of 10 000th element

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // TEST_UNNAMED_LAMBDAS

namespace ex = oneapi::dpl::experimental;

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Reference values
    std::uint_fast32_t philox4_32_ref = 1955073260;
    std::uint_fast64_t philox4_64_ref = 3409172418970261260;
    int err = 0;

    // Generate 10 000th element for philox4_32
    err += test<ex::philox4x32, 10000, 1>(queue) != philox4_32_ref;
#if TEST_LONG_RUN
    err += test<ex::philox4x32_vec<1>, 10000, 1>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<2>, 10000, 2>(queue) != philox4_32_ref;
    // In case of philox4x32_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<ex::philox4x32_vec<3>, 10002, 3>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<4>, 10000, 4>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<8>, 10000, 8>(queue) != philox4_32_ref;
    err += test<ex::philox4x32_vec<16>, 10000, 16>(queue) != philox4_32_ref;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // Generate 10 000th element for philox4_64
    err += test<ex::philox4x64, 10000, 1>(queue) != philox4_64_ref;
#if TEST_LONG_RUN
    err += test<ex::philox4x64_vec<1>, 10000, 1>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<2>, 10000, 2>(queue) != philox4_64_ref;
    // In case of philox4x64_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<ex::philox4x64_vec<3>, 10002, 3>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<4>, 10000, 4>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<8>, 10000, 8>(queue) != philox4_64_ref;
    err += test<ex::philox4x64_vec<16>, 10000, 16>(queue) != philox4_64_ref;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
