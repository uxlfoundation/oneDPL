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

namespace ex = oneapi::dpl::experimental;

using philox2x32 = ex::philox_engine<std::uint_fast32_t, 32, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64 = ex::philox_engine<std::uint_fast64_t, 64, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;

template <int _N>
using philox2x32_vec = ex::philox_engine<sycl::vec<uint_fast32_t, _N>, 32, 2, 10, 0xD2511F53, 0x9E3779B9>;

template <int _N>
using philox2x64_vec =
    ex::philox_engine<sycl::vec<uint_fast64_t, _N>, 64, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
#endif // TEST_UNNAMED_LAMBDAS

int
main()
{

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Reference values from p2075 paper series
    std::uint_fast32_t philox4_32_ref = 1955073260U;
    std::uint_fast64_t philox4_64_ref = 3409172418970261260U;
    // Reference values, generated using reference implementation from https://github.com/DEShawResearch/random123
    std::uint_fast32_t philox2_32_ref = 2942762615U;
    std::uint_fast64_t philox2_64_ref = 14685864013162917916U;

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

    // Generate 10 000th element for philox2_32
    err += test<philox2x32, 10000, 1>(queue) != philox2_32_ref;
#if TEST_LONG_RUN
    err += test<philox2x32_vec<1>, 10000, 1>(queue) != philox2_32_ref;
    err += test<philox2x32_vec<2>, 10000, 2>(queue) != philox2_32_ref;
    // In case of philox2x32_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<philox2x32_vec<3>, 10002, 3>(queue) != philox2_32_ref;
    err += test<philox2x32_vec<4>, 10000, 4>(queue) != philox2_32_ref;
    err += test<philox2x32_vec<8>, 10000, 8>(queue) != philox2_32_ref;
    err += test<philox2x32_vec<16>, 10000, 16>(queue) != philox2_32_ref;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // Generate 10 000th element for philox2_64
    err += test<philox2x64, 10000, 1>(queue) != philox2_64_ref;
#if TEST_LONG_RUN
    err += test<philox2x64_vec<1>, 10000, 1>(queue) != philox2_64_ref;
    err += test<philox2x64_vec<2>, 10000, 2>(queue) != philox2_64_ref;
    // In case of philox2x64_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<philox2x64_vec<3>, 10002, 3>(queue) != philox2_64_ref;
    err += test<philox2x64_vec<4>, 10000, 4>(queue) != philox2_64_ref;
    err += test<philox2x64_vec<8>, 10000, 8>(queue) != philox2_64_ref;
    err += test<philox2x64_vec<16>, 10000, 16>(queue) != philox2_64_ref;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
