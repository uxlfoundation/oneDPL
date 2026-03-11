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

using philox2x32 = oneapi::dpl::philox_engine<std::uint_fast32_t, 32, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64 = oneapi::dpl::philox_engine<std::uint_fast64_t, 64, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;

template <int _N>
using philox2x32_vec = oneapi::dpl::philox_engine<sycl::vec<uint_fast32_t, _N>, 32, 2, 10, 0xD2511F53, 0x9E3779B9>;
template <int _N>
using philox2x64_vec =
    oneapi::dpl::philox_engine<sycl::vec<uint_fast64_t, _N>, 64, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
#endif // TEST_UNNAMED_LAMBDAS

/* Declarations for Philox engine with non-standard word size */
using philox2x32_w5 = oneapi::dpl::philox_engine<std::uint_fast32_t, 5, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64_w5 = oneapi::dpl::philox_engine<std::uint_fast64_t, 5, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
using philox4x32_w5 =
    oneapi::dpl::philox_engine<std::uint_fast32_t, 5, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x64_w5 = oneapi::dpl::philox_engine<std::uint_fast64_t, 5, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                                 0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

using philox2x32_w15 = oneapi::dpl::philox_engine<std::uint_fast32_t, 15, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64_w15 =
    oneapi::dpl::philox_engine<std::uint_fast64_t, 15, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
using philox4x32_w15 =
    oneapi::dpl::philox_engine<std::uint_fast32_t, 15, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x64_w15 = oneapi::dpl::philox_engine<std::uint_fast64_t, 15, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                                  0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

using philox2x32_w18 = oneapi::dpl::philox_engine<std::uint_fast32_t, 18, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox2x64_w18 =
    oneapi::dpl::philox_engine<std::uint_fast64_t, 18, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
using philox4x32_w18 =
    oneapi::dpl::philox_engine<std::uint_fast32_t, 18, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
using philox4x64_w18 = oneapi::dpl::philox_engine<std::uint_fast64_t, 18, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                                  0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

using philox2x32_w30 = oneapi::dpl::philox_engine<std::uint_fast32_t, 30, 2, 10, 0xD2511F53, 0x9E3779B9>;
using philox4x32_w30 =
    oneapi::dpl::philox_engine<std::uint_fast32_t, 30, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;

using philox2x64_w25 =
    oneapi::dpl::philox_engine<std::uint_fast64_t, 25, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
using philox4x64_w25 = oneapi::dpl::philox_engine<std::uint_fast64_t, 25, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                                  0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

using philox2x64_w49 =
    oneapi::dpl::philox_engine<std::uint_fast64_t, 49, 2, 10, 0xD2B74407B1CE6E93, 0x9E3779B97F4A7C15>;
using philox4x64_w49 = oneapi::dpl::philox_engine<std::uint_fast64_t, 49, 4, 10, 0xCA5A826395121157, 0x9E3779B97F4A7C15,
                                                  0xD2E7470EE14C6C93, 0xBB67AE8584CAA73B>;

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

    // Reference values, generated using std::philox_engine from libstdc++
    std::uint_fast32_t philox2_32_w5_ref = 15U;
    std::uint_fast64_t philox2_64_w5_ref = 1U;
    std::uint_fast32_t philox4_32_w5_ref = 3U;
    std::uint_fast64_t philox4_64_w5_ref = 22U;
    // Reference values for w=15
    std::uint_fast32_t philox2_32_w15_ref = 14415U;
    std::uint_fast64_t philox2_64_w15_ref = 31535U;
    std::uint_fast32_t philox4_32_w15_ref = 14456U;
    std::uint_fast64_t philox4_64_w15_ref = 7360U;
    // Reference values for w=18
    std::uint_fast32_t philox2_32_w18_ref = 171106U;
    std::uint_fast64_t philox2_64_w18_ref = 233114U;
    std::uint_fast32_t philox4_32_w18_ref = 677U;
    std::uint_fast64_t philox4_64_w18_ref = 46070U;
    // Reference values for w=30
    std::uint_fast32_t philox2_32_w30_ref = 727289928U;
    std::uint_fast32_t philox4_32_w30_ref = 915043975U;
    // Reference values for w=25
    std::uint_fast64_t philox2_64_w25_ref = 9005821U;
    std::uint_fast64_t philox4_64_w25_ref = 28261501U;
    // Reference values for w=49
    std::uint_fast64_t philox2_64_w49_ref = 311726971455743U;
    std::uint_fast64_t philox4_64_w49_ref = 224210067519518U;

    int err = 0;

    // Generate 10 000th element for philox4_32
    err += test<oneapi::dpl::philox4x32, 10000, 1>(queue) != philox4_32_ref;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::philox4x32_vec<1>, 10000, 1>(queue) != philox4_32_ref;
    err += test<oneapi::dpl::philox4x32_vec<2>, 10000, 2>(queue) != philox4_32_ref;
    // In case of philox4x32_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::philox4x32_vec<3>, 10002, 3>(queue) != philox4_32_ref;
    err += test<oneapi::dpl::philox4x32_vec<4>, 10000, 4>(queue) != philox4_32_ref;
    err += test<oneapi::dpl::philox4x32_vec<8>, 10000, 8>(queue) != philox4_32_ref;
    err += test<oneapi::dpl::philox4x32_vec<16>, 10000, 16>(queue) != philox4_32_ref;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    // Generate 10 000th element for philox4_64
    err += test<oneapi::dpl::philox4x64, 10000, 1>(queue) != philox4_64_ref;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::philox4x64_vec<1>, 10000, 1>(queue) != philox4_64_ref;
    err += test<oneapi::dpl::philox4x64_vec<2>, 10000, 2>(queue) != philox4_64_ref;
    // In case of philox4x64_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::philox4x64_vec<3>, 10002, 3>(queue) != philox4_64_ref;
    err += test<oneapi::dpl::philox4x64_vec<4>, 10000, 4>(queue) != philox4_64_ref;
    err += test<oneapi::dpl::philox4x64_vec<8>, 10000, 8>(queue) != philox4_64_ref;
    err += test<oneapi::dpl::philox4x64_vec<16>, 10000, 16>(queue) != philox4_64_ref;
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

    // Generate 10 000th element for philox with non-standard word size
    // w=5
    err += test<philox2x32_w5, 10000, 1>(queue) != philox2_32_w5_ref;
    err += test<philox2x64_w5, 10000, 1>(queue) != philox2_64_w5_ref;
    err += test<philox4x32_w5, 10000, 1>(queue) != philox4_32_w5_ref;
    err += test<philox4x64_w5, 10000, 1>(queue) != philox4_64_w5_ref;
    EXPECT_TRUE(!err, "Test FAILED");
    // w=15
    err += test<philox2x32_w15, 10000, 1>(queue) != philox2_32_w15_ref;
    err += test<philox2x64_w15, 10000, 1>(queue) != philox2_64_w15_ref;
    err += test<philox4x32_w15, 10000, 1>(queue) != philox4_32_w15_ref;
    err += test<philox4x64_w15, 10000, 1>(queue) != philox4_64_w15_ref;
    EXPECT_TRUE(!err, "Test FAILED");
    // w=18
    err += test<philox2x32_w18, 10000, 1>(queue) != philox2_32_w18_ref;
    err += test<philox2x64_w18, 10000, 1>(queue) != philox2_64_w18_ref;
    err += test<philox4x32_w18, 10000, 1>(queue) != philox4_32_w18_ref;
    err += test<philox4x64_w18, 10000, 1>(queue) != philox4_64_w18_ref;
    EXPECT_TRUE(!err, "Test FAILED");
    // w=30
    err += test<philox2x32_w30, 10000, 1>(queue) != philox2_32_w30_ref;
    err += test<philox4x32_w30, 10000, 1>(queue) != philox4_32_w30_ref;
    EXPECT_TRUE(!err, "Test FAILED");
    // w=25
    err += test<philox2x64_w25, 10000, 1>(queue) != philox2_64_w25_ref;
    err += test<philox4x64_w25, 10000, 1>(queue) != philox4_64_w25_ref;
    EXPECT_TRUE(!err, "Test FAILED");
    // w=49
    err += test<philox2x64_w49, 10000, 1>(queue) != philox2_64_w49_ref;
    err += test<philox4x64_w49, 10000, 1>(queue) != philox4_64_w49_ref;
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
