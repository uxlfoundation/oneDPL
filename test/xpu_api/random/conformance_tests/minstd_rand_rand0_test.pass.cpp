// -*- C++ -*-
//===-- minstd_rand_rand0_test.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of minst_rand and minstd_rand0 engines - comparison of 10 000th element
//
// using minstd_rand0 = linear_congruential_engine<uint_fast32_t, 16’807, 0, 2’147’483’647>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type minstd_rand0
//     produces the value 1043618065.
//
// using minstd_rand = linear_congruential_engine<uint_fast32_t, 48’271, 0, 2’147’483’647>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type minstd_rand
//     produces the value 399268537

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Reference values
    uint_fast32_t minstd_rand0_ref_sample = 1043618065;
    uint_fast32_t minstd_rand_ref_sample = 399268537;
    int err = 0;

    // Generate 10 000th element for minstd_rand0
    err += test<oneapi::dpl::minstd_rand0,        10000, 1>(queue)  != minstd_rand0_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::minstd_rand0_vec<1>, 10000, 1>(queue)  != minstd_rand0_ref_sample;
    err += test<oneapi::dpl::minstd_rand0_vec<2>, 10000, 2>(queue)  != minstd_rand0_ref_sample;
    // In case of minstd_rand0_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::minstd_rand0_vec<3>, 10002, 3>(queue)  != minstd_rand0_ref_sample;
    err += test<oneapi::dpl::minstd_rand0_vec<4>, 10000, 4>(queue)  != minstd_rand0_ref_sample;
    err += test<oneapi::dpl::minstd_rand0_vec<8>, 10000, 8>(queue)  != minstd_rand0_ref_sample;
    err += test<oneapi::dpl::minstd_rand0_vec<16>,10000, 16>(queue) != minstd_rand0_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    err += test<oneapi::dpl::minstd_rand,        10000, 1>(queue)  != minstd_rand_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::minstd_rand_vec<1>, 10000, 1>(queue)  != minstd_rand_ref_sample;
    err += test<oneapi::dpl::minstd_rand_vec<2>, 10000, 2>(queue)  != minstd_rand_ref_sample;
    // In case of minstd_rand_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::minstd_rand_vec<3>, 10002, 3>(queue)  != minstd_rand_ref_sample;
    err += test<oneapi::dpl::minstd_rand_vec<4>, 10000, 4>(queue)  != minstd_rand_ref_sample;
    err += test<oneapi::dpl::minstd_rand_vec<8>, 10000, 8>(queue)  != minstd_rand_ref_sample;
    err += test<oneapi::dpl::minstd_rand_vec<16>,10000, 16>(queue) != minstd_rand_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
