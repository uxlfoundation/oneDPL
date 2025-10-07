// -*- C++ -*-
//===-- ranlux_24_48_test.cpp ----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of ranlux24 and ranlux48 engines - comparison of 10 000th element
//
// using ranlux24 = discard_block_engine<ranlux24_base, 223, 23>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux24
//     produces the value 9901578.
//
// using ranlux48 = discard_block_engine<ranlux48_base, 389, 11>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux48
//     produces the value 249142670248501

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Reference values
    uint_fast32_t ranlux24_ref_sample = 9901578;
    uint_fast64_t ranlux48_ref_sample = 249142670248501;
    int err = 0;

    // Generate 10 000th element for ranlux24
    err += test<oneapi::dpl::ranlux24,        10000, 1>(queue)  !=ranlux24_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::ranlux24_vec<1>, 10000, 1>(queue)  !=ranlux24_ref_sample;
    err += test<oneapi::dpl::ranlux24_vec<2>, 10000, 2>(queue)  !=ranlux24_ref_sample;
    // In case of ranlux24_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::ranlux24_vec<3>, 10002, 3>(queue)  !=ranlux24_ref_sample;
    err += test<oneapi::dpl::ranlux24_vec<4>, 10000, 4>(queue)  !=ranlux24_ref_sample;
    err += test<oneapi::dpl::ranlux24_vec<8>, 10000, 8>(queue)  !=ranlux24_ref_sample;
    err += test<oneapi::dpl::ranlux24_vec<16>,10000, 16>(queue) !=ranlux24_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    err += test<oneapi::dpl::ranlux48,        10000, 1>(queue)  !=ranlux48_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::ranlux48_vec<1>, 10000, 1>(queue)  !=ranlux48_ref_sample;
    err += test<oneapi::dpl::ranlux48_vec<2>, 10000, 2>(queue)  !=ranlux48_ref_sample;
    // In case of ranlux48_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::ranlux48_vec<3>, 10002, 3>(queue)  !=ranlux48_ref_sample;
    err += test<oneapi::dpl::ranlux48_vec<4>, 10000, 4>(queue)  !=ranlux48_ref_sample;
    err += test<oneapi::dpl::ranlux48_vec<8>, 10000, 8>(queue)  !=ranlux48_ref_sample;
    err += test<oneapi::dpl::ranlux48_vec<16>,10000, 16>(queue) !=ranlux48_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
