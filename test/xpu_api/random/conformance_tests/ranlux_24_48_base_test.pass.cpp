// -*- C++ -*-
//===-- ranlux_24_48_base_test.cpp ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Abstract:
//
// Test of ranlux24_base and ranlux48_base engines - comparison of 10 000th element
//
// using ranlux24_base = subtract_with_carry_engine<uint_fast32_t, 24, 10, 24>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux24_base
//     produces the value 7937952.
//
// using ranlux48_base = subtract_with_carry_engine<uint_fast64_t, 48, 5, 12>;
// Required behavior:
//     The 10000th consecutive invocation of a default-constructed object of type ranlux48_base
//     produces the value 61839128582725

#include "support/utils.h"

#if TEST_UNNAMED_LAMBDAS
#include "common_for_conformance_tests.hpp"
#include <oneapi/dpl/random>
#endif // TEST_UNNAMED_LAMBDAS

int main() {

#if TEST_UNNAMED_LAMBDAS

    sycl::queue queue = TestUtils::get_test_queue();

    // Reference values
    uint_fast32_t ranlux24_base_ref_sample = 7937952;
    uint_fast64_t ranlux48_base_ref_sample = 61839128582725;
    int err = 0;

    // Generate 10 000th element for ranlux24_base
    err += test<oneapi::dpl::ranlux24_base,        10000, 1>(queue)  != ranlux24_base_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::ranlux24_base_vec<1>, 10000, 1>(queue)  != ranlux24_base_ref_sample;
    err += test<oneapi::dpl::ranlux24_base_vec<2>, 10000, 2>(queue)  != ranlux24_base_ref_sample;
    // In case of ranlux24_base_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::ranlux24_base_vec<3>, 10002, 3>(queue)  != ranlux24_base_ref_sample;
    err += test<oneapi::dpl::ranlux24_base_vec<4>, 10000, 4>(queue)  != ranlux24_base_ref_sample;
    err += test<oneapi::dpl::ranlux24_base_vec<8>, 10000, 8>(queue)  != ranlux24_base_ref_sample;
    err += test<oneapi::dpl::ranlux24_base_vec<16>,10000, 16>(queue) != ranlux24_base_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

    err += test<oneapi::dpl::ranlux48_base,        10000, 1>(queue)  != ranlux48_base_ref_sample;
#if TEST_LONG_RUN
    err += test<oneapi::dpl::ranlux48_base_vec<1>, 10000, 1>(queue)  != ranlux48_base_ref_sample;
    err += test<oneapi::dpl::ranlux48_base_vec<2>, 10000, 2>(queue)  != ranlux48_base_ref_sample;
    // In case of ranlux48_base_vec<3> engine generate 10002 values as 10000 % 3 != 0
    err += test<oneapi::dpl::ranlux48_base_vec<3>, 10002, 3>(queue)  != ranlux48_base_ref_sample;
    err += test<oneapi::dpl::ranlux48_base_vec<4>, 10000, 4>(queue)  != ranlux48_base_ref_sample;
    err += test<oneapi::dpl::ranlux48_base_vec<8>, 10000, 8>(queue)  != ranlux48_base_ref_sample;
    err += test<oneapi::dpl::ranlux48_base_vec<16>,10000, 16>(queue) != ranlux48_base_ref_sample;
#endif // TEST_LONG_RUN
    EXPECT_TRUE(!err, "Test FAILED");

#endif // TEST_UNNAMED_LAMBDAS

    return TestUtils::done(TEST_UNNAMED_LAMBDAS);
}
