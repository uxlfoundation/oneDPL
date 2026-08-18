// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING

// The test harness compares the whole target/reference ranges element-wise. The algorithm is unstable.
// To guarantee the equivalence, generate only one unique value per true/false partitions.

// pred1 (val > 0).
struct gen_alternate { int operator()(auto i) const { return i % 2 ? 1 : 0;        } };
struct gen_blocked   { int operator()(auto i) const { return (i / 64) % 2 ? 1 : 0; } };
struct gen_all_true  { int operator()(auto)   const { return 1;                    } };
struct gen_all_false { int operator()(auto)   const { return 0;                    } };
struct gen_one_true  { int operator()(auto i) const { return i == 0 ? 1 : 0;       } };
struct gen_one_false { int operator()(auto i) const { return i == 0 ? 0 : 1;       } };

// pred2 (val == 4) with the identity projection.
struct gen_eq4 { int operator()(auto i) const { return i % 3 ? 7 : 4; } };

// pred2 (val == 4) with the 'proj' projection (val * 2).
struct gen_eq4_proj { int operator()(auto i) const { return i % 3 ? 7 : 2; } };

// pred3 (val < 0).
struct gen_negative { int operator()(auto i) const { return i % 2 ? -5 : 5; } };

#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto partition_checker = TEST_PREPARE_CALLABLE(std::ranges::partition);

    // Different data generators with the same predicate: balanced, blocked and degenerate cases.
    test_range_algo<0, int, data_in, gen_alternate>{big_sz}(dpl_ranges::partition, partition_checker, pred1);
#if ONEDPL_STD_RANGES_TEST_ALL_PERMUTATIONS
    test_range_algo<1, int, data_in, gen_blocked>{}(dpl_ranges::partition, partition_checker, pred1);
    test_range_algo<2, int, data_in, gen_all_true>{}(dpl_ranges::partition, partition_checker, pred1);
    test_range_algo<3, int, data_in, gen_all_false>{}(dpl_ranges::partition, partition_checker, pred1);
    test_range_algo<4, int, data_in, gen_one_true>{}(dpl_ranges::partition, partition_checker, pred1);
    test_range_algo<5, int, data_in, gen_one_false>{}(dpl_ranges::partition, partition_checker, pred1);

    // Projections: a callable one and the pointer-to-data-member/pointer-to-member-function ones.
    test_range_algo<6, P2, data_in, gen_alternate>{}(dpl_ranges::partition, partition_checker, pred1, &P2::x);
    test_range_algo<7, P2, data_in, gen_blocked>{}(dpl_ranges::partition, partition_checker, pred1, &P2::proj);
#endif
    test_range_algo<8, int, data_in, gen_eq4_proj>{}(dpl_ranges::partition, partition_checker, pred2, proj);

    // Other predicates.
    test_range_algo<9, int, data_in, gen_eq4>{}(dpl_ranges::partition, partition_checker, pred2);
    test_range_algo<10, int, data_in, gen_negative>{}(dpl_ranges::partition, partition_checker, pred3);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
