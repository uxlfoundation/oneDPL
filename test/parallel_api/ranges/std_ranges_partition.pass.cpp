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

#include <algorithm>
#include <functional>
#include <ranges>
#include <utility>

namespace test_std_ranges
{

// std::ranges::partition is not a stable algorithm: it only guarantees that the elements satisfying
// the predicate precede those that do not. The parallel and device specializations rearrange the
// elements differently from std::ranges::partition, so the element-wise comparison against the
// reference implementation, which the test harness performs, is only meaningful for data where the
// partitioned sequence is unique. That is why every data generator below produces just two distinct
// values: one satisfying the tested predicate and one not satisfying it. The generators differ in
// how these values are distributed over the sequence, which exercises different balances of the
// partition implementation.

// pred1 (val > 0): alternating, blocked, all-true, all-false and almost-all-true/false patterns.
auto gen_alternate = [](auto i) { return i % 2 ? 1 : 0;        };
auto gen_blocked   = [](auto i) { return (i / 64) % 2 ? 1 : 0; };
auto gen_all_true  = [](auto)   { return 1;                    };
auto gen_all_false = [](auto)   { return 0;                    };
auto gen_one_true  = [](auto i) { return i == 0 ? 1 : 0;       };
auto gen_one_false = [](auto i) { return i == 0 ? 0 : 1;       };

// pred2 (val == 4) with the identity projection.
auto gen_eq4 = [](auto i) { return i % 3 ? 7 : 4; };

// pred2 (val == 4) with the 'proj' projection (val * 2).
auto gen_eq4_proj = [](auto i) { return i % 3 ? 7 : 2; };

// pred3 (val < 0).
auto gen_negative = [](auto i) { return i % 2 ? -5 : 5; };

// A wrapper around the tested algorithm which is passed to the harness instead of the algorithm
// itself: besides the element-wise comparison against std::ranges::partition made by the harness, it
// verifies the partition post-conditions, which do not depend on a particular permutation produced
// by the parallel or the device implementation.
struct partition_checked_fn
{
    template <typename Policy, typename R, typename... Args>
    std::ranges::borrowed_subrange_t<R>
    operator()(Policy&& exec, R&& r, Args... args) const
    {
        std::ranges::borrowed_subrange_t<R> res =
            oneapi::dpl::ranges::partition(std::forward<Policy>(exec), std::forward<R>(r), args...);

        // An r-value range is only used in an unevaluated context (a return type check), so the data
        // is available for inspection whenever the range is passed as an l-value.
        if constexpr (std::is_lvalue_reference_v<R>)
        {
            EXPECT_TRUE(std::ranges::is_partitioned(r, args...), "the range is not partitioned");

            if constexpr (std::ranges::borrowed_range<R>)
            {
                EXPECT_TRUE(std::ranges::end(res) == std::ranges::end(r), "wrong end of the returned subrange");
                EXPECT_TRUE(std::ranges::none_of(res, args...),
                            "the returned subrange contains elements satisfying the predicate");
            }
        }

        return res;
    }
};

inline constexpr partition_checked_fn partition_checked{};

} // namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto partition_checker = TEST_PREPARE_CALLABLE(std::ranges::partition);

    // Different data generators with the same predicate: balanced, blocked and degenerate cases.
    test_range_algo<0, int, data_in, decltype(gen_alternate)>{big_sz}(partition_checked, partition_checker, pred1);
    test_range_algo<1, int, data_in, decltype(gen_blocked  )>{      }(partition_checked, partition_checker, pred1);
    test_range_algo<2, int, data_in, decltype(gen_all_true )>{      }(partition_checked, partition_checker, pred1);
    test_range_algo<3, int, data_in, decltype(gen_all_false)>{      }(partition_checked, partition_checker, pred1);
    test_range_algo<4, int, data_in, decltype(gen_one_true )>{      }(partition_checked, partition_checker, pred1);
    test_range_algo<5, int, data_in, decltype(gen_one_false)>{      }(partition_checked, partition_checker, pred1);

    // Other predicates.
    test_range_algo<6, int, data_in, decltype(gen_eq4     )>{}(partition_checked, partition_checker, pred2);
    test_range_algo<7, int, data_in, decltype(gen_negative)>{}(partition_checked, partition_checker, pred3);

    // Projections: a callable one and the pointer-to-data-member/pointer-to-member-function ones.
    test_range_algo<8, int, data_in, decltype(gen_eq4_proj )>{}(partition_checked, partition_checker, pred2, proj);
    test_range_algo<9,  P2, data_in, decltype(gen_alternate)>{}(partition_checked, partition_checker, pred1, &P2::x);
    test_range_algo<10, P2, data_in, decltype(gen_blocked  )>{}(partition_checked, partition_checker, pred1, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
