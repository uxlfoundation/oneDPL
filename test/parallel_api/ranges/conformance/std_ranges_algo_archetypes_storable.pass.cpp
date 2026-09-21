// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include "std_ranges_archetypes.h"
#include "std_ranges_algo_archetypes_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The storable archetype family: the algorithms which return an element by value and are therefore
    // constrained by std::indirectly_copyable_storable, i.e. min, max and minmax. Covers them first
    // with const comparators, then with comparators taking their arguments by non-const reference, and
    // finally without a comparator at all, i.e. with the default std::ranges::less.

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; }, "minmax");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    // KSATODO: the vectorized brick of min_element, __simd_min_element at unseq_backend_simd.h:654,655,
    // keeps the current extremum in a _ValueType copy and compares it with the copy held by another
    // _ComplexType, which the reduction hands over as a const lvalue; the scalar tail at
    // unseq_backend_simd.h:671 compares against a const local as well. std::indirect_strict_weak_order
    // only asks the comparator to accept iter_reference_t, i.e. a non-const lvalue, so a comparator
    // taking its arguments by non-const reference does not compile. Holding the reduction argument by
    // non-const reference, as the mutable state it in fact is, is enough to fix it.
    // The device side is conforming here, and so are the calls with the const comparator above.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto res) { return res.val == 0; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST
        run_algo_host_policies<storable_archetype>(call, check, "min, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 3>(call, check, "min, non-const comparator");
#endif
    }

    // KSATODO: max reaches the very same __simd_min_element with the comparator wrapped in
    // __reorder_pred (utils.h:116), so it hits the const lvalue of the note above.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MAX_HOST
        run_algo_host_policies<storable_archetype>(call, check, "max, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 4>(call, check, "max, non-const comparator");
#endif
    }

    // KSATODO: __simd_minmax_element (unseq_backend_simd.h:715,720,727,732,748,753) stores both extrema
    // the very same way as __simd_min_element, see the note above min.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MINMAX_HOST
        run_algo_host_policies<storable_archetype>(call, check, "minmax, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 5>(call, check, "minmax, non-const comparator");
#endif
    }

    //----------------------------------------------------------------------------------------------
    // The same algorithms called without a comparator at all, i.e. with the default
    // std::ranges::less, which requires the element type itself to be std::totally_ordered.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 6>(
        [](auto&& policy, auto&& view) { return dpl_ranges::min(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto res) { return res.val == 0; }, "min, default comparator");

    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 7>(
        [](auto&& policy, auto&& view) { return dpl_ranges::max(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max, default comparator");

    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 8>(
        [](auto&& policy, auto&& view) { return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; },
        "minmax, default comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
