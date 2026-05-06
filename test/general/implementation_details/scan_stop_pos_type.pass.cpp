// -*- C++ -*-
//===-- tuple_unit.pass.cpp -----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/execution>
#include "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h"
#endif

#include "support/utils.h"

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    using R1 = oneapi::dpl::__ranges::guard_view<int*>;
    using R2 = oneapi::dpl::__ranges::guard_view<float*>;
    using R3 = oneapi::dpl::__ranges::guard_view<double*>;

    // Derive zip_view type from make_zip_view � avoids hardcoding internal type names
    // and automatically matches the specialization in __scan_stop_pos_t
    using Zip2 = std::decay_t<decltype(oneapi::dpl::__ranges::make_zip_view(std::declval<R1>(), std::declval<R2>()))>;
    using Zip3 = std::decay_t<decltype(oneapi::dpl::__ranges::make_zip_view(std::declval<R1>(), std::declval<R2>(),
                                                                            std::declval<R3>()))>;

    // Scalar size type: what __size returns for a single plain range
    using SZ = decltype(oneapi::dpl::__ranges::__size(std::declval<R1>()));

    using oneapi::dpl::__par_backend_hetero::__scan_stop_pos_t;

    // 1. Single plain range -> scalar
    static_assert(std::is_same_v<__scan_stop_pos_t<R1>, std::tuple<SZ>>,
                  "Single range must yield tuple<SZ>");

    // 2. zip_view<R1,R2> -> tuple<SZ,SZ>
    static_assert(std::is_same_v<__scan_stop_pos_t<Zip2>, std::tuple<SZ, SZ>>,
                  "zip_view<R1,R2> must yield tuple<SZ,SZ>");

    // 3. zip_view<R1,R2,R3> -> tuple<SZ,SZ,SZ>
    static_assert(std::is_same_v<__scan_stop_pos_t<Zip3>, std::tuple<SZ, SZ, SZ>>,
                  "zip_view<R1,R2,R3> must yield tuple<SZ,SZ,SZ>");

    // 4. R1, R2 -> flat tuple<SZ,SZ> (NOT nested)
    static_assert(std::is_same_v<__scan_stop_pos_t<R1, R2>, std::tuple<SZ, SZ>>,
                  "Two plain ranges must yield flat tuple<SZ,SZ>");

    // 5. R1, R2, R3 -> flat tuple<SZ,SZ,SZ> (NOT nested)
    static_assert(std::is_same_v<__scan_stop_pos_t<R1, R2, R3>, std::tuple<SZ, SZ, SZ>>,
                  "Three plain ranges must yield flat tuple<SZ,SZ,SZ>");

    // 6. Zip2 + R3 -> zip grouped (nested), R3 flat
    static_assert(std::is_same_v<__scan_stop_pos_t<Zip2, R3>, std::tuple<std::tuple<SZ, SZ>, SZ>>,
                  "zip_view<R1,R2> + R3 must yield tuple<tuple<SZ,SZ>, SZ>");

    // 7. R1 + Zip2 -> R1 flat, zip nested at tail
    static_assert(std::is_same_v<__scan_stop_pos_t<R1, Zip2>, std::tuple<SZ, std::tuple<SZ, SZ>>>,
                  "R1 + zip_view<R1,R2> must yield tuple<SZ, tuple<SZ,SZ>>");

    // 8. Zip2 + R3 + R3 -> outer flat, zip nested at head
    static_assert(
        std::is_same_v<__scan_stop_pos_t<Zip2, R3, R3>, std::tuple<std::tuple<SZ, SZ>, SZ, SZ>>,
        "zip_view<R1,R2> + R3 + R3 must yield tuple<tuple<SZ,SZ>, SZ, SZ>");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}