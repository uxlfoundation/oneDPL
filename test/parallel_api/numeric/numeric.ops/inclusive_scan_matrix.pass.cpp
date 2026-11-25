// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"
#include "support/utils_scan.h"

#include <cstdint>

using namespace TestUtils;

template <typename Mtx, typename BinaryOp>
void
test_matrix(Mtx init, BinaryOp binary_op, Mtx trash)
{
    for (size_t n = 0; n <= TestUtils::get_scan_test_max_n(); n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        auto gen = [](size_t k) { return Mtx(k, k + 1); };
        Sequence<Mtx> in(n, gen);
        Sequence<Mtx> out(n);
        Sequence<Mtx> expected(n);

        invoke_on_all_policies<4>()(test_inclusive_scan_with_binary_op<Mtx>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<5>()(test_inclusive_scan_with_binary_op<Mtx>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<6>()(test_inclusive_scan_with_binary_op<Mtx>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), in.size(), binary_op, trash);
        invoke_on_all_policies<7>()(test_inclusive_scan_with_binary_op<Mtx>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), in.size(), binary_op, trash);

        invoke_on_all_policies<8>()(test_inclusive_scan_with_binary_op<Mtx, /*in_place*/true>(), in.begin(), in.end(),
                                    in.begin(), in.end(), expected.begin(), in.size(), init, binary_op, gen);
        invoke_on_all_policies<9>()(test_inclusive_scan_with_binary_op<Mtx, /*in_place*/true>(), in.begin(), in.end(),
                                    in.begin(), in.end(), expected.begin(), in.size(), binary_op, gen);
    }
}

int
main()
{
#if !_PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN
    // Test with highly restricted type and associative but not commutative operation
    test_matrix(Matrix2x2<std::int32_t>(), multiply_matrix<std::int32_t>(), Matrix2x2<std::int32_t>(-666, 666));
#endif

    return done();
}
