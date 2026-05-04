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

template <typename In, typename Out, typename BinaryOp>
void
test_matrix(Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k, k + 1); });

        Sequence<Out> out(n, [&](size_t) { return trash; });
        Sequence<Out> expected(n, [&](size_t) { return trash; });

        auto __scan_invoker = [&](Sequence<Out>& out) {
        invoke_on_all_policies<4>()(test_inclusive_scan_with_binary_op<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<5>()(test_inclusive_scan_with_binary_op<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<6>()(test_inclusive_scan_with_binary_op<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), binary_op, trash);
        invoke_on_all_policies<7>()(test_inclusive_scan_with_binary_op<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), binary_op, trash);
        };

        //perform regular a scan algorithm
        __scan_invoker(out);

        //perform an in-place scan algorithm
        __scan_invoker(in);
    }
}

int
main()
{
#if !_PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN
    // Test with highly restricted type and associative but not commutative operation
    test_matrix<Matrix2x2<std::int32_t>, Matrix2x2<std::int32_t>>(Matrix2x2<std::int32_t>(), multiply_matrix<std::int32_t>(),
                                                            Matrix2x2<std::int32_t>(-666, 666));
#endif

    return done();
}
