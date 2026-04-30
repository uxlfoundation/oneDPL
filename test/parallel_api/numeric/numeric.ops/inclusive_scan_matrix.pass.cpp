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
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

#include <cstdint>

using namespace TestUtils;

template <typename Type>
struct test_inclusive_scan_with_binary_op
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        inclusive_scan_serial(in_first, in_last, expected_first, binary_op, init);
        auto orr = inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, binary_op, init);

        EXPECT_TRUE(out_last == orr, "inclusive_scan with binary operator returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan with binary operator");
        std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, BinaryOp binary_op, T trash)
    {
        using namespace std;

        inclusive_scan_serial(in_first, in_last, expected_first, binary_op);
        auto orr = inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, binary_op);

        EXPECT_TRUE(out_last == orr, "inclusive_scan with binary operator without init returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan with binary operator without init");
        std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */, Iterator2 /* out_last */,
               Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */, BinaryOp /* binary_op */, T /* trash */)
    {
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */, Iterator2 /* out_last */,
               Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

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
