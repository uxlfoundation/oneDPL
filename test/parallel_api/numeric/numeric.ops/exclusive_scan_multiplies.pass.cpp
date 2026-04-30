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

#include <random>
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace TestUtils;

template <typename Type>
struct test_exclusive_scan_with_binary_op
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        exclusive_scan_serial(in_first, in_last, expected_first, init, binary_op);

        auto orr = exclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, init, binary_op);

        EXPECT_TRUE(out_last == orr, "exclusive_scan with binary operator returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from exclusive_scan with binary operator");
        std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */,
               Iterator2 /* out_last */, Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */,
               T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

template <typename T>
void
test_with_multiplies()
{
#if TEST_DPCPP_BACKEND_PRESENT
    T trash = 666;
    T init = 1;
    const std::size_t custom_item_count = 10;

    for (size_t n = custom_item_count; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> out(n, [&](size_t) { return trash; });
        Sequence<T> expected(n, [&](size_t) { return trash; });

        Sequence<T> in(n, [](size_t /*index*/) { return 1; });
        std::size_t counter = 0;
        std::generate_n(in.begin(), custom_item_count, [&counter]() { return (counter++) % 3 + 2; });
        std::default_random_engine gen{42};
        std::shuffle(in.begin(), in.end(), gen);

        invoke_on_all_hetero_policies<21>()(test_exclusive_scan_with_binary_op<T>(), in.begin(), in.end(), out.begin(),
                                            out.end(), expected.begin(), expected.end(), in.size(), init,
                                            std::multiplies{}, trash);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
}

int
main()
{
    test_with_multiplies<std::uint64_t>();

    return done();
}
