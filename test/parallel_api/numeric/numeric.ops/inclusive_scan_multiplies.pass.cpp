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

#include <random>
#include <cstdint>

using namespace TestUtils;

template <typename T>
void
test_with_multiplies()
{
#if TEST_DPCPP_BACKEND_PRESENT
    T init = 1;
    const std::size_t custom_item_count = 10;

    for (size_t n = custom_item_count; n <= TestUtils::get_scan_test_max_n(); n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> out(n);
        Sequence<T> expected(n);

        auto one = [](size_t /*index*/) { return 1; };
        Sequence<T> in(n, one);
        std::size_t counter = 0;
        std::generate_n(in.begin(), custom_item_count, [&counter]() { return (counter++) % 3 + 2; } );
        std::default_random_engine gen{42};
        std::shuffle(in.begin(), in.end(), gen);

        invoke_on_all_hetero_policies<20>()(test_inclusive_scan_with_binary_op<>(), in.begin(), in.end(),
                                            out.begin(), out.end(), expected.begin(), in.size(), init,
                                            std::multiplies{}, T{0xDEAD}, one);
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
}

int
main()
{
    test_with_multiplies<std::uint64_t>();

    return done(TEST_DPCPP_BACKEND_PRESENT);
}
