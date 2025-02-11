//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_complex.h"

//  cyl_bessel_jl
#include <cmath>

void
test()
{
    long double nul = 1.0L / 3.0L, xl = 0.5L;

    std::cyl_bessel_jl(nul, xl);
}

ONEDPL_TEST_NUM_MAIN
{
    IF_LONG_DOUBLE_SUPPORT(test())

    return 0;
}
