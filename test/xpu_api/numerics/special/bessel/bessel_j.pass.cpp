//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_bessel.h"

//  cyl_bessel_j
#include <cmath>

void
test()
{
    double nud = 1.0 / 3.0, xd = 0.5;

    std::cyl_bessel_j(nud, xd);
}

ONEDPL_TEST_NUM_MAIN
{
    IF_DOUBLE_SUPPORT(test())

    return 0;
}
