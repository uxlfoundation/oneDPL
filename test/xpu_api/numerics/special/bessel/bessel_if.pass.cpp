//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_bessel.h"

//  cyl_bessel_if
#include <cmath>

void
test(float)
{
    float nuf = 1.0F / 3.0F, xf = 0.5F;

    std::cyl_bessel_if(nuf, xf);
}

void
test(double)
{
    double nud = 1.0 / 3.0, xd = 0.5;

    std::cyl_bessel_if(nud, xd);
}

ONEDPL_TEST_NUM_MAIN
{
    test(float{});
    IF_DOUBLE_SUPPORT(test(double{}))

    return 0;
}
