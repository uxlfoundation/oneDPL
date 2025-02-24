//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_bessel.h"

void
test01()
{
    float xf = std::numeric_limits<float>::quiet_NaN();
    double xd = std::numeric_limits<double>::quiet_NaN();

    float nuf = 0.0F;
    double nud = 0.0;

    float a = std::cyl_bessel_i(nuf, xf);
    float b = std::cyl_bessel_if(nuf, xf);
    double c = std::cyl_bessel_i(nud, xd);

    assert(std::isnan(a));
    assert(std::isnan(b));
    assert(std::isnan(c));
}

void
test02()
{
    float xf = 1.0F;
    double xd = 1.0;

    float nuf = std::numeric_limits<float>::quiet_NaN();
    double nud = std::numeric_limits<double>::quiet_NaN();

    float a = std::cyl_bessel_i(nuf, xf);
    float b = std::cyl_bessel_if(nuf, xf);
    double c = std::cyl_bessel_i(nud, xd);

    assert(std::isnan(a));
    assert(std::isnan(b));
    assert(std::isnan(c));
}

ONEDPL_TEST_NUM_MAIN
{
    IF_DOUBLE_SUPPORT(test01())
    IF_DOUBLE_SUPPORT(test02())

    return 0;
}
