//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_bessel.h"

template <typename T>
void
test01();

template <typename T>
void
test02();

template <>
void
test01<float>()
{
    float xf = std::numeric_limits<float>::quiet_NaN();
    float nuf = 0.0F;

    float a = std::cyl_bessel_j(nuf, xf);
    float b = std::cyl_bessel_jf(nuf, xf);

    assert(std::isnan(a));
    assert(std::isnan(b));
}

template <>
void
test01<double>()
{
    double xd = std::numeric_limits<double>::quiet_NaN();
    double nud = 0.0;
    double c = std::cyl_bessel_j(nud, xd);

    assert(std::isnan(c));
}

template <>
void
test02<float>()
{
    float xf = 1.0F;
    float nuf = std::numeric_limits<float>::quiet_NaN();

    float a = std::cyl_bessel_j(nuf, xf);
    float b = std::cyl_bessel_jf(nuf, xf);

    assert(std::isnan(a));
    assert(std::isnan(b));
}

template <>
void
test02<double>()
{
    double xd = 1.0;
    double nud = std::numeric_limits<double>::quiet_NaN();
    double c = std::cyl_bessel_j(nud, xd);

    assert(std::isnan(c));
}

ONEDPL_TEST_NUM_MAIN
{
#if _PSTL_TEST_BESSEL_STD_LIB_IMPL_COMPLIANT
    test01<float>();
    test02<float>();

    IF_DOUBLE_SUPPORT(test01<double>())
    IF_DOUBLE_SUPPORT(test02<double>())
#endif

    return 0;
}
