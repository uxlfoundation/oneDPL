//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_bessel.h"

//  cyl_bessel_jl
#include <cmath>

void
test()
{
    float nuf = 1.0F / 3.0F, xf = 0.5F;

    std::cyl_bessel_kf(nuf, xf);
}

ONEDPL_TEST_NUM_MAIN
{
#if _PSTL_TEST_BESSEL_STD_LIB_IMPL_COMPLIANT
    test();
#endif

    return 0;
}
