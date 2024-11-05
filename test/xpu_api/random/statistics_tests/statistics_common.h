// -*- C++ -*-
//===----------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//

#include <iostream>
#include <vector>

template <typename ScalarIntType>
int
compare_moments(int nsamples, const std::vector<ScalarIntType>& samples, double tM, double tD, double tQ)
{
    // sample moments
    double sum = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < nsamples; i++)
    {
        sum += samples[i];
        sum2 += samples[i] * samples[i];
    }
    double sM = sum / nsamples;
    double sD = sum2 / nsamples - sM * sM;

    // comparison of theoretical and sample moments
    double tD2 = tD * tD;
    double s = ((tQ - tD2) / nsamples) - (2 * (tQ - 2.0 * tD2) / (nsamples * nsamples)) +
               ((tQ - 3.0 * tD2) / (nsamples * nsamples * nsamples));

    double DeltaM = (tM - sM) / sqrt(tD / nsamples);
    double DeltaD = (tD - sD) / sqrt(s);

    if (fabs(DeltaM) > 3.0 || fabs(DeltaD) > 3.0)
    {
        std::cout << "Error: sample moments (mean= " << sM << ", variance= " << sD
                  << ") disagree with theory (mean=" << tM << ", variance= " << tD << ")";
        return 1;
    }

    return 0;
}