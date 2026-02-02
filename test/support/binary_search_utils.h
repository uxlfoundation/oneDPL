// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

using namespace TestUtils;

// TODO: replace data generation with random data and update check to compare result to
// the result of the serial algorithm
template <typename Accessor1, typename Accessor2, typename Accessor3, typename Size>
void
initialize_data(Accessor1 data, Accessor2 value, Accessor3 result, Size n)
{
    const int num_values = n > 100 ? n / 100 : 1; // # search values expected to be << n

    for (int i = 0; i < n; i += 2)
    {
        data[i] = i;
        if (i + 1 < n)
        {
            data[i + 1] = i;
        }
        if (i < num_values * 2)
        {
            // value = {0, 2, 5, 6, 9, 10, 13...}
            // result will alternate true/false after initial true
            value[i / 2] = i + (i != 0 && i % 4 == 0 ? 1 : 0);
        }
        result[i / 2] = 0;
    }
}
