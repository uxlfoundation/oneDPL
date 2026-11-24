// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "batched_stable_sort.h"

template <bool IsAscending>
using SortConfig = kt::gpu::merge_wg_resident_tag<DefaultKeyCompare<IsAscending>>;

int
main()
{
    return test<SortConfig, /*WgResidentOnly=*/true>();
}
