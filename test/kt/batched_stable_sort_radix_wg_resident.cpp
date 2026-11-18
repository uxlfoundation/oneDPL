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
using SortConfig = kt::gpu::radix_wg_resident_tag<IsAscending, TestRadixBits>;

int
main()
{
#if TEST_SYCL_WG_RESIDENT_RADIX_SORT_KT_AVAILABLE
    return test<SortConfig, /*WgResidentOnly=*/true>();
#else
    return TestUtils::done(false);
#endif
}
