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
using SortConfig = kt::gpu::radix_multi_wg_tag<IsAscending, TestRadixBits>;

int
main()
{
#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE
    return test<SortConfig, /*WgResidentOnly=*/false>();
#else
    return TestUtils::done(false);
#endif
}
