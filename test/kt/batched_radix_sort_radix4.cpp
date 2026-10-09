// -*- C++ -*-
//===-- batched_radix_sort_radix4.cpp -------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With 4 radix bits, only segments which fit in a work-group tile are supported
#define TEST_RADIX_BITS 4
#include "batched_radix_sort.cpp"
