// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_DEFS_H
#define _ONEDPL_KT_DEFS_H

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/sycl_forward_progress.h"

#if _ONEDPL_COOPERATIVE_KERNELS_PRESENT && _ONEDPL_LIBSYCL_SUB_GROUP_MASK_PRESENT
#    define _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT 1
#endif

#endif
