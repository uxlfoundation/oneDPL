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

// Check if cooperative kernels are supported (forward progress and root group extensions)
// Requires an intel/llvm compiler after 2025.1.0 where all required functionality is implemented.
// Open-source compiler builds prior to this functionality becoming sufficient (September 2024)
// do not have a reliable detection method but are unlikely to be used.
#if defined(SYCL_EXT_ONEAPI_FORWARD_PROGRESS) && defined(SYCL_EXT_ONEAPI_ROOT_GROUP) &&                                \
    (!defined(__INTEL_LLVM_COMPILER) || __INTEL_LLVM_COMPILER >= 20250100)
#    define _ONEDPL_KT_COOPERATIVE_KERNELS_PRESENT 1
#endif

#if _ONEDPL_KT_COOPERATIVE_KERNELS_PRESENT && _ONEDPL_LIBSYCL_SUB_GROUP_MASK_PRESENT
#    define _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT 1
#endif

#endif
