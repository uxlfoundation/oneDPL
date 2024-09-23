// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "support/test_config.h"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/array>

#if TEST_DPCPP_BACKEND_PRESENT
#   include <oneapi/dpl/async>
#endif

#include <oneapi/dpl/cmath>
#include <oneapi/dpl/complex>
#include <oneapi/dpl/cstddef>
#include <oneapi/dpl/cstring>

#if TEST_DPCPP_BACKEND_PRESENT
#   include <oneapi/dpl/dynamic_selection>
#endif

#include <oneapi/dpl/execution>
#include <oneapi/dpl/functional>
#include <oneapi/dpl/iterator>

#if TEST_DPCPP_BACKEND_PRESENT
#   include <oneapi/dpl/experimental/kernel_templates>
#endif

#include <oneapi/dpl/limits>
#include <oneapi/dpl/memory>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/optional>

#if TEST_DPCPP_BACKEND_PRESENT
#   include <oneapi/dpl/random>
#endif

#if _ENABLE_RANGES_TESTING
#   include <oneapi/dpl/ranges>
#endif

#include <oneapi/dpl/ratio>
#include <oneapi/dpl/tuple>
#include <oneapi/dpl/type_traits>
#include <oneapi/dpl/utility>
