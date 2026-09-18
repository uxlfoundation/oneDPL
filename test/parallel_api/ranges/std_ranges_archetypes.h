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

#ifndef _STD_RANGES_ARCHETYPES_H
#define _STD_RANGES_ARCHETYPES_H

#if _ENABLE_STD_RANGES_TESTING

//------------------------------------------------------------------------------------------------
// Archetypes for the algorithms of glue_algorithm_ranges_impl.h
//
// Every algorithm there constrains its range parameters with std::ranges::random_access_range and
// std::ranges::sized_range only; all the remaining requirements are expressed as indirect concepts
// on the iterators. The element archetypes below therefore drop everything a "regular" type would
// have and add back exactly the operations one concept family needs. archetype_view is reused as the
// range, so the ranges are random access and sized but neither contiguous nor common.
//------------------------------------------------------------------------------------------------

#include "std_ranges_archetypes_base.h"
#include "std_ranges_archetypes_memory.h"
#include "std_ranges_archetypes_read.h"
#include "std_ranges_archetypes_value.h"
#include "std_ranges_archetypes_write.h"
#include "std_ranges_archetypes_permute.h"
#include "std_ranges_archetypes_merge.h"
#include "std_ranges_archetypes_storable.h"

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_ARCHETYPES_H
