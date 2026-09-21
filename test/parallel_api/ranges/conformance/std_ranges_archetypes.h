// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
