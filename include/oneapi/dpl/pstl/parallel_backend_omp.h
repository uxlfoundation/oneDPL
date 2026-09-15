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

// This header guard is used to check inclusion of OpenMP backend.
// Changing this macro may result in broken tests.
#ifndef _ONEDPL_PARALLEL_BACKEND_OMP_H
#define _ONEDPL_PARALLEL_BACKEND_OMP_H

//------------------------------------------------------------------------
// parallel_invoke
//------------------------------------------------------------------------

#include "./omp/parallel_invoke.h"

//------------------------------------------------------------------------
// parallel_for
//------------------------------------------------------------------------

#include "./omp/parallel_for.h"

//------------------------------------------------------------------------
// parallel_for_each
//------------------------------------------------------------------------

#include "./omp/parallel_for_each.h"

//------------------------------------------------------------------------
// parallel_reduce
//------------------------------------------------------------------------

#include "./omp/parallel_reduce.h"
#include "./omp/parallel_transform_reduce.h"

//------------------------------------------------------------------------
// parallel_scan
//------------------------------------------------------------------------

#include "./omp/parallel_scan.h"
#include "./omp/parallel_transform_scan.h"

//------------------------------------------------------------------------
// parallel_stable_sort
//------------------------------------------------------------------------

#include "./omp/parallel_stable_partial_sort.h"
#include "./omp/parallel_stable_sort.h"

#endif //_ONEDPL_PARALLEL_BACKEND_OMP_H
