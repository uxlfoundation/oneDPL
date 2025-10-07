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
#ifndef _ONEDPL_PARALLEL_BACKEND_H
#define _ONEDPL_PARALLEL_BACKEND_H
#include "onedpl_config.h"

// Select a parallel backend
#if ONEDPL_USE_TBB_BACKEND || (!defined(ONEDPL_USE_TBB_BACKEND) && !ONEDPL_USE_OPENMP_BACKEND && _ONEDPL_TBB_AVAILABLE)
#    define _ONEDPL_PAR_BACKEND_TBB 1
#    include "parallel_backend_tbb.h"
#elif ONEDPL_USE_OPENMP_BACKEND || (!defined(ONEDPL_USE_OPENMP_BACKEND) && _ONEDPL_OPENMP_AVAILABLE)
#    define _ONEDPL_PAR_BACKEND_OPENMP 1
#    include "parallel_backend_omp.h"
#else
#    define _ONEDPL_PAR_BACKEND_SERIAL 1
#    include "parallel_backend_serial.h"
#endif

#if _ONEDPL_BACKEND_SYCL
#    include "hetero/dpcpp/parallel_backend_sycl.h"
#    if _ONEDPL_FPGA_DEVICE
#        include "hetero/dpcpp/parallel_backend_sycl_fpga.h"
#    endif
#endif

namespace oneapi
{
namespace dpl
{
#if _ONEDPL_PAR_BACKEND_TBB
namespace __par_backend = __tbb_backend;
#elif _ONEDPL_PAR_BACKEND_OPENMP
namespace __par_backend = __omp_backend;
#elif _ONEDPL_PAR_BACKEND_SERIAL
namespace __par_backend = __serial_backend;
#else
#    error "Parallel backend was not specified"
#endif
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_H
