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

#ifndef _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H
#define _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H

#include <cstddef>

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

template <class _Index, class _Fp>
void
__parallel_for_body(_Index __first, _Index __last, _Fp __f, std::size_t __grainsize)
{
    // initial partition of the iteration space into chunks
    auto __policy = oneapi::dpl::__omp_backend::__chunk_partitioner(__first, __last, __grainsize);

#if ITT_PRESENT_TMP && defined(DEBUG_CHUNKS)
    __itt_pause();
    std::cerr << "Total number of chunks: " << __policy.__n_chunks << std::endl;
    std::cerr << "Chunk size: " << __policy.__chunk_size << std::endl;
    std::cerr << "First chunk size: " << __policy.__first_chunk_size << std::endl;
    std::cerr << "Grainsize: " << __grainsize << std::endl;
    std::cerr << "N: " << __last - __first << std::endl;
    __itt_resume();
#endif

    // To avoid over-subscription we use taskloop for the nested parallelism
    _ONEDPL_PRAGMA(omp taskloop untied mergeable)
    for (std::size_t __chunk = 0; __chunk < __policy.__n_chunks; ++__chunk)
    {
#if ITT_PRESENT_TMP && defined(DEBUG_CHUNKS)
    __itt_pause();
    auto start_time = std::chrono::high_resolution_clock::now();
    __itt_resume();
#endif
        oneapi::dpl::__omp_backend::__process_chunk(__policy, __first, __chunk, __f);
#if ITT_PRESENT_TMP && defined(DEBUG_CHUNKS)
    __itt_pause();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    #pragma omp critical
    {
        std::cerr << elapsed_time << std::endl;
    }
    __itt_resume();
#endif
    }
}

//------------------------------------------------------------------------
// Notation:
// Evaluation of brick f[i,j) for each subrange [i,j) of [first, last)
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _Index, class _Fp>
void
__parallel_for(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last, _Fp __f,
               std::size_t __grainsize = __default_chunk_size)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f, __grainsize);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _ONEDPL_PRAGMA(omp parallel)
        _ONEDPL_PRAGMA(omp single nowait)
        {
            oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f, __grainsize);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H
