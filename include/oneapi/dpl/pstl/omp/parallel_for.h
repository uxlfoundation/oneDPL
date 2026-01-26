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

// Chunk size selection strategy for parallel for algorithms
// It primarily depends on a functor complexity, which is out of our control,
// but some tuning is still possible based on the input size and number of threads
inline std::size_t
__get_optimal_chunk_size(std::size_t __size, int __num_threads)
{
    // Minimal/multiple are selected to allow vectorization inside a chunk with AVX-512 or narrower vector instructions
    constexpr std::size_t __min_chunk = 64;
    constexpr std::size_t __multiple_chunk = 64;

    // TODO: investigate if a larger max chunk size makes sense
    constexpr std::size_t __max_chunk = 16384;

    constexpr std::size_t __large_size_optimal_chunk = 2048;

    // Fill in all threads.
    // It prioritizes performance with complex functors.
    // Trivial functors perform much better with larger chunks and using a portion of available threads,
    // but due to a small overall execution time the impact is not that significant.
    if (__size <= __max_chunk)
    {
        return std::max(__min_chunk, (__size / __num_threads) / __multiple_chunk * __multiple_chunk);
    }
    else if (__size <= 2097152)
    {
        return std::min(__max_chunk, (__size / __num_threads) / __multiple_chunk * __multiple_chunk);
    }
    // When reaching ~1-4M elements, the optimal chunk is smaller
    // It is probably due to exhausted L1/L2 caches requiring different memory access patterns
    else
    {
        return __large_size_optimal_chunk;
    }
}

template <class _Index, class _Fp>
void
__parallel_for_body(_Index __first, _Index __last, _Fp __f)
{
    std::size_t __grainsize = __get_optimal_chunk_size(__last - __first, omp_get_max_threads());

    // initial partition of the iteration space into chunks
    auto __policy = oneapi::dpl::__omp_backend::__chunk_partitioner(__first, __last, __grainsize);

    // To avoid over-subscription we use taskloop for the nested parallelism
    _ONEDPL_PRAGMA(omp taskloop untied mergeable)
    for (std::size_t __chunk = 0; __chunk < __policy.__n_chunks; ++__chunk)
    {
        oneapi::dpl::__omp_backend::__process_chunk(__policy, __first, __chunk, __f);
    }
}

//------------------------------------------------------------------------
// Notation:
// Evaluation of brick f[i,j) for each subrange [i,j) of [first, last)
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _Index, class _Fp>
void
__parallel_for(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last, _Fp __f)
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _ONEDPL_PRAGMA(omp parallel)
        _ONEDPL_PRAGMA(omp single nowait)
        {
            oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H
