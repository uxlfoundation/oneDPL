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

struct __grain_selector_any_workload
{
    inline std::size_t operator()(std::size_t __size, int __num_threads) const
    {
        // Multiple is selected to allow vectorization inside a chunk with AVX-512 or narrower vector instructions
        // Min/Max are found empirically
        constexpr std::size_t __min_chunk = 256;
        constexpr std::size_t __max_chunk = 16384;
        constexpr std::size_t __multiple_chunk = 64;

        // Aim for 3 tasks per thread for better load balancing
        std::size_t __grainsize = __size / (__num_threads * 3);
        if (__grainsize < __min_chunk)
            __grainsize = __min_chunk;
        else if (__grainsize > __max_chunk)
            __grainsize = __max_chunk;
        // Round up to avoid too avoid small uneven chunk at the end
        return ((__grainsize + __multiple_chunk - 1) / __multiple_chunk) * __multiple_chunk;
    }
};

struct __grain_selector_for_small_workload
{
    inline std::size_t operator()(std::size_t __size, int __num_threads) const
    {
        // Multiple is selected to allow vectorization inside a chunk with AVX-512 or narrower vector instructions
        // Min/Max are found empirically
        constexpr std::size_t __min_chunk = 2048;
        constexpr std::size_t __max_chunk = 16384;
        constexpr std::size_t __multiple_chunk = 64;

        // Aim for 3 tasks per thread for better load balancing
        std::size_t __grainsize = __size / (__num_threads * 3);
        if (__grainsize < __min_chunk)
            __grainsize = __min_chunk;
        else if (__grainsize > __max_chunk)
            __grainsize = __max_chunk;
        // Round up to avoid too avoid small uneven chunk at the end
        return ((__grainsize + __multiple_chunk - 1) / __multiple_chunk) * __multiple_chunk;
    }
};

struct __grain_selector_for_large_workload
{
    inline std::size_t operator()(std::size_t __size, int __num_threads) const
    {
        // Multiple is selected to allow vectorization inside a chunk with AVX-512 or narrower vector instructions
        // Min/Max are found empirically
        constexpr std::size_t __min_chunk = 64;
        constexpr std::size_t __max_chunk = 1024;
        constexpr std::size_t __multiple_chunk = 64;

        // Aim for 3 tasks per thread for better load balancing
        std::size_t __grainsize = __size / (__num_threads * 3);
        if (__grainsize < __min_chunk)
            __grainsize = __min_chunk;
        else if (__grainsize > __max_chunk)
            __grainsize = __max_chunk;
        // Round up to avoid too avoid small uneven chunk at the end
        return ((__grainsize + __multiple_chunk - 1) / __multiple_chunk) * __multiple_chunk;
    }
};

template <class _Index, class _Fp, class _GrainSelector>
void
__parallel_for_body(_Index __first, _Index __last, _Fp __f, _GrainSelector __grain_selector)
{
    std::size_t __grainsize = __grain_selector(__last - __first, omp_get_num_threads());

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

template <class _ExecutionPolicy, class _Index, class _Fp, class _GrainSelector = __grain_selector_any_workload>
void
__parallel_for(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last, _Fp __f,
               _GrainSelector __grain_selector = _GrainSelector())
{
    if (omp_in_parallel())
    {
        // we don't create a nested parallel region in an existing parallel
        // region: just create tasks
        oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f, __grain_selector);
    }
    else
    {
        // in any case (nested or non-nested) one parallel region is created and
        // only one thread creates a set of tasks
        _ONEDPL_PRAGMA(omp parallel)
        _ONEDPL_PRAGMA(omp single nowait)
        {
            oneapi::dpl::__omp_backend::__parallel_for_body(__first, __last, __f, __grain_selector);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H
