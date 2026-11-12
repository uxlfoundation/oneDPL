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

#include <algorithm> // std::max
#include <cstddef>   // std::size_t

#include "util.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

//------------------------------------------------------------------------
// Notation:
// Evaluation of brick f[i,j) for each subrange [i,j) of [first, last)
//------------------------------------------------------------------------

template <class _ExecutionPolicy, class _Index, class _Fp>
void
__parallel_for(oneapi::dpl::__internal::__omp_backend_tag, _ExecutionPolicy&&, _Index __first, _Index __last, _Fp __f,
               std::size_t __grainsize = __default_chunk_size)
{
    const std::size_t __n = __last - __first;
    if (__n <= __grainsize)
    {
        __f(__first, __last);
        return;
    }

    if (omp_in_parallel())
    {
        // We don't create a nested parallel region in an existing parallel region:
        // use taskloop instead to avoid over-subscription
        const auto __policy =
            oneapi::dpl::__omp_backend::__chunk_partitioner(__first, __last, omp_get_num_threads(), __grainsize);

        _ONEDPL_PRAGMA(omp taskloop untied mergeable)
        for (std::size_t __chunk = 0; __chunk < __policy.__n_chunks; ++__chunk)
        {
            oneapi::dpl::__omp_backend::__process_chunk(__policy, __first, __chunk, __f);
        }
        return;
    }

    // Conservative empirical estimate:
    // the number of elements per thread up to which dynamic scheduling costs more
    // than the imbalance it removes.
    constexpr std::size_t __static_schedule_max_elements_per_thread = 8192;
    const std::size_t __num_threads = omp_get_max_threads();

    if (__n <= __num_threads * __static_schedule_max_elements_per_thread)
    {
        // Static scheduling has almost no cost per chunk, so finer-grained chunks can be used:
        // they can use more threads in case small number of elements
        constexpr std::size_t __static_schedule_min_chunk_size = 64;
        const auto __static_policy = oneapi::dpl::__omp_backend::__chunk_partitioner(
            __first, __last, __num_threads, __static_schedule_min_chunk_size);

        _ONEDPL_PRAGMA(omp parallel for schedule(static, 1))
        for (std::size_t __chunk = 0; __chunk < __static_policy.__n_chunks; ++__chunk)
        {
            oneapi::dpl::__omp_backend::__process_chunk(__static_policy, __first, __chunk, __f);
        }
    }
    else
    {
        const auto __policy =
            oneapi::dpl::__omp_backend::__chunk_partitioner(__first, __last, __num_threads, __grainsize);

        // Coarsen the dispatches by grouping several chunks once the chunks stop growing.
        // It reduces the scheduling overhead as if the chunks continued growing,
        // while the capped chunk size keeps early exits fine-grained.
        // TODO: move it to __chunk_partitioner if it is beneficial in other patterns.
        const std::size_t __chunks_per_dispatch = std::max<std::size_t>(
            1, __policy.__n_chunks / (__num_threads * oneapi::dpl::__omp_backend::__nominal_chunks_per_thread));

        // In libiomp, monotonic scheduling empirically shows better results
        // even on CPUs with hybrid cores with uneven workload.
        // In libgomp, there is no observable difference between monotonic and nonmonotonic modifiers.
        // Generally, nonmonotonic (default dynamic) scheduling should be more performant,
        // thus enabling it by default with other implementations.
#if __INTEL_LLVM_COMPILER
        _ONEDPL_PRAGMA(omp parallel for schedule(monotonic : dynamic, __chunks_per_dispatch))
#else
        _ONEDPL_PRAGMA(omp parallel for schedule(dynamic, __chunks_per_dispatch))
#endif
        for (std::size_t __chunk = 0; __chunk < __policy.__n_chunks; ++__chunk)
        {
            oneapi::dpl::__omp_backend::__process_chunk(__policy, __first, __chunk, __f);
        }
    }
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi
#endif // _ONEDPL_INTERNAL_OMP_PARALLEL_FOR_H
