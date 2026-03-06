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

#ifndef _ONEDPL_INTERNAL_OMP_UTIL_H
#define _ONEDPL_INTERNAL_OMP_UTIL_H

#include <iterator>    // std::iterator_traits, std::distance
#include <cstddef>     // std::size_t
#include <memory>      // std::allocator
#include <type_traits> // std::decay, is_integral_v, enable_if_t, common_type_t
#include <utility>     // std::forward
#include <omp.h>

#include "../parallel_backend_utils.h"
#include "../unseq_backend_simd.h"
#include "../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __omp_backend
{

//------------------------------------------------------------------------
// use to cancel execution
//------------------------------------------------------------------------
inline void
__cancel_execution(oneapi::dpl::__internal::__omp_backend_tag)
{
    // TODO: Figure out how to make cancellation work.
}

//------------------------------------------------------------------------
// raw buffer
//------------------------------------------------------------------------

template <typename _Tp>
using __buffer = oneapi::dpl::__utils::__buffer_impl<_Tp, std::allocator>;

// Preliminary size of each chunk: requires further discussion
constexpr std::size_t __default_chunk_size = 2048;

// Convenience function to determine when we should run serial.
template <typename _Iterator, std::enable_if_t<!std::is_integral_v<_Iterator>, bool> = true>
constexpr auto
__should_run_serial(_Iterator __first, _Iterator __last, const std::size_t __cutoff) -> bool
{
    using _difference_type = typename std::iterator_traits<_Iterator>::difference_type;
    auto __size = std::distance(__first, __last);
    return __size <= static_cast<_difference_type>(__cutoff);
}

template <typename _Index, std::enable_if_t<std::is_integral_v<_Index>, bool> = true>
constexpr auto
__should_run_serial(_Index __first, _Index __last, const std::size_t __cutoff) -> bool
{
    using _difference_type = _Index;
    auto __size = __last - __first;
    return __size <= static_cast<_difference_type>(__cutoff);
}

struct __chunk_metrics
{
    std::size_t __n_chunks;
    std::size_t __chunk_size;
    // early chunks to get the leftover elements; these chunks will have size __chunk_size + 1
    std::size_t __n_larger_chunks;
};

template <class _RandomAccessIterator>
auto
__chunk_partitioner(_RandomAccessIterator __first, _RandomAccessIterator __last, const std::size_t __num_threads,
                    const std::size_t __preferred_min_chunk_size) -> __chunk_metrics
{
    // For sufficiently large inputs, this algorithm creates a number of chunks that is
    // a multiple of the thread count for optimal load balancing.
    // Leftover elements are evenly distributed across earlier chunks to improve processor prefetch utilization.

    using _Size = std::common_type_t<decltype(__last - __first), std::size_t>;
    const _Size __n = __last - __first;

    std::size_t __n_chunks = 1;
    std::size_t __chunk_size = __n;
    std::size_t __n_larger_chunks = 0;

    if (__n <= __preferred_min_chunk_size)
    {
        return __chunk_metrics{__n_chunks, __chunk_size, __n_larger_chunks};
    }

    // Aim for 3 chunks per thread for better load balancing
    constexpr std::size_t __nominal_chunks_per_thread = 3;
    std::size_t __nominal_n_chunks = __num_threads * __nominal_chunks_per_thread;
    std::size_t __nominal_chunk_size = __n / __nominal_n_chunks;

    // Large input. Limit chunk size to aid early-exit algorithms which cannot exit mid-way through a chunk.
    constexpr std::size_t __preferred_max_chunk_size = 32768;
    if (__nominal_chunk_size >= __preferred_max_chunk_size)
    {
        __n_chunks =     // compute the number of chunks per thread, multiply by the number of threads
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __preferred_max_chunk_size * __num_threads) * __num_threads;
        __chunk_size = __n / __n_chunks;
    }
    // Enough input for all threads with the nominal per-thread chunk count.
    else if (__nominal_chunk_size >= __preferred_min_chunk_size)
    {
        __n_chunks = __nominal_n_chunks;
        __chunk_size = __nominal_chunk_size;
    }
    // Enough input for all threads but not enough for the nominal per-thread chunk count.
    else if (__n >= __num_threads * __preferred_min_chunk_size)
    {
        __n_chunks = (__n / (__preferred_min_chunk_size * __num_threads)) * __num_threads;
        __chunk_size = __n / __n_chunks;
    }
    // Not enough input even for one chunk per thread.
    else
    {
        __chunk_size = __preferred_min_chunk_size;
        __n_chunks = __n / __chunk_size;
    }
    __n_larger_chunks = __n % __n_chunks;
    return __chunk_metrics{__n_chunks, __chunk_size, __n_larger_chunks};
}

template <typename _Iterator, typename _Index, typename _Func>
void
__process_chunk(const __chunk_metrics& __metrics, _Iterator __base, _Index __chunk_index, _Func __f)
{
    bool __is_larger_chunk = __chunk_index < __metrics.__n_larger_chunks;
    auto __this_chunk_size = __metrics.__chunk_size + __is_larger_chunk;
    auto __n_previous_larger_chunks = __is_larger_chunk ? __chunk_index : __metrics.__n_larger_chunks;
    auto __index = __chunk_index * __metrics.__chunk_size + __n_previous_larger_chunks;
    auto __first = __base + __index;
    auto __last = __first + __this_chunk_size;
    __f(__first, __last);
}

namespace __detail
{

// Workaround for VS 2017: declare an alias to the CRTP base template
template <typename _ValueType, typename... _Args>
struct __enumerable_thread_local_storage;

template <typename... _Ts>
using __etls_base = __utils::__enumerable_thread_local_storage_base<__enumerable_thread_local_storage, _Ts...>;

template <typename _ValueType, typename... _Args>
struct __enumerable_thread_local_storage : public __etls_base<_ValueType, _Args...>
{

    template <typename... _LocalArgs>
    __enumerable_thread_local_storage(_LocalArgs&&... __args)
        : __etls_base<_ValueType, _Args...>({std::forward<_LocalArgs>(__args)...})
    {
    }

    static std::size_t
    get_num_threads()
    {
        return omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
    }

    static std::size_t
    get_thread_num()
    {
        return omp_get_thread_num();
    }
};

} // namespace __detail

// enumerable thread local storage should only be created with this make function
template <typename _ValueType, typename... _Args>
__detail::__enumerable_thread_local_storage<_ValueType, std::remove_reference_t<_Args>...>
__make_enumerable_tls(_Args&&... __args)
{
    return __detail::__enumerable_thread_local_storage<_ValueType, std::remove_reference_t<_Args>...>(
        std::forward<_Args>(__args)...);
}

} // namespace __omp_backend
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_INTERNAL_OMP_UTIL_H
