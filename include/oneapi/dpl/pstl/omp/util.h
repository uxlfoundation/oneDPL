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

#include <iterator>    //std::iterator_traits, std::distance
#include <cstddef>     //std::size_t
#include <memory>      //std::allocator
#include <type_traits> // std::decay, is_integral_v, enable_if_t
#include <utility>     // std::forward
#include <omp.h>

#include "../parallel_backend_utils.h"
#include "../unseq_backend_simd.h"
#include "../utils.h"

// Portability "#pragma" definition
#ifdef _MSC_VER
#    define _PSTL_PRAGMA(x) __pragma(x)
#else
#    define _PSTL_PRAGMA(x) _Pragma(#    x)
#endif

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

template <typename _ExecutionPolicy, typename _Tp>
using __buffer = oneapi::dpl::__utils::__buffer_impl<std::decay_t<_ExecutionPolicy>, _Tp, std::allocator>;

// Preliminary size of each chunk: requires further discussion
constexpr std::size_t __default_chunk_size = 2048;

// Convenience function to determine when we should run serial.
template <typename _Iterator, std::enable_if_t<!std::is_integral_v<_Iterator>, bool> = true>
constexpr auto
__should_run_serial(_Iterator __first, _Iterator __last) -> bool
{
    using _difference_type = typename std::iterator_traits<_Iterator>::difference_type;
    auto __size = std::distance(__first, __last);
    return __size <= static_cast<_difference_type>(__default_chunk_size);
}

template <typename _Index, std::enable_if_t<std::is_integral_v<_Index>, bool> = true>
constexpr auto
__should_run_serial(_Index __first, _Index __last) -> bool
{
    using _difference_type = _Index;
    auto __size = __last - __first;
    return __size <= static_cast<_difference_type>(__default_chunk_size);
}

struct __chunk_metrics
{
    std::size_t __n_chunks;
    std::size_t __chunk_size;
    std::size_t __first_chunk_size;
};

// The iteration space partitioner according to __requested_chunk_size
template <class _RandomAccessIterator, class _Size = std::size_t>
auto
__chunk_partitioner(_RandomAccessIterator __first, _RandomAccessIterator __last,
                    _Size __requested_chunk_size = __default_chunk_size) -> __chunk_metrics
{
    /*
     * This algorithm improves distribution of elements in chunks by avoiding
     * small tail chunks. The leftover elements that do not fit neatly into
     * the chunk size are redistributed to early chunks. This improves
     * utilization of the processor's prefetch and reduces the number of
     * tasks needed by 1.
     */

    const _Size __n = __last - __first;
    _Size __n_chunks = 0;
    _Size __chunk_size = 0;
    _Size __first_chunk_size = 0;
    if (__n < __requested_chunk_size)
    {
        __chunk_size = __n;
        __first_chunk_size = __n;
        __n_chunks = 1;
        return __chunk_metrics{__n_chunks, __chunk_size, __first_chunk_size};
    }

    __n_chunks = (__n / __requested_chunk_size) + 1;
    __chunk_size = __n / __n_chunks;
    __first_chunk_size = __chunk_size;
    const _Size __n_leftover_items = __n - (__n_chunks * __chunk_size);

    if (__n_leftover_items == __chunk_size)
    {
        __n_chunks += 1;
        return __chunk_metrics{__n_chunks, __chunk_size, __first_chunk_size};
    }
    else if (__n_leftover_items == 0)
    {
        __first_chunk_size = __chunk_size;
        return __chunk_metrics{__n_chunks, __chunk_size, __first_chunk_size};
    }

    const _Size __n_extra_items_per_chunk = __n_leftover_items / __n_chunks;
    const _Size __n_final_leftover_items = __n_leftover_items - (__n_extra_items_per_chunk * __n_chunks);

    __chunk_size += __n_extra_items_per_chunk;
    __first_chunk_size = __chunk_size + __n_final_leftover_items;

    return __chunk_metrics{__n_chunks, __chunk_size, __first_chunk_size};
}

template <typename _Iterator, typename _Index, typename _Func>
void
__process_chunk(const __chunk_metrics& __metrics, _Iterator __base, _Index __chunk_index, _Func __f)
{
    auto __this_chunk_size = __chunk_index == 0 ? __metrics.__first_chunk_size : __metrics.__chunk_size;
    auto __index = __chunk_index == 0 ? 0
                                      : (__chunk_index * __metrics.__chunk_size) +
                                            (__metrics.__first_chunk_size - __metrics.__chunk_size);
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
