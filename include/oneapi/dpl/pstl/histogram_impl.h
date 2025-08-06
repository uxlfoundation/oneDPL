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

#ifndef _ONEDPL_HISTOGRAM_IMPL_H
#define _ONEDPL_HISTOGRAM_IMPL_H

#include "histogram_extension_defs.h"
#include "histogram_binhash_utils.h"
#include "execution_impl.h"
#include "iterator_impl.h"
#include "algorithm_fwd.h"

#if _ONEDPL_HETERO_BACKEND
#    include "hetero/histogram_impl_hetero.h"
#endif

namespace oneapi
{
namespace dpl
{

namespace __internal
{

template <class _ForwardIterator, class _IdxHashFunc, class _RandomAccessIterator, class _IsVector>
void
__brick_histogram(_ForwardIterator __first, _ForwardIterator __last, _IdxHashFunc __func,
                  _RandomAccessIterator __histogram_first, _IsVector) noexcept
{
    for (; __first != __last; ++__first)
    {
        std::int32_t __bin = __func.get_bin(*__first);
        if (__bin >= 0)
        {
            ++__histogram_first[__bin];
        }
    }
}

template <class _Tag, class _ExecutionPolicy, class _ForwardIterator, class _Size, class _IdxHashFunc,
          class _RandomAccessIterator>
void
__pattern_histogram(_Tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                    _Size __num_bins, _IdxHashFunc __func, _RandomAccessIterator __histogram_first)
{
    using _HistogramValueT = typename std::iterator_traits<_RandomAccessIterator>::value_type;
    static_assert(oneapi::dpl::__internal::__is_serial_tag_v<_Tag> ||
                  oneapi::dpl::__internal::__is_parallel_forward_tag_v<_Tag>);
    __pattern_fill(_Tag{}, std::forward<_ExecutionPolicy>(__exec), __histogram_first, __histogram_first + __num_bins,
                   _HistogramValueT{0});
    __brick_histogram(__first, __last, __func, __histogram_first, typename _Tag::__is_vector{});
}

template <class _IsVector, class _ExecutionPolicy, class _RandomAccessIterator1, class _Size, class _IdxHashFunc,
          class _RandomAccessIterator2>
void
__pattern_histogram(oneapi::dpl::__internal::__parallel_tag<_IsVector>, _ExecutionPolicy&& __exec,
                    _RandomAccessIterator1 __first, _RandomAccessIterator1 __last, _Size __num_bins,
                    _IdxHashFunc __func, _RandomAccessIterator2 __histogram_first)
{
    using __backend_tag = typename oneapi::dpl::__internal::__parallel_tag<_IsVector>::__backend_tag;
    using _HistogramValueT = typename std::iterator_traits<_RandomAccessIterator2>::value_type;
    using _DiffType = typename std::iterator_traits<_RandomAccessIterator2>::difference_type;

    _DiffType __n = __last - __first;
    if (__n == 0)
    {
        // when n == 0, we must fill the output histogram with zeros
        __pattern_fill(oneapi::dpl::__internal::__parallel_tag<_IsVector>{}, std::forward<_ExecutionPolicy>(__exec),
                       __histogram_first, __histogram_first + __num_bins, _HistogramValueT{0});
    }
    else
    {
        auto __tls =
            __par_backend::__make_enumerable_tls<std::vector<_HistogramValueT>>(__num_bins, _HistogramValueT{0});

        //main histogram loop
        //TODO: add defaulted grain-size option for __parallel_for and use larger one here to account for overhead
        __par_backend::__parallel_for(
            __backend_tag{}, __exec, __first, __last,
            [__func, &__tls](_RandomAccessIterator1 __first_local, _RandomAccessIterator1 __last_local) {
                __internal::__brick_histogram(__first_local, __last_local, __func,
                                              __tls.get_for_current_thread().begin(), _IsVector{});
            });
        // now accumulate temporary storage into output histogram
        const std::size_t __num_temporary_copies = __tls.size();
        __par_backend::__parallel_for(
            __backend_tag{}, std::forward<_ExecutionPolicy>(__exec), _Size{0}, __num_bins,
            [__num_temporary_copies, __histogram_first, &__tls](auto __hist_start_id, auto __hist_end_id) {
                const _DiffType __local_n = __hist_end_id - __hist_start_id;
                //initialize output histogram with first local histogram via assign
                __internal::__brick_walk2_n(__tls.get_with_id(0).begin() + __hist_start_id, __local_n,
                                            __histogram_first + __hist_start_id,
                                            oneapi::dpl::__internal::__pstl_assign(), _IsVector{});
                for (std::size_t __i = 1; __i < __num_temporary_copies; ++__i)
                {
                    //accumulate into output histogram with other local histogram via += operator
                    __internal::__brick_walk2_n(
                        __tls.get_with_id(__i).begin() + __hist_start_id, __local_n,
                        __histogram_first + __hist_start_id,
                        [](_HistogramValueT __x, _HistogramValueT& __y) { __y += __x; }, _IsVector{});
                }
            });
    }
}

} // namespace __internal

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _Size, typename _ValueType,
          typename _RandomAccessIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator2>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last, _Size num_bins,
          _ValueType first_bin_min_val, _ValueType last_bin_max_val, _RandomAccessIterator2 histogram_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, histogram_first);

    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__evenly_divided_binhash<_ValueType>(first_bin_min_val, last_bin_max_val, num_bins),
        histogram_first);
    return histogram_first + num_bins;
}

template <typename _ExecutionPolicy, typename _RandomAccessIterator1, typename _RandomAccessIterator2,
          typename _RandomAccessIterator3>
oneapi::dpl::__internal::__enable_if_execution_policy<_ExecutionPolicy, _RandomAccessIterator3>
histogram(_ExecutionPolicy&& exec, _RandomAccessIterator1 first, _RandomAccessIterator1 last,
          _RandomAccessIterator2 boundary_first, _RandomAccessIterator2 boundary_last,
          _RandomAccessIterator3 histogram_first)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(exec, first, boundary_first, histogram_first);

    ::std::ptrdiff_t num_bins = boundary_last - boundary_first - 1;
    oneapi::dpl::__internal::__pattern_histogram(
        __dispatch_tag, ::std::forward<_ExecutionPolicy>(exec), first, last, num_bins,
        oneapi::dpl::__internal::__custom_boundary_binhash{boundary_first, boundary_last}, histogram_first);
    return histogram_first + num_bins;
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_IMPL_H
