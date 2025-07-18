// -*- C++ -*-
//===-- algorithm_impl_hetero.h -------------------------------------------===//
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

#ifndef _ONEDPL_ALGORITHM_IMPL_HETERO_H
#define _ONEDPL_ALGORITHM_IMPL_HETERO_H

#include "../../functional"
#include "../algorithm_fwd.h"

#include "../parallel_backend.h"
#include "../utils_ranges.h"
#include "utils_hetero.h"

#if _ONEDPL_BACKEND_SYCL
#    include "dpcpp/execution_sycl_defs.h"
#    include "dpcpp/parallel_backend_sycl_utils.h"
#    include "dpcpp/unseq_backend_sycl.h"
#endif

#include <cstddef> // std::nullptr_t
#include <utility> // std::forward

namespace oneapi
{
namespace dpl
{
namespace __internal
{

//------------------------------------------------------------------------
// walk1
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
void
__pattern_walk1(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator __first, _ForwardIterator __last,
                _Function __f)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return;

    auto __keep =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator>();
    auto __buf = __keep(__first, __last);

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::walk_n_vectors_or_scalars<_Function>{__f, static_cast<std::size_t>(__n)}, __n, __buf.all_view())
        .__checked_deferrable_wait();
}

//------------------------------------------------------------------------
// walk1_n
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Size,
          typename _Function>
_ForwardIterator
__pattern_walk1_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n,
                  _Function __f)
{
    __pattern_walk1(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __n, __f);
    return __first + __n;
}

//------------------------------------------------------------------------
// walk2
//------------------------------------------------------------------------

// TODO: A tag _WaitMode is used for provide a patterns call pipeline, where the last one should be synchronous
// Probably it should be re-designed by a pipeline approach, when a pattern returns some sync objects
// and ones are combined into a "pipeline" (probably like Range pipeline)
//
// A note on access mode types below: the __vector_path_impl in unseq_backend::walk_n_vectors_or_scalars only respects
// the default template arguments:
// __acc_mode1 = __par_backend_hetero::access_mode::read
// __acc_mode2 = __par_backend_hetero::access_mode::write
// For any provided _Function object, the default access modes should be respected even if other access modes are
// required due to dependency / synchronization issues. For a detailed explanation, see:
// https://github.com/uxlfoundation/oneDPL/issues/1272
template <typename _WaitMode = __par_backend_hetero::__deferrable_mode,
          __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::write,
          typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Function>
_ForwardIterator2
__pattern_walk2(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return __first2;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future = oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::walk_n_vectors_or_scalars<_Function>{__f, static_cast<std::size_t>(__n)}, __n, __buf1.all_view(),
        __buf2.all_view());

    // Call no wait, wait or deferrable wait depending on _WaitMode
    __future.wait(_WaitMode{});

    return __first2 + __n;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _Size,
          typename _ForwardIterator2, typename _Function>
_ForwardIterator2
__pattern_walk2_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _Size __n,
                  _ForwardIterator2 __first2, _Function __f)
{
    return __pattern_walk2(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __first1 + __n, __first2, __f);
}

//------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Function>
_ForwardIterator2
__pattern_swap(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
               _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __f)
{
    const auto __n = __last1 - __first1;
    if (__n == 0)
        return __first2;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);

    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);

    auto __future = oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__brick_swap<_Function>{__f, static_cast<std::size_t>(__n)}, __n, __buf1.all_view(),
        __buf2.all_view());
    __future.wait(__par_backend_hetero::__deferrable_mode{});
    return __first2 + __n;
}

//------------------------------------------------------------------------
// walk3
//------------------------------------------------------------------------

// A note on access mode types below: the __vector_path_impl in unseq_backend::walk_n_vectors_or_scalars only respects
// the default template arguments:
// __acc_mode1 = __par_backend_hetero::access_mode::read
// __acc_mode2 = __par_backend_hetero::access_mode::read
// __acc_mode3 __par_backend_hetero::access_mode::write
// For any provided _Function object, the default access modes should be respected even if other access modes are
// required due to dependency / synchronization issues. For a detailed explanation, see:
// https://github.com/uxlfoundation/oneDPL/issues/1272
template <typename _BackendTag, __par_backend_hetero::access_mode __acc_mode1 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode2 = __par_backend_hetero::access_mode::read,
          __par_backend_hetero::access_mode __acc_mode3 = __par_backend_hetero::access_mode::write,
          typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2, typename _ForwardIterator3,
          typename _Function>
_ForwardIterator3
__pattern_walk3(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3, _Function __f)
{
    auto __n = __last1 - __first1;
    if (__n <= 0)
        return __first3;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode1, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode2, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __n);
    auto __keep3 = oneapi::dpl::__ranges::__get_sycl_range<__acc_mode3, _ForwardIterator3>();
    auto __buf3 = __keep3(__first3, __first3 + __n);

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::walk_n_vectors_or_scalars<_Function>{__f, static_cast<std::size_t>(__n)}, __n, __buf1.all_view(),
        __buf2.all_view(), __buf3.all_view())
        .__checked_deferrable_wait();

    return __first3 + __n;
}

//------------------------------------------------------------------------
// walk_brick, walk_brick_n
//------------------------------------------------------------------------

template <typename _Name>
struct __walk_brick_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Function>
void
__pattern_walk_brick(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                     _ForwardIterator __last, _Function __f)
{
    if (__last - __first <= 0)
        return;

    __pattern_walk1(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first, __last, __f);
}

template <typename _Name>
struct __walk_brick_n_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Size,
          typename _Function>
_ForwardIterator
__pattern_walk_brick_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __n,
                       _Function __f)
{
    __pattern_walk1(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk_brick_n_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first, __first + __n, __f);
    return __first + __n;
}

//------------------------------------------------------------------------
// walk2_brick, walk2_brick_n
//------------------------------------------------------------------------

template <typename _Name>
struct __walk2_brick_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Brick>
_ForwardIterator2
__pattern_walk2_brick(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                      _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Brick __brick)
{
    return __pattern_walk2(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __brick);
}

template <typename _Name>
struct __walk2_brick_n_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _Size,
          typename _ForwardIterator2, typename _Brick>
_ForwardIterator2
__pattern_walk2_brick_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                        _Size __n, _ForwardIterator2 __first2, _Brick __brick)
{
    return __pattern_walk2(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk2_brick_n_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __first1 + __n, __first2, __brick);
}

//------------------------------------------------------------------------
// transform_if
//------------------------------------------------------------------------

template <typename _Name>
struct __walk2_transform_if_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Function>
_ForwardIterator2
__pattern_walk2_transform_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                             _ForwardIterator1 __last1, _ForwardIterator2 __first2, _Function __func)
{
    // Require `read_write` access mode for output sequence to force a copy in for host iterators to capture incoming
    // values of the output sequence for elements where the predicate is false. We never actually read from the output
    // sequence, so there is no risk when ran with the vectorized path of walk_n_vector_or_scalars. For more info,
    // please see the comment above __pattern_walk2 and https://github.com/uxlfoundation/oneDPL/issues/1272.
    return __pattern_walk2</*_WaitMode*/ __par_backend_hetero::__deferrable_mode,
                           __par_backend_hetero::access_mode::read, __par_backend_hetero::access_mode::read_write>(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk2_transform_if_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __func);
}

template <typename _Name>
struct __walk3_transform_if_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _ForwardIterator3, typename _Function>
_ForwardIterator3
__pattern_walk3_transform_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                             _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator3 __first3,
                             _Function __func)
{
    // Require `read_write` access mode for output sequence to force a copy in for host iterators to capture incoming
    // values of the output sequence for elements where the predicate is false. We never actually read from the output
    // sequence, so there is no risk when ran with the vectorized path of walk_n_vector_or_scalars. For more info,
    // please see the comment above __pattern_walk3 and https://github.com/uxlfoundation/oneDPL/issues/1272.
    return __pattern_walk3<_BackendTag, __par_backend_hetero::access_mode::read,
                           __par_backend_hetero::access_mode::read, __par_backend_hetero::access_mode::read_write>(
        __tag,
        __par_backend_hetero::make_wrapped_policy<__walk3_transform_if_wrapper>(
            ::std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __first3, __func);
}

//------------------------------------------------------------------------
// fill
//------------------------------------------------------------------------

template <typename _SourceT>
struct fill_functor
{
    _SourceT __value;
    template <typename _TargetT>
    void
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _T>
_ForwardIterator
__pattern_fill(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
               _ForwardIterator __last, const _T& __value)
{
    __pattern_walk1(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                    fill_functor<_T>{__value});
    return __last;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, class _Size, typename _T>
_ForwardIterator
__pattern_fill_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first, _Size __count,
                 const _T& __value)
{
    return __pattern_fill(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __count, __value);
}

//------------------------------------------------------------------------
// generate
//------------------------------------------------------------------------

template <typename _Generator>
struct generate_functor
{
    _Generator __g;

    template <typename _TargetT>
    void
    operator()(_TargetT& value) const
    {
        value = __g();
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, typename _Generator>
_ForwardIterator
__pattern_generate(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                   _ForwardIterator __last, _Generator __g)
{
    __pattern_walk1(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__first),
                    __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__last),
                    generate_functor<_Generator>{__g});
    return __last;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator, class _Size, typename _Generator>
_ForwardIterator
__pattern_generate_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator __first,
                     _Size __count, _Generator __g)
{
    return __pattern_generate(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __first + __count, __g);
}

//------------------------------------------------------------------------
// brick_copy, brick_move
//------------------------------------------------------------------------

template <typename _BackendTag>
struct __brick_copy_n<__hetero_tag<_BackendTag>>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::forward<_SourceT>(__source);
    }
};

template <typename _BackendTag>
struct __brick_copy<__hetero_tag<_BackendTag>>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::forward<_SourceT>(__source);
    }
};

template <typename _BackendTag>
struct __brick_move<__hetero_tag<_BackendTag>>
{
    template <typename _SourceT, typename _TargetT>
    void
    operator()(_SourceT&& __source, _TargetT&& __target) const
    {
        __target = ::std::move(__source);
    }
};

template <typename _BackendTag, typename _SourceT>
struct __brick_fill<__hetero_tag<_BackendTag>, _SourceT>
{
    _SourceT __value;
    template <typename _TargetT>
    void
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

template <typename _BackendTag, typename _SourceT>
struct __brick_fill_n<__hetero_tag<_BackendTag>, _SourceT>
{
    _SourceT __value;
    template <typename _TargetT>
    void
    operator()(_TargetT& __target) const
    {
        __target = __value;
    }
};

//------------------------------------------------------------------------
// min_element, max_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare>
_Iterator
__pattern_min_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                      _Compare __comp)
{
    if (__first == __last)
        return __last;

    using _IteratorValueType = typename ::std::iterator_traits<_Iterator>::value_type;
    using _IndexValueType = ::std::make_unsigned_t<typename ::std::iterator_traits<_Iterator>::difference_type>;
    using _ReduceValueType = tuple<_IndexValueType, _IteratorValueType>;
    // Commutativity of the reduction operator depends on the compilation target (see __reduce_fn below);
    // __spirv_target_conditional postpones deciding on commutativity to the device code where the
    // target can be correctly tested.
    using _Commutative = oneapi::dpl::__internal::__spirv_target_conditional</*_SpirvT*/ ::std::false_type,
                                                                             /*_NonSpirvT*/ ::std::true_type>;
    __pattern_min_element_reduce_fn<_ReduceValueType, _Compare> __reduce_fn{__comp};
    oneapi::dpl::__internal::__pattern_min_element_transform_fn<_ReduceValueType> __transform_fn;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    auto __ret_idx = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType, _Commutative>(
                         _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
                         unseq_backend::__no_init_value{}, // no initial value
                         __buf.all_view())
                         .get();

    return __first + ::std::get<0>(__ret_idx);
}

// TODO:
//   The following minmax_element implementation
//   has at worst 2N applications of the predicate
//   whereas the standard says about (3/2)N applications.
//
//   The issue is in the first reduce iteration which make N comparison instead of possible N/2.
//   It takes place due to the way we initialize buffer in transform stage:
//      each ReduceValueType consists of {min_element_index, max_element_index, min_element_value, max_element_value}
//      and in the initial stage `__identity_init_fn` we take the same buffer element as the min element and max element
//      Thus, in the first iteration we have N element buffer to make N comparisons (min and max for each two ReduceValueType's)
//
//   One of possible solution for it is to make initial reduce of each two elements
//   to get N/2 element buffer with ReduceValueType's
//   resulting in N/2 comparisons in the first iteration (one comparison with stride=2 for N)
//   Thus, there will be sum( N/2 + N/2 + N/4 + N/8 + ... ) or (N/2 + N) comparisons
//   However the solution requires use of custom pattern or substantial redesign of existing parallel_transform_reduce.
//

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare>
::std::pair<_Iterator, _Iterator>
__pattern_minmax_element(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                         _Compare __comp)
{
    if (__first == __last)
        return ::std::make_pair(__first, __first);

    using _IteratorValueType = typename ::std::iterator_traits<_Iterator>::value_type;
    using _IndexValueType = ::std::make_unsigned_t<typename ::std::iterator_traits<_Iterator>::difference_type>;
    using _ReduceValueType = ::std::tuple<_IndexValueType, _IndexValueType, _IteratorValueType, _IteratorValueType>;

    // This operator doesn't track the lowest found index in case of equal min. values and the highest found index in
    // case of equal max. values. Thus, this operator is not commutative.
    oneapi::dpl::__internal::__pattern_minmax_element_reduce_fn<_Compare, _ReduceValueType> __reduce_fn{__comp};

    // TODO: Doesn't work with `zip_iterator`.
    //       In that case the first and the second arguments of `_ReduceValueType` will be
    //       a `tuple` of `difference_type`, not the `difference_type` itself.
    oneapi::dpl::__internal::__pattern_minmax_element_transform_fn<_ReduceValueType> __transform_fn;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    auto __ret = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                                ::std::false_type /*is_commutative*/>(
                     _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
                     unseq_backend::__no_init_value{}, // no initial value
                     __buf.all_view())
                     .get();

    return ::std::make_pair<_Iterator, _Iterator>(__first + ::std::get<0>(__ret), __first + ::std::get<1>(__ret));
}

//------------------------------------------------------------------------
// adjacent_find
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _BinaryPredicate,
          typename _OrFirstTag>
_Iterator
__pattern_adjacent_find(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                        _BinaryPredicate __pred, _OrFirstTag __is_or_semantic)
{
    const auto __n = __last - __first;
    if (__n < 2)
        return __last;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_BinaryPredicate>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last); //a scope lifetime of this instance should be.

    auto __view = __buf.all_view();
    auto __view1 = oneapi::dpl::__ranges::take_view_simple(__view, __view.size() - 1);
    auto __view2 = oneapi::dpl::__ranges::drop_view_simple(__view, 1);

    assert(__view1.size() == __view2.size());

    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_Iterator>::difference_type>;
    using _TagType = std::conditional_t<__is_or_semantic(), oneapi::dpl::__par_backend_hetero::__parallel_or_tag,
                                        oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<_IndexType>>;

    auto result =
        __par_backend_hetero::__parallel_find_or(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                 _Predicate{__pred}, _TagType{}, __size_calc{}, __view1, __view2);

    // inverted conditional because of
    // reorder_predicate in glue_algorithm_impl.h
    if constexpr (__is_or_semantic())
        return result ? __first : __last;
    else
        return __first + (result == __n - 1 ? __n : result);
}

//------------------------------------------------------------------------
// count, count_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
typename ::std::iterator_traits<_Iterator>::difference_type
__pattern_count(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                _Predicate __predicate)
{
    if (__first == __last)
        return 0;

    using _ReduceValueType = typename ::std::iterator_traits<_Iterator>::difference_type;

    auto __reduce_fn = ::std::plus<_ReduceValueType>{};
    // int is being implicitly casted to difference_type
    // otherwise we can only pass the difference_type as a functor template parameter
    oneapi::dpl::__internal::__pattern_count_transform_fn<_Predicate> __transform_fn{__predicate};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    return oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                          ::std::true_type /*is_commutative*/>(
               _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
               unseq_backend::__no_init_value{}, // no initial value
               __buf.all_view())
        .get();
}

//------------------------------------------------------------------------
// any_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Pred>
bool
__pattern_any_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                 _Pred __pred)
{
    if (__first == __last)
        return false;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_Pred>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred},
        __par_backend_hetero::__parallel_or_tag{}, __size_calc{}, __buf.all_view());
}

//------------------------------------------------------------------------
// equal
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
bool
__pattern_equal(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1,
                _Iterator2 __first2, _Iterator2 __last2, _Pred __pred)
{
    if (__last1 - __first1 != __last2 - __first2)
        return false;

    if (__last1 == __first1)
        return true; //both sequences are empty

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<oneapi::dpl::__internal::__not_pred<_Pred>>;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf1 = __keep(__first1, __last1);
    auto __buf2 = __keep(__first2, __last2);

    assert(__last1 - __first1 == __last2 - __first2);

    using size_calc = oneapi::dpl::__ranges::__first_size_calc;

    return !__par_backend_hetero::__parallel_find_or(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                     _Predicate{oneapi::dpl::__internal::__not_pred<_Pred>{__pred}},
                                                     __par_backend_hetero::__parallel_or_tag{}, size_calc{},
                                                     __buf1.all_view(), __buf2.all_view());
}

//------------------------------------------------------------------------
// equal version for sequences with equal length
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
bool
__pattern_equal(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1,
                _Iterator2 __first2, _Pred __pred)
{
    return oneapi::dpl::__internal::__pattern_equal(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first1, __last1,
                                                    __first2, __first2 + (__last1 - __first1), __pred);
}

//------------------------------------------------------------------------
// find_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Pred>
_Iterator
__pattern_find_if(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                  _Pred __pred)
{
    if (__first == __last)
        return __last;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_Iterator>::difference_type>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        __buf.all_view());

    return __first + __res;
}

//------------------------------------------------------------------------
// find_end
//------------------------------------------------------------------------

template <typename Name>
struct equal_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
_Iterator1
__pattern_find_end(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                   _Iterator2 __s_first, _Iterator2 __s_last, _Pred __pred)
{
    if (__first == __last || __s_last == __s_first || __last - __first < __s_last - __s_first)
        return __last;

    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __pattern_equal(
            __tag, __par_backend_hetero::make_wrapped_policy<equal_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
            __first, __last, __s_first, __pred);
        return __res ? __first : __last;
    }
    else
    {
        using _Predicate = unseq_backend::multiple_match_pred<_Pred>;
        using _IndexType = typename std::iterator_traits<_Iterator1>::difference_type;
        using _TagType = __par_backend_hetero::__parallel_find_backward_tag<_IndexType>;
        using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

        auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
        auto __buf1 = __keep(__first, __last);
        auto __buf2 = __keep(__s_first, __s_last);

        auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
            _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
            __buf1.all_view(), __buf2.all_view());

        return __first + __res;
    }
}

//------------------------------------------------------------------------
// find_first_of
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
_Iterator1
__pattern_find_first_of(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                        _Iterator2 __s_first, _Iterator2 __s_last, _Pred __pred)
{
    if (__first == __last || __s_last == __s_first)
        return __last;

    using _Predicate = unseq_backend::first_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_Iterator1>::difference_type>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    // TODO: To check whether it makes sense to iterate over the second sequence in case of
    // distance(__first, __last) < distance(__s_first, __s_last).

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf1 = __keep(__first, __last);
    auto __buf2 = __keep(__s_first, __s_last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        __buf1.all_view(), __buf2.all_view());

    return __first + __res;
}

//------------------------------------------------------------------------
// search
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
_Iterator1
__pattern_search(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                 _Iterator2 __s_first, _Iterator2 __s_last, _Pred __pred)
{
    if (__s_last == __s_first)
        return __first;

    if (__last - __first < __s_last - __s_first)
        return __last;

    if (__last - __first == __s_last - __s_first)
    {
        const bool __res = __pattern_equal(
            __tag, __par_backend_hetero::make_wrapped_policy<equal_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
            __first, __last, __s_first, __pred);
        return __res ? __first : __last;
    }

    using _Predicate = unseq_backend::multiple_match_pred<_Pred>;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_Iterator1>::difference_type>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf1 = __keep(__first, __last);
    auto __buf2 = __keep(__s_first, __s_last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred}, _TagType{}, __size_calc{},
        __buf1.all_view(), __buf2.all_view());

    return __first + __res;
}

//------------------------------------------------------------------------
// search_n
//------------------------------------------------------------------------

template <typename _Tp, typename _Pred>
struct __search_n_unary_predicate
{
    _Tp __value_;
    _Pred __pred_;

    template <typename _Value>
    bool
    operator()(const _Value& __val) const
    {
        return !__pred_(__val, __value_);
    }
};

template <typename Name>
struct any_of_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Size, typename _Tp,
          typename _BinaryPredicate>
_Iterator
__pattern_search_n(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                   _Size __count, const _Tp& __value, _BinaryPredicate __pred)
{
    if (__count <= 0)
        return __first;

    if (__last - __first < __count)
        return __last;

    if (__last - __first == __count)
    {
        return (!__internal::__pattern_any_of(
                   __tag,
                   __par_backend_hetero::make_wrapped_policy<any_of_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
                   __first, __last, __search_n_unary_predicate<_Tp, _BinaryPredicate>{__value, __pred}))
                   ? __first
                   : __last;
    }

    using _Predicate = unseq_backend::n_elem_match_pred<_BinaryPredicate, _Tp, _Size>;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_Iterator>::difference_type>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__pred, __value, __count}, _TagType{},
        __size_calc{}, __buf.all_view());

    return __first + __res;
}

//------------------------------------------------------------------------
// mismatch
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Pred>
::std::pair<_Iterator1, _Iterator2>
__pattern_mismatch(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1,
                   _Iterator2 __first2, _Iterator2 __last2, _Pred __pred)
{
    auto __n = std::min(__last1 - __first1, __last2 - __first2);
    if (__n <= 0)
        return std::make_pair(__first1, __first2);

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred<oneapi::dpl::__internal::__not_pred<_Pred>>;
    using _IndexType =
        std::make_unsigned_t<std::common_type_t<typename std::iterator_traits<_Iterator1>::difference_type,
                                                typename std::iterator_traits<_Iterator2>::difference_type>>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__min_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf1 = __keep(__first1, __last1);
    auto __buf2 = __keep(__first2, __last2);

    __n = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        _Predicate{oneapi::dpl::__internal::__not_pred<_Pred>{__pred}}, _TagType{}, __size_calc{}, __buf1.all_view(),
        __buf2.all_view());

    return std::make_pair(__first1 + __n, __first2 + __n);
}

//------------------------------------------------------------------------
// copy_if
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _Predicate>
_Iterator2
__pattern_copy_if(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                  _Iterator2 __result_first, _Predicate __pred)
{
    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;

    if (__first == __last)
        return __result_first;

    _It1DifferenceType __n = __last - __first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
    auto __buf2 = __keep2(__result_first, __result_first + __n);

    auto __res = __par_backend_hetero::__parallel_copy_if(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __buf1.all_view(), __buf2.all_view(), __n, __pred);

    ::std::size_t __num_copied = __res.get(); //is a blocking call
    return __result_first + __num_copied;
}

//------------------------------------------------------------------------
// partition_copy
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _Iterator3, typename _UnaryPredicate>
::std::pair<_Iterator2, _Iterator3>
__pattern_partition_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first,
                         _Iterator1 __last, _Iterator2 __result1, _Iterator3 __result2, _UnaryPredicate __pred)
{
    if (__first == __last)
        return ::std::make_pair(__result1, __result2);

    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;

    _It1DifferenceType __n = __last - __first;

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);

    auto __zipped_res = __par_backend_hetero::zip(
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__result1),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__result2));

    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, decltype(__zipped_res)>();
    auto __buf2 = __keep2(__zipped_res, __zipped_res + __n);

    auto __result = oneapi::dpl::__par_backend_hetero::__parallel_partition_copy(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(), __buf2.all_view(), __pred);

    _It1DifferenceType __num_true = __result.get(); // blocking call

    return std::make_pair(__result1 + __num_true, __result2 + (__last - __first - __num_true));
}

//------------------------------------------------------------------------
// unique_copy
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _BinaryPredicate>
_Iterator2
__pattern_unique_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first, _Iterator1 __last,
                      _Iterator2 __result_first, _BinaryPredicate __pred)
{
    using _It1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;

    _It1DifferenceType __n = __last - __first;

    if (__n == 0)
        return __result_first;
    if (__n == 1)
    {
        // For a sequence of size 1, we can just copy the only element to the result.
        oneapi::dpl::__internal::__pattern_walk2_brick(
            __hetero_tag<_BackendTag>{}, std::forward<_ExecutionPolicy>(__exec), __first, __last, __result_first,
            oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
        return __result_first + 1;
    }

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator2>();
    auto __buf2 = __keep2(__result_first, __result_first + __n);

    auto __result = oneapi::dpl::__par_backend_hetero::__parallel_unique_copy(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), __buf1.all_view(), __buf2.all_view(), __pred);

    return __result_first + __result.get(); // is a blocking call
}

template <typename _Name>
struct copy_back_wrapper;

template <typename _Name>
struct copy_back_wrapper2;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
_Iterator
__pattern_remove_if(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                    _Predicate __pred)
{
    if (__last == __first)
        return __last;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__last - __first);
    auto __copy_first = __buf.get();

    auto __copy_last = __pattern_copy_if(__tag, __exec, __first, __last, __copy_first, __not_pred<_Predicate>{__pred});

    //TODO: To optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer
    // __pattern_copy_if above may be async due to there is implicit synchronization on sycl::buffer and the accessors

    // The temporary buffer is constructed from a range, therefore it's destructor will not block, therefore
    // we must call __pattern_walk2 in a way which provides blocking synchronization for this pattern.
    return __pattern_walk2(
        __tag, __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_copy<__hetero_tag<_BackendTag>>{});
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _BinaryPredicate>
_Iterator
__pattern_unique(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                 _BinaryPredicate __pred)
{
    if (__last - __first < 2)
        return __last;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__last - __first);
    auto __copy_first = __buf.get();
    auto __copy_last = __pattern_unique_copy(__tag, __exec, __first, __last, __copy_first, __pred);

    //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer

    // The temporary buffer is constructed from a range, therefore it's destructor will not block, therefore
    // we must call __pattern_walk2 in a way which provides blocking synchronization for this pattern.
    // We never actually write to the sequence, so there is no risk when ran with the vectorized path of
    // walk_n_vector_or_scalars. For more info, please see the comment above __pattern_walk2 and
    // https://github.com/uxlfoundation/oneDPL/issues/1272.
    return __pattern_walk2</*_WaitMode*/ __par_backend_hetero::__deferrable_mode,
                           __par_backend_hetero::access_mode::read_write,
                           __par_backend_hetero::access_mode::read_write>(
        __tag, __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_copy<__hetero_tag<_BackendTag>>{});
}

//------------------------------------------------------------------------
// is_partitioned
//------------------------------------------------------------------------

enum _IsPartitionedReduceType : signed char
{
    __broken,
    __all_true,
    __all_false,
    __true_false
};

template <typename _ReduceValueType>
struct __pattern_is_partitioned_reduce_fn
{
    _IsPartitionedReduceType
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        _ReduceValueType __table[] = {__broken,     __broken,     __broken,     __broken, __broken,    __all_true,
                                      __true_false, __true_false, __broken,     __broken, __all_false, __broken,
                                      __broken,     __broken,     __true_false, __broken};
        return __table[__a * 4 + __b];
    }
};

template <typename _Predicate>
struct __pattern_is_partitioned_transform_fn
{
    _Predicate __predicate;

    template <typename _TGroupIdx, typename _TAcc>
    _IsPartitionedReduceType
    operator()(_TGroupIdx __gidx, _TAcc __acc) const
    {
        return (__predicate(__acc[__gidx]) ? __all_true : __all_false);
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Predicate>
bool
__pattern_is_partitioned(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                         _Predicate __predicate)
{
    if (__last - __first < 2)
        return true;

    using _ReduceValueType = _IsPartitionedReduceType;
    __pattern_is_partitioned_reduce_fn<_ReduceValueType> __reduce_fn;
    __pattern_is_partitioned_transform_fn<_Predicate> __transform_fn{__predicate};

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                                ::std::false_type /*is_commutative*/>(
                     _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
                     unseq_backend::__no_init_value{}, // no initial value
                     __buf.all_view())
                     .get();

    return __broken != __reduce_fn(_ReduceValueType{__all_true}, __res);
}

//------------------------------------------------------------------------
// is_heap / is_heap_until
//------------------------------------------------------------------------

template <class _Comp>
struct __is_heap_check
{
    mutable _Comp __comp_;

    template <class _Idx, class _Accessor>
    bool
    operator()(const _Idx __idx, const _Accessor& __acc) const
    {
        // Make sure that we have a signed integer here to avoid getting negative value when __idx == 0
        using _SignedIdx = ::std::make_signed_t<_Idx>;
        return __comp_(__acc[(static_cast<_SignedIdx>(__idx) - 1) / 2], __acc[__idx]);
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
_RandomAccessIterator
__pattern_is_heap_until(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                        _RandomAccessIterator __last, _Compare __comp)
{
    if (__last - __first < 2)
        return __last;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred_by_idx<__is_heap_check<_Compare>>;
    using _IndexType = std::make_unsigned_t<typename std::iterator_traits<_RandomAccessIterator>::difference_type>;
    using _TagType = __par_backend_hetero::__parallel_find_forward_tag<_IndexType>;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__par_backend_hetero::__parallel_find_or(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), _Predicate{__comp}, _TagType{}, __size_calc{},
        __buf.all_view());

    return __first + __res;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _RandomAccessIterator, typename _Compare>
bool
__pattern_is_heap(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _RandomAccessIterator __first,
                  _RandomAccessIterator __last, _Compare __comp)
{
    if (__last - __first < 2)
        return true;

    using _Predicate = oneapi::dpl::unseq_backend::single_match_pred_by_idx<__is_heap_check<_Compare>>;
    using _TagType = __par_backend_hetero::__parallel_or_tag;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf = __keep(__first, __last);

    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                                  _Predicate{__comp}, _TagType{}, __size_calc{},
                                                                  __buf.all_view());
}

//------------------------------------------------------------------------
// merge
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _Iterator3, typename _Compare>
_Iterator3
__pattern_merge(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __first1, _Iterator1 __last1,
                _Iterator2 __first2, _Iterator2 __last2, _Iterator3 __d_first, _Compare __comp)
{
    auto __n1 = __last1 - __first1;
    auto __n2 = __last2 - __first2;
    auto __n = __n1 + __n2;
    if (__n == 0)
        return __d_first;

    //To consider the direct copying pattern call in case just one of sequences is empty.
    if (__n1 == 0)
        oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __d_first, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    else if (__n2 == 0)
        oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<copy_back_wrapper2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __d_first, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    else
    {
        auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
        auto __buf1 = __keep1(__first1, __last1);
        auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
        auto __buf2 = __keep2(__first2, __last2);

        auto __keep3 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _Iterator3>();
        auto __buf3 = __keep3(__d_first, __d_first + __n);

        __par_backend_hetero::__parallel_merge(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                               __buf1.all_view(), __buf2.all_view(), __buf3.all_view(), __comp)
            .__checked_deferrable_wait();
    }
    return __d_first + __n;
}

//------------------------------------------------------------------------
// inplace_merge
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare>
void
__pattern_inplace_merge(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first,
                        _Iterator __middle, _Iterator __last, _Compare __comp)
{
    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    if (__first == __middle || __middle == __last || __first == __last)
        return;

    assert(__first < __middle && __middle < __last);

    auto __n = __last - __first;
    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__n);
    auto __copy_first = __buf.get();
    auto __copy_last = __copy_first + __n;

    __pattern_merge(
        __tag, __exec, __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__middle),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__middle),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read>(__last),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::write>(__copy_first), __comp);

    //TODO: optimize copy back depending on Iterator, i.e. set_final_data for host iterator/pointer

    // The temporary buffer is constructed from a range, therefore it's destructor will not block, therefore
    // we must call __pattern_walk2 in a way which provides blocking synchronization for this pattern.
    __pattern_walk2(
        __tag, __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(::std::forward<_ExecutionPolicy>(__exec)),
        __copy_first, __copy_last, __first, __brick_move<__hetero_tag<_BackendTag>>{});
}

//------------------------------------------------------------------------
// sort
//------------------------------------------------------------------------
template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare, typename _Proj>
void
__stable_sort_with_projection(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                              _Compare __comp, _Proj __proj)
{
    if (__last - __first < 2)
        return;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    __par_backend_hetero::__parallel_stable_sort(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                 __buf.all_view(), __comp, __proj)
        .__checked_deferrable_wait();
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare,
          typename _LeafSort = std::nullptr_t>
void
__pattern_sort(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
               _Compare __comp, _LeafSort = {})
{
    __stable_sort_with_projection(__tag, std::forward<_ExecutionPolicy>(__exec), __first, __last, __comp,
                                  oneapi::dpl::identity{});
}

//------------------------------------------------------------------------
// sort_by_key
//------------------------------------------------------------------------

struct __pattern_sort_by_key_fn
{
    template <typename _Arg>
    auto
    operator()(const _Arg& __a) const
    {
        return std::get<0>(__a);
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Compare,
          typename _LeafSort = std::nullptr_t>
void
__pattern_sort_by_key(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __keys_first,
                      _Iterator1 __keys_last, _Iterator2 __values_first, _Compare __comp, _LeafSort = {})
{
    static_assert(std::is_move_constructible_v<typename std::iterator_traits<_Iterator1>::value_type> &&
                      std::is_move_constructible_v<typename std::iterator_traits<_Iterator2>::value_type>,
                  "The keys and values should be move constructible in case of parallel execution.");

    auto __beg = oneapi::dpl::make_zip_iterator(__keys_first, __values_first);
    auto __end = __beg + (__keys_last - __keys_first);
    __stable_sort_with_projection(__tag, std::forward<_ExecutionPolicy>(__exec), __beg, __end, __comp,
                                  __pattern_sort_by_key_fn{});
}

//------------------------------------------------------------------------
// stable_partition
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _UnaryPredicate>
_Iterator
__pattern_stable_partition(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first,
                           _Iterator __last, _UnaryPredicate __pred)
{
    if (__last == __first)
        return __last;
    else if (__last - __first < 2)
        return __pattern_any_of(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __pred) ? __last
                                                                                                          : __first;

    using _ValueType = typename ::std::iterator_traits<_Iterator>::value_type;

    auto __n = __last - __first;

    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __true_buf(__n);
    oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __false_buf(__n);
    auto __true_result = __true_buf.get();
    auto __false_result = __false_buf.get();

    auto copy_result = __pattern_partition_copy(__tag, __exec, __first, __last, __true_result, __false_result, __pred);
    auto true_count = copy_result.first - __true_result;

    //TODO: optimize copy back if possible (inplace, decrease number of submits)
    __pattern_walk2(__tag, __par_backend_hetero::make_wrapped_policy<copy_back_wrapper>(__exec), __true_result,
                    copy_result.first, __first, __brick_move<__hetero_tag<_BackendTag>>{});

    __pattern_walk2(
        __tag, __par_backend_hetero::make_wrapped_policy<copy_back_wrapper2>(::std::forward<_ExecutionPolicy>(__exec)),
        __false_result, copy_result.second, __first + true_count, __brick_move<__hetero_tag<_BackendTag>>{});

    //TODO: A buffer is constructed from a range, the destructor does not need to block.
    // The synchronization between these patterns is not required due to the data are being processed independently.
    // So, sycl::event::wait(event1, event2) should be call. __pattern_walk2 calls above should be asynchronous and
    // return event1 and event2.

    return __first + true_count;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _UnaryPredicate>
_Iterator
__pattern_partition(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                    _UnaryPredicate __pred)
{
    //TODO: consider nonstable approaches
    return __pattern_stable_partition(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __last, __pred);
}

//------------------------------------------------------------------------
// lexicographical_compare
//------------------------------------------------------------------------

template <typename _ReduceValueType>
struct __pattern_lexicographical_compare_reduce_fn
{
    auto
    operator()(_ReduceValueType __a, _ReduceValueType __b) const
    {
        bool __is_mismatched = __a != 0;
        return __a * __is_mismatched + __b * !__is_mismatched;
    }
};

template <typename _Compare, typename _ReduceValueType>
struct __pattern_lexicographical_compare_transform_fn
{
    _Compare __comp;

    template <typename _TGroupIdx, typename _TAcc1, typename _TAcc2>
    _ReduceValueType
    operator()(_TGroupIdx __gidx, _TAcc1 __acc1, _TAcc2 __acc2) const
    {
        auto const& __s1_val = __acc1[__gidx];
        auto const& __s2_val = __acc2[__gidx];

        ::std::int32_t __is_s1_val_less = __comp(__s1_val, __s2_val);
        ::std::int32_t __is_s1_val_greater = __comp(__s2_val, __s1_val);

        // 1 if __s1_val <  __s2_val, -1 if __s1_val <  __s2_val, 0 if __s1_val == __s2_val
        return _ReduceValueType{1 * __is_s1_val_less - 1 * __is_s1_val_greater};
    }
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2, typename _Compare>
bool
__pattern_lexicographical_compare(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator1 __first1,
                                  _Iterator1 __last1, _Iterator2 __first2, _Iterator2 __last2, _Compare __comp)
{
    //trivial pre-checks
    if (__first2 == __last2)
        return false;
    if (__first1 == __last1)
        return true;

    using _Iterator1DifferenceType = typename ::std::iterator_traits<_Iterator1>::difference_type;
    using _ReduceValueType = int32_t;

    __pattern_lexicographical_compare_reduce_fn<_ReduceValueType> __reduce_fn;
    __pattern_lexicographical_compare_transform_fn<_Compare, _ReduceValueType> __transform_fn{__comp};

    auto __shared_size = ::std::min(__last1 - __first1, (_Iterator1DifferenceType)(__last2 - __first2));

    auto __keep1 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __buf1 = __keep1(__first1, __first1 + __shared_size);

    auto __keep2 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __buf2 = __keep2(__first2, __first2 + __shared_size);

    auto __ret_idx =
        oneapi::dpl::__par_backend_hetero::__parallel_transform_reduce<_ReduceValueType,
                                                                       ::std::false_type /*is_commutative*/>(
            _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec), __reduce_fn, __transform_fn,
            unseq_backend::__no_init_value{}, // no initial value
            __buf1.all_view(), __buf2.all_view())
            .get();

    return __ret_idx ? __ret_idx == 1 : (__last1 - __first1) < (__last2 - __first2);
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _Compare>
bool
__pattern_includes(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                   _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2, _Compare __comp)
{
    //according to the spec
    if (__first2 == __last2)
        return true;

    const auto __n1 = __last1 - __first1;
    const auto __n2 = __last2 - __first2;
    //optimization; {1} - the first sequence, {2} - the second sequence
    //{1} is empty or size_of{2} > size_of{1}
    if (__n1 == 0 || __n2 > __n1)
        return false;

    using __brick_include_type = unseq_backend::__brick_includes<_Compare, decltype(__n1), decltype(__n2)>;
    using _TagType = __par_backend_hetero::__parallel_or_tag;
    using __size_calc = oneapi::dpl::__ranges::__first_size_calc;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read>();
    auto __buf1 = __keep(__first1, __last1);
    auto __buf2 = __keep(__first2, __last2);

    return !oneapi::dpl::__par_backend_hetero::__parallel_find_or(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                                  __brick_include_type{__comp, __n1, __n2}, _TagType{},
                                                                  __size_calc{}, __buf2.all_view(), __buf1.all_view());
}

//------------------------------------------------------------------------
// partial_sort
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare>
void
__pattern_partial_sort(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __mid,
                       _Iterator __last, _Compare __comp)
{
    if (__last - __first < 2)
        return;

    __par_backend_hetero::__parallel_partial_sort(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__first),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__mid),
        __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__last), __comp)
        .__checked_deferrable_wait();
}

//------------------------------------------------------------------------
// partial_sort_copy
//------------------------------------------------------------------------

template <typename _KernelName>
struct __initial_copy_1
{
};

template <typename _KernelName>
struct __initial_copy_2
{
};

template <typename _KernelName>
struct __copy_back
{
};

template <typename _KernelName>
struct __partial_sort_1
{
};

template <typename _KernelName>
struct __partial_sort_2
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _InIterator, typename _OutIterator,
          typename _Compare>
_OutIterator
__pattern_partial_sort_copy(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _InIterator __first,
                            _InIterator __last, _OutIterator __out_first, _OutIterator __out_last, _Compare __comp)
{
    using _ValueType = typename ::std::iterator_traits<_InIterator>::value_type;

    auto __in_size = __last - __first;
    auto __out_size = __out_last - __out_first;

    if (__in_size == 0 || __out_size == 0)
        return __out_first;

    // TODO: we can avoid a separate __pattern_walk2 for initial copy: it can be done during sort itself
    // like it's done for CPU version, but it's better to be done together with merge cutoff implementation
    // as it uses a similar mechanism.
    if (__in_size <= __out_size)
    {
        // If our output buffer is larger than the input buffer, simply copy elements to the output and use
        // full sort on them.
        auto __out_end = __pattern_walk2<__par_backend_hetero::__sync_mode>(
            __tag, __par_backend_hetero::make_wrapped_policy<__initial_copy_1>(__exec), __first, __last, __out_first,
            __brick_copy<__hetero_tag<_BackendTag>>{});

        // TODO: __pattern_walk2 is a blocking call here, so there is a synchronization between the patterns.
        // But, when the input iterators are a kind of hetero iterator on top of sycl::buffer, SYCL
        // runtime makes a dependency graph. In that case the call of __pattern_walk2 could be changed to
        // be asynchronous for better performance.

        // Use regular sort as partial_sort isn't required to be stable.
        //__pattern_sort is a blocking call.
        __pattern_sort(
            __tag,
            __par_backend_hetero::make_wrapped_policy<__partial_sort_1>(::std::forward<_ExecutionPolicy>(__exec)),
            __out_first, __out_end, __comp);

        return __out_end;
    }
    else
    {
        // If our input buffer is smaller than the input buffer do the following:
        // - create a temporary buffer and copy all the elements from the input buffer there
        // - run partial sort on the temporary buffer
        // - copy k elements from the temporary buffer to the output buffer.
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __buf(__in_size);

        auto __buf_first = __buf.get();

        auto __buf_last = __pattern_walk2<__par_backend_hetero::__async_mode>(
            __tag, __par_backend_hetero::make_wrapped_policy<__initial_copy_2>(__exec), __first, __last, __buf_first,
            __brick_copy<__hetero_tag<_BackendTag>>{});

        auto __buf_mid = __buf_first + __out_size;

        // An explicit wait between the patterns isn't required here because we are working a with temporary
        // sycl::buffer and sycl accessors. SYCL runtime makes a dependency graph to prevent the races between
        // the patterns: __pattern_walk2, __parallel_partial_sort and __pattern_walk2.

        __par_backend_hetero::__parallel_partial_sort(
            _BackendTag{}, __par_backend_hetero::make_wrapped_policy<__partial_sort_2>(__exec),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_first),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_mid),
            __par_backend_hetero::make_iter_mode<__par_backend_hetero::access_mode::read_write>(__buf_last), __comp);

        return __pattern_walk2(
            __tag, __par_backend_hetero::make_wrapped_policy<__copy_back>(::std::forward<_ExecutionPolicy>(__exec)),
            __buf_first, __buf_mid, __out_first, __brick_copy<__hetero_tag<_BackendTag>>{});

        // The temporary buffer is constructed from a range, therefore it's destructor will not block, therefore
        // we must call __pattern_walk2 in a way which provides blocking synchronization for this pattern.
    }
}

//------------------------------------------------------------------------
// nth_element
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator, typename _Compare>
void
__pattern_nth_element(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __nth,
                      _Iterator __last, _Compare __comp)
{
    if (__first == __last || __nth == __last)
        return;

    // TODO: check partition-based implementation
    // - try to avoid host dereference issue
    // - measure performance of the issue-free implementation
    __pattern_partial_sort(__tag, ::std::forward<_ExecutionPolicy>(__exec), __first, __nth + 1, __last, __comp);
}

//------------------------------------------------------------------------
// reverse
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator>
void
__pattern_reverse(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last)
{
    auto __n = __last - __first;
    if (__n <= 1)
        return;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__reverse_functor<typename std::iterator_traits<_Iterator>::difference_type>{__n}, __n / 2,
        __buf.all_view())
        .__checked_deferrable_wait();
}

//------------------------------------------------------------------------
// reverse_copy
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _BidirectionalIterator, typename _ForwardIterator>
_ForwardIterator
__pattern_reverse_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _BidirectionalIterator __first,
                       _BidirectionalIterator __last, _ForwardIterator __result)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __result;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _BidirectionalIterator>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator>();
    auto __buf2 = __keep2(__result, __result + __n);
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__reverse_copy<typename std::iterator_traits<_BidirectionalIterator>::difference_type>{__n}, __n,
        __buf1.all_view(), __buf2.all_view())
        .__checked_deferrable_wait();

    return __result + __n;
}

//------------------------------------------------------------------------
// rotate
//------------------------------------------------------------------------
//Advantages over "3x reverse" version of algorithm:
//1:Not sensitive to size of shift
//  (With 3x reverse was large variance)
//2:The average time is better until ~10e8 elements
//Wrapper needed to avoid kernel problems
template <typename Name>
struct __rotate_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator>
_Iterator
__pattern_rotate(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __new_first,
                 _Iterator __last)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __first;

    using _Tp = typename ::std::iterator_traits<_Iterator>::value_type;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);
    auto __temp_buf = oneapi::dpl::__par_backend_hetero::__buffer<_Tp>(__n);

    auto __temp_rng_w =
        oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::write>(__temp_buf.get_buffer());

    const auto __shift = __new_first - __first;
    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__rotate_wrapper>(__exec),
        unseq_backend::__rotate_copy<typename std::iterator_traits<_Iterator>::difference_type>{__n, __shift}, __n,
        __buf.all_view(), __temp_rng_w);

    //An explicit wait isn't required here because we are working with a temporary sycl::buffer and sycl accessors and
    //SYCL runtime makes a dependency graph to prevent the races between two __parallel_for patterns.

    using _Function = __brick_move<__hetero_tag<_BackendTag>>;
    auto __temp_rng_rw =
        oneapi::dpl::__ranges::all_view<_Tp, __par_backend_hetero::access_mode::read_write>(__temp_buf.get_buffer());
    auto __brick = unseq_backend::walk_n_vectors_or_scalars<_Function>{_Function{}, static_cast<std::size_t>(__n)};
    oneapi::dpl::__par_backend_hetero::__parallel_for(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec), __brick,
                                                      __n, __temp_rng_rw, __buf.all_view())
        .__checked_deferrable_wait();

    // The temporary buffer is constructed from a range, therefore it's destructor will not block, therefore
    // we must call __parallel_for with wait() to provide the blocking synchronization for this pattern.

    return __first + (__last - __new_first);
}

//------------------------------------------------------------------------
// rotate_copy
//------------------------------------------------------------------------

template <typename _BackendTag, typename _ExecutionPolicy, typename _BidirectionalIterator, typename _ForwardIterator>
_ForwardIterator
__pattern_rotate_copy(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _BidirectionalIterator __first,
                      _BidirectionalIterator __new_first, _BidirectionalIterator __last, _ForwardIterator __result)
{
    auto __n = __last - __first;
    if (__n <= 0)
        return __result;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _BidirectionalIterator>();
    auto __buf1 = __keep1(__first, __last);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _ForwardIterator>();
    auto __buf2 = __keep2(__result, __result + __n);

    const auto __shift = __new_first - __first;

    oneapi::dpl::__par_backend_hetero::__parallel_for(
        _BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
        unseq_backend::__rotate_copy<typename std::iterator_traits<_BidirectionalIterator>::difference_type>{__n,
                                                                                                             __shift},
        __n, __buf1.all_view(), __buf2.all_view())
        .__checked_deferrable_wait();

    return __result + __n;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare, typename _SetTag>
_OutputIterator
__pattern_hetero_set_op(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                        _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                        _OutputIterator __result, _Compare __comp, _SetTag __set_tag)
{
    typedef typename std::iterator_traits<_ForwardIterator1>::difference_type _Size1;

    const _Size1 __n1 = std::distance(__first1, __last1);
    _Size1 __output_size = __n1;
    if constexpr (_SetTag::__can_write_from_rng2_v)
    {
        const _Size1 __n2 = std::distance(__first2, __last2);
        // one shot algorithm can write from set 1 or set 2, whereas old algorithm can only write from set 1.
        __output_size = __n1 + __n2;
    }

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator1>();
    auto __buf1 = __keep1(__first1, __last1);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _ForwardIterator2>();
    auto __buf2 = __keep2(__first2, __last2);

    auto __keep3 = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutputIterator>();
    auto __buf3 = __keep3(__result, __result + __output_size);

    auto __result_size = __par_backend_hetero::__parallel_set_op(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                                 __buf1.all_view(), __buf2.all_view(),
                                                                 __buf3.all_view(), __comp, __set_tag)
                             .get();

    return __result + __result_size;
}

template <typename Name>
struct __set_intersection_scan_then_propagate
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare>
_OutputIterator
__pattern_set_intersection(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                           _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                           _OutputIterator __result, _Compare __comp)
{
    // intersection is empty
    if (__first1 == __last1 || __first2 == __last2)
        return __result;
    if (__par_backend_hetero::__can_set_op_write_from_set_b(_BackendTag{}, __exec))
    {
        return __pattern_hetero_set_op(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                       __last2, __result, __comp, unseq_backend::_IntersectionTag<std::true_type>());
    }
    return __pattern_hetero_set_op(
        __tag,
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_intersection_scan_then_propagate>(
            std::forward<_ExecutionPolicy>(__exec)),
        __first1, __last1, __first2, __last2, __result, __comp, unseq_backend::_IntersectionTag<std::false_type>());
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_difference_copy_case_1
{
};

template <typename Name>
struct __set_difference_scan_then_propagate
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare>
_OutputIterator
__pattern_set_difference(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                         _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                         _OutputIterator __result, _Compare __comp)
{
    // {} \ {2}: the difference is empty
    if (__first1 == __last1)
        return __result;

    // {1} \ {}: the difference is {1}
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_difference_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    }
    if (__par_backend_hetero::__can_set_op_write_from_set_b(_BackendTag{}, __exec))
    {
        return __pattern_hetero_set_op(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_difference_scan_then_propagate>(
                std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __first2, __last2, __result, __comp, unseq_backend::_DifferenceTag<std::true_type>());
    }
    else
    {
        return __pattern_hetero_set_op(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                       __last2, __result, __comp, unseq_backend::_DifferenceTag<std::false_type>());
    }
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_union_copy_case_1
{
};

template <typename Name>
struct __set_union_copy_case_2
{
};

template <typename Name>
struct __set_union_scan_then_propagate
{
};

template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare>
_OutputIterator
__pattern_set_union(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _ForwardIterator1 __first1,
                    _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2,
                    _OutputIterator __result, _Compare __comp)
{
    if (__first1 == __last1 && __first2 == __last2)
        return __result;

    //{1} is empty
    if (__first1 == __last1)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __result, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    }

    //{2} is empty
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    }

    if (__par_backend_hetero::__can_set_op_write_from_set_b(_BackendTag{}, __exec))
    {
        return __pattern_hetero_set_op(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                       __last2, __result, __comp, unseq_backend::_UnionTag<std::true_type>());
    }
    else
    {
        using _ValueType = typename std::iterator_traits<_OutputIterator>::value_type;

        // temporary buffer to store intermediate result
        const auto __n2 = __last2 - __first2;
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __diff(__n2);
        auto __buf = __diff.get();

        //1. Calc difference {2} \ {1}
        const auto __n_diff =
            oneapi::dpl::__internal::__pattern_hetero_set_op(
                __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_scan_then_propagate>(__exec),
                __first2, __last2, __first1, __last1, __buf, __comp, unseq_backend::_DifferenceTag<std::false_type>()) -
            __buf;

        //2. Merge {1} and the difference
        return oneapi::dpl::__internal::__pattern_merge(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_union_copy_case_2>(
                std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __buf, __buf + __n_diff, __result, __comp);
    }
}

//Dummy names to avoid kernel problems
template <typename Name>
struct __set_symmetric_difference_copy_case_1
{
};

template <typename Name>
struct __set_symmetric_difference_copy_case_2
{
};

template <typename Name>
struct __set_symmetric_difference_phase_1
{
};

template <typename Name>
struct __set_symmetric_difference_phase_2
{
};

//------------------------------------------------------------------------
// set_symmetric_difference
//------------------------------------------------------------------------
// At the moment the algo implementation based on 3 phases:
// 1. Calc difference {1} \ {2}
// 2. Calc difference {2} \ {1}
// 3. Merge the differences
template <typename _BackendTag, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare>
_OutputIterator
__pattern_set_symmetric_difference(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec,
                                   _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                   _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    if (__first1 == __last1 && __first2 == __last2)
        return __result;

    //{1} is empty
    if (__first1 == __last1)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_1>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first2, __last2, __result, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    }

    //{2} is empty
    if (__first2 == __last2)
    {
        return oneapi::dpl::__internal::__pattern_walk2_brick(
            __tag,
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_copy_case_2>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __first1, __last1, __result, oneapi::dpl::__internal::__brick_copy<__hetero_tag<_BackendTag>>{});
    }

    if (__par_backend_hetero::__can_set_op_write_from_set_b(_BackendTag{}, __exec))
    {
        return __pattern_hetero_set_op(__tag, std::forward<_ExecutionPolicy>(__exec), __first1, __last1, __first2,
                                       __last2, __result, __comp,
                                       unseq_backend::_SymmetricDifferenceTag<std::true_type>());
    }
    else
    {
        typedef typename std::iterator_traits<_OutputIterator>::value_type _ValueType;

        // temporary buffers to store intermediate result
        const auto __n1 = __last1 - __first1;
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __diff_1(__n1);
        auto __buf_1 = __diff_1.get();
        const auto __n2 = __last2 - __first2;
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueType> __diff_2(__n2);
        auto __buf_2 = __diff_2.get();

        //1. Calc difference {1} \ {2}
        const auto __n_diff_1 =
            oneapi::dpl::__internal::__pattern_hetero_set_op(
                __tag,
                oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_phase_1>(__exec),
                __first1, __last1, __first2, __last2, __buf_1, __comp,
                unseq_backend::_DifferenceTag<std::false_type>()) -
            __buf_1;

        //2. Calc difference {2} \ {1}
        const auto __n_diff_2 =
            oneapi::dpl::__internal::__pattern_hetero_set_op(
                __tag,
                oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__set_symmetric_difference_phase_2>(__exec),
                __first2, __last2, __first1, __last1, __buf_2, __comp,
                unseq_backend::_DifferenceTag<std::false_type>()) -
            __buf_2;

        //3. Merge the differences
        return oneapi::dpl::__internal::__pattern_merge(__tag, std::forward<_ExecutionPolicy>(__exec), __buf_1,
                                                        __buf_1 + __n_diff_1, __buf_2, __buf_2 + __n_diff_2, __result,
                                                        __comp);
    }
}

template <typename _Name>
struct __shift_left_right;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Range>
oneapi::dpl::__internal::__difference_t<_Range>
__pattern_shift_left(__hetero_tag<_BackendTag>, _ExecutionPolicy&& __exec, _Range __rng,
                     oneapi::dpl::__internal::__difference_t<_Range> __n)
{
    //If (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first. Otherwise, returns last.
    using _DiffType = oneapi::dpl::__internal::__difference_t<_Range>;
    _DiffType __size = __rng.size();

    assert(__n > 0 && __n < __size);

    _DiffType __mid = __size / 2 + __size % 2;
    _DiffType __size_res = __size - __n;

    //1. n >= size/2; 'size - _n' parallel copying
    if (__n >= __mid)
    {
        using _Function = __brick_move<__hetero_tag<_BackendTag>>;

        //TODO: to consider use just "read" access mode for a source range and just "write" - for a destination range.
        auto __src = oneapi::dpl::__ranges::drop_view_simple<_Range, _DiffType>(__rng, __n);
        auto __dst = oneapi::dpl::__ranges::take_view_simple<_Range, _DiffType>(__rng, __size_res);

        auto __brick =
            unseq_backend::walk_n_vectors_or_scalars<_Function>{_Function{}, static_cast<std::size_t>(__size_res)};

        oneapi::dpl::__par_backend_hetero::__parallel_for(_BackendTag{}, ::std::forward<_ExecutionPolicy>(__exec),
                                                          __brick, __size_res, __src, __dst)
            .__checked_deferrable_wait();
    }
    else //2. n < size/2; 'n' parallel copying
    {
        auto __brick = unseq_backend::__brick_shift_left<_DiffType>{__size, __n};
        oneapi::dpl::__par_backend_hetero::__parallel_for(
            _BackendTag{},
            oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__shift_left_right>(
                ::std::forward<_ExecutionPolicy>(__exec)),
            __brick, __n, __rng)
            .__checked_deferrable_wait();
    }

    return __size_res;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator>
_Iterator
__pattern_shift_left(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                     typename ::std::iterator_traits<_Iterator>::difference_type __n)
{
    //If (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first. Otherwise, returns last.
    auto __size = __last - __first;
    if (__n <= 0)
        return __last;
    if (__n >= __size)
        return __first;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    auto __res = oneapi::dpl::__internal::__pattern_shift_left(__tag, ::std::forward<_ExecutionPolicy>(__exec),
                                                               __buf.all_view(), __n);
    return __first + __res;
}

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator>
_Iterator
__pattern_shift_right(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator __first, _Iterator __last,
                      typename ::std::iterator_traits<_Iterator>::difference_type __n)
{
    //If (n > 0 && n < m), returns first + n. Otherwise, if n  > 0, returns last. Otherwise, returns first.
    auto __size = __last - __first;
    if (__n <= 0)
        return __first;
    if (__n >= __size)
        return __last;

    auto __keep = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator>();
    auto __buf = __keep(__first, __last);

    //A shift right is the shift left with a reverse logic.
    auto __rng = oneapi::dpl::__ranges::reverse_view_simple<decltype(__buf.all_view())>{__buf.all_view()};
    auto __res =
        oneapi::dpl::__internal::__pattern_shift_left(__tag, ::std::forward<_ExecutionPolicy>(__exec), __rng, __n);

    return __last - __res;
}

template <typename _Name>
struct __copy_keys_values_wrapper;

template <typename _BackendTag, typename _ExecutionPolicy, typename _Iterator1, typename _Iterator2,
          typename _Iterator3, typename _Iterator4, typename _BinaryPredicate, typename _BinaryOperator>
typename std::iterator_traits<_Iterator3>::difference_type
__pattern_reduce_by_segment(__hetero_tag<_BackendTag> __tag, _ExecutionPolicy&& __exec, _Iterator1 __keys_first,
                            _Iterator1 __keys_last, _Iterator2 __values_first, _Iterator3 __out_keys_first,
                            _Iterator4 __out_values_first, _BinaryPredicate __binary_pred, _BinaryOperator __binary_op)
{
    const std::size_t __n = std::distance(__keys_first, __keys_last);

    if (__n == 0)
        return 0;

    if (__n == 1)
    {
        __brick_copy<__hetero_tag<_BackendTag>> __copy_op{};

        oneapi::dpl::__internal::__pattern_walk2_n(
            __tag, oneapi::dpl::__par_backend_hetero::make_wrapped_policy<__copy_keys_values_wrapper>(__exec),
            oneapi::dpl::make_zip_iterator(__keys_first, __values_first), 1,
            oneapi::dpl::make_zip_iterator(__out_keys_first, __out_values_first), __copy_op);

        return 1;
    }

    auto __keep_keys = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator1>();
    auto __keys = __keep_keys(__keys_first, __keys_last);
    auto __keep_values = oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _Iterator2>();
    auto __values = __keep_values(__values_first, __values_first + __n);
    auto __keep_key_outputs =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator3>();
    auto __out_keys = __keep_key_outputs(__out_keys_first, __out_keys_first + __n);
    auto __keep_value_outputs =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read_write, _Iterator4>();
    auto __out_values = __keep_value_outputs(__out_values_first, __out_values_first + __n);
    return oneapi::dpl::__par_backend_hetero::__parallel_reduce_by_segment(
        _BackendTag{}, std::forward<_ExecutionPolicy>(__exec), __keys.all_view(), __values.all_view(),
        __out_keys.all_view(), __out_values.all_view(), __binary_pred, __binary_op);
}

template <typename _BackendTag, typename _Policy, typename _InputIterator1, typename _InputIterator2,
          typename _OutputIterator, typename _T, typename _BinaryPredicate, typename _Operator, typename _Inclusive>
_OutputIterator
__pattern_scan_by_segment(__hetero_tag<_BackendTag>, _Policy&& __policy, _InputIterator1 __first1,
                          _InputIterator1 __last1, _InputIterator2 __first2, _OutputIterator __result, _T __init,
                          _BinaryPredicate __binary_pred, _Operator __binary_op, _Inclusive)
{
    const auto __n = std::distance(__first1, __last1);

    // Check for empty element ranges
    if (__n <= 0)
        return __result;

    namespace __bknd = oneapi::dpl::__par_backend_hetero;

    auto __keep_keys = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, _InputIterator1>();
    auto __key_buf = __keep_keys(__first1, __last1);
    auto __keep_values = oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read, _InputIterator2>();
    auto __value_buf = __keep_values(__first2, __first2 + __n);
    auto __keep_value_outputs =
        oneapi::dpl::__ranges::__get_sycl_range<__bknd::access_mode::read_write, _OutputIterator>();
    auto __value_output_buf = __keep_value_outputs(__result, __result + __n);
    using _IterValueType = typename std::iterator_traits<_InputIterator2>::value_type;

    // Currently, this pattern requires a known identity for the binary operator.
    static_assert(unseq_backend::__has_known_identity<_Operator, _IterValueType>::value,
                  "Calls to __pattern_scan_by_segment require a known identity for the provided binary operator");
    constexpr _IterValueType __identity = unseq_backend::__known_identity<_Operator, _IterValueType>;

    __bknd::__parallel_scan_by_segment<_Inclusive::value>(
        _BackendTag{}, std::forward<_Policy>(__policy), __key_buf.all_view(), __value_buf.all_view(),
        __value_output_buf.all_view(), __binary_pred, __binary_op, __init, __identity);
    return __result + __n;
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_ALGORITHM_IMPL_HETERO_H
