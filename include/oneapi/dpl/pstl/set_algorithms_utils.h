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

#ifndef __ONEDPL_SET_ALGORITHMS_UTILS_H
#define __ONEDPL_SET_ALGORITHMS_UTILS_H

#include <cstdint>     // for std::uint8_t
#include <memory>      // for std::addressof
#include <tuple>       // for std::tuple
#include <type_traits> // for std::is_same_v
#include <algorithm>

#if _ONEDPL_CPP20_RANGES_PRESENT
#    include <ranges> // for std::ranges::in_in_out_result, std::ranges::in_out_result, std::ranges::borrowed_iterator_t
#endif

#include "functional_impl.h"
#include "iterator_impl.h" // for oneapi::dpl::discard_iterator

namespace oneapi
{
namespace dpl
{
namespace __utils
{
enum class __parallel_set_op_mask : std::uint8_t
{
    none = 0x00,     // initial state
    data1 = 0x01,    // mask for first input data item usage
    data2 = 0x02,    // mask for second input data item usage
    data_out = 0x04, // mask for output data item usage

    both = 0x03,      // data1 | data2: mask for both input data items usage
    data1_out = 0x05, // data1 | data_out: mask for copy data item from the first data set into output
    data2_out = 0x06, // data2 | data_out: mask for copy data item from the second data set into output
    both_out = 0x07   // both  | data_out: mask for copy data item from the first and the second data set into output
};

namespace __internal
{
inline std::nullptr_t
__set_iterator_mask(std::nullptr_t, __parallel_set_op_mask) noexcept
{
    return nullptr;
}

inline __parallel_set_op_mask*
__set_iterator_mask(__parallel_set_op_mask* __mask, __parallel_set_op_mask __state) noexcept
{
    *__mask = __state;
    return ++__mask;
}

template <typename _Size>
inline std::nullptr_t
__set_iterator_mask_n(std::nullptr_t, __parallel_set_op_mask, _Size) noexcept
{
    return nullptr;
}

template <typename _Size>
inline __parallel_set_op_mask*
__set_iterator_mask_n(__parallel_set_op_mask* __mask, __parallel_set_op_mask __state, _Size __count) noexcept
{
    for (_Size __i = 0; __i < __count; ++__i)
        __mask[__i] = __state;

    return __mask + __count;
}

template <typename _InputIterator, typename _OutputIterator>
void
__uninitialized_copy_or_discard(_InputIterator __it_in, _OutputIterator __it_out)
{
    using _OutValueType = typename std::iterator_traits<_OutputIterator>::value_type;
    if constexpr (!std::is_same_v<_OutputIterator, oneapi::dpl::discard_iterator>)
    {
        // We should use placement new here because this method really works with raw uninitialized memory
        new (std::addressof(*__it_out)) _OutValueType(*__it_in);
    }
}
} // namespace __internal

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator, typename _MaskIterator>
using __set_construct_return_t = std::tuple<_ForwardIterator1, _ForwardIterator2, _OutputIterator, _MaskIterator>;

// ATTENTION.
// Real data generation and mask generation in this function depend on _OutputIterator and _MaskIterator.
// If _OutputIterator is oneapi::dpl::discard_iterator, no data will be generated.
// If _MaskIterator is std::nullptr_t, no mask will be generated.
// It is expected that in this case the caller doesn't need a mask at all and simply ignores it.
// The same behavior applies to all four set-op functions: __set_union_construct, __set_intersection_construct,
// __set_difference_construct and __set_symmetric_difference_construct.
template <typename _CopyConstructRange, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare, typename _Proj1, typename _Proj2, typename _MaskIterator>
__set_construct_return_t<_ForwardIterator1, _ForwardIterator2, _OutputIterator, _MaskIterator>
__set_union_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                      _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _Proj1 __proj1,
                      _Proj2 __proj2, _MaskIterator __mask)
{
    _CopyConstructRange __cc_range;

    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
        {
            __mask = __internal::__set_iterator_mask_n(__mask, __parallel_set_op_mask::data1_out, __last1 - __first1);
            __result = __cc_range(__first1, __last1, __result);

            return {__last1, __first2, __result, __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
        {
            __internal::__uninitialized_copy_or_discard(__first2, __result);
            ++__first2;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data2_out);
        }
        else
        {
            __internal::__uninitialized_copy_or_discard(__first1, __result);
            if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
            {
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data1_out);
            }
            else
            {
                ++__first2;
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::both_out);
            }
            ++__first1;
        }
    }

    __mask = __internal::__set_iterator_mask_n(__mask, __parallel_set_op_mask::data2_out, __last2 - __first2);
    __result = __cc_range(__first2, __last2, __result);

    return {__first1, __last2, __result, __mask};
}

template <typename _ForwardIterator1, typename _ForwardIterator2, typename _OutputIterator,
          typename _Compare, typename _Proj1, typename _Proj2, typename _MaskIterator>
__set_construct_return_t<_ForwardIterator1, _ForwardIterator2, _OutputIterator, _MaskIterator>
__set_intersection_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                             _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _Proj1 __proj1,
                             _Proj2 __proj2, _MaskIterator __mask)
{
    while (__first1 != __last1 && __first2 != __last2)
    {
        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            ++__first1;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data1);
        }
        else if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
        {
            ++__first2;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data2);
        }
        else
        {
            __internal::__uninitialized_copy_or_discard(__first1, __result);
            ++__first1;
            ++__first2;
            ++__result;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::both_out);
        }
    }

    return {__first1, __first2, __result, __mask};
}

template <typename _CopyConstructRange, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare, typename _Proj1, typename _Proj2, typename _MaskIterator>
__set_construct_return_t<_ForwardIterator1, _ForwardIterator2, _OutputIterator, _MaskIterator>
__set_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                           _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp, _Proj1 __proj1,
                           _Proj2 __proj2, _MaskIterator __mask)
{
    _CopyConstructRange __cc_range;

    while (__first1 != __last1)
    {
        if (__first2 == __last2)
        {
            __mask = __internal::__set_iterator_mask_n(__mask, __parallel_set_op_mask::data1_out, __last1 - __first1);
            __result = __cc_range(__first1, __last1, __result);

            return {__last1, __first2, __result, __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            __internal::__uninitialized_copy_or_discard(__first1, __result);
            ++__result;
            ++__first1;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data1_out);
        }
        else
        {
            if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
            {
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data2);
            }
            else
            {
                ++__first1;
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::both);
            }
            ++__first2;
        }
    }

    return {__first1, __first2, __result, __mask};
}

template <typename _CopyConstructRange, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _Compare, typename _Proj1, typename _Proj2, typename _MaskIterator>
__set_construct_return_t<_ForwardIterator1, _ForwardIterator2, _OutputIterator, _MaskIterator>
__set_symmetric_difference_construct(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                                     _ForwardIterator2 __last2, _OutputIterator __result, _Compare __comp,
                                     _Proj1 __proj1, _Proj2 __proj2, _MaskIterator __mask)
{
    _CopyConstructRange __cc_range;

    while (__first1 != __last1)
    {
        if (__first2 == __last2)
        {
            __mask = __internal::__set_iterator_mask_n(__mask, __parallel_set_op_mask::data1_out, __last1 - __first1);
            __result = __cc_range(__first1, __last1, __result);

            return {__last1, __first2, __result, __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            __internal::__uninitialized_copy_or_discard(__first1, __result);
            ++__result;
            ++__first1;
            __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data1_out);
        }
        else
        {
            if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
            {
                __internal::__uninitialized_copy_or_discard(__first2, __result);
                ++__result;
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::data2_out);
            }
            else
            {
                ++__first1;
                __mask = __internal::__set_iterator_mask(__mask, __parallel_set_op_mask::both);
            }
            ++__first2;
        }
    }

    __mask = __internal::__set_iterator_mask_n(__mask, __parallel_set_op_mask::data2_out, __last2 - __first2);
    __result = __cc_range(__first2, __last2, __result);

    return {__first1, __last2, __result, __mask};
}

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessOutputIterator>
struct __set_operations_result
{
    _RandomAccessIterator1 __in1;
    _RandomAccessIterator2 __in2;
    _RandomAccessOutputIterator __it_out;

#if _ONEDPL_CPP20_RANGES_PRESENT
    operator std::ranges::in_in_out_result<_RandomAccessIterator1, _RandomAccessIterator2,
                                           _RandomAccessOutputIterator>() const
    {
        return {__in1, __in2, __it_out};
    }
#endif
};

#if _ONEDPL_CPP20_RANGES_PRESENT
#    if ONEDPL_SET_RANGE_ALGS_CPP26_LIKE
template <typename _R1, typename _R2, typename _OutRange>
using __set_difference_return_t =
    std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_R2>,
                                  std::ranges::borrowed_iterator_t<_OutRange>>;
#    else
template <typename _R1, typename _R2, typename _OutRange>
using __set_difference_return_t =
    std::ranges::in_out_result<std::ranges::borrowed_iterator_t<_R1>, std::ranges::borrowed_iterator_t<_OutRange>>;
#    endif

// Helper function to create the appropriate return type for oneapi::dpl::ranges::set_difference based on C++26 compatibility mode.
// In C++23, set_difference returns in_out_result with the second input iterator omitted, as it is not needed by the caller.
template <typename _It1, typename _It2, typename _ItOut>
#    if ONEDPL_SET_RANGE_ALGS_CPP26_LIKE
std::ranges::in_in_out_result<_It1, _It2, _ItOut>
#    else
std::ranges::in_out_result<_It1, _ItOut>
#    endif
__create_set_difference_result(_It1 __it1, [[maybe_unused]] _It2 __it2, _ItOut __it_out)
{
#    if ONEDPL_SET_RANGE_ALGS_CPP26_LIKE
    return {__it1, __it2, __it_out};
#    else
    return {__it1, __it_out};
#    endif
}
#endif // _ONEDPL_CPP20_RANGES_PRESENT

} // namespace __utils
} // namespace dpl
} // namespace oneapi

#endif // __ONEDPL_SET_ALGORITHMS_UTILS_H
