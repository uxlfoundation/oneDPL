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

#ifndef _ONEDPL_PARALLEL_BACKEND_UTILS_H
#define _ONEDPL_PARALLEL_BACKEND_UTILS_H

#include <atomic>
#include <cstddef>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include <cassert>
#include <cstdint> // for std::uint8_t
#include "utils.h"
#include "memory_fwd.h"
#include "functional_impl.h" // for oneapi::dpl::identity, std::invoke
#include "utils_ranges.h" // KSATODO does this include really needed?

namespace oneapi
{
namespace dpl
{
namespace __utils
{

//------------------------------------------------------------------------
// raw buffer (with specified _TAllocator)
//------------------------------------------------------------------------

template <typename _Tp, template <typename _T> typename _TAllocator>
class __buffer_impl
{
    _TAllocator<_Tp> _M_allocator;
    _Tp* _M_ptr = nullptr;
    const ::std::size_t _M_buf_size = 0;

    __buffer_impl(const __buffer_impl&) = delete;
    void
    operator=(const __buffer_impl&) = delete;

  public:
    //! Try to obtain buffer of given size to store objects of _Tp type
    __buffer_impl(const std::size_t __n) : _M_allocator(), _M_ptr(_M_allocator.allocate(__n)), _M_buf_size(__n) {}
    //! True if buffer was successfully obtained, zero otherwise.
    operator bool() const { return _M_ptr != nullptr; }
    //! Return pointer to buffer, or nullptr if buffer could not be obtained.
    _Tp*
    get() const
    {
        return _M_ptr;
    }
    //! Destroy buffer
    ~__buffer_impl() { _M_allocator.deallocate(_M_ptr, _M_buf_size); }
};

//! Destroy sequence [xs,xe)
struct __serial_destroy
{
    template <typename _RandomAccessIterator>
    void
    operator()(_RandomAccessIterator __zs, _RandomAccessIterator __ze)
    {
        using _ValueType = typename std::iterator_traits<_RandomAccessIterator>::value_type;
        while (__zs != __ze)
        {
            --__ze;
            (*__ze).~_ValueType();
        }
    }
};

//! Merge sequences [__xs,__xe) and [__ys,__ye) to output sequence [__zs,(__xe-__xs)+(__ye-__ys)), using ::std::move
struct __serial_move_merge
{
    const ::std::size_t _M_nmerge;

    explicit __serial_move_merge(::std::size_t __nmerge) : _M_nmerge(__nmerge) {}
    template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3, class _Compare,
              class _MoveValueX, class _MoveValueY, class _MoveSequenceX, class _MoveSequenceY>
    void
    operator()(_RandomAccessIterator1 __xs, _RandomAccessIterator1 __xe, _RandomAccessIterator2 __ys,
               _RandomAccessIterator2 __ye, _RandomAccessIterator3 __zs, _Compare __comp, _MoveValueX __move_value_x,
               _MoveValueY __move_value_y, _MoveSequenceX __move_sequence_x, _MoveSequenceY __move_sequence_y)
    {
        constexpr bool __same_move_val = ::std::is_same_v<_MoveValueX, _MoveValueY>;
        constexpr bool __same_move_seq = ::std::is_same_v<_MoveSequenceX, _MoveSequenceY>;

        std::size_t __n = _M_nmerge;
        assert(__n > 0);

        auto __nx = __xe - __xs;
        //auto __ny = __ye - __ys;
        _RandomAccessIterator3 __zs_beg = __zs;

        if (__xs != __xe)
        {
            if (__ys != __ye)
            {
                for (;;)
                {
                    if (std::invoke(__comp, *__ys, *__xs))
                    {
                        const auto __i = __zs - __zs_beg;
                        if (__i < __nx)
                            __move_value_x(__ys, __zs);
                        else
                            __move_value_y(__ys, __zs);
                        ++__zs;
                        --__n;
                        if (++__ys == __ye)
                        {
                            break;
                        }
                        else if (__n == 0)
                        {
                            if constexpr (__same_move_seq)
                            {
                                __zs = __move_sequence_x(__ys, __ye, __zs);
                            }
                            else
                            {
                                const auto __j = __zs - __zs_beg;
                                if (__j < __nx)
                                    __zs = __move_sequence_x(__ys, __ye, __zs);
                                else
                                    __zs = __move_sequence_y(__ys, __ye, __zs);
                            }
                            break;
                        }
                    }
                    else
                    {
                        if constexpr (__same_move_val)
                        {
                            __move_value_x(__xs, __zs);
                        }
                        else
                        {
                            const auto __i = __zs - __zs_beg;
                            if (__i < __nx)
                                __move_value_x(__xs, __zs);
                            else
                                __move_value_y(__xs, __zs);
                        }

                        ++__zs;
                        --__n;
                        if (++__xs == __xe)
                        {
                            if constexpr (__same_move_seq)
                            {
                                __move_sequence_x(__ys, __ye, __zs);
                            }
                            else
                            {
                                const auto __j = __zs - __zs_beg;
                                if (__j < __nx)
                                    __move_sequence_x(__ys, __ye, __zs);
                                else
                                    __move_sequence_y(__ys, __ye, __zs);
                            }
                            return;
                        }
                        else if (__n == 0)
                        {
                            if constexpr (__same_move_seq)
                            {
                                __zs = __move_sequence_x(__xs, __xe, __zs);
                                __move_sequence_x(__ys, __ye, __zs);
                            }
                            else
                            {
                                const auto __j = __zs - __zs_beg;
                                if (__j < __nx)
                                {
                                    __zs = __move_sequence_x(__xs, __xe, __zs);
                                    __move_sequence_x(__ys, __ye, __zs);
                                }
                                else
                                {
                                    __zs = __move_sequence_y(__xs, __xe, __zs);
                                    __move_sequence_y(__ys, __ye, __zs);
                                }
                            }
                            return;
                        }
                    }
                }
            }
            __ys = __xs;
            __ye = __xe;
        }
        if constexpr (__same_move_seq)
        {
            __move_sequence_x(__ys, __ye, __zs);
        }
        else
        {
            const auto __i = __zs - __zs_beg;
            if (__i < __nx)
                __move_sequence_x(__ys, __ye, __zs);
            else
                __move_sequence_y(__ys, __ye, __zs);
        }
    }
};

enum class __parallel_set_op_mask : std::uint8_t
{
    eData1 = 0x10,          // mask for first input data item usage
    eData2 = 0x01,          // mask for second input data item usage
    eBoth = eData1 | eData2 // mask for both input data items usage
};

inline std::nullptr_t
__set_iterator_mask(std::nullptr_t, __parallel_set_op_mask)
{
    return nullptr;
}

inline
__parallel_set_op_mask*
__set_iterator_mask(__parallel_set_op_mask* __mask, __parallel_set_op_mask __state)
{
    *__mask = __state;
    return ++__mask;
}

template <class _IsVector, typename _ExecutionPolicy, typename _Size>
std::nullptr_t
__set_iterator_mask_n(oneapi::dpl::__internal::__parallel_tag<_IsVector>, _ExecutionPolicy&&, std::nullptr_t,
                      __parallel_set_op_mask, _Size)
{
    return nullptr;
}

template <class _IsVector, typename _ExecutionPolicy, typename _Size>
__parallel_set_op_mask*
__set_iterator_mask_n(oneapi::dpl::__internal::__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                      __parallel_set_op_mask* __mask, __parallel_set_op_mask __state, _Size __count)
{
    oneapi::dpl::__internal::__pattern_fill_n(__tag, std::forward<_ExecutionPolicy>(__exec), __mask, __count, __state);
    return __mask + __count;
}

template <class _IsVector, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _MaskIterator, typename _CopyConstructRange, typename _Compare, typename _Proj1, typename _Proj2>
std::tuple<_OutputIterator, _MaskIterator>
__set_union_construct(oneapi::dpl::__internal::__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                      _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                      _ForwardIterator2 __last2, _OutputIterator __result, _MaskIterator __mask,
                      _CopyConstructRange __cc_range, _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    assert(__first1 <= __last1);
    assert(__first2 <= __last2);

    // This implementation should be aligned with https://eel.is/c++draft/set.union

    using _Tp = typename std::iterator_traits<_OutputIterator>::value_type;

    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
        {
            __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData1, __last1 - __first1);
            return {__cc_range(__first1, __last1, __result), __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
        {
            new (std::addressof(*__result)) _Tp(*__first2);
            ++__first2;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData2);
        }
        else
        {
            new (std::addressof(*__result)) _Tp(*__first1);
            if (!std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
            {
                ++__first2;
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eBoth);
            }
            else
            {
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData1);
            }
            ++__first1;
        }
    }

    __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData2, __last2 - __first2);
    return {__cc_range(__first2, __last2, __result), __mask};
}

template <class _IsVector, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _MaskIterator, typename _CopyFunc, typename _CopyFromFirstSet,
          typename _Compare, typename _Proj1, typename _Proj2>
std::tuple<_OutputIterator, _MaskIterator>
__set_intersection_construct(oneapi::dpl::__internal::__parallel_tag<_IsVector> __tag,
                             _ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                             _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                             _MaskIterator __mask, _CopyFunc _copy, _CopyFromFirstSet,
                             _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    // This implementation should be aligned with https://eel.is/c++draft/set.intersection

    while (__first1 != __last1 && __first2 != __last2)
    {
        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            ++__first1;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData1);
        }
        else if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
        {
            ++__first2;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData2);
        }
        else
        {
            if constexpr (_CopyFromFirstSet::value)
                _copy(*__first1, *__result);
            else
                _copy(*__first2, *__result);

            ++__first1;
            ++__first2;
            ++__result;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eBoth);
        }
    }

    // This needed to save in mask that we processed all data till the end
    __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData1, __last1 - __first1);
    __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData2, __last2 - __first2);

    return {__result, __mask};
}

template <class _IsVector, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _MaskIterator, typename _CopyConstructRange, typename _Compare,
          typename _Proj1, typename _Proj2>
std::tuple<_OutputIterator, _MaskIterator>
__set_difference_construct(oneapi::dpl::__internal::__parallel_tag<_IsVector> __tag, _ExecutionPolicy&& __exec,
                           _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
                           _ForwardIterator2 __last2, _OutputIterator __result, _MaskIterator __mask,
                           _CopyConstructRange __cc_range, _Compare __comp, _Proj1 __proj1, _Proj2 __proj2)
{
    // This implementation should be aligned with https://eel.is/c++draft/set.difference

    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    while (__first1 != __last1)
    {
        if (__first2 == __last2)
        {
            __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData1, __last1 - __first1);
            return {__cc_range(__first1, __last1, __result), __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            new (std::addressof(*__result)) _Tp(*__first1);
            ++__result;
            ++__first1;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData1);
        }
        else
        {
            if (!std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
            {
                ++__first1;
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eBoth);
            }
            else
            {
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData2);
            }
            ++__first2;
        }
    }

    return {__result, __mask};
}

template <class _IsVector, typename _ExecutionPolicy, typename _ForwardIterator1, typename _ForwardIterator2,
          typename _OutputIterator, typename _MaskIterator, typename _CopyConstructRange, typename _Compare,
          typename _Proj1, typename _Proj2>
std::tuple<_OutputIterator, _MaskIterator>
__set_symmetric_difference_construct(oneapi::dpl::__internal::__parallel_tag<_IsVector> __tag,
                                     _ExecutionPolicy&& __exec, _ForwardIterator1 __first1, _ForwardIterator1 __last1,
                                     _ForwardIterator2 __first2, _ForwardIterator2 __last2, _OutputIterator __result,
                                     _MaskIterator __mask, _CopyConstructRange __cc_range, _Compare __comp,
                                     _Proj1 __proj1, _Proj2 __proj2)
{
    // This implementation should be aligned with https://eel.is/c++draft/set.symmetric.difference

    using _Tp = typename ::std::iterator_traits<_OutputIterator>::value_type;

    while (__first1 != __last1)
    {
        if (__first2 == __last2)
        {
            __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData1, __last1 - __first1);
            return {__cc_range(__first1, __last1, __result), __mask};
        }

        if (std::invoke(__comp, std::invoke(__proj1, *__first1), std::invoke(__proj2, *__first2)))
        {
            new (std::addressof(*__result)) _Tp(*__first1);
            ++__result;
            ++__first1;
            __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData1);
        }
        else
        {
            if (std::invoke(__comp, std::invoke(__proj2, *__first2), std::invoke(__proj1, *__first1)))
            {
                new (std::addressof(*__result)) _Tp(*__first2);
                ++__result;
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eData2);
            }
            else
            {
                ++__first1;
                __mask = __set_iterator_mask(__mask, __parallel_set_op_mask::eBoth);
            }
            ++__first2;
        }
    }

    __mask = __set_iterator_mask_n(__tag, __exec, __mask, __parallel_set_op_mask::eData2, __last2 - __first2);
    return {__cc_range(__first2, __last2, __result), __mask};
}

template <template <typename, typename...> typename _Concrete, typename _ValueType, typename... _Args>
struct __enumerable_thread_local_storage_base
{
    using _Derived = _Concrete<_ValueType, _Args...>;

    __enumerable_thread_local_storage_base(std::tuple<_Args...> __tp)
        : __thread_specific_storage(_Derived::get_num_threads()), __num_elements(0), __args(__tp)
    {
    }

    // Note: size should not be used concurrently with parallel loops which may instantiate storage objects, as it may
    // not return an accurate count of instantiated storage objects in lockstep with the number allocated and stored.
    // This is because the count is not atomic with the allocation and storage of the storage objects.
    std::size_t
    size() const
    {
        // only count storage which has been instantiated
        return __num_elements.load(std::memory_order_relaxed);
    }

    // Note: get_with_id should not be used concurrently with parallel loops which may instantiate storage objects,
    // as its operation may provide an out of date view of the stored objects based on the timing new object creation
    // and incrementing of the size.
    // TODO: Consider replacing this access with a visitor pattern.
    _ValueType&
    get_with_id(std::size_t __i)
    {
        assert(__i < size());

        if (size() == __thread_specific_storage.size())
        {
            return *__thread_specific_storage[__i];
        }

        std::size_t __j = 0;
        for (std::size_t __count = 0; __j < __thread_specific_storage.size() && __count <= __i; ++__j)
        {
            // Only include storage from threads which have instantiated a storage object
            if (__thread_specific_storage[__j])
            {
                ++__count;
            }
        }
        // Need to back up one once we have found a valid storage object
        return *__thread_specific_storage[__j - 1];
    }

    _ValueType&
    get_for_current_thread()
    {
        const std::size_t __i = _Derived::get_thread_num();
        std::optional<_ValueType>& __local = __thread_specific_storage[__i];
        if (!__local)
        {
            // create temporary storage on first usage to avoid extra parallel region and unnecessary instantiation
            std::apply([&__local](_Args... __arg_pack) { __local.emplace(__arg_pack...); }, __args);
            __num_elements.fetch_add(1, std::memory_order_relaxed);
        }
        return *__local;
    }

    std::vector<std::optional<_ValueType>> __thread_specific_storage;
    std::atomic_size_t __num_elements;
    const std::tuple<_Args...> __args;
};

template <typename _RandomAccessIterator1, typename _RandomAccessIterator2, typename _RandomAccessOutputIterator>
struct __set_operations_result
{
    _RandomAccessIterator1 __in1;
    _RandomAccessIterator2 __in2;
    _RandomAccessOutputIterator __it_out;

    // Get reached input1 and output iterators
    template <typename TResult>
    TResult
    __get_reached_in1_out() const
    {
        return {__in1, __it_out};
    }

    // Get reached input1, input2 and output iterators
    template <typename TResult>
    TResult
    __get_reached_in1_in2_out() const
    {
        return {__in1, __in2, __it_out};
    }

    // Get reached output iterator
    _RandomAccessOutputIterator
    __get_reached_out() const
    {
        return __it_out;
    }

    __set_operations_result<_RandomAccessIterator1, _RandomAccessIterator2, _RandomAccessOutputIterator>
    operator+(std::tuple<typename std::iterator_traits<_RandomAccessIterator1>::difference_type,
                         typename std::iterator_traits<_RandomAccessIterator2>::difference_type,
                         typename std::iterator_traits<_RandomAccessOutputIterator>::difference_type> __offsets) const
    {
        return {__in1 + std::get<0>(__offsets),
                __in2 + std::get<1>(__offsets),
                __it_out + std::get<2>(__offsets)};
    }
};

// Move it1 forward by n, but not beyond it2
template <typename _RandomAccessIterator, typename Size = typename std::iterator_traits<_RandomAccessIterator>::difference_type>
_RandomAccessIterator
__advance_clamped(_RandomAccessIterator it1, Size n, _RandomAccessIterator it2)
{
    return it1 + (it2 >= it1 ? std::min(it2 - it1, n) : 0);
}

} // namespace __utils
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_UTILS_H
