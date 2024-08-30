/*
 *  Copyright (c) Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef _ONEDPL_EXCLUSIVE_SCAN_BY_SEGMENT_IMPL_H
#define _ONEDPL_EXCLUSIVE_SCAN_BY_SEGMENT_IMPL_H

#include "../pstl/parallel_backend.h"
#include "function.h"
#include "by_segment_extension_defs.h"
#include "../pstl/utils.h"
#include "scan_by_segment_impl.h"

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class ExclusiveScan1;
template <typename Name>
class ExclusiveScan2;

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename T, typename BinaryPredicate, typename Operator>
OutputIterator
pattern_exclusive_scan_by_segment(_Tag, Policy&& policy, InputIterator1 first1, InputIterator1 last1,
                                  InputIterator2 first2, OutputIterator result, T init, BinaryPredicate binary_pred,
                                  Operator binary_op)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    const auto n = ::std::distance(first1, last1);

    // Check for empty and single element ranges
    if (n <= 0)
        return result;
    if (n == 1)
    {
        *result = init;
        return result + 1;
    }

    typedef typename ::std::iterator_traits<OutputIterator>::value_type OutputType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef unsigned int FlagType;

    InputIterator2 last2 = first2 + n;

    // compute head flags
    oneapi::dpl::__par_backend::__buffer<Policy, FlagType> _flags(policy, n);
    auto flags = _flags.get();
    flags[0] = 1;

    transform(policy, first1, last1 - 1, first1 + 1, _flags.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPredicate>(binary_pred));

    // shift input one to the right and initialize segments with init
    oneapi::dpl::__par_backend::__buffer<Policy, OutputType> _temp(policy, n);
    auto temp = _temp.get();

    temp[0] = init;

    // TODO : add stencil form of replace_copy_if to oneDPL if the
    // transform call here is difficult to understand and maintain.
#if 1
    transform(policy, first2, last2 - 1, _flags.get() + 1, _temp.get() + 1,
              internal::replace_if_fun<T, ::std::negate<FlagType>>(::std::negate<FlagType>(), init));
#else
    replace_copy_if(policy1, first2, last2 - 1, _flags.get() + 1, _temp.get() + 1, ::std::negate<FlagType>(), init);
#endif

    // scan key-flag tuples
    inclusive_scan(::std::forward<Policy>(policy), make_zip_iterator(_temp.get(), _flags.get()),
                   make_zip_iterator(_temp.get(), _flags.get()) + n, make_zip_iterator(result, _flags.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, Operator>(binary_op));
    return result + n;
}

#if _ONEDPL_BACKEND_SYCL
template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename T, typename BinaryPredicate, typename Operator>
OutputIterator
exclusive_scan_by_segment_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                               InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
                               BinaryPredicate binary_pred, Operator binary_op,
                               ::std::true_type /* has_known_identity*/)
{
    return internal::__scan_by_segment_impl_common(__tag, ::std::forward<Policy>(policy), first1, last1, first2, result,
                                                   init, binary_pred, binary_op, ::std::false_type{});
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename T, typename BinaryPredicate, typename Operator>
OutputIterator
exclusive_scan_by_segment_impl(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 first1,
                               InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
                               BinaryPredicate binary_pred, Operator binary_op,
                               ::std::false_type /* has_known_identity*/)
{

    const auto n = ::std::distance(first1, last1);
    typedef typename ::std::iterator_traits<OutputIterator>::value_type OutputType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;
    typedef unsigned int FlagType;

    InputIterator2 last2 = first2 + n;

    // compute head flags
    oneapi::dpl::__par_backend_hetero::__buffer<Policy, FlagType> _flags(policy, n);
    {
        auto flag_buf = _flags.get_buffer();
        auto flags = flag_buf.get_host_access(sycl::read_write);
        flags[0] = 1;
    }

    transform(policy, first1, last1 - 1, first1 + 1, _flags.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPredicate>(binary_pred));

    // shift input one to the right and initialize segments with init
    oneapi::dpl::__par_backend_hetero::__buffer<Policy, OutputType> _temp(policy, n);
    {
        auto temp_buf = _temp.get_buffer();
        auto temp = temp_buf.get_host_access(sycl::read_write);

        temp[0] = init;
    }

    auto policy1 = oneapi::dpl::__par_backend_hetero::make_wrapped_policy<ExclusiveScan1>(policy);

    // TODO : add stencil form of replace_copy_if to oneDPL if the
    // transform call here is difficult to understand and maintain.
#    if 1
    transform(::std::move(policy1), first2, last2 - 1, _flags.get() + 1, _temp.get() + 1,
              internal::replace_if_fun<T, ::std::negate<FlagType>>(::std::negate<FlagType>(), init));
#    else
    replace_copy_if(::std::move(policy1), first2, last2 - 1, _flags.get() + 1, _temp.get() + 1,
                    ::std::negate<FlagType>(), init);
#    endif

    auto policy2 =
        oneapi::dpl::__par_backend_hetero::make_wrapped_policy<ExclusiveScan2>(::std::forward<Policy>(policy));

    // scan key-flag tuples
    transform_inclusive_scan(::std::move(policy2), make_zip_iterator(_temp.get(), _flags.get()),
                             make_zip_iterator(_temp.get(), _flags.get()) + n, make_zip_iterator(result, _flags.get()),
                             internal::segmented_scan_fun<ValueType, FlagType, Operator>(binary_op),
                             oneapi::dpl::__internal::__no_op(),
                             oneapi::dpl::__internal::make_tuple(init, FlagType(1)));
    return result + n;
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename T, typename BinaryPredicate, typename Operator>
OutputIterator
pattern_exclusive_scan_by_segment(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                                  InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
                                  BinaryPredicate binary_pred, Operator binary_op)
{
    return internal::exclusive_scan_by_segment_impl(
        __tag, ::std::forward<Policy>(policy), first1, last1, first2, result, init, binary_pred, binary_op,
        typename unseq_backend::__has_known_identity<
            Operator, typename ::std::iterator_traits<InputIterator2>::value_type>::type{});
}

#endif
} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate, typename Operator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init, BinaryPredicate binary_pred, Operator binary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, first1, first2, result);

    return internal::pattern_exclusive_scan_by_segment(__dispatch_tag, ::std::forward<Policy>(policy), first1, last1,
                                                       first2, result, init, binary_pred, binary_op);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init, BinaryPredicate binary_pred)
{
    typedef typename ::std::iterator_traits<InputIterator2>::value_type V2;
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, init, binary_pred,
                                     ::std::plus<V2>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init)
{
    typedef typename ::std::iterator_traits<InputIterator1>::value_type V1;
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, init,
                                     ::std::equal_to<V1>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result)
{
    typedef typename ::std::iterator_traits<InputIterator2>::value_type V2;
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, V2(0));
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate, typename Operator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result, T init, BinaryPredicate binary_pred, Operator binary_op)
{
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, init, binary_pred,
                                     binary_op);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result, T init, BinaryPredicate binary_pred)
{
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, init, binary_pred);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result, T init)
{
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, init);
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result)
{
    return exclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result);
}

} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_EXCLUSIVE_SCAN_BY_SEGMENT_IMPL_H
