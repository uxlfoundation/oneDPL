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

#ifndef _ONEDPL_INCLUSIVE_SCAN_BY_SEGMENT_IMPL_H
#define _ONEDPL_INCLUSIVE_SCAN_BY_SEGMENT_IMPL_H

#include "by_segment_extension_defs.h"

#include "../pstl/glue_numeric_defs.h"
#include "../pstl/glue_numeric_impl.h"

#include "../pstl/parallel_backend.h"
#include "function.h"
#include "../pstl/utils.h"
#include "scan_by_segment_impl.h"

namespace oneapi
{
namespace dpl
{
namespace internal
{

template <typename Name>
class InclusiveScan1;

template <class _Tag, typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
OutputIterator
pattern_inclusive_scan_by_segment(_Tag, Policy&& policy, InputIterator1 first1, InputIterator1 last1,
                                  InputIterator2 first2, OutputIterator result, BinaryPredicate binary_pred,
                                  BinaryOperator binary_op)
{
    static_assert(__internal::__is_host_dispatch_tag_v<_Tag>);

    const auto n = ::std::distance(first1, last1);

    // Check for empty and single element ranges
    if (n <= 0)
        return result;
    if (n == 1)
    {
        *result = *first2;
        return result + 1;
    }

    typedef unsigned int FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;

    oneapi::dpl::__par_backend::__buffer<Policy, FlagType> _mask(policy, n);
    auto mask = _mask.get();

    mask[0] = 1;

    transform(policy, first1, last1 - 1, first1 + 1, _mask.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPredicate>(binary_pred));

    inclusive_scan(::std::forward<Policy>(policy), make_zip_iterator(first2, _mask.get()),
                   make_zip_iterator(first2, _mask.get()) + n, make_zip_iterator(result, _mask.get()),
                   internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op));

    return result + n;
}

#if _ONEDPL_BACKEND_SYCL
template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryPredicate, typename BinaryOperator>
OutputIterator
inclusive_scan_by_segment_impl(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                               InputIterator1 last1, InputIterator2 first2, OutputIterator result,
                               BinaryPredicate binary_pred, BinaryOperator binary_op,
                               ::std::true_type /* has_known_identity */)
{
    using iter_value_t = typename ::std::iterator_traits<InputIterator2>::value_type;
    iter_value_t identity = unseq_backend::__known_identity<BinaryOperator, iter_value_t>;
    return internal::__scan_by_segment_impl_common(__tag, ::std::forward<Policy>(policy), first1, last1, first2, result,
                                                   identity, binary_pred, binary_op, ::std::true_type{});
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryPredicate, typename BinaryOperator>
OutputIterator
inclusive_scan_by_segment_impl(__internal::__hetero_tag<_BackendTag>, Policy&& policy, InputIterator1 first1,
                               InputIterator1 last1, InputIterator2 first2, OutputIterator result,
                               BinaryPredicate binary_pred, BinaryOperator binary_op,
                               ::std::false_type /* has_known_identity */)
{

    typedef unsigned int FlagType;
    typedef typename ::std::iterator_traits<InputIterator2>::value_type ValueType;

    const auto n = ::std::distance(first1, last1);

    // Check for empty element ranges
    if (n <= 0)
        return result;

    FlagType initial_mask = 1;

    oneapi::dpl::__par_backend_hetero::__buffer<Policy, FlagType> _mask(policy, n);
    {
        auto mask_buf = _mask.get_buffer();
        auto mask = mask_buf.get_host_access(sycl::read_write);

        mask[0] = initial_mask;
    }

    transform(::std::forward<Policy>(policy), first1, last1 - 1, first1 + 1, _mask.get() + 1,
              oneapi::dpl::__internal::__not_pred<BinaryPredicate>(binary_pred));

    auto policy1 = oneapi::dpl::__par_backend_hetero::make_wrapped_policy<InclusiveScan1>(policy);
    transform_inclusive_scan(::std::move(policy1), make_zip_iterator(first2, _mask.get()),
                             make_zip_iterator(first2, _mask.get()) + n, make_zip_iterator(result, _mask.get()),
                             internal::segmented_scan_fun<ValueType, FlagType, BinaryOperator>(binary_op),
                             oneapi::dpl::__internal::__no_op());
    return result + n;
}

template <typename _BackendTag, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryPredicate, typename BinaryOperator>
OutputIterator
pattern_inclusive_scan_by_segment(__internal::__hetero_tag<_BackendTag> __tag, Policy&& policy, InputIterator1 first1,
                                  InputIterator1 last1, InputIterator2 first2, OutputIterator result,
                                  BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    return internal::inclusive_scan_by_segment_impl(
        __tag, ::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred, binary_op,
        typename unseq_backend::__has_known_identity<
            BinaryOperator, typename ::std::iterator_traits<InputIterator2>::value_type>::type{});
}

#endif
} // namespace internal

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    const auto __dispatch_tag = oneapi::dpl::__internal::__select_backend(policy, first1, first2, result);

    return internal::pattern_inclusive_scan_by_segment(__dispatch_tag, ::std::forward<Policy>(policy), first1, last1,
                                                       first2, result, binary_pred, binary_op);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter, typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_segment(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result,
                          BinaryPredicate binary_pred)
{
    using T = typename ::std::iterator_traits<InputIter2>::value_type;

    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred,
                                     ::std::plus<T>());
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_segment(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result)
{
    using T = typename ::std::iterator_traits<InputIter1>::value_type;

    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result,
                                     ::std::equal_to<T>());
}

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_key(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                      OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred,
                                     binary_op);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter, typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_key(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result,
                      BinaryPredicate binary_pred)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result, binary_pred);
}

template <typename Policy, typename InputIter1, typename InputIter2, typename OutputIter>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIter>
inclusive_scan_by_key(Policy&& policy, InputIter1 first1, InputIter1 last1, InputIter2 first2, OutputIter result)
{
    return inclusive_scan_by_segment(::std::forward<Policy>(policy), first1, last1, first2, result);
}
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_INCLUSIVE_SCAN_BY_SEGMENT_IMPL_H
