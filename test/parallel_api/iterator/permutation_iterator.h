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

#ifndef _PERMUTATION_ITERATOR_H
#define _PERMUTATION_ITERATOR_H

#include "support/utils_const.h"
#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#endif // TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_test_base.h"

#include <type_traits>
#include <vector>
#include <iterator>

// Index of permutation iterator is based on counting iterator
struct perm_it_index_tags_counting { };

// Index of permutation iterator is based on host iterator
struct perm_it_index_tags_host { };

#if TEST_DPCPP_BACKEND_PRESENT
// Index of permutation iterator is based on USM shared memory
struct perm_it_index_tags_usm_shared { };
// Test case is for USM device memory is unavailable to implement due to indexes
// cannot be initialized on the host (USM device is not accessible on the host)
#endif // TEST_DPCPP_BACKEND_PRESENT

// Index of permutation iterator is based on transform iterator
struct perm_it_index_tags_transform_iterator { };

// Index of permutation iterator is based on callable object
struct perm_it_index_tags_callable_object { };


////////////////////////////////////////////////////////////////////////////////
/**
 * @param template typename TSourceIterator - source iterator type
 * @param typename TSourceDataSize - type of source data size
 * @param typename PermItIndexTag - tag permutation iterator base kind:
 *          - perm_it_index_tags_counting;
 *          - perm_it_index_tags_host;
 *          - perm_it_index_tags_usm_shared;
 *          - perm_it_index_tags_transform_iterator.
 */
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator_data
{
    TSourceIterator itSource;                // First source iterator
    const TSourceDataSize src_data_size = 0; // Source data items count
};

////////////////////////////////////////////////////////////////////////////////
struct multiply_index_by_two
{
    template <typename Index>
    Index
    operator()(const Index& i) const
    {
        return i * 2;
    }

    std::size_t
    eval_items_count(std::size_t src_items_count) const
    {
        return src_items_count / 2 + src_items_count % 2;
    }
};
const auto kDefaultIndexStepOp = multiply_index_by_two{};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize, typename PermItIndexTag>
struct test_through_permutation_iterator
{
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags_counting>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename Policy, typename Operand>
    void
    operator()(Policy&& exec, Operand op)
    {
        auto indexes_begin = dpl::counting_iterator<TSourceDataSize>(0);

        auto permItBegin = dpl::make_permutation_iterator(data.itSource, indexes_begin);
        auto permItEnd = permItBegin + data.src_data_size;

        op(std::forward<Policy>(exec), permItBegin, permItEnd);
    }
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags_host>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename Policy, typename Operand>
    void
    operator()(Policy&& exec, Operand op)
    {
        ::std::vector<TSourceDataSize> indexes;

        for (TSourceDataSize perm_idx_step = 1; perm_idx_step < data.src_data_size; perm_idx_step = kDefaultIndexStepOp(perm_idx_step))
        {
            const TSourceDataSize idx_size = data.src_data_size / perm_idx_step;
            indexes.resize(idx_size);
            for (TSourceDataSize idx = 0, val = 0; idx < idx_size; ++idx, val += perm_idx_step)
                indexes[idx] = val;

            auto permItBegin = dpl::make_permutation_iterator(data.itSource, indexes.begin());
            auto permItEnd = permItBegin + indexes.size();

            op(CLONE_TEST_POLICY(exec), permItBegin, permItEnd);
        }
    }
};

#if TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags_usm_shared>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename Policy, typename Operand>
    void
    operator()(Policy&& exec, Operand op)
    {
        using TestBaseData = TestUtils::test_base_data_usm<sycl::usm::alloc::shared, TSourceDataSize>;

        std::cout<<"Testing through usm_shared map"<<TestUtils::max_n<<", "<<TestUtils::inout1_offset<<"\n";

        TestBaseData test_base_data(TestUtils::get_test_queue(), {{TestUtils::max_n, TestUtils::inout1_offset}});
        TSourceDataSize* itIndexStart = test_base_data.get_start_from(TestUtils::UDTKind::eKeys);

        for (TSourceDataSize perm_idx_step = 1; perm_idx_step <= data.src_data_size;
             perm_idx_step = kDefaultIndexStepOp(perm_idx_step))
        {

            const TSourceDataSize idx_size = data.src_data_size / perm_idx_step;

            auto indexes = test_base_data.get_start_from(TestUtils::UDTKind::eKeys);
            for (TSourceDataSize idx = 0, val = 0; idx < idx_size; ++idx, val += perm_idx_step)
            {
                indexes[idx] = val;
                std::cout<<val<<" ";
            }
            auto permItBegin = dpl::make_permutation_iterator(data.itSource, itIndexStart);
            auto permItEnd = permItBegin + indexes.size();

            op(CLONE_TEST_POLICY(exec), permItBegin, permItEnd);
        }
    }
};
#endif // TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags_transform_iterator>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    using ValueType = typename ::std::iterator_traits<TSourceIterator>::value_type;

    // Using callable object instead of lambda here to ensure transform iterator would be
    // default constructible, that is part of the Forward Iterator requirements in the C++ standard.
    struct NoTransform
    {
        ValueType operator()(const ValueType& val) const
        {
            return val;
        }
    };

    template <typename Policy, typename Operand>
    void
    operator()(Policy&& exec, Operand op)
    {
        auto indexes_begin = dpl::counting_iterator<TSourceDataSize>(0);
        auto itTransformBegin = dpl::make_transform_iterator(indexes_begin, NoTransform{});
        auto permItBegin = dpl::make_permutation_iterator(data.itSource, itTransformBegin);
        auto permItEnd = permItBegin + data.src_data_size;

        op(std::forward<Policy>(exec), permItBegin, permItEnd);
    }
};

////////////////////////////////////////////////////////////////////////////////
template <typename TSourceIterator, typename TSourceDataSize>
struct test_through_permutation_iterator<TSourceIterator, TSourceDataSize, perm_it_index_tags_callable_object>
{
    test_through_permutation_iterator_data<TSourceIterator, TSourceDataSize> data;

    test_through_permutation_iterator(TSourceIterator itSource, const TSourceDataSize src_data_size)
        : data{itSource, src_data_size}
    {
    }

    template <typename Policy, typename Operand>
    void
    operator()(Policy&& exec, Operand op)
    {
        auto permItBegin = dpl::make_permutation_iterator(data.itSource, kDefaultIndexStepOp);
        auto permItEnd = permItBegin + kDefaultIndexStepOp.eval_items_count(data.src_data_size);

        op(std::forward<Policy>(exec), permItBegin, permItEnd);
    }
};

#endif // _PERMUTATION_ITERATOR_H
