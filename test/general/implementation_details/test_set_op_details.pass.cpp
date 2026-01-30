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

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include <oneapi/dpl/pstl/parallel_backend_utils.h>

#include <vector>
#include <functional>

template <typename Container1, typename Container2>
std::size_t
evalContainerSize(const Container1& cont1, const Container2& cont2)
{
    return cont1.size() + cont2.size();
}

template <typename Container1, typename Container2>
std::size_t
evalMaskSize(const Container1& cont1, const Container2& cont2)
{
    return cont1.size() + cont2.size();
}

// For details please see description of the enum oneapi::dpl::__utils::__parallel_set_op_mask
using MaskContainer = std::vector<oneapi::dpl::__utils::__parallel_set_op_mask>;

constexpr oneapi::dpl::__utils::__parallel_set_op_mask  D1 = oneapi::dpl::__utils::__parallel_set_op_mask::eData1;
constexpr oneapi::dpl::__utils::__parallel_set_op_mask  D2 = oneapi::dpl::__utils::__parallel_set_op_mask::eData2;
constexpr oneapi::dpl::__utils::__parallel_set_op_mask D12 = oneapi::dpl::__utils::__parallel_set_op_mask::eBoth;

// The rules for testing set_union described at https://eel.is/c++draft/set.union
void
test_set_union_construct()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;
    
    // the first case - output range has enough capacity
    {
        const Container       cont1 = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container       cont2 = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer maskExp = {       D1,        D1,       D12,       D12,       D12,        D2,        D2};
        const Container  contOutExp = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}, {6, 3, 2}, {7, 4, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container       cont1 = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container       cont2 = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container  contOutExp = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_union_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1       = { };
        const Container cont2       = { };
        const MaskContainer maskExp = { };
        const Container contOutExp  = { };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container is empty
    {
        const Container cont1       = {                               };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,        D2,        D2};
        const Container contOutExp  = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the second container is empty
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {                               };
        const MaskContainer maskExp = {       D1,        D1,        D1};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the first container
    {
        const Container cont1       = {           {2, 0, 1}           };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,       D12,        D2};
        const Container contOutExp  = {{1, 0, 2}, {2, 0, 1}, {3, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the second container
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {           {2, 0, 2}           };
        const MaskContainer maskExp = {       D1,       D12,        D1};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1       = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2       = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp  = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: both containers have the same items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {      D12,       D12,       D12};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2       = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer maskExp = {       D1,        D1,        D1,        D2,        D2,        D2};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2       = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer maskExp = {       D1,       D12,        D1,       D12,        D2};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_union_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

// The rules for testing set_union described at https://eel.is/c++draft/set.intersection
void
test_set_intersection_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1       = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const Container cont2       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const MaskContainer maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp  = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}                      };

        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_construct");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1       = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp  = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}                      };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_intersection_construct_edge_cases()
{
    constexpr auto CopyFromFirstRange = std::true_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1       = { };
        const Container cont2       = { };
        const MaskContainer maskExp = { };
        const Container contOutExp  = { };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container is empty
    {
        const Container cont1       = {                               };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,        D2,        D2};
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the second container is empty
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {                               };
        const MaskContainer maskExp = {       D1,        D1,        D1};
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the first container
    {
        const Container cont1       = {           {2, 0, 1}           };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,       D12,        D2};
        const Container contOutExp  = {           {2, 0, 1}           };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_intersection_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_difference_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2       = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer maskExp = {       D1,        D1,       D12,       D12,       D12                      };
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}                                                       };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1       = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp  = {                                                       {6, 3, 1}, {7, 4, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_difference_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1       = { };
        const Container cont2       = { };
        const MaskContainer maskExp = { };
        const Container contOutExp  = { };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container is empty
    {
        const Container cont1       = {                               };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {                               };
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the second container is empty
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {                               };
        const MaskContainer maskExp = {       D1,        D1,        D1};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the first container
    {
        const Container cont1       = {           {2, 0, 1}           };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,       D12           };
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the second container
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {           {2, 0, 2}           };
        const MaskContainer maskExp = {       D1,       D12,        D1};
        const Container contOutExp  = {{1, 0, 1},            {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1       = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2       = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp  = {                                 {3, 3, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: both containers have the same items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {      D12,       D12,       D12};
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2       = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer maskExp = {       D1,        D1,        D1                                 };
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2       = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer maskExp = {       D1,       D12,        D1,       D12           };
        const Container contOutExp  = {{1, 0, 1},            {2, 2, 1}                      };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_symmetric_difference_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2       = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer maskExp = {       D1,        D1,       D12,       D12,       D12,        D2,        D2};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1},                                  {6, 3, 2}, {7, 4, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1       = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp  = {{1, 0, 2}, {2, 1, 2},                                  {6, 3, 1}, {7, 4, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}

void
test_set_symmetric_difference_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: the first container is empty
    {
        const Container cont1       = {                               };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,        D2,        D2};
        const Container contOutExp  = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the second container is empty
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {                               };
        const MaskContainer maskExp = {       D1,        D1,        D1};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the first container
    {
        const Container cont1       = {           {2, 0, 1}           };
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {       D2,       D12,        D2};
        const Container contOutExp  = {{1, 0, 2},            {3, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: one item in the second container
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {           {2, 0, 2}           };
        const MaskContainer maskExp = {       D1,       D12,        D1};
        const Container contOutExp  = {{1, 0, 1},            {3, 2, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1       = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2       = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp  = {                                 {3, 3, 1}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: both containers have the same items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2       = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer maskExp = {      D12,       D12,       D12};
        const Container contOutExp  = {                               };
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2       = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer maskExp = {       D1,        D1,        D1,        D2,        D2,        D2};
        const Container contOutExp  = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1       = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2       = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer maskExp = {       D1,       D12,        D1,       D12,        D2};
        const Container contOutExp  = {{1, 0, 1},            {2, 2, 1},            {4, 2, 2}};
        Container contOut(evalContainerSize(cont1, cont2));

        MaskContainer mask(evalMaskSize(cont1, cont2));
        auto mask_b = mask.data();

        auto [out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ_RANGES(contOutExp, std::ranges::subrange(contOut.begin(), out), "Incorrect result data state");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");
    }
}
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING
    test_set_union_construct();
    test_set_union_construct_edge_cases();

    test_set_intersection_construct();
    test_set_intersection_construct_edge_cases();

    test_set_difference_construct();
    test_set_difference_construct_edge_cases();

    test_set_symmetric_difference_construct();
    test_set_symmetric_difference_construct_edge_cases();

    bProcessed = true;
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(bProcessed);
}
