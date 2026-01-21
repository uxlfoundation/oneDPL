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

#include <oneapi/dpl/pstl/parallel_backend_utils.h>

#include <vector>
#include <functional>
#include <array>

constexpr std::size_t kOutputSize = 10;

template <typename Container>
auto
summ(const Container& container)
{
    return std::accumulate(std::begin(container), std::end(container), 0, std::plus{});
}

// For details please see decsciption of the enum oneapi::dpl::__utils::__parallel_set_op_mask
template <std::size_t Size>
using MaskContainer = std::array<oneapi::dpl::__utils::__parallel_set_op_mask, Size>;
using MaskResultsContainer = std::vector<oneapi::dpl::__utils::__parallel_set_op_mask>;

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
        const Container          cont1 = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container          cont2 = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<7> maskExp = {       D1,        D1,       D12,       D12,       D12,        D2,        D2};
        const Container     contOutExp = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}, {6, 3, 2}, {7, 4, 2}};
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container          cont1 = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container          cont2 = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<7> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container     contOutExp = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range hasn't enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<5> maskExp = {       D1,        D1,       D12,       D12,       D12                      };
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        Container contOut(5);          // +++++++++  +++++++++  +++++++++  +++++++++  +++++++++  <-- out of range -->
        //                                                                                       {6, 3, 2}, {7, 4, 2}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<5> maskExp = {       D2,        D2,       D12,       D12,       D12                      };
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}, {5, 2, 1}                      };
        Container contOut(5);          // +++++++++  +++++++++  +++++++++  +++++++++  +++++++++  <-- out of range -->
        //                                                                                       {6, 3, 1}, {7, 4, 1}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<4> maskExp = {       D2,        D2,       D12,       D12                                 };
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}                                 };
        Container contOut(4);          // +++++++++  +++++++++  +++++++++  +++++++++  <--------- out of range ------>
        //                                                                            {5, 2, 1}, {6, 3, 1}, {7, 4, 1}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}           };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2},                       {8, 5, 2}};
        const MaskContainer<4> maskExp = {       D2,        D2,       D12,       D12                                            };
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}                                            };
        Container contOut(4);          // +++++++++  +++++++++  +++++++++  +++++++++  <--------------- out of range ----------->
        //                                                                            {5, 2, 1}, {6, 3, 1}, {7, 4, 1}, {8, 5, 2}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1},                       {8, 5, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}, {6, 3, 2}, {7, 4, 2}           };
        const MaskContainer<4> maskExp = {       D2,        D2,       D12,       D12                                            };
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 0, 1}, {4, 1, 1}                                            };
        Container contOut(4);          // +++++++++  +++++++++  +++++++++  +++++++++  <--------------- out of range ----------->
        //                                                                            {5, 2, 1}, {6, 3, 2}, {7, 4, 2}, {8, 5, 1}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }
}

void
test_set_union_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1          = { };
        const Container cont2          = { };
        const MaskContainer<0> maskExp = { };
        const Container contOutExp     = { };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: the first container is empty
    {
        const Container cont1          = {                               };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {       D2,        D2,        D2};
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: the second container is empty
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {                               };
        const MaskContainer<3> maskExp = {       D1,        D1,        D1};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: one item in the first container
    {
        const Container cont1          = {           {2, 0, 1}           };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {       D2,       D12,        D2};
        const Container contOutExp     = {{1, 0, 2}, {2, 0, 1}, {3, 2, 2}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: one item in the second container
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {           {2, 0, 2}           };
        const MaskContainer<3> maskExp = {       D1,       D12,        D1};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1          = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2          = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer<4> maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp     = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: both containers have the same items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {      D12,       D12,       D12};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2          = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer<6> maskExp = {       D1,        D1,        D1,        D2,        D2,        D2};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: output container has zero capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<0> maskExp = {                                                     };
        const Container contOutExp     = {                                                     };
        //                               {<-------------------- out of range ----------------->}
        Container contOut(0); //          {1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 1, 2}, {5, 2, 2}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: output container has one element capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<1> maskExp = {       D1                                            };
        const Container contOutExp     = {{1, 0, 1}                                            };
        //                               {+++++++++  <---------------- out of range ---------->}
        Container contOut(1);   //                   {2, 1, 1}, {3, 2, 1}, {4, 1, 2}, {5, 2, 2}

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2          = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer<5> maskExp = {       D1,       D12,        D1,       D12,        D2};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}, {4, 2, 2}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_union_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(contOutExp.size(), std::distance(contOut.begin(), out), "incorrect state of out for __set_union_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_union_bounded_construct");
    }
}

// The rules for testing set_union described at https://eel.is/c++draft/set.intersection
void
test_set_intersection_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};
    constexpr auto CopyFromSecondRange = std::false_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const Container cont2          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const MaskContainer<5> maskExp = {       D2,        D2,       D12,       D12,       D12                      };
        const Container contOutExp     = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}                      };

        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");
        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<5> maskExp = {       D2,        D2,       D12,       D12,       D12                      };
        const Container contOutExp     = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}                      };
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        // Here we have 3 but not 5 because we testing __set_intersection_bounded_construct
        // and iterators moved to the end in __pattern_set_intersection for __hetero_tag
        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // the first case - output range hasn't enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<4> maskExp = {       D1,        D1,       D12,       D12                                 };
        const Container contOutExp     = {                      {3, 2, 1}                                            };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(1, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<5> maskExp = {       D2,        D2,       D12,       D12,       D12                      };
        const Container contOutExp     = {                      {3, 0, 1}, {4, 1, 1}                                 };
        Container contOut(2);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(2, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(4, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<4> maskExp = {       D2,        D2,       D12,       D12                                 };
        const Container contOutExp     = {                      {3, 0, 1}                                            };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}, {8, 5, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2},                       {8, 5, 2}};
        const MaskContainer<8> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1,       D12};
        const Container contOutExp     = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}                                 };
        Container contOut(3);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }
}

void
test_set_intersection_construct_edge_cases()
{
    constexpr auto CopyFromFirstRange = std::true_type{};
    constexpr auto CopyFromSecondRange = std::false_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1          = { };
        const Container cont2          = { };
        const MaskContainer<0> maskExp = { };
        const Container contOutExp     = { };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // The case: the first container is empty
    {
        const Container cont1          = {                               };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<0> maskExp = {                               };
        const Container contOutExp     = {                               };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        // Here we have 3 but not 5 because we testing __set_intersection_bounded_construct
        // and iterators moved to the end in __pattern_set_intersection for __hetero_tag
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // The case: the second container is empty
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {                               };
        const MaskContainer<0> maskExp = {                               };
        const Container contOutExp     = {                               };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }

    // The case: one item in the first container
    {
        const Container cont1          = {           {2, 0, 1}           };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<2> maskExp = {       D2,       D12           };
        const Container contOutExp     = {           {2, 0, 1}           };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_intersection_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__op_uninitialized_copy<int>{},
            CopyFromFirstRange,
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_intersection_bounded_construct");
        EXPECT_EQ(2, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_intersection_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_intersection_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_intersection_bounded_construct");
    }
}

void
test_set_difference_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};
    constexpr auto CopyFromSecondRange = std::false_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<5> maskExp = {       D1,        D1,       D12,       D12,       D12                      };
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}                                                       };
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<7> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp     = {                                                       {6, 3, 1}, {7, 4, 1}};
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<2> maskExp = {       D1,        D1                                                       };
        const Container contOutExp     = {{1, 0, 1}                                                                  };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<6> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1           };
        const Container contOutExp     = {                                                       {6, 3, 1}           };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
         contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<7> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp     = {                                                       {6, 3, 1}, {7, 4, 1}};
        Container contOut(2);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}, {8, 5, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2},                       {8, 5, 2}};
        const MaskContainer<8> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1,       D12};
        const Container contOutExp     = {                                                       {6, 3, 1}, {7, 4, 1}           };
        Container contOut(2);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(6, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(6, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}, {8, 5, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2},                       {8, 5, 2}};
        const MaskContainer<6> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1                      };
        const Container contOutExp     = {                                                       {6, 3, 1}                      };
        Container contOut(1);          //                                                        +++++++++  <-- out of range -->

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }
}

void
test_set_difference_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1          = { };
        const Container cont2          = { };
        const MaskContainer<0> maskExp = { };
        const Container contOutExp     = { };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: the first container is empty
    {
        const Container cont1          = {                               };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<0> maskExp = {                               };
        const Container contOutExp     = {                               };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: the second container is empty
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {                               };
        const MaskContainer<3> maskExp = {       D1,        D1,        D1};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: one item in the first container
    {
        const Container cont1          = {           {2, 0, 1}           };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<2> maskExp = {       D2,       D12           };
        const Container contOutExp     = {                               };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: one item in the second container
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {           {2, 0, 2}           };
        const MaskContainer<3> maskExp = {       D1,       D12,        D1};
        const Container contOutExp     = {{1, 0, 1},            {3, 2, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1          = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2          = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer<4> maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp     = {                                 {3, 3, 1}};
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: both containers have the same items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {      D12,       D12,       D12};
        const Container contOutExp     = {                               };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2          = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer<3> maskExp = {       D1,        D1,        D1                                 };
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: output container has zero capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<1> maskExp = {       D1                                            };
        const Container contOutExp     = {                                                     };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: output container has one element capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<2> maskExp = {       D1,        D1                                 };
        const Container contOutExp     = {{1, 0, 1}                                            };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2          = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer<4> maskExp = {       D1,       D12,        D1,       D12           };
        const Container contOutExp     = {{1, 0, 1},            {2, 2, 1}                      };
        Container contOut(kOutputSize);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }

    // The case: no intersections and empty output
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}                                 };
        const Container cont2          = {                      {3, 0, 2}, {3, 1, 2}, {4, 2, 2}};
        const MaskContainer<1> maskExp = {       D1                                            };
        const Container contOutExp     = {                                                     };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_difference_bounded_construct");
    }
}

void
test_set_symmetric_difference_construct()
{
    constexpr auto CopyFromFirstRange = std::true_type{};
    constexpr auto CopyFromSecondRange = std::false_type{};

    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // the first case - output range has enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<7> maskExp = {       D1,        D1,       D12,       D12,       D12,        D2,        D2};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1},                                  {6, 3, 2}, {7, 4, 2}};
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(4, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // the first case - output range has enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<7> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1,        D1};
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2},                                  {6, 3, 1}, {7, 4, 1}};
        Container contOut(cont1.size() + cont2.size());

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(5, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(4, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 3, 1}, {5, 4, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}, {6, 3, 2}, {7, 4, 2}};
        const MaskContainer<2> maskExp = {       D1,        D1                                                       };
        const Container contOutExp     = {{1, 0, 1}                                                                  };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<2> maskExp = {       D2,        D2                                                       };
        const Container contOutExp     = {{1, 0, 2}                                                                  };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // the first case - output range hasn't enough capacity - SWAP input ranges data
    {
        const Container cont1          = {                      {3, 0, 1}, {4, 1, 1}, {5, 2, 1}, {6, 3, 1}, {7, 4, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}, {4, 3, 2}, {5, 4, 2}                      };
        const MaskContainer<6> maskExp = {       D2,        D2,       D12,       D12,       D12,        D1           };
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2},                                  {6, 3, 1}           };
        Container contOut(3);          // +++++++++  +++++++++                                   +++++++++   <--oor-->

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(5, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }
}

void
test_set_symmetric_difference_construct_edge_cases()
{
    using DataType = TestUtils::SetDataItem<int>;
    using Container = std::vector<DataType>;

    // The case: both containers are empty
    {
        const Container cont1          = { };
        const Container cont2          = { };
        const MaskContainer<0> maskExp = { };
        const Container contOutExp     = { };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: the first container is empty
    {
        const Container cont1          = {                               };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {       D2,        D2,        D2};
        const Container contOutExp     = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        Container contOut(3);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: the second container is empty
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {                               };
        const MaskContainer<3> maskExp = {       D1,        D1,        D1};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        Container contOut(3);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: one item in the first container
    {
        const Container cont1          = {           {2, 0, 1}           };
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {       D2,       D12,        D2};
        const Container contOutExp     = {{1, 0, 2},            {3, 2, 2}};
        Container contOut(2);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: one item in the second container
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {           {2, 0, 2}           };
        const MaskContainer<3> maskExp = {       D1,       D12,        D1};
        const Container contOutExp     = {{1, 0, 1},            {3, 2, 1}};
        Container contOut(2);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(2, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: all items are equal but the last item in the first container is unique
    {
        const Container cont1          = {{2, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}};
        const Container cont2          = {{2, 0, 2}, {2, 1, 2}, {2, 2, 2}           };
        const MaskContainer<4> maskExp = {      D12,       D12,       D12,        D1};
        const Container contOutExp     = {                                 {3, 3, 1}};
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: both containers have the same items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}};
        const Container cont2          = {{1, 0, 2}, {2, 1, 2}, {3, 2, 2}};
        const MaskContainer<3> maskExp = {      D12,       D12,       D12};
        const Container contOutExp     = {                               };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: all items in the first container less then in the second one
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                                 };
        const Container cont2          = {                                 {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        const MaskContainer<6> maskExp = {       D1,        D1,        D1,        D2,        D2,        D2};
        const Container contOutExp     = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}, {4, 0, 2}, {5, 1, 2}, {6, 2, 2}};
        Container contOut(6);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(3, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(6, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: output container has zero capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<1> maskExp = {       D1                                            };
        const Container contOutExp     = {                                                     };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: output container has one element capacity
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {3, 2, 1}                      };
        const Container cont2          = {                      {3, 0, 2}, {4, 1, 2}, {5, 2, 2}};
        const MaskContainer<2> maskExp = {       D1,        D1                                 };
        const Container contOutExp     = {{1, 0, 1}                                            };
        Container contOut(1);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(1, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(1, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: the first container has duplicated items
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}, {2, 2, 1}, {3, 3, 1}           };
        const Container cont2          = {           {2, 0, 2},            {3, 1, 2}, {4, 2, 2}};
        const MaskContainer<5> maskExp = {       D1,       D12,        D1,       D12,        D2};
        const Container contOutExp     = {{1, 0, 1},            {2, 2, 1},            {4, 2, 2}};
        Container contOut(3);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(4, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(3, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }

    // The case: no intersections and empty output
    {
        const Container cont1          = {{1, 0, 1}, {2, 1, 1}                                 };
        const Container cont2          = {                      {3, 0, 2}, {3, 1, 2}, {4, 2, 2}};
        const MaskContainer<1> maskExp = {       D1                                            };
        const Container contOutExp     = {                                                     };
        Container contOut(0);

        MaskResultsContainer mask(cont1.size() + cont2.size());
        auto mask_b = mask.data();

        auto [in1, in2, out, mask_e] = oneapi::dpl::__utils::__set_symmetric_difference_bounded_construct(
            cont1.begin(), cont1.end(),
            cont2.begin(), cont2.end(),
            contOut.begin(), contOut.end(),
            mask_b,
            oneapi::dpl::__internal::__BrickCopyConstruct<std::false_type>{},
            std::less{}, TestUtils::SetDataItemProj{}, TestUtils::SetDataItemProj{});

        EXPECT_EQ(0, std::distance(cont1.begin(),   in1), "incorrect state of in1 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(cont2.begin(),   in2), "incorrect state of in2 for __set_symmetric_difference_bounded_construct");
        EXPECT_EQ(0, std::distance(contOut.begin(), out), "incorrect state of out for __set_symmetric_difference_bounded_construct");

        EXPECT_EQ_RANGES(maskExp, std::ranges::subrange(mask_b, mask_e), "Incorrect mask state");

        // Truncate output from out till the end to avoid compare error
        contOut.erase(out, contOut.end());
        EXPECT_EQ_RANGES(contOutExp, contOut, "wrong result of result contOut after __set_symmetric_difference_bounded_construct");
    }
}

int
main()
{
    test_set_union_construct();
    test_set_union_construct_edge_cases();

    test_set_intersection_construct();
    test_set_intersection_construct_edge_cases();

    test_set_difference_construct();
    test_set_difference_construct_edge_cases();

    test_set_symmetric_difference_construct();
    test_set_symmetric_difference_construct_edge_cases();

    return TestUtils::done();
}
