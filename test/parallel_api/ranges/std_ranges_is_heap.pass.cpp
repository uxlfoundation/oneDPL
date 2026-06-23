// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"
#include "std_ranges_heap_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto is_heap_checker = TEST_PREPARE_CALLABLE(std::ranges::is_heap);

    // Dimension 1: tests data generators
    const auto generators = std::make_tuple(test_std_ranges::MaxHeapGenerator{},
                                            test_std_ranges::ThroughParentHeapGenerator{},
                                            test_std_ranges::NonHeapGenerator{},
                                            test_std_ranges::CorruptedHeapGenerator<test_std_ranges::MaxHeapGenerator>{
                                                test_std_ranges::MaxHeapGenerator{},
                                                /*broken element index*/ 42});

    // Dimension 2: tests comparators
    const auto comparators = std::make_tuple(std::ranges::less{},
                                             std::ranges::greater{}, 
                                             test_std_ranges::CustomLess<test_std_ranges::element_t>{},
                                             test_std_ranges::CustomGreat<test_std_ranges::element_t>{});

    // Dimension 3: tests projections (object to describe test data type, projection function)
    const auto projections = std::make_tuple(std::make_tuple(test_std_ranges::element_t{}, std::identity{}),
                                             std::make_tuple(test_std_ranges::P2{}, &test_std_ranges::P2::x),
                                             std::make_tuple(test_std_ranges::P2{}, &test_std_ranges::P2::proj));

    using GeneratorsT = decltype(generators);
    using ComparatorsT = decltype(comparators);
    using ProjectionsT = decltype(projections);

    // Iterate through the elements of dimension 1: tests data generators
    test_std_ranges::for_each_in_tuple<GeneratorsT>{}([&]<class Generator, std::size_t GeneratorIdx>() {

        // Iterate through the elements of dimension 2: tests comparators
        test_std_ranges::for_each_in_tuple<ComparatorsT>{}([&]<class Comparator, std::size_t ComparatorIdx>() {

            // Iterate through the elements of dimension 3: tests projections (object to describe test data type, projection function)
            test_std_ranges::for_each_in_tuple<ProjectionsT>{}([&]<class Projection, std::size_t ProjectionIdx>() {

                auto generator = std::get<GeneratorIdx>(generators);
                using generator_t = std::decay_t<decltype(generator)>;

                auto comp = std::get<ComparatorIdx>(comparators);
                using comp_t = std::decay_t<decltype(comp)>;

                auto proj_info = std::get<ProjectionIdx>(projections);
                using test_data_t = std::decay_t<decltype(std::get<0>(proj_info))>;
                auto proj = std::get<1>(proj_info);
                using proj_t = std::decay_t<decltype(proj)>;

                constexpr int call_id = GeneratorIdx * 100 + ComparatorIdx * 10 + ProjectionIdx;
                test_std_ranges::test_range_algo<call_id, test_data_t, test_std_ranges::data_in, generator_t>{}(dpl_ranges::is_heap, is_heap_checker, comp, proj);

                if constexpr (std::is_same_v<proj_t, std::identity>)
                {
                    constexpr int call_id1 = call_id * 10;
                    test_std_ranges::test_range_algo<call_id1, test_data_t, test_std_ranges::data_in, generator_t>{}(dpl_ranges::is_heap, is_heap_checker, comp);

                    if constexpr (std::is_same_v<comp_t, std::ranges::less>)
                    {
                        test_std_ranges::test_range_algo<call_id1 + 1, test_data_t, test_std_ranges::data_in, generator_t>{}(dpl_ranges::is_heap, is_heap_checker);
                    }
                }
            });
        });
    });
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
