// -*- C++ -*-
//===-- input_data_sweep.h ------------------------------------------------===//
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

#ifndef _INPUT_DATA_SWEEP_H
#define _INPUT_DATA_SWEEP_H

#include "support/utils.h"
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "support/utils_invoke.h"

#if TEST_DPCPP_BACKEND_PRESENT
template <int __recurse, int __reverses, bool __read = true, bool __reset_read = true, bool __write = true,
          bool __check_write = true, bool __usable_as_perm_map = true, bool __usable_as_perm_src = true,
          bool __is_reversible = true, typename Policy, typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename OriginalIterator1, typename OriginalIterator2, typename ExpectedIterator,
          typename T>
void
wrap_recurse(Policy&& exec, InputIterator1 first, InputIterator1 last, InputIterator2 copy_from_first,
             OutputIterator copy_to_first, OriginalIterator1 orig_first, OriginalIterator2 orig_out_first,
             ExpectedIterator expected_first, T trash, const std::string& input_descr)
{
    auto exec1 = TestUtils::create_new_policy_idx<0>(exec);
    auto exec2 = TestUtils::create_new_policy_idx<1>(exec);
    auto exec3 = TestUtils::create_new_policy_idx<2>(exec);
    auto exec4 = TestUtils::create_new_policy_idx<3>(exec);
    auto exec5 = TestUtils::create_new_policy_idx<4>(exec);
    auto exec6 = TestUtils::create_new_policy_idx<5>(exec);
    auto exec7 = TestUtils::create_new_policy_idx<6>(exec);
    auto exec8 = TestUtils::create_new_policy_idx<7>(exec);
    auto exec9 = TestUtils::create_new_policy_idx<8>(exec);
    auto exec10 = TestUtils::create_new_policy_idx<9>(exec);
    auto exec11 = TestUtils::create_new_policy_idx<10>(exec);
    auto exec12 = TestUtils::create_new_policy_idx<11>(exec);
    auto exec13 = TestUtils::create_new_policy_idx<12>(exec);

    oneapi::dpl::counting_iterator<size_t> counting(size_t{0});

    const auto n = last - first;

    //Run the tests
    auto get_expect = [n](auto exp) {
        (void)n; // avoid unused-lambda-capture, bug https://bugs.llvm.org/show_bug.cgi?id=35450
        if constexpr (__reverses % 2 == 0)
        {
            return exp;
        }
        else
        {
            return std::make_reverse_iterator(exp + n);
        }
    };

#    if _ONEDPL_DEBUG_SYCL
    std::cout << input_descr << ":";
#    endif // _ONEDPL_DEBUG_SYCL

    if constexpr (__read)
    {
        oneapi::dpl::fill(exec1, orig_out_first, orig_out_first + n, trash);
        if constexpr (__reset_read)
        {
            //Reset data if required
            oneapi::dpl::copy(exec2, expected_first, expected_first + n, orig_first);
        }

        //Run test
        oneapi::dpl::copy(exec3, first, last, copy_to_first);

        //get expected sequence with proper number of reverses
        auto expect = get_expect(expected_first);
        std::string msg = std::string("wrong read effect from ") + input_descr;
        //verify result using original unwrapped output
        EXPECT_EQ_N(expect, orig_out_first, n, msg.c_str());
#    if _ONEDPL_DEBUG_SYCL
        std::cout << " read pass,";
#    endif // _ONEDPL_DEBUG_SYCL
    }
    if constexpr (__write)
    {
        //Reset data
        if constexpr (__check_write)
        {
            //only reset output data if we intend to check it afterward
            oneapi::dpl::fill(exec4, orig_first, orig_first + n, trash);
        }

        oneapi::dpl::copy(exec5, copy_from_first, copy_from_first + n, first);
        //check write if required (ignore discard iterator)
        if constexpr (__check_write)
        {
            //copy back data from original unwrapped sequence
            std::vector<T> copy_back(n);
            oneapi::dpl::copy(exec6, orig_first, orig_first + n, copy_back.begin());

            //get expected sequence with proper number of reverses
            auto expect = get_expect(expected_first);
            std::string msg = std::string("wrong write effect from ") + input_descr;
            //verify copied back data
            EXPECT_EQ_N(expect, copy_back.begin(), n, msg.c_str());
#    if _ONEDPL_DEBUG_SYCL
            std::cout << " write pass";
#    endif // _ONEDPL_DEBUG_SYCL
        }
        else
        {
#    if _ONEDPL_DEBUG_SYCL
            std::cout << " write pass (no check)";
#    endif // _ONEDPL_DEBUG_SYCL
        }
    }
    if constexpr (!__read && !__write)
    {
#    if _ONEDPL_DEBUG_SYCL
        std::cout << " has no valid tests";
#    endif // _ONEDPL_DEBUG_SYCL
    }
#    if _ONEDPL_DEBUG_SYCL
    std::cout << std::endl;
#    endif // _ONEDPL_DEBUG_SYCL

    // Now recurse with a layer of wrappers if requested
    if constexpr (__recurse > 0)
    {
#    if _ONEDPL_DEBUG_SYCL
        std::cout << std::endl << "Recursing on " << input_descr << ":" << std::endl;
#    endif // _ONEDPL_DEBUG_SYCL
        oneapi::dpl::discard_iterator discard{};
        // iterate through all wrappers and recurse - 1
        auto noop = [](auto i) { return i; };

        if constexpr (__is_reversible)
        { // std::reverse_iterator(it)
            auto reversed_first = ::std::make_reverse_iterator(last);
            auto reversed_last = ::std::make_reverse_iterator(first);
            std::string new_input_descr = std::string("std::reverse(") + input_descr + std::string(")");
            //TODO: Look at device copyability of std::reverse_iterator and re-enable recurse
            wrap_recurse<0, __reverses + 1, __read, __reset_read, __write, __check_write, __usable_as_perm_map,
                         __usable_as_perm_src, __is_reversible>(exec7, reversed_first, reversed_last, copy_from_first,
                                                                copy_to_first, orig_first, orig_out_first,
                                                                expected_first, trash, new_input_descr);
        }

        { //transform_iterator(it,noop{})
            auto trans = oneapi::dpl::make_transform_iterator(first, noop);
            std::string new_input_descr = std::string("transform_iterator(") + input_descr + std::string(", noop)");
            wrap_recurse<__recurse - 1, __reverses, __read, __reset_read, /*__write=*/false, __check_write,
                         __usable_as_perm_map, __usable_as_perm_src, __is_reversible>(
                exec8, trans, trans + n, discard, copy_to_first, orig_first, orig_out_first, expected_first, trash,
                new_input_descr);
        }

        if constexpr (__usable_as_perm_src)
        { //permutation_iteartor(it,noop{})
            std::string new_input_descr = std::string("permutation_iterator(") + input_descr + std::string(", noop)");
            auto perm = oneapi::dpl::make_permutation_iterator(first, noop);
            wrap_recurse<__recurse - 1, __reverses, __read, __reset_read, __write, __check_write, __usable_as_perm_map,
                         __usable_as_perm_src, __is_reversible>(exec9, perm, perm + n, copy_from_first, copy_to_first,
                                                                orig_first, orig_out_first, expected_first, trash,
                                                                new_input_descr);
        }

        if constexpr (__usable_as_perm_src)
        { //permutation_iterator(it,counting_iter)
            std::string new_input_descr =
                std::string("permutation_iterator(") + input_descr + std::string(", counting_iterator)");
            auto perm = oneapi::dpl::make_permutation_iterator(first, counting);
            wrap_recurse<__recurse - 1, __reverses, __read, __reset_read, __write, __check_write, __usable_as_perm_map,
                         __usable_as_perm_src, __is_reversible>(exec11, perm, perm + n, copy_from_first, copy_to_first,
                                                                orig_first, orig_out_first, expected_first, trash,
                                                                new_input_descr);
        }

        if constexpr (__usable_as_perm_map)
        { //permutation_iterator(counting_iterator,it)
            std::string new_input_descr =
                std::string("permutation_iterator(counting_iterator,") + input_descr + std::string(")");
            auto perm = oneapi::dpl::make_permutation_iterator(counting, first);
            wrap_recurse<__recurse - 1, __reverses, __read, __reset_read, /*__write=*/false, __check_write,
                         __usable_as_perm_map, __usable_as_perm_src, __is_reversible>(
                exec10, perm, perm + n, discard, copy_to_first, orig_first, orig_out_first, expected_first, trash,
                new_input_descr);
        }

        { //zip_iterator(counting_iterator,it)
            std::string new_input_descr =
                std::string("zip_iterator(counting_iterator,") + input_descr + std::string(")");
            auto zip = oneapi::dpl::make_zip_iterator(counting, first);
            auto zip_out = oneapi::dpl::make_zip_iterator(discard, copy_to_first);
            wrap_recurse<__recurse - 1, __reverses, __read, __reset_read, /*__write=*/false, __check_write,
                         /*__usable_as_perm_map=*/false, __usable_as_perm_src, __is_reversible>(
                exec12, zip, zip + n, discard, zip_out, orig_first, orig_out_first, expected_first, trash,
                new_input_descr);
        }

        { //zip_iterator(it, discard_iterator)
            std::string new_input_descr =
                std::string("zip_iterator(") + input_descr + std::string(", discard_iterator)");
            auto zip = oneapi::dpl::make_zip_iterator(first, discard);
            auto zip_in = oneapi::dpl::make_zip_iterator(copy_from_first, counting);
            wrap_recurse<__recurse - 1, __reverses, /*__read=*/false, false, __write, __check_write,
                         /*__usable_as_perm_map=*/false, __usable_as_perm_src, __is_reversible>(
                exec13, zip, zip + n, zip_in, discard, orig_first, orig_out_first, expected_first, trash,
                new_input_descr);
        }
    }
}

#endif //TEST_DPCPP_BACKEND_PRESENT

#endif //_INPUT_DATA_SWEEP_H
