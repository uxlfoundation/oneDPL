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

#ifndef _STD_RANGES_ALGO_ARCHETYPES_TEST_H
#define _STD_RANGES_ALGO_ARCHETYPES_TEST_H

#include <oneapi/dpl/execution>

#include "support/test_config.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING

#include "std_ranges_archetypes.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace test_std_ranges
{

// The archetypes are neither copyable nor movable, so they cannot live in a container: the storage
// is raw memory with in-place constructed elements, wrapped into archetype_view, which is random
// access and sized but neither contiguous nor common. This leaves the implementation no way to fall
// back to raw pointer arithmetic or to a hidden copy of the elements.
inline constexpr std::size_t archetype_test_size = 1000;

// Runs a one-range algorithm and checks the result with __checker(view, result).
template <typename _Elem, typename _Alloc, typename _Policy, typename _Algo, typename _Checker>
void
run_algo(_Alloc __alloc, _Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    archetypes::archetype_storage<_Elem, _Alloc> __storage(__alloc, archetype_test_size,
                                                           [](std::size_t __i) { return (int)__i; });
    auto __view = __storage.view();

    auto __res = __algo(std::forward<_Policy>(__policy), __view);

    EXPECT_TRUE(__checker(__view, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

// Runs a two-range algorithm and checks the result with __checker(view1, view2, result).
template <typename _Elem1, typename _Elem2, typename _Alloc1, typename _Alloc2, typename _Policy, typename _Algo,
          typename _Checker>
void
run_algo2(_Alloc1 __alloc1, _Alloc2 __alloc2, _Policy&& __policy, _Algo __algo, _Checker __checker,
          const char* __algo_name)
{
    archetypes::archetype_storage<_Elem1, _Alloc1> __storage1(__alloc1, archetype_test_size,
                                                              [](std::size_t __i) { return (int)__i; });
    archetypes::archetype_storage<_Elem2, _Alloc2> __storage2(__alloc2, archetype_test_size,
                                                              [](std::size_t __i) { return (int)__i; });
    auto __view1 = __storage1.view();
    auto __view2 = __storage2.view();

    auto __res = __algo(std::forward<_Policy>(__policy), __view1, __view2);

    EXPECT_TRUE(__checker(__view1, __view2, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

// Runs a one-range algorithm with the host policies only. A value argument which is neither
// copyable nor movable cannot be passed to a device kernel, so such an archetype is meaningful for
// the host policies only, where the implementation is required to keep a reference to the value.
template <typename _Elem, typename _Algo, typename _Checker>
void
run_algo_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    std::allocator<_Elem> __alloc;
    run_algo<_Elem>(__alloc, oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo<_Elem>(__alloc, oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo<_Elem>(__alloc, oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo<_Elem>(__alloc, oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

// Runs a two-range algorithm with the host policies only, see run_algo_host_policies.
template <typename _Elem1, typename _Elem2, typename _Algo, typename _Checker>
void
run_algo2_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    std::allocator<_Elem1> __alloc1;
    std::allocator<_Elem2> __alloc2;
    run_algo2<_Elem1, _Elem2>(__alloc1, __alloc2, oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(__alloc1, __alloc2, oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(__alloc1, __alloc2, oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(__alloc1, __alloc2, oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

// _CallId makes the SYCL kernel name of the device call unique: every instantiation of the harness
// submits its own kernel, and with -fno-sycl-unnamed-lambda two kernels sharing a name are a
// "definition with same mangled name" error.
template <typename _Elem, int _CallId, typename _Algo, typename _Checker>
void
run_algo_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo_host_policies<_Elem>(__algo, __checker, __algo_name);

#if TEST_DPCPP_BACKEND_PRESENT
    auto __policy = TestUtils::get_dpcpp_test_policy<_CallId>();
    sycl::usm_allocator<_Elem, sycl::usm::alloc::shared> __q_alloc{__policy.queue()};
    run_algo<_Elem>(__q_alloc, __policy, __algo, __checker, __algo_name);
#endif //TEST_DPCPP_BACKEND_PRESENT
}

template <typename _Elem1, typename _Elem2, int _CallId, typename _Algo, typename _Checker>
void
run_algo2_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_host_policies<_Elem1, _Elem2>(__algo, __checker, __algo_name);

#if TEST_DPCPP_BACKEND_PRESENT
    auto __policy = TestUtils::get_dpcpp_test_policy<_CallId>();
    sycl::usm_allocator<_Elem1, sycl::usm::alloc::shared> __q_alloc1{__policy.queue()};
    sycl::usm_allocator<_Elem2, sycl::usm::alloc::shared> __q_alloc2{__policy.queue()};
    run_algo2<_Elem1, _Elem2>(__q_alloc1, __q_alloc2, __policy, __algo, __checker, __algo_name);
#endif //TEST_DPCPP_BACKEND_PRESENT
}

} //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
#endif //_STD_RANGES_ALGO_ARCHETYPES_TEST_H
