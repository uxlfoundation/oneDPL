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
#include <type_traits>
#include <utility>

namespace test_std_ranges
{

// The archetypes are neither copyable nor movable, so they cannot live in a container: the storage
// is raw memory with in-place constructed elements, wrapped into archetype_view, which is random
// access and sized but neither contiguous nor common. This leaves the implementation no way to fall
// back to raw pointer arithmetic or to a hidden copy of the elements.
inline constexpr std::size_t archetype_test_size = 1000;

// The default content of an input range: 0, 1, 2, ... An output range is filled with zeros instead,
// see make_out_storage below.
inline constexpr auto archetype_index_factory = [](std::size_t __i) { return (int)__i; };

#if TEST_DPCPP_BACKEND_PRESENT
// True for the device policies, i.e. the ones carrying a SYCL queue.
template <typename _Policy>
inline constexpr bool is_device_policy_v = requires(_Policy& __policy) { __policy.queue(); };
#else
template <typename _Policy>
inline constexpr bool is_device_policy_v = false;
#endif

// Builds the storage of one range with the allocator matching the policy: a range touched by a device
// kernel has to live in device accessible memory, while a host policy is happy with std::allocator.
// archetype_storage is immovable, so it is returned as a prvalue and initialized directly into the
// variable of the caller.
template <typename _Elem, typename _Policy, typename _Factory>
auto
make_storage(_Policy&& __policy, std::size_t __n, _Factory __factory)
{
#if TEST_DPCPP_BACKEND_PRESENT
    if constexpr (is_device_policy_v<std::remove_cvref_t<_Policy>>)
    {
        sycl::usm_allocator<_Elem, sycl::usm::alloc::shared> __alloc{__policy.queue()};
        return archetypes::archetype_storage<_Elem, decltype(__alloc)>(__alloc, __n, __factory);
    }
    else
#endif
    {
        return archetypes::archetype_storage<_Elem, std::allocator<_Elem>>(std::allocator<_Elem>{}, __n, __factory);
    }
}

// The storage of the output range of an algorithm which writes into a range of its own (merge, the
// set operations, the binary transform, ...). Every element starts as zero, so a test has to check a
// position the algorithm is expected to write a non-zero value into.
template <typename _Elem, typename _Policy>
auto
make_out_storage(_Policy&& __policy, std::size_t __n)
{
    return make_storage<_Elem>(std::forward<_Policy>(__policy), __n, [](std::size_t) { return 0; });
}

// Runs a one-range algorithm and checks the result with __checker(view, result).
template <typename _Elem, typename _Policy, typename _Algo, typename _Checker>
void
run_algo(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage = make_storage<_Elem>(__policy, archetype_test_size, archetype_index_factory);
    auto __view = __storage.view();

    auto __res = __algo(std::forward<_Policy>(__policy), __view);

    EXPECT_TRUE(__checker(__view, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

// Runs a two-range algorithm and checks the result with __checker(view1, view2, result).
template <typename _Elem1, typename _Elem2, typename _Policy, typename _Algo, typename _Checker>
void
run_algo2(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage1 = make_storage<_Elem1>(__policy, archetype_test_size, archetype_index_factory);
    auto __storage2 = make_storage<_Elem2>(__policy, archetype_test_size, archetype_index_factory);
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
    run_algo<_Elem>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

// Runs a two-range algorithm with the host policies only, see run_algo_host_policies.
template <typename _Elem1, typename _Elem2, typename _Algo, typename _Checker>
void
run_algo2_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2<_Elem1, _Elem2>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo2<_Elem1, _Elem2>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

#if TEST_DPCPP_BACKEND_PRESENT
// A device policy passes the element type into a kernel, so the caller is expected to name the
// device copyable archetype (the _dc one) explicitly. Everything else the host only archetype lacks
// (default construction, comparison, ordering, ...) is still missing in the _dc counterpart.
//
// _CallId makes the SYCL kernel name of the device call unique: every instantiation of the harness
// submits its own kernel, and with -fno-sycl-unnamed-lambda two kernels sharing a name are a
// "definition with same mangled name" error. Pass __LINE__, which is unique by construction; the ids
// only have to be unique inside one translation unit, and every test file is its own executable.
template <typename _Elem, int _CallId, typename _Algo, typename _Checker>
void
run_algo_hetero_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo<_Elem>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
}

// Runs a two-range algorithm with the hetero policies, see run_algo_hetero_policies.
template <typename _Elem1, typename _Elem2, int _CallId, typename _Algo, typename _Checker>
void
run_algo2_hetero_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2<_Elem1, _Elem2>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

// Runs one and the same generic lambda with the host and with the hetero policies: _Elem is the host
// only archetype and _ElemDc its device copyable counterpart, so the lambda has to derive every other
// type it needs (a value argument, an output element type) from the element type it is handed.
//
// _RunHost and _RunHetero switch one side off where the implementation is known to ask for more than
// the requires-clause of the algorithm allows. A false branch is discarded by if constexpr, so the
// call is not instantiated at all and the compilation error stays away.
template <typename _Elem, typename _ElemDc, int _CallId, bool _RunHost = true, bool _RunHetero = true,
          typename _Algo, typename _Checker>
void
run_algo_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    if constexpr (_RunHost)
        run_algo_host_policies<_Elem>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    if constexpr (_RunHetero)
        run_algo_hetero_policies<_ElemDc, _CallId>(__algo, __checker, __algo_name);
#endif
}

// Runs a two-range algorithm with both the host and the hetero policies, see run_algo_all_policies.
template <typename _Elem1, typename _Elem2, typename _Elem1Dc, typename _Elem2Dc, int _CallId, bool _RunHost = true,
          bool _RunHetero = true, typename _Algo, typename _Checker>
void
run_algo2_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    if constexpr (_RunHost)
        run_algo2_host_policies<_Elem1, _Elem2>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    if constexpr (_RunHetero)
        run_algo2_hetero_policies<_Elem1Dc, _Elem2Dc, _CallId>(__algo, __checker, __algo_name);
#endif
}

} //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
#endif //_STD_RANGES_ALGO_ARCHETYPES_TEST_H
