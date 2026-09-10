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

// The content of the second range of a two-range algorithm when the check has to tell the two ranges
// apart: 1000, 1001, 1002, ... With it the values of the two ranges are disjoint, so a check of a copy,
// a move or a swap cannot pass just because the position of the second range already held the value
// which is expected there. See run_algo2_offset.
inline constexpr auto archetype_offset_index_factory = [](std::size_t __i) {
    return (int)(__i + archetype_test_size);
};

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

// The same as run_algo2, with the second range filled with the offset values instead of the ascending
// ones, see archetype_offset_index_factory. The algorithms which write the element of one range into
// the other one (copy, move, swap_ranges) are run this way, because with one and the same fill in both
// ranges the value the check reads is the one the position held from the start anyway.
template <typename _Elem1, typename _Elem2, typename _Policy, typename _Algo, typename _Checker>
void
run_algo2_offset(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage1 = make_storage<_Elem1>(__policy, archetype_test_size, archetype_index_factory);
    auto __storage2 = make_storage<_Elem2>(__policy, archetype_test_size, archetype_offset_index_factory);
    auto __view1 = __storage1.view();
    auto __view2 = __storage2.view();

    auto __res = __algo(std::forward<_Policy>(__policy), __view1, __view2);

    EXPECT_TRUE(__checker(__view1, __view2, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

// The same two runners over plain_archetype_view instead of archetype_view: the range then provides
// nothing but begin() and end(), because it does not derive from std::ranges::view_interface, so an
// implementation which calls size(), operator[] or any other member of it does not compile. Only a few
// representative algorithms are run this way - the range shape is a property of the implementation of
// the dispatch and not of the individual algorithm, so one call per pattern shape is enough.
template <typename _Elem, typename _Policy, typename _Algo, typename _Checker>
void
run_algo_plain(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage = make_storage<_Elem>(__policy, archetype_test_size, archetype_index_factory);
    auto __view = __storage.template view<archetypes::plain_archetype_view>();

    auto __res = __algo(std::forward<_Policy>(__policy), __view);

    EXPECT_TRUE(__checker(__view, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

template <typename _Elem1, typename _Elem2, typename _Policy, typename _Algo, typename _Checker>
void
run_algo2_plain(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage1 = make_storage<_Elem1>(__policy, archetype_test_size, archetype_index_factory);
    auto __storage2 = make_storage<_Elem2>(__policy, archetype_test_size, archetype_index_factory);
    auto __view1 = __storage1.template view<archetypes::plain_archetype_view>();
    auto __view2 = __storage2.template view<archetypes::plain_archetype_view>();

    auto __res = __algo(std::forward<_Policy>(__policy), __view1, __view2);

    EXPECT_TRUE(__checker(__view1, __view2, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

// Runs a one-range algorithm with every host policy. Each of them reaches its own implementation
// branch: the vectorized ones (unseq, par_unseq) go through the SIMD bricks and the parallel ones
// (par, par_unseq) through the parallel patterns, so all four are needed to have every branch
// compiled. A call site whose host side is broken wraps this call in an #if on the _HOST gap macro of
// the algorithm, which switches all four off together, whichever of the branches is the broken one.
//
// Calling this and not run_algo_all_policies also covers the archetypes which are meaningful for the
// host policies only: a value argument which is neither copyable nor movable cannot be passed into a
// device kernel, and the host implementation is required to refer to the value of the user.
template <typename _Elem, typename _Algo, typename _Checker>
void
run_algo_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo<_Elem>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

// Runs a two-range algorithm with every host policy, see run_algo_host_policies.
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
// "definition with same mangled name" error. The ids only have to be unique inside one translation
// unit, and every test file is its own executable, so each file numbers its calls from zero.
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
// A call site which has to skip a policy because of a known implementation gap does not use this
// helper: it spells out the policies it does run, so that the #if on the gap macro covers exactly the
// broken part and every other branch of the implementation stays compiled.
template <typename _Elem, typename _ElemDc, int _CallId, typename _Algo, typename _Checker>
void
run_algo_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo_host_policies<_Elem>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<_ElemDc, _CallId>(__algo, __checker, __algo_name);
#endif
}

// Runs a two-range algorithm with both the host and the hetero policies, see run_algo_all_policies.
template <typename _Elem1, typename _Elem2, typename _Elem1Dc, typename _Elem2Dc, int _CallId, typename _Algo,
          typename _Checker>
void
run_algo2_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_host_policies<_Elem1, _Elem2>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<_Elem1Dc, _Elem2Dc, _CallId>(__algo, __checker, __algo_name);
#endif
}

// The host policies with the offset fill of the second range, for the call sites which run the device
// side separately or not at all. See run_algo2_offset.
template <typename _Elem1, typename _Elem2, typename _Algo, typename _Checker>
void
run_algo2_offset_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_offset<_Elem1, _Elem2>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo2_offset<_Elem1, _Elem2>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo2_offset<_Elem1, _Elem2>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo2_offset<_Elem1, _Elem2>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

#if TEST_DPCPP_BACKEND_PRESENT
template <typename _Elem1, typename _Elem2, int _CallId, typename _Algo, typename _Checker>
void
run_algo2_offset_hetero_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_offset<_Elem1, _Elem2>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

// The same as run_algo2_all_policies with the offset fill of the second range, see run_algo2_offset.
template <typename _Elem1, typename _Elem2, typename _Elem1Dc, typename _Elem2Dc, int _CallId, typename _Algo,
          typename _Checker>
void
run_algo2_offset_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_offset_host_policies<_Elem1, _Elem2>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_offset<_Elem1Dc, _Elem2Dc>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
#endif
}

// The same two helpers over the plain range without view_interface, see run_algo_plain.
template <typename _Elem, typename _ElemDc, int _CallId, typename _Algo, typename _Checker>
void
run_algo_plain_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo_plain<_Elem>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo_plain<_Elem>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo_plain<_Elem>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo_plain<_Elem>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_plain<_ElemDc>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
#endif
}

template <typename _Elem1, typename _Elem2, typename _Elem1Dc, typename _Elem2Dc, int _CallId, typename _Algo,
          typename _Checker>
void
run_algo2_plain_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2_plain<_Elem1, _Elem2>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo2_plain<_Elem1, _Elem2>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo2_plain<_Elem1, _Elem2>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo2_plain<_Elem1, _Elem2>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_plain<_Elem1Dc, _Elem2Dc>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
#endif
}

} //namespace test_std_ranges

#endif //_ENABLE_STD_RANGES_TESTING
#endif //_STD_RANGES_ALGO_ARCHETYPES_TEST_H
