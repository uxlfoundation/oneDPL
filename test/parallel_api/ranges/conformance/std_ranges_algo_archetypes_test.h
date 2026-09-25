// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

inline constexpr std::size_t archetype_test_size = 1000;

inline constexpr auto archetype_index_factory = [](std::size_t __i) { return (int)__i; };

inline constexpr auto archetype_offset_index_factory = [](std::size_t __i) {
    return (int)(__i + archetype_test_size);
};

#if TEST_DPCPP_BACKEND_PRESENT
template <typename _Policy>
inline constexpr bool is_device_policy_v = requires(_Policy& __policy) { __policy.queue(); };
#else
template <typename _Policy>
inline constexpr bool is_device_policy_v = false;
#endif

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

template <typename _Elem, typename _Policy>
auto
make_out_storage(_Policy&& __policy, std::size_t __n)
{
    return make_storage<_Elem>(std::forward<_Policy>(__policy), __n, [](std::size_t) { return 0; });
}

template <typename _Elem, typename _Policy, typename _Algo, typename _Checker>
void
run_algo(_Policy&& __policy, _Algo __algo, _Checker __checker, const char* __algo_name)
{
    auto __storage = make_storage<_Elem>(__policy, archetype_test_size, archetype_index_factory);
    auto __view = __storage.view();

    auto __res = __algo(std::forward<_Policy>(__policy), __view);

    EXPECT_TRUE(__checker(__view, __res), (std::string("wrong result from ") + __algo_name).c_str());
}

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

template <typename _Elem, typename _Algo, typename _Checker>
void
run_algo_host_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo<_Elem>(oneapi::dpl::execution::seq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::unseq, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par, __algo, __checker, __algo_name);
    run_algo<_Elem>(oneapi::dpl::execution::par_unseq, __algo, __checker, __algo_name);
}

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
template <typename _Elem, int _CallId, typename _Algo, typename _Checker>
void
run_algo_hetero_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo<_Elem>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
}

template <typename _Elem1, typename _Elem2, int _CallId, typename _Algo, typename _Checker>
void
run_algo2_hetero_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo2<_Elem1, _Elem2>(TestUtils::get_dpcpp_test_policy<_CallId>(), __algo, __checker, __algo_name);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename _Elem, typename _ElemDc, int _CallId, typename _Algo, typename _Checker>
void
run_algo_all_policies(_Algo __algo, _Checker __checker, const char* __algo_name)
{
    run_algo_host_policies<_Elem>(__algo, __checker, __algo_name);
#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<_ElemDc, _CallId>(__algo, __checker, __algo_name);
#endif
}

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
