// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING

#include <algorithm>
#include <functional>
#include <vector>

using namespace test_std_ranges;
namespace dpl_ranges = oneapi::dpl::ranges;

// std::ranges::partition is not a stable algorithm: it only guarantees that elements satisfying the
// predicate precede those that do not, without preserving relative order. The parallel and device
// specializations rearrange elements differently from std::ranges::partition, so we verify the
// algorithm post-conditions instead of comparing the whole range element-wise against a reference.
template <typename Range, typename Pred, typename Proj>
void
check_partition_effect(const std::vector<int>& original_keys, Range&& r, int n, Pred pred, Proj proj,
                       decltype(std::ranges::begin(std::declval<Range&>())) res_begin, const char* msg)
{
    auto holds = [&](auto&& v) { return bool(std::invoke(pred, std::invoke(proj, v))); };

    // The partition point: number of elements satisfying the predicate.
    const int k = static_cast<int>(std::count_if(original_keys.begin(), original_keys.end(),
                                                  [&](int key) { return bool(std::invoke(pred, key)); }));

    // Returned subrange must start exactly at the partition point and end at the range end.
    EXPECT_TRUE(res_begin == std::ranges::begin(r) + k, (std::string("wrong partition point: ") + msg).c_str());

    // All elements before the partition point satisfy the predicate, all after do not.
    for (int i = 0; i < k; ++i)
        EXPECT_TRUE(holds(r[i]), (std::string("wrong left partition: ") + msg).c_str());
    for (int i = k; i < n; ++i)
        EXPECT_TRUE(!holds(r[i]), (std::string("wrong right partition: ") + msg).c_str());

    // The result must be a permutation of the input: compare the multiset of projected keys.
    std::vector<int> result_keys(n);
    for (int i = 0; i < n; ++i)
        result_keys[i] = std::invoke(proj, r[i]);
    std::vector<int> expected_keys = original_keys;
    std::sort(expected_keys.begin(), expected_keys.end());
    std::sort(result_keys.begin(), result_keys.end());
    EXPECT_TRUE(expected_keys == result_keys, (std::string("result is not a permutation: ") + msg).c_str());
}

// Data generator producing a mix of elements satisfying and not satisfying pred1 (val > 0),
// including zeros and duplicates to exercise partition ties.
inline auto partition_gen = [](int i) { return (i % 7) - 3; };

template <int CallId, typename T, typename Proj = std::identity>
struct test_partition
{
    template <typename Algo, typename Pred>
    void
    operator()(Algo algo, Pred pred, Proj proj = {}) const
    {
        for (int n : {0, 1, 2, 3, 7, 20, small_size, medium_size})
            host_case(algo, n, pred, proj);

        for (int n : {0, 1, 2, 3, 7, 20, small_size})
            host_view_case(algo, n, pred, proj);

        // A non-borrowed rvalue range must yield std::ranges::dangling as the return type.
        using rvalue_ret_t = decltype(algo(oneapi::dpl::execution::par, std::declval<std::vector<T>>(),
                                            std::declval<Pred>(), std::declval<Proj>()));
        static_assert(all_dangling_in_result_v<rvalue_ret_t>);

#if TEST_DPCPP_BACKEND_PRESENT
        // Pointer-to-member-function projections are not supported inside device kernels.
        if constexpr (!std::is_member_function_pointer_v<Proj>)
        {
#if _PSTL_LAMBDA_PTR_TO_MEMBER_WINDOWS_BROKEN
            if constexpr (!std::is_member_pointer_v<Proj>)
#endif
            {
                auto policy = TestUtils::get_dpcpp_test_policy();
                for (int n : {0, 1, small_size, medium_size})
                    device_case(policy, algo, n, pred, proj);
            }
        }
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

  private:
    static std::vector<int>
    make_keys(int n)
    {
        std::vector<int> keys(n);
        for (int i = 0; i < n; ++i)
            keys[i] = partition_gen(i);
        return keys;
    }

    static std::vector<T>
    make_data(const std::vector<int>& keys)
    {
        std::vector<T> data(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i)
            data[i] = T(keys[i]);
        return data;
    }

    template <typename Algo, typename Pred>
    void
    host_case(Algo algo, int n, Pred pred, Proj proj) const
    {
        const std::string msg = "host, partition<" + std::to_string(CallId) + ">";
        auto run = [&](auto&& policy)
        {
            std::vector<int> keys = make_keys(n);
            std::vector<T> data = make_data(keys);
            auto res = algo(policy, data, pred, proj);
            check_partition_effect(keys, data, n, pred, proj, res.begin(), msg.c_str());
            EXPECT_TRUE(res.end() == data.end(), (std::string("wrong subrange end: ") + msg).c_str());
        };
        run(oneapi::dpl::execution::seq);
        run(oneapi::dpl::execution::unseq);
        run(oneapi::dpl::execution::par);
        run(oneapi::dpl::execution::par_unseq);
    }

    template <typename Algo, typename Pred>
    void
    host_view_case(Algo algo, int n, Pred pred, Proj proj) const
    {
        const std::string msg = "host view, partition<" + std::to_string(CallId) + ">";
        auto run_subrange = [&](auto&& policy)
        {
            std::vector<int> keys = make_keys(n);
            std::vector<T> data = make_data(keys);
            auto view = std::ranges::subrange(data.begin(), data.end());
            auto res = algo(policy, view, pred, proj);
            check_partition_effect(keys, view, n, pred, proj, res.begin(), msg.c_str());
            EXPECT_TRUE(res.end() == view.end(), (std::string("wrong subrange end (subrange): ") + msg).c_str());
        };
        run_subrange(oneapi::dpl::execution::seq);
        run_subrange(oneapi::dpl::execution::par);

#if TEST_CPP20_SPAN_PRESENT
        auto run_span = [&](auto&& policy)
        {
            std::vector<int> keys = make_keys(n);
            std::vector<T> data = make_data(keys);
            std::span<T> view(data.data(), data.size());
            auto res = algo(policy, view, pred, proj);
            check_partition_effect(keys, view, n, pred, proj, res.begin(), msg.c_str());
            EXPECT_TRUE(res.end() == view.end(), (std::string("wrong subrange end (span): ") + msg).c_str());
        };
        run_span(oneapi::dpl::execution::seq);
        run_span(oneapi::dpl::execution::par);
#endif // TEST_CPP20_SPAN_PRESENT
    }

#if TEST_DPCPP_BACKEND_PRESENT
    template <typename ExecutionPolicy, typename Algo, typename Pred>
    void
    device_case(ExecutionPolicy&& policy, Algo algo, int n, Pred pred, Proj proj) const
    {
        const std::string msg = "device, partition<" + std::to_string(CallId) + ">";
        std::vector<int> keys = make_keys(n);
        std::vector<T> host = make_data(keys);
        usm_vector<T> usm(policy, host.data(), n);
        auto& vec = usm();
        auto res = algo(CLONE_TEST_POLICY_IDX(policy, CallId), vec, pred, proj);
        check_partition_effect(keys, vec, n, pred, proj, res.begin(), msg.c_str());
        EXPECT_TRUE(res.end() == vec.end(), (std::string("wrong subrange end: ") + msg).c_str());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
};

#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    // pred = pred1 (val > 0), projection = identity: plain integer keys.
    test_partition<0, int>{}(dpl_ranges::partition, pred1);

    // Projection applied to integer keys (proj doubles the value).
    test_partition<1, int, decltype(proj)>{}(dpl_ranges::partition, pred1, proj);

    // Member-data projection (P2::x): exercised on host and device.
    test_partition<2, P2, int P2::*>{}(dpl_ranges::partition, pred1, &P2::x);

    // Member-function projection (P2::proj): host only (skipped inside device kernels).
    test_partition<3, P2, int (P2::*)() const>{}(dpl_ranges::partition, pred1, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
