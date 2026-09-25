// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include "std_ranges_archetypes.h"
#include "std_ranges_algo_archetypes_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    run_algo_all_policies<writable_archetype, writable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");

    run_algo2_offset_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc,
                                  1>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(out_view);
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val &&
                   std::ranges::begin(out_view)[n - 1].val == std::ranges::begin(in_view)[n - 1].val;
        },
        "copy");

    run_algo2_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc, 2>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::reverse_copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == std::ranges::begin(in_view)[n - 1].val &&
                   std::ranges::begin(out_view)[n - 1].val == std::ranges::begin(in_view)[0].val;
        },
        "reverse_copy");

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::rotate_copy(std::forward<decltype(policy)>(policy), in_view,
                                           std::ranges::begin(in_view) + 10, out_view);
        };
        auto check = [](auto&& in_view, auto&& out_view, auto) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == std::ranges::begin(in_view)[10].val &&
                   std::ranges::begin(out_view)[n - 10].val == std::ranges::begin(in_view)[0].val;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "rotate_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_ROTATE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 3>(call, check, "rotate_copy");
#endif
    }

    run_algo2_offset_all_policies<move_in_archetype, move_out_archetype, move_in_archetype_dc, move_out_archetype_dc,
                                  4>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");

    run_algo2_offset_all_policies<swap_archetype, swap_archetype, swap_archetype_dc, swap_archetype_dc, 5>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == (int)archetype_test_size + 7 &&
                   std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 6>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 7>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, projection");

    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 8>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary");

    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 9>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_projected_binary_op{}, transform_proj{}, transform_proj{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, projections");

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "copy_if");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 10>(call, check, "copy_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == 1 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "remove_copy_if");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 11>(call, check, "remove_copy_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::unique_copy(std::forward<decltype(policy)>(policy), in_view, out_view, copy_equiv{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "unique_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_UNIQUE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 12>(call, check, "unique_copy");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_true_view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(in_view)>>;
            auto out_false_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_false_view = out_false_storage.view();
            auto res = dpl_ranges::partition_copy(std::forward<decltype(policy)>(policy), in_view, out_true_view,
                                                  out_false_view, copy_pred{});
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_true_view)[1].val == 3 && std::ranges::begin(out_false_view)[0].val == 1 &&
                   (std::size_t)(res.out1 - std::ranges::begin(out_true_view)) == (n + 2) / 3 &&
                   (std::size_t)(res.out2 - std::ranges::begin(out_false_view)) == n - (n + 2) / 3;
        };
        auto check = [](auto&&, auto&&, auto res) { return res; };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "partition_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 13>(call, check, "partition_copy");
#endif
    }

    run_algo_all_policies<writable_archetype, writable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace_if(std::forward<decltype(policy)>(policy), view, write_pred{},
                                          typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 && std::ranges::begin(view)[1].val == 1 &&
                   std::ranges::begin(view)[3].val == 42;
        },
        "replace_if");

    run_algo_all_policies<replaceable_archetype, replaceable_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace(std::forward<decltype(policy)>(policy), view, search_value{3},
                                       typename elem_t::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[3].val == 42 && std::ranges::begin(view)[2].val == 2;
        },
        "replace");

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[3].val == 4 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - 1;
        };

        run_algo2_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check, "remove_copy");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_HETERO
        run_algo2_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 16>(call, check, "remove_copy");
#endif
    }

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred{},
                                               typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[0].val == 42 && std::ranges::begin(out_view)[3].val == 42 &&
                   std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST
        run_algo2_offset_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "replace_copy_if");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 17>(call, check,
                                                                                         "replace_copy_if");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                            typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[3].val == 42 && std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST
        run_algo2_offset_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check, "replace_copy");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 18>(call, check,
                                                                                                "replace_copy");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST || TEST_DPCPP_BACKEND_PRESENT

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 19>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; },
        "transform, non-const callable");

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, 20>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, non-const projection");

    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 21>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op_mut{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const callable");

    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, 22>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                      transform_projected_binary_op{}, transform_proj_mut{}, transform_proj_mut{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const projections");

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy_if(std::forward<decltype(policy)>(policy), in_view, out_view, copy_pred_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "copy_if, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 23>(call, check,
                                                                                   "copy_if, non-const predicate");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view,
                                              copy_pred_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[0].val == 1 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check,
                                                                       "remove_copy_if, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_IF_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 24>(
            call, check, "remove_copy_if, non-const predicate");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::unique_copy(std::forward<decltype(policy)>(policy), in_view, out_view,
                                           copy_equiv_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[1].val == 3 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == (n + 2) / 3;
        };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check, "unique_copy, non-const relation");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_UNIQUE_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 25>(call, check,
                                                                                   "unique_copy, non-const relation");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_true_view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(in_view)>>;
            auto out_false_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_false_view = out_false_storage.view();
            auto res = dpl_ranges::partition_copy(std::forward<decltype(policy)>(policy), in_view, out_true_view,
                                                  out_false_view, copy_pred_mut{});
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_true_view)[1].val == 3 && std::ranges::begin(out_false_view)[0].val == 1 &&
                   (std::size_t)(res.out1 - std::ranges::begin(out_true_view)) == (n + 2) / 3 &&
                   (std::size_t)(res.out2 - std::ranges::begin(out_false_view)) == n - (n + 2) / 3;
        };
        auto check = [](auto&&, auto&&, auto res) { return res; };

        run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(call, check,
                                                                       "partition_copy, non-const predicate");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_COPY_HETERO
        run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 26>(
            call, check, "partition_copy, non-const predicate");
#endif
    }

    run_algo_all_policies<writable_archetype, writable_archetype_dc, 27>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace_if(std::forward<decltype(policy)>(policy), view, write_pred_mut{},
                                          typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 && std::ranges::begin(view)[1].val == 1 &&
                   std::ranges::begin(view)[3].val == 42;
        },
        "replace_if, non-const predicate");

    run_algo_all_policies<replaceable_archetype, replaceable_archetype_dc, 28>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::replace(std::forward<decltype(policy)>(policy), view, search_value{3},
                                       typename elem_t::value_arg{42}, replace_proj_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[3].val == 42 && std::ranges::begin(view)[2].val == 2;
        },
        "replace, non-const projection");

    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::remove_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                           replace_proj_mut{});
        };
        auto check = [](auto&& in_view, auto&& out_view, auto res) {
            const auto n = std::ranges::size(in_view);
            return std::ranges::begin(out_view)[3].val == 4 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == n - 1;
        };

        run_algo2_host_policies<remove_copy_in_archetype, copy_out_archetype>(call, check,
                                                                             "remove_copy, non-const projection");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_COPY_HETERO
        run_algo2_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 29>(
            call, check, "remove_copy, non-const projection");
#endif
    }

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy_if(std::forward<decltype(policy)>(policy), in_view, out_view,
                                               copy_pred_mut{}, typename out_t::value_arg{42});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[0].val == 42 && std::ranges::begin(out_view)[3].val == 42 &&
                   std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST
        run_algo2_offset_host_policies<copy_in_archetype, copy_out_archetype>(call, check,
                                                                             "replace_copy_if, non-const predicate");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 30>(
            call, check, "replace_copy_if, non-const predicate");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_IF_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& in_view, auto&& out_view) {
            using out_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(out_view)>>;
            return dpl_ranges::replace_copy(std::forward<decltype(policy)>(policy), in_view, out_view, search_value{3},
                                            typename out_t::value_arg{42}, replace_proj_mut{});
        };
        auto check = [](auto&&, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[3].val == 42 && std::ranges::begin(out_view)[2].val == 2;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST
        run_algo2_offset_host_policies<remove_copy_in_archetype, copy_out_archetype>(
            call, check, "replace_copy, non-const projection");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo2_offset_hetero_policies<remove_copy_in_archetype_dc, copy_out_archetype_dc, 31>(
            call, check, "replace_copy, non-const projection");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REPLACE_COPY_HOST || TEST_DPCPP_BACKEND_PRESENT

    run_algo2_plain_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc,
                                 32>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto res) {
            return res.in == std::ranges::begin(in_view) + std::ranges::size(in_view) &&
                   res.out == std::ranges::begin(out_view) + std::ranges::size(out_view);
        },
        "copy, plain ranges");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
