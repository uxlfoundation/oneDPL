// -*- C++ -*-
//===-- rts_bridge_mat_3g_inline_function.pass.cpp ------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3g (Matrix2x2): Copies the entire body of
// __parallel_transform_reduce_then_scan verbatim into a local function and
// calls it with the same args as 4b. If this fails while 3e passes, the
// compiler makes different optimization decisions when the full function is
// compiled together (submitter instantiation + loop + all parameters).

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <oneapi/dpl/pstl/utils.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

using TestUtils::Matrix2x2;
using TestUtils::multiply_matrix;

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

// Kernel name tags
struct Bridge3gReduceKernel;
struct Bridge3gScanKernel;

// Verbatim copy of __parallel_transform_reduce_then_scan as a local function
template <std::uint32_t __bytes_per_work_item_iter, typename _CustomName, typename _InRng, typename _OutRng,
          typename _GenReduceInput, typename _ReduceOp, typename _GenScanInput, typename _ScanInputTransform,
          typename _WriteOp, typename _InitType, typename _Inclusive, typename _IsUniquePattern>
rts::__future<sycl::event, rts::__result_and_scratch_storage<typename _InitType::__value_type>>
local_parallel_transform_reduce_then_scan(sycl::queue& __q, const std::size_t __n, _InRng&& __in_rng, _OutRng&& __out_rng,
                                          _GenReduceInput __gen_reduce_input, _ReduceOp __reduce_op,
                                          _GenScanInput __gen_scan_input, _ScanInputTransform __scan_input_transform,
                                          _WriteOp __write_op, _InitType __init, _Inclusive, _IsUniquePattern,
                                          sycl::event __prior_event = {})
{
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        rts::__reduce_then_scan_reduce_kernel<_CustomName>>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
        rts::__reduce_then_scan_scan_kernel<_CustomName>>;
    using _ValueType = typename _InitType::__value_type;

    const auto __supported_sg_sizes = __q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    const std::uint8_t __min_sub_group_size =
        *std::min_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());
    const std::uint8_t __max_sub_group_size =
        *std::max_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());
    constexpr std::uint16_t __max_inputs_per_item =
        std::max(std::uint16_t{1}, std::uint16_t{512 / __bytes_per_work_item_iter});
    constexpr bool __inclusive = _Inclusive::value;
    constexpr bool __is_unique_pattern_v = _IsUniquePattern::value;

    const std::uint32_t __wg_size_cap = __q.get_device().is_gpu() ? 1024 : 256;
    const std::uint32_t __max_work_group_size = oneapi::dpl::__internal::__max_work_group_size(__q, __wg_size_cap);
    const std::uint32_t __work_group_size = (__max_work_group_size / __max_sub_group_size) * __max_sub_group_size;

    const std::uint32_t __num_work_groups =
        oneapi::dpl::__internal::__dpl_bit_ceil(
            __q.get_device().template get_info<sycl::info::device::max_compute_units>()) /
        (__q.get_device().is_gpu() ? 4 : 1);

    const std::uint32_t __max_num_sub_groups_local = __work_group_size / __min_sub_group_size;
    const std::uint32_t __max_num_sub_groups_global = __max_num_sub_groups_local * __num_work_groups;
    const std::uint32_t __max_inputs_per_work_group = __work_group_size * __max_inputs_per_item;
    const std::uint32_t __max_inputs_per_block = __max_inputs_per_work_group * __num_work_groups;
    std::size_t __inputs_remaining = __n;
    if constexpr (__is_unique_pattern_v)
    {
        __inputs_remaining -= 1;
    }
    assert(__inputs_remaining > 0);
    std::uint32_t __inputs_per_item =
        __inputs_remaining >= __max_inputs_per_block
            ? __max_inputs_per_item
            : oneapi::dpl::__internal::__dpl_ceiling_div(oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                                                         __num_work_groups * __work_group_size);
    const std::size_t __block_size = std::min(__inputs_remaining, std::size_t{__max_inputs_per_block});
    const std::size_t __num_blocks = __inputs_remaining / __block_size + (__inputs_remaining % __block_size != 0);

    rts::__result_and_scratch_storage<_ValueType> __result_and_scratch{__q, __max_num_sub_groups_global + 2};

    const bool __use_slm_for_comm = !std::is_trivially_copyable_v<_ValueType> || !__q.get_device().is_gpu();

    using _ReduceSubmitter =
        rts::__parallel_reduce_then_scan_reduce_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v,
                                                     _GenReduceInput, _ReduceOp, _InitType, _ReduceKernel>;
    using _ScanSubmitter =
        rts::__parallel_reduce_then_scan_scan_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v, _ReduceOp,
                                                   _GenScanInput, _ScanInputTransform, _WriteOp, _InitType,
                                                   _ScanKernel>;
    _ReduceSubmitter __reduce_submitter{__num_work_groups,
                                        __work_group_size,
                                        __max_inputs_per_block,
                                        __max_num_sub_groups_local,
                                        __n,
                                        __use_slm_for_comm,
                                        __gen_reduce_input,
                                        __reduce_op,
                                        __init};
    _ScanSubmitter __scan_submitter{__num_work_groups,
                                    __work_group_size,
                                    __max_inputs_per_block,
                                    __max_num_sub_groups_local,
                                    __max_num_sub_groups_global,
                                    __num_blocks,
                                    __n,
                                    __use_slm_for_comm,
                                    __reduce_op,
                                    __gen_scan_input,
                                    __scan_input_transform,
                                    __write_op,
                                    __init};

    for (std::size_t __b = 0; __b < __num_blocks; ++__b)
    {
        std::uint32_t __workitems_in_block = oneapi::dpl::__internal::__dpl_ceiling_div(
            std::min(__inputs_remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_in_block_round_up_workgroup =
            oneapi::dpl::__internal::__dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __global_range = sycl::range<1>(__workitems_in_block_round_up_workgroup);
        auto __local_range = sycl::range<1>(__work_group_size);
        auto __kernel_nd_range = sycl::nd_range<1>(__global_range, __local_range);

        __prior_event = __reduce_submitter(__q, __kernel_nd_range, __in_rng, __result_and_scratch, __prior_event,
                                           __inputs_remaining, __b);
        __prior_event = __scan_submitter(__q, __kernel_nd_range, __in_rng, __out_rng, __result_and_scratch,
                                         __prior_event, __inputs_remaining, __b);
        __inputs_remaining -= std::min(__inputs_remaining, __block_size);
        if (__b + 2 == __num_blocks)
        {
            __inputs_per_item = __inputs_remaining >= __max_inputs_per_block
                                    ? __max_inputs_per_item
                                    : oneapi::dpl::__internal::__dpl_ceiling_div(
                                          oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                                          __num_work_groups * __work_group_size);
        }
    }
    return rts::__future{std::move(__prior_event), std::move(__result_and_scratch)};
}

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = multiply_matrix<std::int32_t>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    using _WriteOp = rts::__simple_write_to_id;

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);

    _BinaryOp mat_op{};
    std::vector<T> h_expected(N);
    h_expected[0] = h_input[0];
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = mat_op(h_expected[i - 1], h_input[i]);

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[bridge3g_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    all_view<T, sycl::access::mode::read> in_view(buf_in, 0, N);
    all_view<T, sycl::access::mode::read_write> out_view(buf_out, 0, N);

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _InitType init{};

    std::printf("[bridge3g_mat] Calling local_parallel_transform_reduce_then_scan\n");

    auto future = local_parallel_transform_reduce_then_scan<sizeof(T), Bridge3gReduceKernel>(
        q, N, in_view, out_view,
        gen_input, binary_op, gen_input, scan_xform, write_op,
        init, std::true_type{}, std::false_type{});

    future.__checked_deferrable_wait();

    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (!(h_out[i] == h_expected[i])) {
            if (errors < 20)
                std::printf("[bridge3g_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3g_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

    return errors ? 1 : 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main() {
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_test();
    if (result != 0) return result;
#endif
    return TestUtils::done();
}
