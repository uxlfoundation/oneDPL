// -*- C++ -*-
//===-- rts_bridge_mat_3o_buffer_only.pass.cpp ----------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3o (Matrix2x2): Same as 3e but forces scratch storage to use
// the buffer-only path (no USM pointer in __combi_accessor). If this passes
// while 3e fails, the USM/buffer dual path in __combi_accessor is the trigger.
//
// Approach: We construct a custom scratch storage that wraps a sycl::buffer
// and always returns buffer-backed __combi_accessor (with nullptr USM ptr).
// This exercises the real submitter kernel body templates but with different
// scratch accessor internals.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
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

// Buffer-only scratch storage that mimics __result_and_scratch_storage interface
// but NEVER uses USM. Always goes through the sycl::buffer path.
template <typename _T, std::size_t _NResults = 1>
struct __buffer_only_scratch_storage {
    using __sycl_buffer_t = sycl::buffer<_T, 1>;

    __sycl_buffer_t __sycl_buf;
    std::size_t __scratch_n;

    __buffer_only_scratch_storage(sycl::queue /*__q*/, std::size_t __scratch_n_arg)
        : __sycl_buf(__scratch_n_arg + _NResults), __scratch_n(__scratch_n_arg)
    {
    }

    template <typename _Acc>
    static auto
    __get_usm_or_buffer_accessor_ptr(const _Acc& __acc, std::size_t = 0)
    {
        // For buffer-only mode, __combi_accessor::__data() returns &__acc[0]
        // since __ptr is nullptr. This static method matches the interface.
        return __acc.__data();
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_scratch_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
        // Pass nullptr as USM pointer to force buffer path in __combi_accessor
        return oneapi::dpl::__par_backend_hetero::__combi_accessor<_T, _AccessMode>(
            __cgh, const_cast<__sycl_buffer_t&>(__sycl_buf), static_cast<_T*>(nullptr), __prop_list);
    }

    template <sycl::access_mode _AccessMode = sycl::access_mode::read_write>
    auto
    __get_result_acc(sycl::handler& __cgh, const sycl::property_list& __prop_list = {}) const
    {
        // Results are at offset __scratch_n, size _NResults
        return oneapi::dpl::__par_backend_hetero::__combi_accessor<_T, _AccessMode>(
            __cgh, const_cast<__sycl_buffer_t&>(__sycl_buf), static_cast<_T*>(nullptr),
            __scratch_n, _NResults, __prop_list);
    }
};

// Kernel name tags
struct Bridge3oReduceKernel;
struct Bridge3oScanKernel;

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = multiply_matrix<std::int32_t>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    using _WriteOp = rts::__simple_write_to_id;

    constexpr std::size_t N = 20000;
    constexpr std::uint16_t __max_inputs_per_item =
        std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});
    constexpr bool __inclusive = true;
    constexpr bool __is_unique_pattern_v = false;

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
    std::printf("[bridge3o_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    // Use the same parameter calculation as __parallel_transform_reduce_then_scan
    const auto __supported_sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    const std::uint8_t __min_sub_group_size =
        *std::min_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());
    const std::uint8_t __max_sub_group_size =
        *std::max_element(__supported_sg_sizes.begin(), __supported_sg_sizes.end());

    const std::uint32_t __wg_size_cap = dev.is_gpu() ? 1024 : 256;
    const std::uint32_t __max_work_group_size = oneapi::dpl::__internal::__max_work_group_size(q, __wg_size_cap);
    const std::uint32_t __work_group_size = (__max_work_group_size / __max_sub_group_size) * __max_sub_group_size;

    const std::uint32_t __num_work_groups =
        oneapi::dpl::__internal::__dpl_bit_ceil(
            dev.get_info<sycl::info::device::max_compute_units>()) /
        (dev.is_gpu() ? 4 : 1);

    const std::uint32_t __max_num_sub_groups_local = __work_group_size / __min_sub_group_size;
    const std::uint32_t __max_num_sub_groups_global = __max_num_sub_groups_local * __num_work_groups;
    const std::uint32_t __max_inputs_per_block = __work_group_size * __max_inputs_per_item * __num_work_groups;

    std::size_t __inputs_remaining = N;
    std::uint32_t __inputs_per_item =
        __inputs_remaining >= __max_inputs_per_block
            ? __max_inputs_per_item
            : oneapi::dpl::__internal::__dpl_ceiling_div(
                  oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                  __num_work_groups * __work_group_size);
    const std::size_t __block_size = std::min(__inputs_remaining, std::size_t{__max_inputs_per_block});
    const std::size_t __num_blocks = __inputs_remaining / __block_size + (__inputs_remaining % __block_size != 0);

    const bool __use_slm_for_comm = !std::is_trivially_copyable_v<T> || !dev.is_gpu();

    std::printf("[bridge3o_mat] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u use_slm=%d\n",
                __work_group_size, __num_work_groups, __inputs_per_item, __num_blocks,
                __max_num_sub_groups_global + 2, (int)__use_slm_for_comm);

    // Allocate scratch using BUFFER-ONLY storage (no USM)
    __buffer_only_scratch_storage<T> __result_and_scratch{q, __max_num_sub_groups_global + 2};

    // Create buffer-backed views (same as 3e)
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _InitType init{};

    // Instantiate the REAL submitter types
    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<Bridge3oReduceKernel>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<Bridge3oScanKernel>;

    using _ReduceSubmitter =
        rts::__parallel_reduce_then_scan_reduce_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v,
                                                          _GenInput, _BinaryOp, _InitType, _ReduceKernel>;
    using _ScanSubmitter =
        rts::__parallel_reduce_then_scan_scan_submitter<__max_inputs_per_item, __inclusive, __is_unique_pattern_v,
                                                        _BinaryOp, _GenInput, _ScanInputTransform, _WriteOp, _InitType,
                                                        _ScanKernel>;

    _ReduceSubmitter __reduce_submitter{__num_work_groups, __work_group_size, __max_inputs_per_block,
                                        __max_num_sub_groups_local, N, __use_slm_for_comm,
                                        gen_input, binary_op, init};
    _ScanSubmitter __scan_submitter{__num_work_groups, __work_group_size, __max_inputs_per_block,
                                    __max_num_sub_groups_local, __max_num_sub_groups_global,
                                    __num_blocks, N, __use_slm_for_comm,
                                    binary_op, gen_input, scan_xform, write_op, init};

    sycl::event __prior_event;

    for (std::size_t __b = 0; __b < __num_blocks; ++__b) {
        auto in_rng = all_view<T, sycl::access::mode::read>(buf_in, 0, N);
        auto out_rng = all_view<T, sycl::access::mode::read_write>(buf_out, 0, N);

        std::uint32_t __workitems_in_block = oneapi::dpl::__internal::__dpl_ceiling_div(
            std::min(__inputs_remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_round_up =
            oneapi::dpl::__internal::__dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __kernel_nd_range = sycl::nd_range<1>(sycl::range<1>(__workitems_round_up),
                                                    sycl::range<1>(__work_group_size));

        __prior_event = __reduce_submitter(q, __kernel_nd_range, in_rng, __result_and_scratch,
                                           __prior_event, __inputs_remaining, __b);
        __prior_event = __scan_submitter(q, __kernel_nd_range, in_rng, out_rng, __result_and_scratch,
                                         __prior_event, __inputs_remaining, __b);

        __inputs_remaining -= std::min(__inputs_remaining, __block_size);
        if (__b + 2 == __num_blocks) {
            __inputs_per_item = __inputs_remaining >= __max_inputs_per_block
                ? __max_inputs_per_item
                : oneapi::dpl::__internal::__dpl_ceiling_div(
                      oneapi::dpl::__internal::__dpl_bit_ceil(__inputs_remaining),
                      __num_work_groups * __work_group_size);
        }
    }
    q.wait();

    // Read output
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
                std::printf("[bridge3o_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3o_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
