// -*- C++ -*-
//===-- rts_bridge_mat_3m_no_placeholder.pass.cpp -------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3m (Matrix2x2): Same as 3e but avoids placeholder accessors.
// Uses a custom range type backed by non-placeholder accessors that are created
// fresh each iteration (inside the submit lambda scope). The real submitter
// kernel body templates are still exercised. If this passes while 3e fails,
// placeholder accessors are the trigger for the crash.
//
// Approach: We use all_view with placeholder::false_t. This creates a
// non-placeholder accessor at construction time. The __require_access_range
// catch-all overload (no-op for unknown range types) handles it. The submitter
// operator() still calls __require_access, but it becomes a no-op since our
// range type is not recognized as all_view<T,M> (it's all_view<T,M,target,false_t>).

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

// A simple range type that wraps a non-placeholder accessor.
// This is NOT recognized by __require_access_range as all_view<T,M>,
// so the catch-all no-op overload is used. The accessor is already bound
// to the command group at construction time (non-placeholder).
template <typename T, sycl::access::mode M>
struct non_placeholder_range {
    using value_type = T;
    using __return_t = std::conditional_t<M == sycl::access::mode::read, const T, T>;

    sycl::accessor<T, 1, M, sycl::target::device, sycl::access::placeholder::false_t> __acc;

    non_placeholder_range(sycl::buffer<T, 1>& __buf, sycl::handler& __cgh, std::size_t __n)
        : __acc(__buf, __cgh, sycl::range<1>(__n))
    {
    }

    __return_t& operator[](std::size_t i) const { return __acc[i]; }
    std::size_t size() const { return __acc.size(); }
};

// Kernel name tags
struct Bridge3mReduceKernel;
struct Bridge3mScanKernel;

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
    std::printf("[bridge3m_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

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

    std::printf("[bridge3m_mat] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u use_slm=%d\n",
                __work_group_size, __num_work_groups, __inputs_per_item, __num_blocks,
                __max_num_sub_groups_global + 2, (int)__use_slm_for_comm);

    // Allocate scratch using the real storage type
    rts::__result_and_scratch_storage<T> __result_and_scratch{q, __max_num_sub_groups_global + 2};

    // Create buffers (NOT all_view - we'll create non-placeholder ranges inside submit)
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _InitType init{};

    // Instantiate the REAL submitter types - but templated on our non-placeholder range type
    using _InRng = non_placeholder_range<T, sycl::access::mode::read>;
    using _OutRng = non_placeholder_range<T, sycl::access::mode::read_write>;

    using _ReduceKernel = oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<Bridge3mReduceKernel>;
    using _ScanKernel = oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<Bridge3mScanKernel>;

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
        std::uint32_t __workitems_in_block = oneapi::dpl::__internal::__dpl_ceiling_div(
            std::min(__inputs_remaining, std::size_t{__max_inputs_per_block}), __inputs_per_item);
        std::uint32_t __workitems_round_up =
            oneapi::dpl::__internal::__dpl_ceiling_div(__workitems_in_block, __work_group_size) * __work_group_size;
        auto __kernel_nd_range = sycl::nd_range<1>(sycl::range<1>(__workitems_round_up),
                                                    sycl::range<1>(__work_group_size));

        // Reduce phase: submit with non-placeholder range created inside the lambda
        __prior_event = q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__prior_event);
            // Create non-placeholder accessor range inside submit lambda
            _InRng in_rng(buf_in, __cgh, N);

            // Allocate SLM
            __dpl_sycl::__local_accessor<T> __sub_group_partials(__max_num_sub_groups_local, __cgh);
            __dpl_sycl::__local_accessor<T> __comm_slm(__use_slm_for_comm ? __work_group_size : 0, __cgh);
            auto __temp_acc = __result_and_scratch.template __get_scratch_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<Bridge3mReduceKernel>(
                __kernel_nd_range, [=, __rs = __reduce_submitter](sycl::nd_item<1> __ndi)
                    [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(32)]] {
                    using _StorageT = rts::__result_and_scratch_storage<T>;
                    T* __tmp_acc = _StorageT::__get_usm_or_buffer_accessor_ptr(__temp_acc);

                    if (!__use_slm_for_comm)
                        __rs.template __reduce_kernel_body<true>(__ndi, __sub_group_partials, __comm_slm,
                                                                 __tmp_acc, in_rng, __inputs_remaining, __b);
                    else
                        __rs.template __reduce_kernel_body<false>(__ndi, __sub_group_partials, __comm_slm,
                                                                  __tmp_acc, in_rng, __inputs_remaining, __b);
                });
        });

        // Scan phase: submit with non-placeholder ranges created inside the lambda
        std::size_t __num_remaining = N - __b * __max_inputs_per_block;
        std::uint32_t __inputs_in_block = std::min(__num_remaining, std::size_t{__max_inputs_per_block});

        __prior_event = q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__prior_event);
            // Create non-placeholder accessor ranges inside submit lambda
            _InRng in_rng(buf_in, __cgh, N);
            _OutRng out_rng(buf_out, __cgh, N);

            // Allocate SLM
            __dpl_sycl::__local_accessor<T> __sub_group_partials(__max_num_sub_groups_local + 1, __cgh);
            __dpl_sycl::__local_accessor<T> __comm_slm(__use_slm_for_comm ? __work_group_size : 0, __cgh);
            auto __temp_acc = __result_and_scratch.template __get_scratch_acc<sycl::access_mode::read_write>(__cgh);
            auto __res_acc = __result_and_scratch.template __get_result_acc<sycl::access_mode::write>(
                __cgh, __dpl_sycl::__no_init{});

            __cgh.parallel_for<Bridge3mScanKernel>(
                __kernel_nd_range, [=, __ss = __scan_submitter](sycl::nd_item<1> __ndi)
                    [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED(32)]] {
                    using _StorageT = rts::__result_and_scratch_storage<T>;
                    T* __tmp_acc_ptr = _StorageT::__get_usm_or_buffer_accessor_ptr(__temp_acc);
                    T* __res_ptr = _StorageT::__get_usm_or_buffer_accessor_ptr(__res_acc,
                                                                                __max_num_sub_groups_global + 2);

                    if (!__use_slm_for_comm)
                        __ss.template __scan_kernel_body<true>(__ndi, __sub_group_partials, __comm_slm,
                                                               __tmp_acc_ptr, __res_ptr, in_rng, out_rng,
                                                               __inputs_in_block, __inputs_remaining, __b);
                    else
                        __ss.template __scan_kernel_body<false>(__ndi, __sub_group_partials, __comm_slm,
                                                                __tmp_acc_ptr, __res_ptr, in_rng, out_rng,
                                                                __inputs_in_block, __inputs_remaining, __b);
                });
        });

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
                std::printf("[bridge3m_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3m_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
