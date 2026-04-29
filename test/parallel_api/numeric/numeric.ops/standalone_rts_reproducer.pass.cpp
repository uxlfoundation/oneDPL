// -*- C++ -*-
//===-- standalone_rts_reproducer.pass.cpp ---------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Standalone SYCL reproducer for reduce_then_scan crash on Windows CPU debug.
// Flattened from oneDPL's parallel_backend_sycl_reduce_then_scan.h — no oneDPL
// algorithm headers, only sycl/sycl.hpp. Reproduces: inclusive_scan<int32_t>
// with n=20000 using the RTS two-kernel (reduce + scan) code path with
// SLM-based sub-group communication (the CPU path).

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

// ---------- helpers (host + device) ----------

inline std::uint32_t bit_ceil_u32(std::uint32_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline std::uint32_t ceiling_div(std::uint32_t a, std::uint32_t b) {
    return (a + b - 1) / b;
}

inline std::uint32_t ceiling_div(std::size_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>((a + b - 1) / b);
}

// ---------- SLM-based sub-group communication (CPU path) ----------

template <typename T>
T slm_shift_group_right(sycl::sub_group sg, T value, std::uint32_t shift, T* slm) {
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t sg_base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[sg_base + lid] = value;
    sycl::group_barrier(sg);
    T result = slm[sg_base + ((lid >= shift) ? lid - shift : lid)];
    sycl::group_barrier(sg);
    return result;
}

template <typename T>
T slm_group_broadcast(sycl::sub_group sg, T value, std::uint32_t broadcast_id, T* slm) {
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t sg_base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[sg_base + lid] = value;
    sycl::group_barrier(sg);
    T result = slm[sg_base + broadcast_id];
    sycl::group_barrier(sg);
    return result;
}

// ---------- inclusive sub-group scans (SLM path) ----------

template <typename T, typename BinaryOp>
void sg_inclusive_scan(sycl::sub_group sg, T& value, BinaryOp op,
                       bool init_present, T& carry, T* slm) {
    std::uint8_t lid = sg.get_local_linear_id();
    std::uint8_t sg_size = sg.get_max_local_range()[0];
    std::uint8_t broadcast_id = sg_size - 1;

    for (std::uint8_t shift = 1; shift <= sg_size / 2; shift <<= 1) {
        T prev = slm_shift_group_right(sg, value, shift, slm);
        if (lid >= shift)
            value = op(prev, value);
    }
    if (init_present) {
        value = op(carry, value);
        carry = slm_group_broadcast(sg, value, broadcast_id, slm);
    } else {
        carry = slm_group_broadcast(sg, value, broadcast_id, slm);
    }
}

template <typename T, typename BinaryOp>
void sg_inclusive_scan_partial(sycl::sub_group sg, T& value, BinaryOp op,
                               bool init_present, T& carry, std::uint32_t count, T* slm) {
    std::uint8_t lid = sg.get_local_linear_id();
    std::uint8_t sg_size = sg.get_max_local_range()[0];
    std::uint8_t broadcast_id = static_cast<std::uint8_t>(count - 1);

    for (std::uint8_t shift = 1; shift <= sg_size / 2; shift <<= 1) {
        T prev = slm_shift_group_right(sg, value, shift, slm);
        if (lid >= shift && lid < count)
            value = op(prev, value);
    }
    if (init_present) {
        value = op(carry, value);
        carry = slm_group_broadcast(sg, value, broadcast_id, slm);
    } else {
        carry = slm_group_broadcast(sg, value, broadcast_id, slm);
    }
}

// ---------- scan_through_elements: reduce phase (no output) ----------

template <typename T, typename BinaryOp>
void scan_through_reduce(sycl::sub_group sg, BinaryOp op, T& carry, bool& carry_initialized,
                         const T* input, std::size_t start_id, std::size_t n,
                         std::uint32_t iters_per_item, std::size_t subgroup_start_id,
                         std::uint32_t sg_id, std::uint32_t active_subgroups, T* slm) {
    std::uint8_t sg_size = sg.get_max_local_range()[0];
    bool is_full_thread = (subgroup_start_id + iters_per_item * sg_size <= n);

    if (is_full_thread) {
        T v = input[start_id];
        sg_inclusive_scan(sg, v, op, false, carry, slm);
        carry_initialized = true;

        for (std::uint32_t j = 1; j < iters_per_item; j++) {
            v = input[start_id + j * sg_size];
            sg_inclusive_scan(sg, v, op, true, carry, slm);
        }
    } else {
        if (sg_id < active_subgroups) {
            std::uint32_t iters = ceiling_div(static_cast<std::uint32_t>(n - subgroup_start_id),
                                              (std::uint32_t)sg_size);
            if (iters == 1) {
                std::size_t local_id = (start_id < n) ? start_id : n - 1;
                T v = input[local_id];
                sg_inclusive_scan_partial(sg, v, op, false, carry,
                                         static_cast<std::uint32_t>(n - subgroup_start_id), slm);
                carry_initialized = true;
            } else {
                T v = input[start_id];
                sg_inclusive_scan(sg, v, op, false, carry, slm);
                carry_initialized = true;

                for (std::uint32_t j = 1; j < iters - 1; j++) {
                    v = input[start_id + j * sg_size];
                    sg_inclusive_scan(sg, v, op, true, carry, slm);
                }
                std::size_t offset = start_id + (iters - 1) * sg_size;
                std::size_t local_id = (offset < n) ? offset : n - 1;
                v = input[local_id];
                sg_inclusive_scan_partial(sg, v, op, true, carry,
                    static_cast<std::uint32_t>(n - (subgroup_start_id + (iters - 1) * sg_size)), slm);
            }
        }
    }
}

// ---------- scan_through_elements: scan phase (writes output, inclusive) ----------

template <typename T, typename BinaryOp>
void scan_through_scan(sycl::sub_group sg, BinaryOp op, T& carry, bool init_present,
                       const T* input, T* output, std::size_t start_id, std::size_t n,
                       std::uint32_t iters_per_item, std::size_t subgroup_start_id,
                       std::uint32_t sg_id, std::uint32_t active_subgroups, T* slm) {
    std::uint8_t sg_size = sg.get_max_local_range()[0];
    bool is_full_thread = (subgroup_start_id + iters_per_item * sg_size <= n);

    if (is_full_thread) {
        T v = input[start_id];
        sg_inclusive_scan(sg, v, op, init_present, carry, slm);
        output[start_id] = v;

        for (std::uint32_t j = 1; j < iters_per_item; j++) {
            v = input[start_id + j * sg_size];
            sg_inclusive_scan(sg, v, op, true, carry, slm);
            output[start_id + j * sg_size] = v;
        }
    } else {
        if (sg_id < active_subgroups) {
            std::uint32_t iters = ceiling_div(static_cast<std::uint32_t>(n - subgroup_start_id),
                                              (std::uint32_t)sg_size);
            if (iters == 1) {
                std::size_t local_id = (start_id < n) ? start_id : n - 1;
                T v = input[local_id];
                sg_inclusive_scan_partial(sg, v, op, init_present, carry,
                                         static_cast<std::uint32_t>(n - subgroup_start_id), slm);
                if (start_id < n)
                    output[start_id] = v;
            } else {
                T v = input[start_id];
                sg_inclusive_scan(sg, v, op, init_present, carry, slm);
                output[start_id] = v;

                for (std::uint32_t j = 1; j < iters - 1; j++) {
                    std::size_t local_id = start_id + j * sg_size;
                    v = input[local_id];
                    sg_inclusive_scan(sg, v, op, true, carry, slm);
                    output[local_id] = v;
                }
                std::size_t offset = start_id + (iters - 1) * sg_size;
                std::size_t local_id = (offset < n) ? offset : n - 1;
                v = input[local_id];
                sg_inclusive_scan_partial(sg, v, op, true, carry,
                    static_cast<std::uint32_t>(n - (subgroup_start_id + (iters - 1) * sg_size)), slm);
                if (offset < n)
                    output[offset] = v;
            }
        }
    }
}

// ---------- kernel names ----------
class ReduceKernel;
class ScanKernel;

int run_rts_reproducer() {
    using T = std::int32_t;
    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));

    std::vector<T> h_expected(N);
    std::inclusive_scan(h_input.begin(), h_input.end(), h_expected.begin());

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();

    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::uint8_t min_sg_size = *std::min_element(sg_sizes.begin(), sg_sizes.end());
    std::uint8_t max_sg_size = *std::max_element(sg_sizes.begin(), sg_sizes.end());

    constexpr std::uint16_t max_inputs_per_item = std::max(std::uint16_t{1},
                                                           std::uint16_t{512 / sizeof(T)});
    std::uint32_t wg_size_cap = dev.is_gpu() ? 1024 : 256;
    std::uint32_t max_wg_size = std::min(
        static_cast<std::uint32_t>(dev.get_info<sycl::info::device::max_work_group_size>()),
        wg_size_cap);
    std::uint32_t work_group_size = (max_wg_size / max_sg_size) * max_sg_size;

    std::uint32_t num_work_groups = bit_ceil_u32(
        dev.get_info<sycl::info::device::max_compute_units>());

    std::uint32_t max_num_sg_local = work_group_size / min_sg_size;
    std::uint32_t max_num_sg_global = max_num_sg_local * num_work_groups;
    std::uint32_t max_inputs_per_wg = work_group_size * max_inputs_per_item;
    std::uint32_t max_inputs_per_block = max_inputs_per_wg * num_work_groups;

    std::size_t inputs_remaining = N;
    std::uint32_t inputs_per_item =
        inputs_remaining >= max_inputs_per_block
            ? max_inputs_per_item
            : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(inputs_remaining)),
                          num_work_groups * work_group_size);

    std::size_t block_size = std::min(inputs_remaining, (std::size_t)max_inputs_per_block);
    std::size_t num_blocks = inputs_remaining / block_size + (inputs_remaining % block_size != 0);

    std::uint32_t scratch_size = max_num_sg_global + 2;

    std::printf("n=%zu, num_work_groups=%u, work_group_size=%u\n", N, num_work_groups, work_group_size);
    std::printf("min_sg=%u, max_sg=%u, max_num_sg_local=%u, max_num_sg_global=%u\n",
                (unsigned)min_sg_size, (unsigned)max_sg_size, max_num_sg_local, max_num_sg_global);
    std::printf("max_inputs_per_item=%u, inputs_per_item=%u, block_size=%zu, num_blocks=%zu\n",
                (unsigned)max_inputs_per_item, inputs_per_item, block_size, num_blocks);
    std::printf("scratch_size=%u\n", scratch_size);

    T* d_input = sycl::malloc_device<T>(N, q);
    T* d_output = sycl::malloc_device<T>(N, q);
    T* d_scratch = sycl::malloc_device<T>(scratch_size, q);
    T* d_result = sycl::malloc_device<T>(1, q);

    q.memcpy(d_input, h_input.data(), N * sizeof(T));
    q.memset(d_output, 0, N * sizeof(T));
    q.memset(d_scratch, 0, scratch_size * sizeof(T));
    q.wait();

    sycl::event prior_event;
    std::size_t ir = inputs_remaining;

    for (std::size_t b = 0; b < num_blocks; b++) {
        std::uint32_t workitems_in_block = ceiling_div(
            std::min(ir, (std::size_t)max_inputs_per_block), inputs_per_item);
        std::uint32_t workitems_round_up = ceiling_div(workitems_in_block, work_group_size) * work_group_size;

        auto nd_range = sycl::nd_range<1>(sycl::range<1>(workitems_round_up),
                                          sycl::range<1>(work_group_size));

        std::printf("Block %zu: workitems=%u global_range=%u inputs_remaining=%zu\n",
                    b, workitems_in_block, workitems_round_up, ir);

        // ============ REDUCE KERNEL ============
        auto reduce_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior_event);
            sycl::local_accessor<T, 1> sg_partials(max_num_sg_local, cgh);
            sycl::local_accessor<T, 1> comm_slm(work_group_size, cgh);

            auto cap_n = N;
            auto cap_max_num_wg = num_work_groups;
            auto cap_wg_size = work_group_size;
            auto cap_max_block_size = max_inputs_per_block;
            auto cap_ir = ir;
            auto cap_b = b;

            cgh.parallel_for<ReduceKernel>(nd_range, [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint8_t sg_size = sg.get_max_local_range()[0];
                std::uint8_t sg_lid = sg.get_local_linear_id();
                std::uint32_t sg_id = sg.get_group_linear_id();
                std::uint32_t group_id = ndi.get_group(0);

                std::uint32_t num_sg_local = cap_wg_size / sg_size;
                std::uint32_t num_sg_global = num_sg_local * cap_max_num_wg;
                std::uint32_t max_inputs_per_sg = cap_max_block_size / num_sg_global;
                std::size_t bc = bit_ceil_u32(static_cast<std::uint32_t>(cap_ir)) / num_sg_global;
                std::uint32_t evenly_divided = static_cast<std::uint32_t>(bc > sg_size ? bc : sg_size);
                std::uint32_t inputs_per_sg = (cap_ir >= cap_max_block_size) ? max_inputs_per_sg : evenly_divided;
                std::uint32_t ipi = inputs_per_sg / sg_size;

                T* slm = &comm_slm[0];

                std::size_t group_start = (cap_b * cap_max_block_size) +
                    (group_id * inputs_per_sg * num_sg_local);
                std::size_t max_in_group = (std::size_t)inputs_per_sg * num_sg_local;
                std::uint32_t inputs_in_group = static_cast<std::uint32_t>(
                    std::min(cap_n - group_start, max_in_group));
                std::uint32_t active_sg = ceiling_div(inputs_in_group, inputs_per_sg);
                std::size_t sg_start = group_start + (sg_id * inputs_per_sg);
                std::size_t start_id = sg_start + sg_lid;

                T carry{};
                bool carry_init = false;

                if (sg_id < active_sg) {
                    scan_through_reduce(sg, std::plus<T>{}, carry, carry_init,
                                        d_input, start_id, cap_n, ipi, sg_start, sg_id, active_sg, slm);
                    if (sg_lid == 0)
                        sg_partials[sg_id] = carry;
                }
                sycl::group_barrier(ndi.get_group());

                if (sg_id == 0) {
                    std::size_t scratch_start = group_id * num_sg_local;
                    std::uint8_t iters = static_cast<std::uint8_t>(ceiling_div(active_sg, (std::uint32_t)sg_size));

                    T wg_carry{};
                    if (iters == 1) {
                        std::uint32_t load_id = std::min((std::uint32_t)sg_lid, active_sg - 1);
                        T v = sg_partials[load_id];
                        sg_inclusive_scan_partial(sg, v, std::plus<T>{}, false, wg_carry, active_sg, slm);
                        if (sg_lid < active_sg)
                            d_scratch[scratch_start + sg_lid] = v;
                    } else {
                        std::uint32_t rid = sg_lid;
                        T v = sg_partials[rid];
                        sg_inclusive_scan(sg, v, std::plus<T>{}, false, wg_carry, slm);
                        d_scratch[scratch_start + rid] = v;
                        rid += sg_size;

                        for (std::uint32_t i = 1; i < (std::uint32_t)iters - 1; i++) {
                            v = sg_partials[rid];
                            sg_inclusive_scan(sg, v, std::plus<T>{}, true, wg_carry, slm);
                            d_scratch[scratch_start + rid] = v;
                            rid += sg_size;
                        }
                        std::uint32_t load_id = std::min(rid, num_sg_local - 1);
                        v = sg_partials[load_id];
                        sg_inclusive_scan_partial(sg, v, std::plus<T>{}, true, wg_carry,
                                                 active_sg - ((iters - 1) * sg_size), slm);
                        if (rid < num_sg_local)
                            d_scratch[scratch_start + rid] = v;
                    }
                }
            });
        });
        prior_event = reduce_ev;

        // ============ SCAN KERNEL ============
        std::size_t num_remaining = N - b * max_inputs_per_block;
        std::uint32_t inputs_in_block = static_cast<std::uint32_t>(
            std::min(num_remaining, (std::size_t)max_inputs_per_block));

        auto scan_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior_event);
            sycl::local_accessor<T, 1> sg_partials(max_num_sg_local + 1, cgh);
            sycl::local_accessor<T, 1> comm_slm(work_group_size, cgh);

            auto cap_n = N;
            auto cap_max_num_wg = num_work_groups;
            auto cap_wg_size = work_group_size;
            auto cap_max_block_size = max_inputs_per_block;
            auto cap_max_num_sg_global = max_num_sg_global;
            auto cap_ir = ir;
            auto cap_b = b;
            auto cap_inputs_in_block = inputs_in_block;
            auto cap_num_blocks = num_blocks;

            cgh.parallel_for<ScanKernel>(nd_range, [=](sycl::nd_item<1> ndi) {
                auto sg = ndi.get_sub_group();
                std::uint8_t sg_size = sg.get_max_local_range()[0];
                std::uint8_t sg_lid = sg.get_local_linear_id();
                std::uint32_t sg_id = sg.get_group_linear_id();
                std::uint32_t group_id = ndi.get_group(0);

                std::uint32_t num_sg_local = cap_wg_size / sg_size;
                std::uint32_t num_sg_global = num_sg_local * cap_max_num_wg;
                std::uint32_t max_inputs_per_sg = cap_max_block_size / num_sg_global;
                std::size_t bc = bit_ceil_u32(static_cast<std::uint32_t>(cap_ir)) / num_sg_global;
                std::uint32_t evenly_divided = static_cast<std::uint32_t>(bc > sg_size ? bc : sg_size);
                std::uint32_t inputs_per_sg = (cap_ir >= cap_max_block_size) ? max_inputs_per_sg : evenly_divided;
                std::uint32_t ipi = inputs_per_sg / sg_size;

                T* slm = &comm_slm[0];

                std::uint32_t active_groups = ceiling_div(cap_inputs_in_block, inputs_per_sg * num_sg_local);

                std::size_t group_start = (cap_b * cap_max_block_size) +
                    (group_id * inputs_per_sg * num_sg_local);
                std::size_t max_in_group = (std::size_t)inputs_per_sg * num_sg_local;
                std::uint32_t inputs_in_group = static_cast<std::uint32_t>(
                    std::min(cap_n - group_start, max_in_group));
                std::uint32_t active_sg = ceiling_div(inputs_in_group, inputs_per_sg);

                T carry_last{};
                T sub_group_carry{};
                bool sg_carry_initialized = true;

                if (sg_id == 0) {
                    std::uint8_t iters = static_cast<std::uint8_t>(ceiling_div(active_sg, (std::uint32_t)sg_size));
                    std::size_t sg_before = group_id * num_sg_local;
                    std::uint32_t load_id = sg_lid;

                    for (std::uint8_t i = 0; i < iters - 1; i++) {
                        sg_partials[load_id] = d_scratch[sg_before + load_id];
                        load_id += sg_size;
                    }
                    if (load_id < active_sg)
                        sg_partials[load_id] = d_scratch[sg_before + load_id];

                    std::uint32_t offset = num_sg_local - 1;
                    if (group_id > 0) {
                        std::size_t elements_to_process = sg_before / num_sg_local;
                        std::size_t pre_carry_iters = ceiling_div(
                            static_cast<std::uint32_t>(elements_to_process), (std::uint32_t)sg_size);

                        if (pre_carry_iters == 1) {
                            std::size_t proposed = num_sg_local * sg_lid + offset;
                            std::size_t reduction_id = (proposed < sg_before) ? proposed : sg_before - 1;
                            T value = d_scratch[reduction_id];
                            sg_inclusive_scan_partial(sg, value, std::plus<T>{}, false, carry_last,
                                                     static_cast<std::uint32_t>(elements_to_process), slm);
                        } else {
                            std::uint32_t rid = num_sg_local * sg_lid + offset;
                            std::uint32_t rid_inc = num_sg_local * sg_size;

                            T value = d_scratch[rid];
                            sg_inclusive_scan(sg, value, std::plus<T>{}, false, carry_last, slm);
                            rid += rid_inc;

                            for (std::uint32_t i = 1; i < (std::uint32_t)pre_carry_iters - 1; i++) {
                                value = d_scratch[rid];
                                sg_inclusive_scan(sg, value, std::plus<T>{}, true, carry_last, slm);
                                rid += rid_inc;
                            }
                            std::size_t remaining = elements_to_process - ((pre_carry_iters - 1) * sg_size);
                            std::size_t final_rid = std::min((std::size_t)rid, sg_before - 1);
                            value = d_scratch[final_rid];
                            sg_inclusive_scan_partial(sg, value, std::plus<T>{}, true, carry_last,
                                                     static_cast<std::uint32_t>(remaining), slm);
                        }

                        std::size_t carry_off = sg_lid;
                        std::uint8_t ci = 0;
                        for (; ci < iters - 1; ci++) {
                            sg_partials[carry_off] = std::plus<T>{}(carry_last, sg_partials[carry_off]);
                            carry_off += sg_size;
                        }
                        if (ci * sg_size + sg_lid < active_sg) {
                            sg_partials[carry_off] = std::plus<T>{}(carry_last, sg_partials[carry_off]);
                        }
                        if (sg_lid == 0) {
                            sg_partials[active_sg] = carry_last;
                        }
                    }
                }

                sycl::group_barrier(ndi.get_group());

                if (cap_b == 0) {
                    if (sg_id > 0) {
                        sub_group_carry = sg_partials[std::min(sg_id - 1, active_sg - 1)];
                    } else if (group_id > 0) {
                        sub_group_carry = sg_partials[active_sg];
                    } else {
                        sg_carry_initialized = false;
                    }
                } else {
                    T block_carry_in = d_scratch[num_sg_global + (cap_b % 2)];
                    if (sg_id > 0) {
                        sub_group_carry = std::plus<T>{}(
                            block_carry_in, sg_partials[std::min(sg_id - 1, active_sg - 1)]);
                    } else if (group_id > 0) {
                        sub_group_carry = std::plus<T>{}(block_carry_in, sg_partials[active_sg]);
                    } else {
                        sub_group_carry = block_carry_in;
                    }
                }

                std::size_t sg_start = group_start + (sg_id * inputs_per_sg);
                std::size_t start_id = sg_start + sg_lid;

                scan_through_scan(sg, std::plus<T>{}, sub_group_carry, sg_carry_initialized,
                                  d_input, d_output, start_id, cap_n, ipi, sg_start,
                                  sg_id, active_sg, slm);

                if (sg_lid == 0 && (active_groups == group_id + 1) && (active_sg == sg_id + 1)) {
                    if (cap_b + 1 == cap_num_blocks) {
                        d_result[0] = sub_group_carry;
                    } else {
                        d_scratch[num_sg_global + 1 - (cap_b % 2)] = sub_group_carry;
                    }
                }
            });
        });
        prior_event = scan_ev;

        ir -= std::min(ir, block_size);
        if (b + 2 == num_blocks) {
            inputs_per_item = ir >= max_inputs_per_block
                ? max_inputs_per_item
                : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(ir)),
                              num_work_groups * work_group_size);
        }
    }

    q.wait();

    std::vector<T> h_output(N);
    q.memcpy(h_output.data(), d_output, N * sizeof(T)).wait();

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_output[i] != h_expected[i]) {
            if (errors < 20) {
                std::printf("MISMATCH at [%zu]: got %d, expected %d\n", i, h_output[i], h_expected[i]);
            }
            errors++;
        }
    }

    if (errors == 0)
        std::printf("PASS: all %zu elements match.\n", N);
    else
        std::printf("FAIL: %d mismatches out of %zu elements.\n", errors, N);

    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_scratch, q);
    sycl::free(d_result, q);

    return errors ? 1 : 0;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    int result = run_rts_reproducer();
    if (result != 0)
        return result;
#endif
    return TestUtils::done();
}
