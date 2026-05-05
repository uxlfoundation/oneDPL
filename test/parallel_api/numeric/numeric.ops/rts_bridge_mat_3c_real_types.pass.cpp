// -*- C++ -*-
//===-- rts_bridge_mat_3c_real_types.pass.cpp --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 3c (Matrix2x2): Same as 3b (buffer/accessor + submitter structs)
// but uses the real oneDPL types: __gen_transform_input, __simple_write_to_id,
// __no_init_value, and all_view ranges. This matches bridge 4b's type
// complexity in the captured struct while still doing manual kernel submission.
// If 3b passes but 3c fails, the issue is the type layout of these members.

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
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

using TestUtils::Matrix2x2;
using TestUtils::multiply_matrix;

template <typename T>
using lazy = oneapi::dpl::__internal::__lazy_ctor_storage<T>;

template <typename T, sycl::access::mode M = sycl::access::mode::read_write>
using all_view = oneapi::dpl::__ranges::all_view<T, M>;

// ---------- helpers ----------

inline std::uint32_t bit_ceil_u32(std::uint32_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
inline std::uint32_t ceiling_div(std::uint32_t a, std::uint32_t b) { return (a + b - 1) / b; }
inline std::uint32_t ceiling_div(std::size_t a, std::uint32_t b) { return static_cast<std::uint32_t>((a + b - 1) / b); }

// ---------- kernel names ----------
class ReduceK3cMat;
class ScanK3cMat;

// ---------- Submitter-style struct for the reduce kernel ----------
// Uses real oneDPL types: _GenInput = __gen_transform_input, _InitType = __no_init_value
template <typename T, typename _BinaryOp, typename _UnaryOp>
struct reduce_submitter_3c
{
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    static constexpr bool __use_subgroup_ops = false;
    static constexpr bool __is_inclusive = true;
    static constexpr std::uint16_t max_ipi = std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::size_t __n;
    const bool __use_slm_for_comm;

    const _GenInput __gen_reduce_input;
    const _BinaryOp __reduce_op;
    _InitType __init;

    template <typename _InRng, typename ScratchAcc>
    void
    submit_body(sycl::handler& cgh, const sycl::nd_range<1> __nd_range,
                _InRng __in_rng, ScratchAcc __scratch_acc,
                const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        sycl::local_accessor<T, 1> __sub_group_partials(__max_num_sub_groups_local, cgh);
        sycl::local_accessor<T, 1> __comm_slm(__work_group_size, cgh);
        auto cn = __n; auto cwg = __max_num_work_groups; auto cws = __work_group_size;
        auto cmb = __max_block_size; auto cir = __inputs_remaining; auto cb = __block_num;

        cgh.parallel_for<ReduceK3cMat>(__nd_range, [=, *this](sycl::nd_item<1> ndi) {
            auto sg = ndi.get_sub_group();
            std::uint8_t ssz = sg.get_max_local_range()[0];
            std::uint8_t slid = sg.get_local_linear_id();
            std::uint32_t sid = sg.get_group_linear_id();
            std::uint32_t gid = ndi.get_group(0);
            std::uint32_t nsl = cws / ssz;
            std::uint32_t nsg = nsl * cwg;
            std::uint32_t mips = cmb / nsg;
            std::size_t bc = bit_ceil_u32(static_cast<std::uint32_t>(cir)) / nsg;
            std::uint32_t ed = static_cast<std::uint32_t>(bc > ssz ? bc : ssz);
            std::uint32_t ips = (cir >= cmb) ? mips : ed;
            std::uint32_t lip = ips / ssz;
            T* sl = &__comm_slm[0];
            std::size_t gs = (cb * cmb) + (gid * ips * nsl);
            std::size_t mig = (std::size_t)ips * nsl;
            std::uint32_t iig = static_cast<std::uint32_t>(std::min(cn - gs, mig));
            std::uint32_t asg = ceiling_div(iig, ips);
            std::size_t sgs = gs + (sid * ips);
            std::size_t stid = sgs + slid;

            lazy<T> carry;
            if (sid < asg) {
                rts::__scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                    /*__init_present=*/false, /*__capture_output=*/false, max_ipi>(
                    sg, __gen_reduce_input, _ScanInputTransform{}, __reduce_op, nullptr, carry,
                    __in_rng, /*unused*/ __in_rng, stid, cn, lip, sgs, sid, asg, sl);
                if (slid == 0) __sub_group_partials[sid] = carry.__v;
                carry.__destroy();
            }
            sycl::group_barrier(ndi.get_group());

            if (sid == 0) {
                std::size_t ss = gid * nsl;
                std::uint8_t it = static_cast<std::uint8_t>(ceiling_div(asg, (std::uint32_t)ssz));
                lazy<T> wc;
                if (it == 1) {
                    std::uint32_t li = std::min((std::uint32_t)slid, asg - 1);
                    T v = __sub_group_partials[li];
                    rts::__sub_group_scan_partial<__use_subgroup_ops, true, false>(
                        sg, v, __reduce_op, wc, asg, sl);
                    if (slid < asg) __scratch_acc[ss + slid] = v;
                } else {
                    std::uint32_t ri = slid;
                    T v = __sub_group_partials[ri];
                    rts::__sub_group_scan<__use_subgroup_ops, true, false>(
                        sg, v, __reduce_op, wc, sl);
                    __scratch_acc[ss + ri] = v;
                    ri += ssz;
                    for (std::uint32_t i = 1; i < (std::uint32_t)it - 1; i++) {
                        v = __sub_group_partials[ri];
                        rts::__sub_group_scan<__use_subgroup_ops, true, true>(
                            sg, v, __reduce_op, wc, sl);
                        __scratch_acc[ss + ri] = v;
                        ri += ssz;
                    }
                    std::uint32_t li = std::min(ri, nsl - 1);
                    v = __sub_group_partials[li];
                    rts::__sub_group_scan_partial<__use_subgroup_ops, true, true>(
                        sg, v, __reduce_op, wc, asg - ((it - 1) * ssz), sl);
                    if (ri < nsl) __scratch_acc[ss + ri] = v;
                }
                wc.__destroy();
            }
        });
    }
};

// ---------- Submitter-style struct for the scan kernel ----------
template <typename T, typename _BinaryOp, typename _UnaryOp>
struct scan_submitter_3c
{
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
    using _WriteOp = rts::__simple_write_to_id;
    using _ScanInputTransform = oneapi::dpl::identity;
    static constexpr bool __use_subgroup_ops = false;
    static constexpr bool __is_inclusive = true;
    static constexpr std::uint16_t max_ipi = std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});

    const std::uint32_t __max_num_work_groups;
    const std::uint32_t __work_group_size;
    const std::uint32_t __max_block_size;
    const std::uint32_t __max_num_sub_groups_local;
    const std::uint32_t __max_num_sub_groups_global;
    const std::size_t __num_blocks;
    const std::size_t __n;
    const bool __use_slm_for_comm;

    const _GenInput __gen_scan_input;
    const _ScanInputTransform __scan_input_transform;
    const _BinaryOp __reduce_op;
    const _WriteOp __write_op;
    _InitType __init;

    template <typename _InRng, typename _OutRng, typename ScratchAcc, typename ResAcc>
    void
    submit_body(sycl::handler& cgh, const sycl::nd_range<1> __nd_range,
                _InRng __in_rng, _OutRng __out_rng, ScratchAcc __scratch_acc, ResAcc __res_acc,
                const std::size_t __inputs_remaining, const std::size_t __block_num) const
    {
        std::size_t nr = __n - __block_num * __max_block_size;
        std::uint32_t iib = static_cast<std::uint32_t>(std::min(nr, (std::size_t)__max_block_size));

        sycl::local_accessor<T, 1> __sub_group_partials(__max_num_sub_groups_local + 1, cgh);
        sycl::local_accessor<T, 1> __comm_slm(__work_group_size, cgh);
        auto cn = __n; auto cwg = __max_num_work_groups; auto cws = __work_group_size;
        auto cmb = __max_block_size; auto cmsg = __max_num_sub_groups_global;
        auto cir = __inputs_remaining; auto cb = __block_num; auto ciib = iib;
        auto cnb = __num_blocks;

        cgh.parallel_for<ScanK3cMat>(__nd_range, [=, *this](sycl::nd_item<1> ndi) {
            auto sg = ndi.get_sub_group();
            std::uint8_t ssz = sg.get_max_local_range()[0];
            std::uint8_t slid = sg.get_local_linear_id();
            std::uint32_t sid = sg.get_group_linear_id();
            std::uint32_t gid = ndi.get_group(0);
            std::uint32_t nsl = cws / ssz;
            std::uint32_t nsg = nsl * cwg;
            std::uint32_t mips = cmb / nsg;
            std::size_t bc = bit_ceil_u32(static_cast<std::uint32_t>(cir)) / nsg;
            std::uint32_t ed = static_cast<std::uint32_t>(bc > ssz ? bc : ssz);
            std::uint32_t ips = (cir >= cmb) ? mips : ed;
            std::uint32_t lip = ips / ssz;
            T* sl = &__comm_slm[0];
            std::uint32_t ag = ceiling_div(ciib, ips * nsl);
            std::size_t gs = (cb * cmb) + (gid * ips * nsl);
            std::size_t mig = (std::size_t)ips * nsl;
            std::uint32_t iig = static_cast<std::uint32_t>(std::min(cn - gs, mig));
            std::uint32_t asg = ceiling_div(iig, ips);

            lazy<T> carry_last;
            lazy<T> sgc;
            bool sgci = true;

            // Step 1: load sub-group partials from scratch buffer
            if (sid == 0) {
                std::uint8_t it = static_cast<std::uint8_t>(ceiling_div(asg, (std::uint32_t)ssz));
                std::size_t sbm = gid * nsl;
                std::uint32_t lri = slid;
                for (std::uint8_t i = 0; i < it - 1; i++) {
                    __sub_group_partials[lri] = __scratch_acc[sbm + lri]; lri += ssz;
                }
                if (lri < asg) __sub_group_partials[lri] = __scratch_acc[sbm + lri];
                std::uint32_t off = nsl - 1;
                if (gid > 0) {
                    std::size_t etp = sbm / nsl;
                    std::size_t pci = ceiling_div(static_cast<std::uint32_t>(etp), (std::uint32_t)ssz);
                    if (pci == 1) {
                        std::size_t pid = nsl * slid + off;
                        std::size_t rid = (pid < sbm) ? pid : sbm - 1;
                        T val = __scratch_acc[rid];
                        rts::__sub_group_scan_partial<__use_subgroup_ops, true, false>(
                            sg, val, __reduce_op, carry_last, static_cast<std::uint32_t>(etp), sl);
                    } else {
                        std::uint32_t rid = nsl * slid + off;
                        std::uint32_t rinc = nsl * ssz;
                        T val = __scratch_acc[rid];
                        rts::__sub_group_scan<__use_subgroup_ops, true, false>(
                            sg, val, __reduce_op, carry_last, sl);
                        rid += rinc;
                        for (std::uint32_t i = 1; i < (std::uint32_t)pci - 1; i++) {
                            val = __scratch_acc[rid];
                            rts::__sub_group_scan<__use_subgroup_ops, true, true>(
                                sg, val, __reduce_op, carry_last, sl);
                            rid += rinc;
                        }
                        std::size_t rem = etp - ((pci - 1) * ssz);
                        std::size_t frid = std::min((std::size_t)rid, sbm - 1);
                        val = __scratch_acc[frid];
                        rts::__sub_group_scan_partial<__use_subgroup_ops, true, true>(
                            sg, val, __reduce_op, carry_last, static_cast<std::uint32_t>(rem), sl);
                    }
                    std::size_t co = slid;
                    std::uint8_t ci = 0;
                    for (; ci < it - 1; ci++) {
                        __sub_group_partials[co] = __reduce_op(carry_last.__v, __sub_group_partials[co]); co += ssz;
                    }
                    if (ci * ssz + slid < asg)
                        __sub_group_partials[co] = __reduce_op(carry_last.__v, __sub_group_partials[co]);
                    if (slid == 0) __sub_group_partials[asg] = carry_last.__v;
                    carry_last.__destroy();
                }
            }
            sycl::group_barrier(ndi.get_group());

            // Step 2: compute sub-group carry
            if (cb == 0) {
                if (sid > 0) sgc.__setup(__sub_group_partials[std::min(sid - 1, asg - 1)]);
                else if (gid > 0) sgc.__setup(__sub_group_partials[asg]);
                else sgci = false;
            } else {
                T bci = __scratch_acc[cmsg + (cb % 2)];
                if (sid > 0) sgc.__setup(__reduce_op(bci, __sub_group_partials[std::min(sid - 1, asg - 1)]));
                else if (gid > 0) sgc.__setup(__reduce_op(bci, __sub_group_partials[asg]));
                else sgc.__setup(bci);
            }

            // Step 3: scan through elements using real oneDPL types
            std::size_t sgs = gs + (sid * ips);
            std::size_t stid = sgs + slid;

            if (sgci) {
                rts::__scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                    /*__init_present=*/true, /*__capture_output=*/true, max_ipi>(
                    sg, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op, sgc,
                    __in_rng, __out_rng, stid, cn, lip, sgs, sid, asg, sl);
            } else {
                rts::__scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                    /*__init_present=*/false, /*__capture_output=*/true, max_ipi>(
                    sg, __gen_scan_input, __scan_input_transform, __reduce_op, __write_op, sgc,
                    __in_rng, __out_rng, stid, cn, lip, sgs, sid, asg, sl);
            }

            if (slid == 0 && (ag == gid + 1) && (asg == sid + 1)) {
                if (cb + 1 == cnb) __res_acc[0] = sgc.__v;
                else __scratch_acc[cmsg + 1 - (cb % 2)] = sgc.__v;
            }
            sgc.__destroy();
        });
    }
};

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using _UnaryOp = oneapi::dpl::identity;
    using _BinaryOp = multiply_matrix<std::int32_t>;
    using _InitType = oneapi::dpl::unseq_backend::__no_init_value<T>;
    using _GenInput = rts::__gen_transform_input<_UnaryOp, T>;
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
    std::printf("[bridge3c_mat] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::uint8_t min_sg = *std::min_element(sg_sizes.begin(), sg_sizes.end());
    std::uint8_t max_sg = *std::max_element(sg_sizes.begin(), sg_sizes.end());

    std::uint32_t wg_cap = dev.is_gpu() ? 1024 : 256;
    std::uint32_t max_wg = std::min(static_cast<std::uint32_t>(dev.get_info<sycl::info::device::max_work_group_size>()), wg_cap);
    std::uint32_t wg_size = (max_wg / max_sg) * max_sg;
    std::uint32_t num_wg = bit_ceil_u32(dev.get_info<sycl::info::device::max_compute_units>());
    std::uint32_t max_sg_local = wg_size / min_sg;
    std::uint32_t max_sg_global = max_sg_local * num_wg;
    constexpr std::uint16_t max_ipi = std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});
    std::uint32_t max_per_block = wg_size * max_ipi * num_wg;

    std::size_t remaining = N;
    std::uint32_t ipi = remaining >= max_per_block ? max_ipi
        : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(remaining)), num_wg * wg_size);
    std::size_t block_size = std::min(remaining, (std::size_t)max_per_block);
    std::size_t num_blocks = remaining / block_size + (remaining % block_size != 0);
    std::uint32_t scratch_size = max_sg_global + 2;

    std::printf("[bridge3c_mat] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u\n",
                wg_size, num_wg, ipi, num_blocks, scratch_size);

    // Use SYCL buffers with all_view (like bridge 4b)
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};
    sycl::buffer<T, 1> buf_scratch{sycl::range<1>(scratch_size)};
    sycl::buffer<T, 1> buf_result{sycl::range<1>(1)};

    _GenInput gen_input{_UnaryOp{}};
    _BinaryOp binary_op{};
    _WriteOp write_op{};
    _InitType init{};

    reduce_submitter_3c<T, _BinaryOp, _UnaryOp> reduce_sub{
        num_wg, wg_size, max_per_block, max_sg_local, N, true,
        gen_input, binary_op, init
    };
    scan_submitter_3c<T, _BinaryOp, _UnaryOp> scan_sub{
        num_wg, wg_size, max_per_block, max_sg_local, max_sg_global,
        num_blocks, N, true,
        gen_input, oneapi::dpl::identity{}, binary_op, write_op, init
    };

    std::size_t ir = remaining;

    for (std::size_t b = 0; b < num_blocks; b++) {
        std::uint32_t wi = ceiling_div(std::min(ir, (std::size_t)max_per_block), ipi);
        std::uint32_t wi_up = ceiling_div(wi, wg_size) * wg_size;
        auto ndr = sycl::nd_range<1>(sycl::range<1>(wi_up), sycl::range<1>(wg_size));

        // Reduce kernel
        q.submit([&](sycl::handler& cgh) {
            auto in_rng = all_view<T, sycl::access::mode::read>(buf_in, 0, N);
            auto scratch_acc = buf_scratch.template get_access<sycl::access::mode::read_write>(cgh);
            oneapi::dpl::__ranges::__require_access(cgh, in_rng);
            reduce_sub.submit_body(cgh, ndr, in_rng, scratch_acc, ir, b);
        });

        // Scan kernel
        q.submit([&](sycl::handler& cgh) {
            auto in_rng = all_view<T, sycl::access::mode::read>(buf_in, 0, N);
            auto out_rng = all_view<T, sycl::access::mode::read_write>(buf_out, 0, N);
            auto scratch_acc = buf_scratch.template get_access<sycl::access::mode::read_write>(cgh);
            auto res_acc = buf_result.template get_access<sycl::access::mode::write>(cgh);
            oneapi::dpl::__ranges::__require_access(cgh, in_rng, out_rng);
            scan_sub.submit_body(cgh, ndr, in_rng, out_rng, scratch_acc, res_acc, ir, b);
        });

        ir -= std::min(ir, block_size);
        if (b + 2 == num_blocks) {
            ipi = ir >= max_per_block ? max_ipi
                : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(ir)), num_wg * wg_size);
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
                std::printf("[bridge3c_mat] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge3c_mat] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

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
