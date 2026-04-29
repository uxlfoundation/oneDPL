// -*- C++ -*-
//===-- rts_bridge_4a_buffer_ranges.pass.cpp --------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 4a: Same as bridge 3 (oneDPL scan helpers) but replaces
// usm_range<T> with oneDPL's all_view backed by SYCL buffers/accessors.
// Tests whether the buffer/accessor range weight triggers the crash.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include <oneapi/dpl/pstl/utils.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h>
#include <oneapi/dpl/pstl/hetero/dpcpp/utils_ranges_sycl.h>

namespace rts = oneapi::dpl::__par_backend_hetero;

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

// Gen input for all_view ranges
template <typename T>
struct view_gen_input {
    using TempData = rts::__noop_temp_data;
    template <typename Rng>
    T operator()(const Rng& rng, std::size_t id, TempData&) const {
        return rng[id];
    }
};

// Write op for all_view ranges
struct view_write_op {
    using _TempData = rts::__noop_temp_data;
    template <typename Rng, typename T>
    void operator()(const Rng& rng, std::size_t id, const T& v, const _TempData&) const {
        rng[id] = v;
    }
};

// ---------- kernel names ----------
class ReduceK4a;
class ScanK4a;

int run_test() {
    using T = std::int32_t;
    using _BinaryOp = std::plus<T>;
    using _GenInput = view_gen_input<T>;
    using _ScanInputTransform = oneapi::dpl::identity;
    using _WriteOp = view_write_op;
    constexpr bool __use_subgroup_ops = false;
    constexpr bool __is_inclusive = true;
    constexpr std::uint16_t max_ipi = std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});

    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = static_cast<T>((k % 991 + 1) ^ (k % 997 + 2));
    std::vector<T> h_expected(N);
    std::inclusive_scan(h_input.begin(), h_input.end(), h_expected.begin());

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[bridge4a] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::uint8_t min_sg = *std::min_element(sg_sizes.begin(), sg_sizes.end());
    std::uint8_t max_sg = *std::max_element(sg_sizes.begin(), sg_sizes.end());

    std::uint32_t wg_cap = dev.is_gpu() ? 1024 : 256;
    std::uint32_t max_wg = std::min(static_cast<std::uint32_t>(dev.get_info<sycl::info::device::max_work_group_size>()), wg_cap);
    std::uint32_t wg_size = (max_wg / max_sg) * max_sg;
    std::uint32_t num_wg = bit_ceil_u32(dev.get_info<sycl::info::device::max_compute_units>());
    std::uint32_t max_sg_local = wg_size / min_sg;
    std::uint32_t max_sg_global = max_sg_local * num_wg;
    std::uint32_t max_per_block = wg_size * max_ipi * num_wg;

    std::size_t remaining = N;
    std::uint32_t ipi = remaining >= max_per_block ? max_ipi
        : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(remaining)), num_wg * wg_size);
    std::size_t block_size = std::min(remaining, (std::size_t)max_per_block);
    std::size_t num_blocks = remaining / block_size + (remaining % block_size != 0);
    std::uint32_t scratch_size = max_sg_global + 2;

    std::printf("[bridge4a] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u\n",
                wg_size, num_wg, ipi, num_blocks, scratch_size);

    // Use SYCL buffers instead of USM
    sycl::buffer<T, 1> buf_in(h_input.data(), sycl::range<1>(N));
    sycl::buffer<T, 1> buf_out{sycl::range<1>(N)};
    // Still use USM for scratch/result (these aren't passed through scan helpers)
    T* d_scratch = sycl::malloc_device<T>(scratch_size, q);
    T* d_result = sycl::malloc_device<T>(1, q);
    q.memset(d_scratch, 0, scratch_size * sizeof(T));
    q.wait();

    _GenInput gen_input{};
    _ScanInputTransform scan_xform{};
    _WriteOp write_op{};
    _BinaryOp binary_op{};

    sycl::event prior;
    std::size_t ir = remaining;

    // Create all_view objects from buffers
    all_view<T, sycl::access::mode::read> in_view(buf_in, 0, N);
    all_view<T, sycl::access::mode::read_write> out_view(buf_out, 0, N);

    for (std::size_t b = 0; b < num_blocks; b++) {
        std::uint32_t wi = ceiling_div(std::min(ir, (std::size_t)max_per_block), ipi);
        std::uint32_t wi_up = ceiling_div(wi, wg_size) * wg_size;
        auto ndr = sycl::nd_range<1>(sycl::range<1>(wi_up), sycl::range<1>(wg_size));

        // ============ REDUCE KERNEL ============
        auto reduce_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior);
            sycl::local_accessor<T, 1> sg_parts(max_sg_local, cgh);
            sycl::local_accessor<T, 1> comm(wg_size, cgh);
            auto cn = N; auto cwg = num_wg; auto cws = wg_size; auto cmb = max_per_block;
            auto cir = ir; auto cb = b;
            auto in_rng = in_view;
            cgh.require(in_rng.accessor());
            cgh.parallel_for<ReduceK4a>(ndr, [=](sycl::nd_item<1> ndi) {
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
                T* sl = &comm[0];
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
                        sg, gen_input, scan_xform, binary_op, nullptr, carry,
                        in_rng, /*unused*/ in_rng, stid, cn, lip, sgs, sid, asg, sl);
                    if (slid == 0) sg_parts[sid] = carry.__v;
                    carry.__destroy();
                }
                sycl::group_barrier(ndi.get_group());

                if (sid == 0) {
                    std::size_t ss = gid * nsl;
                    std::uint8_t it = static_cast<std::uint8_t>(ceiling_div(asg, (std::uint32_t)ssz));
                    lazy<T> wc;
                    if (it == 1) {
                        std::uint32_t li = std::min((std::uint32_t)slid, asg - 1);
                        T v = sg_parts[li];
                        rts::__sub_group_scan_partial<__use_subgroup_ops, true, false>(
                            sg, v, binary_op, wc, asg, sl);
                        if (slid < asg) d_scratch[ss + slid] = v;
                    } else {
                        std::uint32_t ri = slid;
                        T v = sg_parts[ri];
                        rts::__sub_group_scan<__use_subgroup_ops, true, false>(
                            sg, v, binary_op, wc, sl);
                        d_scratch[ss + ri] = v;
                        ri += ssz;
                        for (std::uint32_t i = 1; i < (std::uint32_t)it - 1; i++) {
                            v = sg_parts[ri];
                            rts::__sub_group_scan<__use_subgroup_ops, true, true>(
                                sg, v, binary_op, wc, sl);
                            d_scratch[ss + ri] = v;
                            ri += ssz;
                        }
                        std::uint32_t li = std::min(ri, nsl - 1);
                        v = sg_parts[li];
                        rts::__sub_group_scan_partial<__use_subgroup_ops, true, true>(
                            sg, v, binary_op, wc, asg - ((it - 1) * ssz), sl);
                        if (ri < nsl) d_scratch[ss + ri] = v;
                    }
                    wc.__destroy();
                }
            });
        });
        prior = reduce_ev;

        // ============ SCAN KERNEL ============
        std::size_t nr = N - b * max_per_block;
        std::uint32_t iib = static_cast<std::uint32_t>(std::min(nr, (std::size_t)max_per_block));

        auto scan_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior);
            sycl::local_accessor<T, 1> sg_parts(max_sg_local + 1, cgh);
            sycl::local_accessor<T, 1> comm(wg_size, cgh);
            auto cn = N; auto cwg = num_wg; auto cws = wg_size; auto cmb = max_per_block;
            auto cmsg = max_sg_global; auto cir = ir; auto cb = b; auto ciib = iib;
            auto cnb = num_blocks;
            auto in_rng = in_view;
            auto o_rng = out_view;
            cgh.require(in_rng.accessor());
            cgh.require(o_rng.accessor());
            cgh.parallel_for<ScanK4a>(ndr, [=](sycl::nd_item<1> ndi) {
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
                T* sl = &comm[0];
                std::uint32_t ag = ceiling_div(ciib, ips * nsl);
                std::size_t gs = (cb * cmb) + (gid * ips * nsl);
                std::size_t mig = (std::size_t)ips * nsl;
                std::uint32_t iig = static_cast<std::uint32_t>(std::min(cn - gs, mig));
                std::uint32_t asg = ceiling_div(iig, ips);

                lazy<T> carry_last;
                lazy<T> sgc;
                bool sgci = true;

                if (sid == 0) {
                    std::uint8_t it = static_cast<std::uint8_t>(ceiling_div(asg, (std::uint32_t)ssz));
                    std::size_t sbm = gid * nsl;
                    std::uint32_t lri = slid;
                    for (std::uint8_t i = 0; i < it - 1; i++) {
                        sg_parts[lri] = d_scratch[sbm + lri]; lri += ssz;
                    }
                    if (lri < asg) sg_parts[lri] = d_scratch[sbm + lri];
                    std::uint32_t off = nsl - 1;
                    if (gid > 0) {
                        std::size_t etp = sbm / nsl;
                        std::size_t pci = ceiling_div(static_cast<std::uint32_t>(etp), (std::uint32_t)ssz);
                        if (pci == 1) {
                            std::size_t pid = nsl * slid + off;
                            std::size_t rid = (pid < sbm) ? pid : sbm - 1;
                            T val = d_scratch[rid];
                            rts::__sub_group_scan_partial<__use_subgroup_ops, true, false>(
                                sg, val, binary_op, carry_last, static_cast<std::uint32_t>(etp), sl);
                        } else {
                            std::uint32_t rid = nsl * slid + off;
                            std::uint32_t rinc = nsl * ssz;
                            T val = d_scratch[rid];
                            rts::__sub_group_scan<__use_subgroup_ops, true, false>(
                                sg, val, binary_op, carry_last, sl);
                            rid += rinc;
                            for (std::uint32_t i = 1; i < (std::uint32_t)pci - 1; i++) {
                                val = d_scratch[rid];
                                rts::__sub_group_scan<__use_subgroup_ops, true, true>(
                                    sg, val, binary_op, carry_last, sl);
                                rid += rinc;
                            }
                            std::size_t rem = etp - ((pci - 1) * ssz);
                            std::size_t frid = std::min((std::size_t)rid, sbm - 1);
                            val = d_scratch[frid];
                            rts::__sub_group_scan_partial<__use_subgroup_ops, true, true>(
                                sg, val, binary_op, carry_last, static_cast<std::uint32_t>(rem), sl);
                        }
                        std::size_t co = slid;
                        std::uint8_t ci = 0;
                        for (; ci < it - 1; ci++) {
                            sg_parts[co] = binary_op(carry_last.__v, sg_parts[co]); co += ssz;
                        }
                        if (ci * ssz + slid < asg)
                            sg_parts[co] = binary_op(carry_last.__v, sg_parts[co]);
                        if (slid == 0) sg_parts[asg] = carry_last.__v;
                        carry_last.__destroy();
                    }
                }
                sycl::group_barrier(ndi.get_group());

                if (cb == 0) {
                    if (sid > 0) sgc.__setup(sg_parts[std::min(sid - 1, asg - 1)]);
                    else if (gid > 0) sgc.__setup(sg_parts[asg]);
                    else sgci = false;
                } else {
                    T bci = d_scratch[cmsg + (cb % 2)];
                    if (sid > 0) sgc.__setup(binary_op(bci, sg_parts[std::min(sid - 1, asg - 1)]));
                    else if (gid > 0) sgc.__setup(binary_op(bci, sg_parts[asg]));
                    else sgc.__setup(bci);
                }

                std::size_t sgs = gs + (sid * ips);
                std::size_t stid = sgs + slid;

                if (sgci) {
                    rts::__scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                        /*__init_present=*/true, /*__capture_output=*/true, max_ipi>(
                        sg, gen_input, scan_xform, binary_op, write_op, sgc,
                        in_rng, o_rng, stid, cn, lip, sgs, sid, asg, sl);
                } else {
                    rts::__scan_through_elements_helper<__use_subgroup_ops, __is_inclusive,
                        /*__init_present=*/false, /*__capture_output=*/true, max_ipi>(
                        sg, gen_input, scan_xform, binary_op, write_op, sgc,
                        in_rng, o_rng, stid, cn, lip, sgs, sid, asg, sl);
                }

                if (slid == 0 && (ag == gid + 1) && (asg == sid + 1)) {
                    if (cb + 1 == cnb) d_result[0] = sgc.__v;
                    else d_scratch[cmsg + 1 - (cb % 2)] = sgc.__v;
                }
                sgc.__destroy();
            });
        });
        prior = scan_ev;

        ir -= std::min(ir, block_size);
        if (b + 2 == num_blocks) {
            ipi = ir >= max_per_block ? max_ipi
                : ceiling_div(bit_ceil_u32(static_cast<std::uint32_t>(ir)), num_wg * wg_size);
        }
    }
    q.wait();

    // Read output back
    std::vector<T> h_out(N);
    {
        auto acc = buf_out.get_host_access();
        for (std::size_t i = 0; i < N; i++)
            h_out[i] = acc[i];
    }

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (h_out[i] != h_expected[i]) {
            if (errors < 20)
                std::printf("[bridge4a] MISMATCH [%zu]: got %d expected %d\n", i, h_out[i], h_expected[i]);
            errors++;
        }
    }
    std::printf("[bridge4a] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

    sycl::free(d_scratch, q); sycl::free(d_result, q);
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
