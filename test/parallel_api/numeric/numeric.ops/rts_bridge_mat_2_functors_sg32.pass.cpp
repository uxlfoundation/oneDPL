// -*- C++ -*-
//===-- rts_bridge_mat_2_functors.pass.cpp ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bridge test 2 (Matrix2x2): Same as rts_bridge_2_functors but uses
// Matrix2x2<int32_t> with multiply_matrix to exercise the SLM communication
// path (non-trivially-copyable type) with a non-commutative binary op.

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <oneapi/dpl/pstl/utils.h>

using TestUtils::Matrix2x2;
using TestUtils::multiply_matrix;

template <typename T>
using lazy = oneapi::dpl::__internal::__lazy_ctor_storage<T>;

// ---------- helpers ----------

inline std::uint32_t bit_ceil_u32(std::uint32_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}
inline std::uint32_t ceiling_div(std::uint32_t a, std::uint32_t b) { return (a + b - 1) / b; }
inline std::uint32_t ceiling_div(std::size_t a, std::uint32_t b) { return static_cast<std::uint32_t>((a + b - 1) / b); }

// ---------- Functor indirection matching oneDPL ----------

struct noop_temp_data {
    template <typename V> void set(std::uint16_t, const V&) const {}
};

template <typename T>
struct gen_input {
    using TempData = noop_temp_data;
    T operator()(const T* data, std::size_t id, TempData&) const {
        return data[id];
    }
};

struct write_output {
    using TempData = noop_temp_data;
    template <typename T>
    void operator()(T* out, std::size_t id, const T& v, const TempData&) const {
        out[id] = v;
    }
};

struct identity_transform {
    template <typename T>
    T operator()(const T& v) const { return v; }
};

// ---------- SLM comm ----------

template <typename T>
T slm_shift_right(sycl::sub_group sg, T value, std::uint32_t shift, T* slm) {
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[base + lid] = value;
    sycl::group_barrier(sg);
    T result = slm[base + ((lid >= shift) ? lid - shift : lid)];
    sycl::group_barrier(sg);
    return result;
}

template <typename T>
T slm_broadcast(sycl::sub_group sg, T value, std::uint32_t bid, T* slm) {
    std::uint32_t lid = sg.get_local_linear_id();
    std::uint32_t base = sg.get_group_linear_id() * sg.get_max_local_range()[0];
    slm[base + lid] = value;
    sycl::group_barrier(sg);
    T result = slm[base + bid];
    sycl::group_barrier(sg);
    return result;
}

// ---------- scan helpers ----------

template <typename T, typename BinaryOp>
void sg_scan(sycl::sub_group sg, T& value, BinaryOp op,
             bool init_present, lazy<T>& carry, T* slm) {
    std::uint8_t lid = sg.get_local_linear_id();
    std::uint8_t sz = sg.get_max_local_range()[0];
    for (std::uint8_t shift = 1; shift <= sz / 2; shift <<= 1) {
        T prev = slm_shift_right(sg, value, shift, slm);
        if (lid >= shift) value = op(prev, value);
    }
    if (init_present) {
        value = op(carry.__v, value);
        carry.__v = slm_broadcast(sg, value, sz - 1, slm);
    } else {
        carry.__setup(slm_broadcast(sg, value, sz - 1, slm));
    }
}

template <typename T, typename BinaryOp>
void sg_scan_partial(sycl::sub_group sg, T& value, BinaryOp op,
                     bool init_present, lazy<T>& carry, std::uint32_t count, T* slm) {
    std::uint8_t lid = sg.get_local_linear_id();
    std::uint8_t sz = sg.get_max_local_range()[0];
    for (std::uint8_t shift = 1; shift <= sz / 2; shift <<= 1) {
        T prev = slm_shift_right(sg, value, shift, slm);
        if (lid >= shift && lid < count) value = op(prev, value);
    }
    if (init_present) {
        value = op(carry.__v, value);
        carry.__v = slm_broadcast(sg, value, count - 1, slm);
    } else {
        carry.__setup(slm_broadcast(sg, value, count - 1, slm));
    }
}

// ---------- reduce phase with functor indirection ----------

template <typename T, typename BinaryOp, typename GenInput, typename ScanTransform>
void scan_through_reduce(sycl::sub_group sg, BinaryOp op, GenInput gen, ScanTransform xform,
                         lazy<T>& carry, bool& carry_init,
                         const T* input, std::size_t start_id, std::size_t n,
                         std::uint32_t iters, std::size_t sg_start,
                         std::uint32_t sg_id, std::uint32_t active_sg, T* slm) {
    typename GenInput::TempData td{};
    std::uint8_t sz = sg.get_max_local_range()[0];
    bool full = (sg_start + iters * sz <= n);

    if (full) {
        T v = xform(gen(input, start_id, td));
        sg_scan(sg, v, op, false, carry, slm);
        carry_init = true;
        for (std::uint32_t j = 1; j < iters; j++) {
            v = xform(gen(input, start_id + j * sz, td));
            sg_scan(sg, v, op, true, carry, slm);
        }
    } else if (sg_id < active_sg) {
        std::uint32_t it = ceiling_div(static_cast<std::uint32_t>(n - sg_start), (std::uint32_t)sz);
        if (it == 1) {
            std::size_t lid = (start_id < n) ? start_id : n - 1;
            T v = xform(gen(input, lid, td));
            sg_scan_partial(sg, v, op, false, carry, static_cast<std::uint32_t>(n - sg_start), slm);
            carry_init = true;
        } else {
            T v = xform(gen(input, start_id, td));
            sg_scan(sg, v, op, false, carry, slm);
            carry_init = true;
            for (std::uint32_t j = 1; j < it - 1; j++) {
                v = xform(gen(input, start_id + j * sz, td));
                sg_scan(sg, v, op, true, carry, slm);
            }
            std::size_t off = start_id + (it - 1) * sz;
            std::size_t lid = (off < n) ? off : n - 1;
            v = xform(gen(input, lid, td));
            sg_scan_partial(sg, v, op, true, carry,
                static_cast<std::uint32_t>(n - (sg_start + (it - 1) * sz)), slm);
        }
    }
}

// ---------- scan phase with functor indirection ----------

template <typename T, typename BinaryOp, typename GenInput, typename ScanTransform, typename WriteOp>
void scan_through_scan(sycl::sub_group sg, BinaryOp op, GenInput gen, ScanTransform xform,
                       WriteOp write, lazy<T>& carry, bool init_present,
                       const T* input, T* output, std::size_t start_id, std::size_t n,
                       std::uint32_t iters, std::size_t sg_start,
                       std::uint32_t sg_id, std::uint32_t active_sg, T* slm) {
    typename GenInput::TempData td{};
    typename WriteOp::TempData wtd{};
    std::uint8_t sz = sg.get_max_local_range()[0];
    bool full = (sg_start + iters * sz <= n);

    if (full) {
        T v = gen(input, start_id, td);
        T sv = xform(v);
        sg_scan(sg, sv, op, init_present, carry, slm);
        write(output, start_id, sv, wtd);
        for (std::uint32_t j = 1; j < iters; j++) {
            v = gen(input, start_id + j * sz, td);
            sv = xform(v);
            sg_scan(sg, sv, op, true, carry, slm);
            write(output, start_id + j * sz, sv, wtd);
        }
    } else if (sg_id < active_sg) {
        std::uint32_t it = ceiling_div(static_cast<std::uint32_t>(n - sg_start), (std::uint32_t)sz);
        if (it == 1) {
            std::size_t lid = (start_id < n) ? start_id : n - 1;
            T v = gen(input, lid, td);
            T sv = xform(v);
            sg_scan_partial(sg, sv, op, init_present, carry, static_cast<std::uint32_t>(n - sg_start), slm);
            if (start_id < n) write(output, start_id, sv, wtd);
        } else {
            T v = gen(input, start_id, td);
            T sv = xform(v);
            sg_scan(sg, sv, op, init_present, carry, slm);
            write(output, start_id, sv, wtd);
            for (std::uint32_t j = 1; j < it - 1; j++) {
                std::size_t lid = start_id + j * sz;
                v = gen(input, lid, td);
                sv = xform(v);
                sg_scan(sg, sv, op, true, carry, slm);
                write(output, lid, sv, wtd);
            }
            std::size_t off = start_id + (it - 1) * sz;
            std::size_t lid = (off < n) ? off : n - 1;
            v = gen(input, lid, td);
            sv = xform(v);
            sg_scan_partial(sg, sv, op, true, carry,
                static_cast<std::uint32_t>(n - (sg_start + (it - 1) * sz)), slm);
            if (off < n) write(output, off, sv, wtd);
        }
    }
}

// ---------- kernel names ----------
class ReduceK2MatSg32;
class ScanK2MatSg32;

int run_test() {
    using T = Matrix2x2<std::int32_t>;
    using BinaryOp = multiply_matrix<std::int32_t>;
    constexpr std::size_t N = 20000;

    std::vector<T> h_input(N);
    for (std::uint32_t k = 0; k < N; k++)
        h_input[k] = T(k % 7 + 1, k % 7 + 2);

    // Manual inclusive scan with matrix multiply
    BinaryOp binary_op{};
    std::vector<T> h_expected(N);
    h_expected[0] = h_input[0];
    for (std::size_t i = 1; i < N; i++)
        h_expected[i] = binary_op(h_expected[i - 1], h_input[i]);

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::printf("[bridge2_mat_sg32] Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());

    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::uint8_t min_sg = *std::min_element(sg_sizes.begin(), sg_sizes.end());
    std::uint8_t max_sg = *std::max_element(sg_sizes.begin(), sg_sizes.end());

    constexpr std::uint16_t max_ipi = std::max(std::uint16_t{1}, std::uint16_t{512 / sizeof(T)});
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

    std::printf("[bridge2_mat_sg32] wg_size=%u num_wg=%u ipi=%u blocks=%zu scratch=%u\n",
                wg_size, num_wg, ipi, num_blocks, scratch_size);

    T* d_in = sycl::malloc_device<T>(N, q);
    T* d_out = sycl::malloc_device<T>(N, q);
    T* d_scratch = sycl::malloc_device<T>(scratch_size, q);
    T* d_result = sycl::malloc_device<T>(1, q);

    q.memcpy(d_in, h_input.data(), N * sizeof(T));

    // Initialize output and scratch via memcpy (can't use memset for non-trivial type)
    std::vector<T> h_out_init(N, T(-666, 666));
    std::vector<T> h_scratch_init(scratch_size, T());
    q.memcpy(d_out, h_out_init.data(), N * sizeof(T));
    q.memcpy(d_scratch, h_scratch_init.data(), scratch_size * sizeof(T));
    q.wait();

    gen_input<T> gen;
    identity_transform xform;
    write_output write;

    sycl::event prior;
    std::size_t ir = remaining;

    for (std::size_t b = 0; b < num_blocks; b++) {
        std::uint32_t wi = ceiling_div(std::min(ir, (std::size_t)max_per_block), ipi);
        std::uint32_t wi_up = ceiling_div(wi, wg_size) * wg_size;
        auto ndr = sycl::nd_range<1>(sycl::range<1>(wi_up), sycl::range<1>(wg_size));

        auto reduce_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior);
            sycl::local_accessor<T, 1> sg_parts(max_sg_local, cgh);
            sycl::local_accessor<T, 1> comm(wg_size, cgh);
            std::size_t cn = N;
            std::uint32_t cwg = num_wg;
            std::uint32_t cws = wg_size;
            std::uint32_t cmb = max_per_block;
            std::size_t cir = ir;
            std::size_t cb = b;
            cgh.parallel_for<ReduceK2MatSg32>(ndr, [=](sycl::nd_item<1> ndi) [[sycl::reqd_sub_group_size(32)]] {
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
                bool ci = false;
                BinaryOp op{};
                if (sid < asg) {
                    scan_through_reduce(sg, op, gen, xform, carry, ci,
                                        d_in, stid, cn, lip, sgs, sid, asg, sl);
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
                        sg_scan_partial(sg, v, op, false, wc, asg, sl);
                        if (slid < asg) d_scratch[ss + slid] = v;
                    } else {
                        std::uint32_t ri = slid;
                        T v = sg_parts[ri];
                        sg_scan(sg, v, op, false, wc, sl);
                        d_scratch[ss + ri] = v;
                        ri += ssz;
                        for (std::uint32_t i = 1; i < (std::uint32_t)it - 1; i++) {
                            v = sg_parts[ri];
                            sg_scan(sg, v, op, true, wc, sl);
                            d_scratch[ss + ri] = v;
                            ri += ssz;
                        }
                        std::uint32_t li = std::min(ri, nsl - 1);
                        v = sg_parts[li];
                        sg_scan_partial(sg, v, op, true, wc, asg - ((it - 1) * ssz), sl);
                        if (ri < nsl) d_scratch[ss + ri] = v;
                    }
                    wc.__destroy();
                }
            });
        });
        prior = reduce_ev;

        std::size_t nr = N - b * max_per_block;
        std::uint32_t iib = static_cast<std::uint32_t>(std::min(nr, (std::size_t)max_per_block));

        auto scan_ev = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prior);
            sycl::local_accessor<T, 1> sg_parts(max_sg_local + 1, cgh);
            sycl::local_accessor<T, 1> comm(wg_size, cgh);
            std::size_t cn = N;
            std::uint32_t cwg = num_wg;
            std::uint32_t cws = wg_size;
            std::uint32_t cmb = max_per_block;
            std::uint32_t cmsg = max_sg_global;
            std::size_t cir = ir;
            std::size_t cb = b;
            std::uint32_t ciib = iib;
            std::size_t cnb = num_blocks;
            cgh.parallel_for<ScanK2MatSg32>(ndr, [=](sycl::nd_item<1> ndi) [[sycl::reqd_sub_group_size(32)]] {
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
                BinaryOp op{};

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
                            sg_scan_partial(sg, val, op, false, carry_last, static_cast<std::uint32_t>(etp), sl);
                        } else {
                            std::uint32_t rid = nsl * slid + off;
                            std::uint32_t rinc = nsl * ssz;
                            T val = d_scratch[rid];
                            sg_scan(sg, val, op, false, carry_last, sl);
                            rid += rinc;
                            for (std::uint32_t i = 1; i < (std::uint32_t)pci - 1; i++) {
                                val = d_scratch[rid];
                                sg_scan(sg, val, op, true, carry_last, sl);
                                rid += rinc;
                            }
                            std::size_t rem = etp - ((pci - 1) * ssz);
                            std::size_t frid = std::min((std::size_t)rid, sbm - 1);
                            val = d_scratch[frid];
                            sg_scan_partial(sg, val, op, true, carry_last, static_cast<std::uint32_t>(rem), sl);
                        }
                        std::size_t co = slid;
                        std::uint8_t ci = 0;
                        for (; ci < it - 1; ci++) {
                            sg_parts[co] = op(carry_last.__v, sg_parts[co]); co += ssz;
                        }
                        if (ci * ssz + slid < asg)
                            sg_parts[co] = op(carry_last.__v, sg_parts[co]);
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
                    if (sid > 0) sgc.__setup(op(bci, sg_parts[std::min(sid - 1, asg - 1)]));
                    else if (gid > 0) sgc.__setup(op(bci, sg_parts[asg]));
                    else sgc.__setup(bci);
                }

                std::size_t sgs = gs + (sid * ips);
                std::size_t stid = sgs + slid;

                scan_through_scan(sg, op, gen, xform, write, sgc, sgci,
                                  d_in, d_out, stid, cn, lip, sgs, sid, asg, sl);

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

    std::vector<T> h_out(N);
    q.memcpy(h_out.data(), d_out, N * sizeof(T)).wait();

    int errors = 0;
    for (std::size_t i = 0; i < N; i++) {
        if (!(h_out[i] == h_expected[i])) {
            if (errors < 20)
                std::printf("[bridge2_mat_sg32] MISMATCH [%zu]: got {%d,%d,%d,%d} expected {%d,%d,%d,%d}\n",
                            i, h_out[i].a00, h_out[i].a01, h_out[i].a10, h_out[i].a11,
                            h_expected[i].a00, h_expected[i].a01, h_expected[i].a10, h_expected[i].a11);
            errors++;
        }
    }
    std::printf("[bridge2_mat_sg32] %s: %d errors out of %zu\n", errors ? "FAIL" : "PASS", errors, N);

    sycl::free(d_in, q); sycl::free(d_out, q); sycl::free(d_scratch, q); sycl::free(d_result, q);
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
