// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
//
// Standalone reproducer for a suspected code-generation issue in the
// reduce-then-scan set_union path.
//
// This file intentionally does NOT include any oneDPL headers. It only uses
// <sycl/sycl.hpp> (via the test support headers) plus the test-suite helpers
// for queue setup and result reporting, so it can run in the same CI
// infrastructure as the rest of the test suite while isolating the kernel
// pattern under investigation from the rest of the library.
//
// Background
// ----------
// The commit under investigation ("further simplification",
// 91a65fc57a36381bc155a4dc9ffab7acffce9d27) rewrote the inner element loop of
// __scan_through_elements_helper_impl. The two forms are, by inspection,
// semantically identical for every reachable state (iters >= 1):
//
//   (A) loop-carried index, reused after the loop:            [BROKEN branch]
//         std::size_t local_id = start_id;
//         for (j = 0; j + 1 < iters; ++j) { use(local_id); local_id += sg; }
//         offset = local_id;                       // == start_id + (iters-1)*sg
//
//   (B) index recomputed from the induction variable:         [PASSING branch]
//         for (j = 0; j + 1 < iters; ++j) { use(start_id + j*sg); }
//         offset = start_id + (iters-1)*sg;
//
// Yet on Windows / level_zero / release -O3 with a 2026.0 toolchain, the set
// operations fail (UR_RESULT_ERROR_DEVICE_LOST) with form (A) and pass with
// form (B). set_union is affected but ordinary scans are not; the distinguishing
// ingredient of the set path is the *variable-count scatter write* after the
// scan (each work item writes a data-dependent number of output elements at a
// prefix-sum offset), which is modeled below.
//
// This reproducer replicates, without oneDPL headers:
//   * a single work group with multiple sub-groups,
//   * a per-sub-group inclusive scan built from shift_group_right + broadcast
//     (the same primitive shape used by __sub_group_scan / _partial),
//   * the loop-carried-index-then-reuse control flow of the suspect helper,
//   * a set_union-like variable-count scatter write keyed off the scan result.
//
// The compile-time macro REPRO_USE_LOOP_CARRY selects form (A) (default, the
// broken form) or form (B) (the known-good form). Build/run both and compare.

#include "support/test_config.h"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl_defs.h" // sycl/sycl.hpp
#include "support/utils_sycl.h"      // get_test_queue

#include <cstdint>
#include <vector>
#include <numeric>

// Select the control-flow form of the inner element loop.
//   1 -> form (A): loop-carried index reused after the loop (suspected broken)
//   0 -> form (B): index recomputed from j (known-good)
#ifndef REPRO_USE_LOOP_CARRY
#define REPRO_USE_LOOP_CARRY 1
#endif

// Fixed geometry: one work group, several sub-groups, requested sub-group size.
// Kept small and deterministic so the result is verifiable against a serial
// reference and so a partial (non-full) trailing sub-group iteration always
// occurs.
inline constexpr int REPRO_SG_SIZE = 16;
inline constexpr int REPRO_NUM_SUB_GROUPS = 4;
inline constexpr int REPRO_WG_SIZE = REPRO_SG_SIZE * REPRO_NUM_SUB_GROUPS;
// Inputs (diagonals) processed per work item across the sub-group iterations.
inline constexpr int REPRO_ITERS_PER_ITEM = 3;
// Max output elements a single input "diagonal" may expand to (set_union writes
// between 1 and diagonal_spacing elements per diagonal).
inline constexpr int REPRO_MAX_EXPANSION = 4;

// Per-sub-group inclusive scan over a value, carrying the running total in
// `carry` across successive calls, matching the shape of __sub_group_masked_scan
// (Hillis-Steele via shift_group_right + a final broadcast of the total).
// `elements` limits participation for a partial trailing iteration, mirroring
// __sub_group_scan_partial.
template <bool Partial>
inline std::int64_t
sub_group_inclusive_scan(const sycl::sub_group& sg, std::int64_t value, std::int64_t& carry,
                         std::uint32_t elements)
{
    const std::uint32_t lid = sg.get_local_linear_id();
    const std::uint32_t sg_size = sg.get_max_local_range()[0];

    for (std::uint32_t shift = 1; shift < sg_size; shift <<= 1)
    {
        std::int64_t partial_in = sycl::shift_group_right(sg, value, shift);
        bool active = lid >= shift;
        if constexpr (Partial)
            active = active && lid < elements;
        if (active)
            value += partial_in;
    }

    // fold in the running carry
    value += carry;

    // broadcast the inclusive total to update the carry for the next iteration
    const std::uint32_t broadcast_id = Partial ? (elements - 1) : (sg_size - 1);
    carry = sycl::group_broadcast(sg, value, broadcast_id);

    return value;
}

// Serial reference for the same computation the kernel performs, per sub-group.
// Each sub-group independently scans its slice of `counts` and produces the
// same scatter into `out` as the kernel should.
static void
compute_reference(const std::vector<std::int64_t>& counts, const std::vector<std::int32_t>& payload,
                  std::size_t n, std::vector<std::int32_t>& out_ref, std::vector<char>& written_ref)
{
    const std::size_t inputs_per_sub_group = std::size_t{REPRO_ITERS_PER_ITEM} * REPRO_SG_SIZE;
    for (int sg = 0; sg < REPRO_NUM_SUB_GROUPS; ++sg)
    {
        std::size_t sg_start = std::size_t(sg) * inputs_per_sub_group;
        if (sg_start >= n)
            continue;
        std::int64_t running = 0;
        for (std::size_t i = sg_start; i < std::min(n, sg_start + inputs_per_sub_group); ++i)
        {
            std::int64_t c = counts[i];
            running += c; // inclusive prefix sum -> output end offset for this input
            std::size_t base = static_cast<std::size_t>(running - c);
            for (std::int64_t k = 0; k < c; ++k)
            {
                out_ref[base + k] = payload[i] + static_cast<std::int32_t>(k);
                written_ref[base + k] = 1;
            }
        }
    }
}

// Runs the reproducer kernel for a given input size (number of "diagonals").
// Returns true if the device output matches the serial reference.
static bool
run_case(sycl::queue& q, std::size_t n)
{
    const std::size_t inputs_per_sub_group = std::size_t{REPRO_ITERS_PER_ITEM} * REPRO_SG_SIZE;

    // Build deterministic, data-dependent counts in [1, REPRO_MAX_EXPANSION].
    // The pattern is chosen so the trailing sub-group iteration is partial and
    // the per-input counts vary across lanes (as they do for set_union).
    std::vector<std::int64_t> counts(n);
    std::vector<std::int32_t> payload(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        counts[i] = static_cast<std::int64_t>(1 + (i * 7 + 3) % REPRO_MAX_EXPANSION);
        payload[i] = static_cast<std::int32_t>(i * 10);
    }

    // Total output size = sum of all counts.
    const std::size_t out_size = static_cast<std::size_t>(std::accumulate(counts.begin(), counts.end(), std::int64_t{0}));

    std::vector<std::int32_t> out_ref(out_size, -1);
    std::vector<char> written_ref(out_size, 0);
    compute_reference(counts, payload, n, out_ref, written_ref);

    std::vector<std::int32_t> out_host(out_size, -1);

    {
        sycl::buffer<std::int64_t> counts_buf(counts.data(), sycl::range<1>(n));
        sycl::buffer<std::int32_t> payload_buf(payload.data(), sycl::range<1>(n));
        sycl::buffer<std::int32_t> out_buf(out_host.data(), sycl::range<1>(out_size));

        q.submit([&](sycl::handler& cgh) {
             sycl::accessor counts_acc(counts_buf, cgh, sycl::read_only);
             sycl::accessor payload_acc(payload_buf, cgh, sycl::read_only);
             sycl::accessor out_acc(out_buf, cgh, sycl::write_only, sycl::no_init);

             auto nd = sycl::nd_range<1>(sycl::range<1>(REPRO_WG_SIZE), sycl::range<1>(REPRO_WG_SIZE));

             cgh.parallel_for(nd, [=](sycl::nd_item<1> item)
                              [[sycl::reqd_sub_group_size(REPRO_SG_SIZE)]] {
                 const sycl::sub_group sg = item.get_sub_group();
                 const std::uint32_t sg_id = sg.get_group_linear_id();
                 const std::uint32_t lid = sg.get_local_linear_id();
                 const std::uint32_t sg_size = sg.get_max_local_range()[0];

                 // Offset of this sub-group's first input, matching __subgroup_start_id.
                 const std::size_t subgroup_start_id = std::size_t(sg_id) * inputs_per_sub_group;

                 // Active-sub-group guard, matching the call sites: skip sub-groups
                 // whose entire slice is past the end.
                 if (subgroup_start_id >= n)
                     return;

                 const std::size_t start_id = subgroup_start_id + lid;

                 // Number of inputs this sub-group actually processes, and the
                 // number of sub-group iterations to cover them (matches
                 // __subgroup_n / __iters).
                 const std::uint32_t subgroup_n = static_cast<std::uint32_t>(
                     std::min<std::size_t>(n - subgroup_start_id, inputs_per_sub_group));
                 const std::uint32_t iters = (subgroup_n + sg_size - 1) / sg_size;

                 std::int64_t carry = 0;

                 // ---- full iterations (all but the last) ----
#if REPRO_USE_LOOP_CARRY
                 // Form (A): loop-carried index, reused after the loop.
                 std::size_t local_id = start_id;
                 for (std::uint32_t j = 0; j + 1 < iters; ++j)
                 {
                     std::int64_t v = counts_acc[local_id];
                     std::int64_t incl = sub_group_inclusive_scan<false>(sg, v, carry, sg_size);
                     std::size_t end_off = static_cast<std::size_t>(incl);
                     std::size_t base = end_off - static_cast<std::size_t>(v);
                     for (std::int64_t k = 0; k < v; ++k)
                         out_acc[base + k] = payload_acc[local_id] + static_cast<std::int32_t>(k);
                     local_id += sg_size;
                 }
#else
                 // Form (B): index recomputed from j each iteration.
                 for (std::uint32_t j = 0; j + 1 < iters; ++j)
                 {
                     std::size_t idx = start_id + std::size_t(j) * sg_size;
                     std::int64_t v = counts_acc[idx];
                     std::int64_t incl = sub_group_inclusive_scan<false>(sg, v, carry, sg_size);
                     std::size_t end_off = static_cast<std::size_t>(incl);
                     std::size_t base = end_off - static_cast<std::size_t>(v);
                     for (std::int64_t k = 0; k < v; ++k)
                         out_acc[base + k] = payload_acc[idx] + static_cast<std::int32_t>(k);
                 }
#endif

                 // ---- trailing (partial) iteration ----
#if REPRO_USE_LOOP_CARRY
                 // Reuse the loop-carried index (this is the pattern under suspicion).
                 std::size_t offset = local_id;
#else
                 std::size_t offset = start_id + static_cast<std::size_t>(iters - 1) * sg_size;
#endif
                 // Clamp the *generation* index to stay in bounds (as the real helper
                 // does), but write at the unclamped offset guarded by a bounds check.
                 std::size_t gen_id = std::min<std::size_t>(offset, n - 1);
                 std::int64_t v = counts_acc[gen_id];
                 std::uint32_t elements_to_process = subgroup_n - (iters - 1) * sg_size;
                 std::int64_t incl = sub_group_inclusive_scan<true>(sg, v, carry, elements_to_process);
                 if (offset < n)
                 {
                     std::size_t end_off = static_cast<std::size_t>(incl);
                     std::size_t base = end_off - static_cast<std::size_t>(v);
                     for (std::int64_t k = 0; k < v; ++k)
                         out_acc[base + k] = payload_acc[offset] + static_cast<std::int32_t>(k);
                 }
             });
         }).wait_and_throw();
    }

    // Verify.
    bool ok = true;
    for (std::size_t i = 0; i < out_size; ++i)
    {
        if (written_ref[i] && out_host[i] != out_ref[i])
        {
            ok = false;
            break;
        }
    }
    return ok;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q = TestUtils::get_test_queue();

    const std::size_t inputs_per_sub_group = std::size_t{REPRO_ITERS_PER_ITEM} * REPRO_SG_SIZE;
    const std::size_t max_inputs = std::size_t{REPRO_NUM_SUB_GROUPS} * inputs_per_sub_group;

    bool all_ok = true;
    // Sweep sizes that force a partial trailing sub-group iteration and a
    // partial trailing sub-group, including the boundary where the last
    // sub-group's slice ends exactly on n.
    for (std::size_t n = 1; n <= max_inputs; ++n)
    {
        bool ok = run_case(q, n);
        if (!ok)
        {
            all_ok = false;
            std::cout << "MISMATCH at n=" << n << "\n";
        }
    }

    EXPECT_TRUE(all_ok, "set_union reduce-then-scan reproducer produced wrong results");
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
