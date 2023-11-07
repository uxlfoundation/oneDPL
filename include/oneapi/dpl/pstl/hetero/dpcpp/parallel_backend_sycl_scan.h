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

#ifndef _ONEDPL_parallel_backend_sycl_scan_H
#define _ONEDPL_parallel_backend_sycl_scan_H

namespace oneapi::dpl::experimental::kt
{

inline namespace igpu {

constexpr size_t SUBGROUP_SIZE = 32;

template<typename _T>
struct __scan_status_flag
{
    using _AtomicRefT = sycl::atomic_ref<::std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>;
    static constexpr std::uint32_t NOT_READY = 0;
    static constexpr std::uint32_t PARTIAL_MASK = 1;
    static constexpr std::uint32_t FULL_MASK = 2;
    static constexpr std::uint32_t OUT_OF_BOUNDS = 4;

    static constexpr int padding = SUBGROUP_SIZE;

    __scan_status_flag(const std::uint32_t tile_id, std::uint32_t* flags_begin, _T* tile_sums,
                       size_t num_elements)
        : atomic_flag(*(flags_begin + tile_id + padding)), scanned_partial_value(tile_sums + tile_id + padding),
          scanned_full_value(tile_sums + tile_id + padding + num_elements), num_elements{num_elements}
    {
    }

    void
    set_partial(_T val)
    {
        (*scanned_partial_value) = val;
        atomic_flag.store(PARTIAL_MASK);
    }

    void
    set_full(_T val)
    {
        (*scanned_full_value) = val;
        atomic_flag.store(FULL_MASK);
    }

    template <typename _Subgroup, typename BinOp>
    _T
    cooperative_lookback(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op, std::uint32_t* flags_begin,
                         _T* tile_sums)
    {
        _T sum = 0;
        int offset = -1;
        int i = 0;
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) + offset; tile >= 0; tile -= SUBGROUP_SIZE)
        {
            _AtomicRefT tile_atomic(*(flags_begin + tile + padding - local_id));
            std::uint32_t flag;
            do
            {
                flag = tile_atomic.load();
            } while (!sycl::all_of_group(subgroup, flag != NOT_READY)); // Loop till all ready

            bool is_full = flag == FULL_MASK;
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            auto lowest_item_with_full = is_full_ballot.find_low();

            // The partial scan results and the full scan sum values are in contiguous memory.
            // Each section of the memory is of size num_elements.
            // The partial sum for a tile is at [i] and the full sum is at [i + num_elements]
            // is_full * num_elements allows to select between the two values without branching the code.
            size_t contrib_offset = tile + padding - local_id + is_full * num_elements;
            _T val = *(tile_sums + contrib_offset);
            _T contribution = local_id <= lowest_item_with_full && (tile - local_id >= 0) ? val : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            sum += sycl::reduce_over_group(subgroup, contribution, bin_op);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (is_full_ballot.any())
                break;

        }

        return sum;
    }

    _AtomicRefT atomic_flag;
    _T* scanned_partial_value;
    _T* scanned_full_value;

    size_t num_elements;
};

template <typename _KernelParam, bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;

    constexpr int status_flag_padding = SUBGROUP_SIZE;
    std::uint32_t status_flags_size = num_wgs + status_flag_padding + 1;
    std::uint32_t tile_sums_size = num_wgs + status_flag_padding;

    uint32_t* status_flags = sycl::malloc_device<uint32_t>(status_flags_size, __queue);
    // First status_flags_size elements: partial scanned values (using _BinaryOp) of each workgroup
    // Second status_flags_size elements: full scanned values, i.e. sum of the previous and current workgroup's partial sums
    _Type* tile_sums = sycl::malloc_device<_Type>(tile_sums_size * 2, __queue);

    auto fill_event = __queue.submit([&](sycl::handler& hdl) {
        hdl.parallel_for<class scan_kt_init>(sycl::range<1>{status_flags_size}, [=](const sycl::item<1>& item)  {
                int id = item.get_linear_id();
                status_flags[id] = id < status_flag_padding ? __scan_status_flag<_Type>::OUT_OF_BOUNDS
                                                            : __scan_status_flag<_Type>::NOT_READY;
        });
    });

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        hdl.depends_on(fill_event);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for<class scan_kt_main>(sycl::nd_range<1>(num_workitems, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            auto group = item.get_group();
            auto subgroup = item.get_sub_group();

            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                sycl::atomic_ref<::std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    idx_atomic(status_flags[status_flags_size - 1]);
                tile_id_lacc[0] = idx_atomic.fetch_add(1);
            }
            sycl::group_barrier(group);
            std::uint32_t tile_id = tile_id_lacc[0];

            auto current_offset = (tile_id*elems_in_tile);
            auto next_offset = ((tile_id+1)*elems_in_tile);
            if (next_offset > n)
                next_offset = n;
            auto in_begin = __in_rng.begin() + current_offset;
            auto in_end = __in_rng.begin() + next_offset;
            auto out_begin = __out_rng.begin() + current_offset;

            if (current_offset >= n)
                return;

            auto local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);
            _Type prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (subgroup.get_group_id() == 0)
            {
                __scan_status_flag<_Type> flag(tile_id, status_flags, tile_sums, status_flags_size);

                if (group.leader())
                    flag.set_partial(local_sum);

                // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                prev_sum = flag.cooperative_lookback(tile_id, subgroup, __binary_op, status_flags, tile_sums);

                if (group.leader())
                    flag.set_full(prev_sum + local_sum);
            }

            prev_sum = sycl::group_broadcast(group, prev_sum, 0);
            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
        });
    });

    event.wait();

    sycl::free(status_flags, __queue);
    sycl::free(tile_sums, __queue);
}

// The generic structure for configuring a kernel
template <std::uint16_t ElemsPerWorkItem, std::uint16_t WorkGroupSize, typename KernelName>
struct kernel_param
{
    static constexpr std::uint16_t elems_per_workitem = ElemsPerWorkItem;
    static constexpr std::uint16_t workgroup_size = WorkGroupSize;
    using kernel_name = KernelName;
};

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */