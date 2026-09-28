// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/utils_sycl.h"
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <limits>
#include <algorithm>
#include <iostream>

auto get_device_parameters(const sycl::device& dvc)
{
    constexpr std::uint32_t num_cu_per_xe_core = 8;

    const std::size_t llc_size = dvc.get_info<sycl::info::device::global_mem_cache_size>();
    const std::uint32_t max_compute_units = dvc.get_info<sycl::info::device::max_compute_units>();
    const std::uint32_t num_xe_cores = std::max(1u, max_compute_units / num_cu_per_xe_core);
    const std::uint32_t max_wg_size = dvc.get_info<sycl::info::device::max_work_group_size>();
    const auto sg_sizes = dvc.get_info<sycl::info::device::sub_group_sizes>();

    std::uint8_t min_sg_size = 0, max_sg_size = 0;
    if (!sg_sizes.empty())
    {
        const auto [min_it, max_it] = std::minmax_element(sg_sizes.begin(), sg_sizes.end());
        min_sg_size = *min_it;
        max_sg_size = *max_it;
    }
   
    return std::tuple{llc_size, max_compute_units, num_xe_cores, max_wg_size, min_sg_size, max_sg_size};
}

auto get_block_limits(const sycl::device& dvc, std::size_t llc_bytes_per_iteration,
                      std::size_t storage_bytes_per_iteration = 0)
{
    constexpr std::uint32_t wg_size_cap = 1024;
    constexpr std::size_t llc_per_cu_fallback_size = 32 * 1024;
    constexpr std::size_t storage_size_cap = 4 * 1024 * 1024;

    auto [llc_size, max_compute_units, num_xe_cores, max_wg_size, min_sg_size, max_sg_size] = get_device_parameters(dvc);
    std::cout << "Device LLC size: " << llc_size << std::endl
              << "Device compute units: " << max_compute_units << std::endl
              << "Device XE cores: " << num_xe_cores << std::endl
              << "Device biggest WG size: " << max_wg_size << std::endl
              << "Device smallest SG size: " << int(min_sg_size) << std::endl
              << "Device biggest SG size: " << int(max_sg_size) << std::endl;

    if (llc_size == 0)
        llc_size = llc_per_cu_fallback_size * max_compute_units;
    const std::size_t llc_target_size = llc_size / 2;

    const std::uint32_t final_wg_size = (std::min(max_wg_size, wg_size_cap) / max_sg_size) * max_sg_size;

    std::uint32_t final_num_work_groups = 0;
    const std::size_t llc_size_required = llc_bytes_per_iteration * final_wg_size * num_xe_cores;
    bool llc_too_small = llc_size < llc_size_required;
    if (llc_too_small)
        final_num_work_groups = num_xe_cores * 2;
    else if (llc_size < 2 * llc_size_required)
        final_num_work_groups = num_xe_cores;
    else
        final_num_work_groups = num_xe_cores * 2;

    const std::size_t final_work_items_per_block = final_num_work_groups * final_wg_size;

    std::uint32_t inputs_per_item_limit = std::numeric_limits<std::uint32_t>::max() / (final_wg_size * 2);
    if (storage_bytes_per_iteration > 0)
        inputs_per_item_limit = storage_size_cap / (storage_bytes_per_iteration * final_work_items_per_block);

    const std::uint32_t max_sub_groups_local = (final_wg_size + min_sg_size - 1) / min_sg_size;
    const std::uint32_t max_sub_groups_global = max_sub_groups_local * final_num_work_groups;

    return std::tuple{llc_size_required, llc_target_size, llc_too_small, final_wg_size, final_num_work_groups,
                      final_work_items_per_block, inputs_per_item_limit, max_sub_groups_local, max_sub_groups_global};
}

void check_scan_block_parameters(const sycl::device& dvc)
{
    using DataType = float;
    // Parameters for in-place remove_if
    constexpr std::size_t llc_bytes_per_iter = 2 * sizeof(DataType);
    constexpr std::size_t storage_bytes_per_iter = 0; // sizeof(DataType);

    auto [llc_size_required, llc_target_size, llc_too_small, final_wg_size, final_work_groups, final_wi_per_block,
          inputs_per_wi_limit, max_sgroups_local, max_sgroups_global]
         = get_block_limits(dvc, llc_bytes_per_iter, storage_bytes_per_iter);
    std::cout << "LLC demand per iteration: " << llc_bytes_per_iter << std::endl
              << "Storage demand per iteration: " << storage_bytes_per_iter << std::endl
              << "LLC size required: " << llc_size_required << std::endl
              << "Targeted LLC size: " << llc_target_size << std::endl
              << "Selected work group size: " << final_wg_size << std::endl
              << "Selected work group number: " << final_work_groups << std::endl
              << "Work items per block: " << final_wi_per_block << std::endl
              << "Iteration limit per WI: " << inputs_per_wi_limit << std::endl
              << "Global max number of subgroups: " << max_sgroups_global << std::endl
              << "Max number of subgroups per WG: " << max_sgroups_local << std::endl;

    std::cout << std::endl << "Input size,Block number,Block size,Inputs per WI,Tail size,Tail inputs per WI" << std::endl;
    
    auto compute_block_params = [=](std::size_t input_size)
    {
        assert((input_size & (input_size - 1)) == 0); // a power of two
        const std::size_t target_num_blocks =
            llc_too_small ? 1 : (input_size * llc_bytes_per_iter + llc_target_size - 1) / llc_target_size;
        const std::size_t max_target_work_items = target_num_blocks * final_wi_per_block;
        const std::uint32_t max_inputs_per_wi = std::min<std::uint32_t>(inputs_per_wi_limit,
            std::max<std::uint32_t>(1, (input_size + max_target_work_items - 1) / max_target_work_items));
        const std::size_t max_inputs_per_block = final_wi_per_block * max_inputs_per_wi;
        const std::size_t num_blocks = (input_size + max_inputs_per_block - 1) / max_inputs_per_block;
        const std::size_t block_size = std::min(input_size, max_inputs_per_block);
        const std::size_t inputs_per_wi =
            input_size >= max_inputs_per_block ? max_inputs_per_wi : (input_size + final_wi_per_block - 1) / final_wi_per_block;
        const std::size_t input_tail = input_size % block_size;
        const std::size_t inputs_per_wi_tail =
            input_tail >= max_inputs_per_block ? max_inputs_per_wi : (input_tail + final_wi_per_block - 1) / final_wi_per_block;

        std::cout << input_size << ","
                  << num_blocks << ","
                  << block_size << ","
                  << inputs_per_wi << ","
                  << input_tail << ","
                  << inputs_per_wi_tail << std::endl;
    };

    for (std::size_t mi = 1; mi <= 256; mi *= 2)
        compute_block_params(mi * 1024 * 1024);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::device dvc = TestUtils::get_test_queue().get_device();
    if (dvc.is_gpu())
        check_scan_block_parameters(dvc);
#endif
    return EXIT_FAILURE; // make it fail for the output to be seen in the logs
    // return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
