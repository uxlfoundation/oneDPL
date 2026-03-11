# GPU Radix Sort Kernel Templates

## Introduction

This RFC describes the design and implementation of GPU radix sort algorithms for oneDPL kernel templates. Two implementations were provided: an ESIMD variant optimized for Intel PVC archiecture and a portable SYCL variant targeting
Intel GPUs that support work-group independent forward progress.

Both implementations use the onesweep radix sort algorithm with decoupled lookback synchronization. This approach addresses the global memory bandwidth bottleneck that typically limits GPU sorting performance by reducing memory traffic compared to traditional multi-pass radix sort algorithms.

The motivation for these implementations was to provide:
- Competitive performance with state-of-the-art GPU sort
- Specialized ESIMD implementation for optimal performance on Intel PVC architecture
- Portable SYCL implementation for Intel Xe GPU support
- Integration with the oneDPL kernel templates framework

## Proposal

### Algorithm Design

#### Onesweep Radix Sort Overview

The onesweep radix sort algorithm processes keys in a single pass per radix stage, unlike traditional radix sort implementations that use separate histogram and reorder phases. For an 8-bit radix, a 32-bit key requires 4 stages to complete the sort after an upfront histogram and scan kernel which computes radix bin offsets across all stages at once.

Each radix stage follows this flow:
1. Load keys from global memory
2. Rank keys locally within each work-group
3. Scan histograms across work-groups
4. Synchronize work-groups using decoupled lookback
5. Reorder keys through shared local memory
6. Scatter keys to global memory

The algorithm processes data in tiles, where each work-group handles a contiguous block of elements. Work-groups execute concurrently and coordinate through a chained scan protocol via decoupled lookback.

#### Memory Traffic and Performance Benefits

The onesweep approach significantly reduces global memory traffic. Assuming temporary storage and decoupled lookback traffic is negligble, we can model the reduction in memory traffic as shown below:

Traditional radix sort implementations perform separate passes for each operation:
- Histogram pass: 1 read through the entire dataset
- Scan: negligble global memory traffic
- Reorder pass: 1 read + 1 write through the entire dataset
- Total: approximately 3n memory accesses per radix iteration

The onesweep approach reduces this traffic:
- Upfront global histogram and scan (one-time cost across all stages)
- Per radix iteration: 1 read + 1 write through the entire dataset
- No separate histogram pass needed per stage

This reduction in memory bandwidth pressure is critical for GPU sort performance, where global memory bandwidth is the primary bottleneck. The upfront histogram enables work-groups to determine global offsets without revisiting the entire dataset at each stage.

#### Decoupled Lookback

Decoupled lookback enables work-groups to process data independently without device-wide barriers and avoid expensive reloading of keys from global memory with separate kernels. Each work-group publishes its local histogram to global memory after computing it. Work-group N then sequentially looks back at the published histograms from work-groups N-1 through 0 to determine the global offset for its data. Work-groups may perform an early exit from the decoupled lookback once a lower indexed work-group has found and published its full incoming histogram.

This mechanism provides two key benefits:
- Work-groups can begin processing as soon as their dependencies are satisfied
- No global synchronization points that would stall all work-groups, freeing hardware resources for higher indexed tiles

The lookback operates by having each work-group spin-wait on global memory locations until the previous work-group has written its partial histogram. Work-groups accumulate these partial results to compute their global offsets.

### Safety and Forward Progress Guarantees

#### The Challenge

The decoupled lookback protocol requires work-group N to spin-wait for work-group N-1 to publish its results. Without guarantees of concurrent execution, this can lead to deadlock if the hardware scheduler does not ensure that work-group N-1 makes progress while work-group N is waiting.

This is a fundamental challenge for any GPU algorithm that uses inter-work-group synchronization within a single kernel launch.

#### ESIMD Approach

The ESIMD implementation relies on specific PVC hardware scheduling characteristics and memory model. Empirical testing was used to verify that work-groups avoid deadlock in the lookback protocol.

#### SYCL Solution and Alternatives

The SYCL implementation uses the oneAPI forward progress extension as a kernel launch property to guarantee safety:

```cpp
auto get(syclex::properties_tag) const
{
    return syclex::properties{
        syclex::work_group_progress<syclex::forward_progress_guarantee::concurrent,
                                    syclex::execution_scope::root_group>,
        syclex::sub_group_size<32>
    };
}
```

This extension guarantees that work-groups within the kernel launch execute concurrently and make independent forward progress. This is the only approach that provides a formal hardware safety guarantee across different GPU schedulers and vendors where the property is supported.

An alternative approach using atomic counters was evaluated but rejected. In this approach, each work-group would increment a global atomic counter to receive a sequential tile ID, and lookback would use this ordering instead of their work-group identifier. This would potentially handle cases where work-groups execute out of dispatch order.

However, this approach has limitations:
- It requires empirical testing and implementation adjustment for each specific architecture to verify correct behavior
- It does not provide a formal guarantee of safety
- During evaluation, hardware hangs were examined on certain hardware configurations
- It relies on scheduler behavior when multiple kernels are concurrently executing that is unspecified by the SYCL programming model

The forward progress extension provides a principled solution that is supported by the oneAPI DPC++ compiler and guarantees correct behavior.

### API Overview

The implementations provide high-level sorting interfaces in the kernel templates framework:

```cpp
namespace oneapi::dpl::experimental::kt::gpu {
    // In-place sort (keys only)
    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam, typename KeysRange>
    sycl::event radix_sort(sycl::queue q, KeysRange&& keys, KernelParam param = {});

    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam, typename KeysIterator>
    sycl::event radix_sort(sycl::queue q, KeysIterator keys_first, KeysIterator keys_last,
                          KernelParam param = {});

    // Out-of-place sort (keys only)
    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam,
              typename KeysInRange, typename KeysOutRange>
    sycl::event radix_sort(sycl::queue q, KeysInRange&& keys_in, KeysOutRange&& keys_out,
                          KernelParam param = {});

    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam, typename KeysIterator>
    sycl::event radix_sort(sycl::queue q, KeysIterator keys_first, KeysIterator keys_last,
                          KeysIterator keys_out_first, KernelParam param = {});

    // In-place sort by key (keys + values)
    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam,
              typename KeysRange, typename ValuesRange>
    sycl::event radix_sort_by_key(sycl::queue q, KeysRange&& keys, ValuesRange&& values,
                                  KernelParam param = {});

    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam,
              typename KeysIterator, typename ValuesIterator>
    sycl::event radix_sort_by_key(sycl::queue q, KeysIterator keys_first, KeysIterator keys_last,
                                  ValuesIterator values_first, KernelParam param = {});

    // Out-of-place sort by key (keys + values)
    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam,
              typename KeysInRange, typename ValsInRange, typename KeysOutRange, typename ValsOutRange>
    sycl::event radix_sort_by_key(sycl::queue q, KeysInRange&& keys_in, ValsInRange&& vals_in,
                                  KeysOutRange&& keys_out, ValsOutRange&& vals_out,
                                  KernelParam param = {});

    template <bool IsAscending = true, uint8_t RadixBits = 8, typename KernelParam,
              typename KeysIterator, typename ValsIterator>
    sycl::event radix_sort_by_key(sycl::queue q, KeysIterator keys_first, KeysIterator keys_last,
                                  ValsIterator vals_first, KeysIterator keys_out_first,
                                  ValsIterator vals_out_first, KernelParam param = {});

    namespace esimd {
        // Same API signatures as gpu namespace
    }
}
```

Template parameters control sort order (`IsAscending`), radix bit width (`RadixBits`), and work-group configuration (via `KernelParam`). All functions return a `sycl::event` for asynchronous execution chaining.

Unified dispatch logic in `radix_sort_dispatchers.h` and `radix_sort_submitters.h` selects the appropriate implementation and optimization path based on the input size and available hardware features.

### Usage Example

```cpp
#include <oneapi/dpl/experimental/kernel_templates>

namespace kt = oneapi::dpl::experimental::kt;

sycl::queue q;
constexpr std::size_t n = 1'000'000;

// SYCL variant: work-group size 1024, 12 elements per work-item
{
    constexpr kt::kernel_param</*data_per_workitem=*/12, /*workgroup_size=*/1024> param;
    uint32_t* keys = sycl::malloc_device<uint32_t>(n, q);
    // ... initialize keys ...
    kt::gpu::radix_sort(q, keys, keys + n, param).wait();
    sycl::free(keys, q);
}

// ESIMD variant on PVC: work-group size 64, 96 elements per work-item
{
    constexpr kt::kernel_param</*data_per_workitem=*/96, /*workgroup_size=*/64> param;
    uint32_t* keys = sycl::malloc_device<uint32_t>(n, q);
    // ... initialize keys ...
    kt::gpu::esimd::radix_sort(q, keys, keys + n, param).wait();
    sycl::free(keys, q);
}
```

### Platform Support

The ESIMD implementation is limited to Intel PVC architecture with a work-group size of 64 and data per work item multiple of 32. The SYCL implementation has been tested on Intel PVC and BMG with work-group sizes of 512 and 1024, requires forward progress extension support, and is expected to work on future Intel GPU architectures supporting the extension.

### Testing

Correctness is validated through a unified test structure that executes across all supported configurations. Tests verify sorting correctness for random, sorted, reverse-sorted, and pathological input distributions.

| Variant | Work-Group Size | Data Per Work-Item | Key Types | Test Variants |
|---------|-----------------|--------------------|-----------| ------------- |
| ESIMD   | 64              | 32-512 (step 32)   | char, uint16_t, int, uint64_t, float, double | In-place, out-of-place, by-key |
| SYCL    | 512, 1024       | 1-16               | char, uint16_t, int, uint64_t, float, double | In-place, out-of-place, by-key |

### Implementation Status

**ESIMD Implementation:**
The ESIMD implementation provides a complete onesweep multi-work-group kernel. It includes a one-work-group optimization for small inputs where the problem size is less than or equal to `workgroup_size × data_per_work_item`. This optimization avoids the lookback synchronization overhead for cases that fit within a single work-group's SLM capacity. The implementation supports both keys-only sorting and key-value pair sorting.

**SYCL Implementation:**
The SYCL implementation provides an onesweep multi-work-group kernel using sub-group primitives for portability. It supports both keys-only sorting and key-value pair sorting. The one-work-group optimization remains an open topic for future work.

Both implementations use an 8-bit radix, resulting in 256 bins per stage.

## Open Questions

The following areas remain open for future investigation and development:

**Radix bit width:** The current implementations use an 8-bit radix (256 bins). Exploring support for other radix widths (e.g., 4 or 6 bits) could improve performance for different key distributions and hardware characteristics. Larger radix widths reduce the number of stages but increase SLM and register file usage. 

**Single work-group kernel template:** The one-work-group optimization is currently embedded within the dispatch logic for ESIMD. Providing a separate kernel template API specifically for single work-group sorting would enable explicit control and specialization for small-data use cases.

**Large input support:** The current implementations use bit masks that limit input sizes to 2^30 elements (approximately 1 billion elements for 4-byte keys). Extending support to larger inputs would require addressing the mask limitations in the offset calculations and histogram data structures.

**Work-group size flexibility:** The SYCL implementation currently supports work-group sizes of 512 and 1024. Supporting more work-group size configurations (e.g. multiples of 128) would enable more flexible tuning for different hardware and problem sizes.

