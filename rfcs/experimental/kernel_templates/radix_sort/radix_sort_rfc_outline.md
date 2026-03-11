# SYCL/ESIMD Radix Sort Kernel Template

## Introduction

High-performance GPU radix sort for oneDPL kernel templates. Provides two implementations: ESIMD (optimized for Intel GPUs) and SYCL (portable across vendors). Uses Onesweep radix sort with decoupled lookback to address the global memory bandwidth bottleneck through reduced memory traffic.

**Motivation:** Traditional GPU radix sort algorithms require multiple passes through data per radix stage (histogram + reorder), resulting in ~3n memory accesses. This implementation reduces memory traffic to ~2n per stage using the Onesweep approach, significantly improving performance on bandwidth-constrained GPUs.

## Proposal

### Algorithm Overview

**Onesweep Radix Sort:**
- Single-pass per radix stage (vs traditional histogram + reorder phases)
- Stage-by-stage processing (e.g., 8-bit radix = 4 stages for 32-bit keys)
- Per-stage flow: load → rank locally → scan histograms → synchronize work-groups → reorder to SLM → scatter to global

**Memory Traffic and Performance Benefits:**
- Traditional radix sort: Separate histogram pass + reorder pass per stage
  - Histogram: 1 read pass through data
  - Reorder: 1 read + 1 write pass
  - Total: ~3n memory accesses per radix iteration
- Onesweep approach:
  - Upfront global histogram + scan (one-time cost)
  - Per radix iteration: ~2n + lookback traffic
  - No separate histogram pass needed per stage
- Reduces computational complexity and memory bandwidth pressure
- Critical for GPU performance where global memory bandwidth is the primary bottleneck

**Decoupled Lookback:**
- Benefits: Work-groups process independently without device-wide barriers
- Mechanism: Each work-group looks back at previous work-groups' published partial histograms
- Enables continuous streaming execution without global synchronization points

### Safety and Forward Progress

**The Challenge:**
- Lookback requires work-group N to spin-wait for work-group N-1 to publish results
- Without forward progress guarantees, potential deadlock if scheduler doesn't ensure concurrent execution

**ESIMD Approach:**
- Relies on specific PVC hardware scheduling characteristics
- Hardware behavior ensures concurrent work-group execution

**SYCL Solution and Alternatives:**

*Forward Progress Extension (Current Implementation):*
- Uses `sycl::ext::oneapi::experimental::work_group_progress<concurrent, root_group>`
- Guarantees concurrent work-groups make forward progress even with spin-loops
- Only approach that guarantees hardware safety across different GPU schedulers

*Alternative: Atomic Counter Approach (Evaluated and Rejected):*
- Use global atomic counter to assign work-group IDs in execution order
- Each work-group increments counter to get its sequential ID
- Lookback uses this ordering instead of dispatch order
- Issues:
  - Requires empirical testing for each specific architecture
  - Does not provide formal guarantee of safety
  - During evaluation, hardware hangs were examined on certain hardware
  - Relies on scheduler behavior, not portable

### API

**Primary Interface:**
- SYCL implementation: `oneapi::dpl::experimental::kt::gpu::radix_sort()`
- ESIMD variant: `oneapi::dpl::experimental::kt::gpu::esimd::radix_sort()`
- Takes `sycl::queue`, input/output ranges, kernel configuration
- Template parameters: ascending/descending order, radix bits, work-group configuration
- Returns `sycl::event` for async execution chaining
- Unified dispatch logic in `radix_sort_dispatchers.h` selects implementation

**Usage Example:**
```cpp
#include <oneapi/dpl/experimental/kt/radix_sort.h>

sycl::queue q;
std::vector<uint32_t> keys(1000000);
// ... initialize keys ...

sycl::buffer<uint32_t> keys_buf(keys.data(), keys.size());

// Sort keys in ascending order using 8-bit radix
auto event = oneapi::dpl::experimental::kt::gpu::radix_sort<
    oneapi::dpl::experimental::kt::order::ascending,
    /* radix_bits = */ 8
>(q, keys_buf);

event.wait();
```

### Dependencies and Platform Support

- Requires SYCL 2020 or later
- SYCL implementation requires `sycl::ext::oneapi::experimental::work_group_progress` extension for forward progress guarantees
- ESIMD implementation: Intel Data Center GPU Max Series (PVC) only
- SYCL implementation: Currently supported on PVC and BMG, with continued expected support for future Intel GPU architectures

### Implementation Status and Next Steps
- **ESIMD**: Complete onesweep multi-work-group implementation
  - One-work-group optimization for small inputs (≤ workgroup_size × data_per_work_item)
  - Supports keys-only and key-value pair sorting
- **SYCL**: Onesweep multi-work-group implementation
  - One-work-group optimization is open topic for future work
  - Supports keys-only and key-value pair sorting

## Open Questions
- Expanding radix bit width support beyond 8 bits
- Separate kernel template API for single work-group optimization
- Support for input sizes larger than 2^30 elements (current mask limitation)
- Extending work-group size support in SYCL implementation (currently fixed at 256)
