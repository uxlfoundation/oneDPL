# `device_vector` for oneDPL

## Introduction

This RFC proposes adding a `device_vector` container to oneDPL that provides
a `std::vector`-like interface for managing device memory. A `device_vector`
is a widely expected companion to parallel/device algorithm libraries, giving
users a familiar, RAII-managed container for data that lives on an accelerator.

### Motivation

- **Migration from CUDA/Thrust.** Thrust's `device_vector` is heavily used
  in CUDA codebases. Providing an equivalent in oneDPL lowers the barrier
  for porting to SYCL backends. SYCLomatic already generates code targeting
  a `dpct::device_vector` compatibility shim, and having an official oneDPL
  type would give that migration a stable target.
- **Ergonomics.** Users currently must manually manage USM allocations or
  SYCL buffers and pair them with raw pointers or iterators. A
  `device_vector` encapsulates allocation, sizing, and lifetime in a
  single object and integrates directly with oneDPL algorithms.
- **Ecosystem alignment.** Multiple SYCL-adjacent projects have already
  implemented their own `device_vector` (see *Existing Implementations*
  below). A single, well-specified type in oneDPL would reduce
  fragmentation.

### Existing Implementations

| Implementation | Source |
|---|---|
| **Thrust** (`thrust::device_vector`) | [NVIDIA/cccl - device_vector.h](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/device_vector.h) |
| **SYCLomatic** (`dpct::device_vector`) | [SYCLomatic - vector.h](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/clang/runtime/dpct-rt/include/dpct/dpl_extras/vector.h) |
| **Distributed Ranges** (`dr::sp::device_vector`) | [distributed-ranges - device_vector.hpp](https://github.com/oneapi-src/distributed-ranges/blob/main/include/dr/sp/device_vector.hpp) |
| **Boost.Compute** (`boost::compute::vector`) | [boostorg/compute - vector.hpp](https://github.com/boostorg/compute/blob/master/include/boost/compute/container/vector.hpp) |

## Proposal

*This section is intentionally left as a rough skeleton for further design
discussion.*

### Strawman API

```cpp
namespace oneapi::dpl {

template <typename T, typename Allocator = /* see below */>
class device_vector {
public:
    // Standard container typedefs
    using value_type      = T;
    using allocator_type  = Allocator;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = device_reference<T>;      // proxy
    using const_reference = device_reference<const T>; // proxy
    using pointer         = device_pointer<T>;
    using const_pointer   = device_pointer<const T>;
    using iterator        = /* device-aware random access iterator */;
    using const_iterator  = /* const version */;

    // Constructors / assignment / destructor -- mirrors std::vector
    // All constructors accept an optional sycl::queue argument
    // (defaulted, e.g. to a global default queue).
    device_vector(sycl::queue q = /* default queue */);
    explicit device_vector(size_type count,
                           sycl::queue q = /* default queue */);
    device_vector(size_type count, const T& value,
                  sycl::queue q = /* default queue */);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last,
                  sycl::queue q = /* default queue */);
    device_vector(std::initializer_list<T> init,
                  sycl::queue q = /* default queue */);
    device_vector(const device_vector&);
    device_vector(device_vector&&) noexcept;
    ~device_vector();

    // Interop with std::vector
    explicit device_vector(const std::vector<T>&);
    explicit operator std::vector<T>() const;

    // Element access (proxy references, implies host-device transfer)
    reference       operator[](size_type pos);
    const_reference operator[](size_type pos) const;
    reference       front();
    reference       back();
    pointer         data() noexcept;

    // Iterators
    iterator begin() noexcept;
    iterator end()   noexcept;
    // + const/reverse variants

    // Queue access
    sycl::queue get_queue() const noexcept;

    // Capacity
    bool      empty()    const noexcept;
    size_type size()     const noexcept;
    size_type capacity() const noexcept;
    void reserve(size_type new_cap);
    void shrink_to_fit();

    // Modifiers
    void clear() noexcept;
    void push_back(const T& value);
    void pop_back();
    void resize(size_type count);
    void resize(size_type count, const T& value);
    void swap(device_vector& other) noexcept;
    // insert, erase, assign ...
};

} // namespace oneapi::dpl
```

### Helper Types

A `device_vector` requires several supporting types (see comparison below):

- **`device_pointer<T>`** -- wraps a raw device pointer; models random
  access iterator; dereference returns `device_reference<T>`.
- **`device_reference<T>`** -- proxy reference for host-side element
  access; reads/writes trigger synchronous `memcpy` on the host path,
  direct dereference on the device path.
- **`device_allocator<T>`** -- allocator using USM device memory
  (or the chosen memory model).

## Comparison of Existing Implementations

### 1. How They Differ

| Aspect | Thrust | SYCLomatic | Distributed Ranges | Boost.Compute |
|---|---|---|---|---|
| **Language/Runtime** | CUDA | SYCL (via DPC++) | SYCL (via DPC++) | OpenCL |
| **Inheritance** | Inherits `detail::vector_base<T,Alloc>` (all logic in base) | Standalone class (no inheritance) | Inherits `dr::sp::vector<T,Alloc>` (thin wrapper) | Standalone class |
| **Default Allocator** | `thrust::device_allocator<T>` (CUDA `cudaMalloc`) | USM: `sycl::usm_allocator<T, shared>` / Buffer: `__buffer_allocator<T>` | None (must be specified; typically `device_allocator<T>`) | `buffer_allocator<T>` (OpenCL buffers) |
| **Memory Model** | **Device memory** -- data lives on device; host access triggers explicit transfers | **Shared/managed memory** -- runtime manages data placement and host-device migration (both USM shared mode and buffer mode) | **Device memory** -- data lives on device; host access triggers explicit transfers | **Device memory** -- data lives in OpenCL buffer; host access via buffer map/unmap |
| **Backing Mechanism** | CUDA device memory (`cudaMalloc`) | USM `sycl::usm::alloc::shared` OR SYCL buffer/accessor (compile-time `#ifdef DPCT_USM_LEVEL_NONE`) | USM device (`sycl::malloc_device`) | OpenCL `cl::Buffer` |
| **Host Element Access** | Via `device_reference` proxy (explicit device-to-host copy) | Via `device_reference` proxy (runtime-managed migration) | Via `device_ref` proxy (explicit `queue.memcpy().wait()`) | Via `buffer_value<T>` proxy (OpenCL buffer read/write commands) |
| **Custom Allocator** | Yes (template parameter) | USM mode: yes / Buffer mode: no | Yes (template parameter) | Yes (template parameter) |
| **std::vector Interop** | Copy constructors from/to `std::vector` | Copy/move + implicit `operator std::vector()` | No direct interop | No direct interop |
| **Multi-device** | No | No | Yes (`rank()` tracks owning device) | No |
| **Queue Association** | Implicit (CUDA stream) | Global default queue | Global default queue | Explicit `command_queue` parameter on constructors and operations |
| **Backend Dispatch** | Tag-based (`device_system_tag`) | Execution policy-based (oneDPL) | Direct SYCL calls | Direct OpenCL calls |
| **Uninitialized Construction** | `default_init_t`, `no_init_t` tags | Not supported | Not supported | Not supported |
| **Buffer Access** | N/A | `get_buffer()` in buffer mode | N/A | `get_buffer()` returns underlying `cl::Buffer` |
| **Allocator Propagation** | Full (`propagate_on_*` traits) | Full (USM mode) | Basic (via base class) | Basic |

### 2. Required Helper Classes

| Helper Type | Thrust | SYCLomatic | Distributed Ranges | Boost.Compute |
|---|---|---|---|---|
| **Proxy reference** | `device_reference<T>` -- inherits CRTP `thrust::reference<T, device_ptr<T>, device_reference<T>>`. Provides all compound assignment, increment/decrement operators. | `device_reference<T>` -- external class. Used as `reference` typedef. | `device_ref<T>` -- constrained to `trivially_copyable` types. Minimal: only `operator T()`, `operator=`. No compound assignment. | `detail::buffer_value<T>` -- proxy that reads/writes via OpenCL buffer commands. |
| **Device pointer** | `device_ptr<T>` -- inherits CRTP `thrust::pointer<T, device_system_tag, ...>`. Wraps raw `T*`. `device_pointer_cast()` factory. Carries `device_system_tag` for backend dispatch. | `device_pointer<T>` -- external class. Used as `pointer` typedef. | `device_ptr<T>` -- random access iterator over raw `T*`. `get_raw_pointer()` accessor. Constrained to `trivially_copyable`. | N/A (uses `buffer_iterator<T>` instead) |
| **Device iterator** | Pointer *is* the iterator (via CRTP pointer base) | `device_iterator<T>` -- separate class | Pointer *is* the iterator (`device_ptr<T>` models random access iterator) | `buffer_iterator<T>` -- random access iterator wrapping buffer + index |
| **Allocator** | `device_allocator<T>` (wraps `cudaMalloc`/`cudaFree`) | USM: `sycl::usm_allocator<T, shared>` / Buffer: `__buffer_allocator<T>` | `device_allocator<T, Alignment>` (wraps `sycl::malloc_device`) | `buffer_allocator<T>` (allocates OpenCL `cl::Buffer` objects) |
| **Allocator traits helper** | None (uses standard `allocator_traits`) | `device_allocator_traits<Alloc>` -- detects `__has_construct`/`__has_destroy`, dispatches to serial or parallel construction | None (uses standard `allocator_traits`) | None |
| **System/dispatch tag** | `device_system_tag` | N/A (uses execution policies) | N/A | N/A (always OpenCL) |
| **Additional** | `detail::vector_base<T,Alloc>` (shared with `host_vector`) | `dpct::internal::is_iterator` (SFINAE trait) | `dr::sp::vector<T,Alloc>` (base class, reusable for other vector types) | Explicit `command_queue` stored as member; `get_buffer()` for raw `cl::Buffer` access |

### Summary of Key Differences

**Thrust** is the most mature and feature-rich: full CRTP hierarchy for
pointer/reference types, tag-based dispatch, rich operator set on proxy
references, and a shared `vector_base` between `device_vector` and
`host_vector`. The trade-off is complexity and CUDA-only semantics.

**SYCLomatic** is the most pragmatic for migration: it supports *both* USM
and buffer backends via compile-time switch, provides `std::vector`
interop (including implicit conversion), and uses oneDPL algorithms
internally for bulk construction/destruction. Unlike Thrust and DR,
SYCLomatic uses a *shared memory* model in both modes -- USM shared
memory where the runtime migrates data between host and device, or SYCL
buffers where the runtime manages data placement via the accessor model.
This means host-side element access does not require an explicit transfer
in the proxy reference; instead, the runtime handles migration
transparently. The trade-off is less predictable performance and
potential migration overhead. SYCLomatic also has a more complex internal
helper (`device_allocator_traits`) and does not support custom allocators
in buffer mode.

**Distributed Ranges** is the most minimal: `device_vector` is a thin
wrapper (~30 lines) over a generic `vector` base, adding only a `rank()`
for multi-device awareness. The `device_ref` is C++20-constrained
(`trivially_copyable`) and provides only basic read/write -- no compound
assignment operators. This simplicity comes at the cost of fewer
convenience features.

**Boost.Compute** is the closest OpenCL analog. Named simply `vector`
(in the `boost::compute` namespace), it stores data in an OpenCL
`cl::Buffer` and requires an explicit `command_queue` for construction
and most operations. Its iterator (`buffer_iterator`) and proxy
reference (`buffer_value`) operate via OpenCL buffer read/write commands.
Notably, Boost.Compute's explicit queue passing on constructors is the
closest precedent to the queue association model proposed here for
oneDPL. It has no separate device pointer type -- the buffer iterator
serves both roles.

## Open Questions

- **Device memory vs. shared memory model?** Thrust and Distributed
  Ranges both use *device memory* -- data lives on the device, and
  host-side access triggers explicit transfers (via proxy references).
  SYCLomatic takes the opposite approach: both its USM mode
  (`sycl::usm::alloc::shared`) and its buffer mode use *shared memory*
  where the runtime manages data placement and migration between host
  and device transparently. The device memory model gives users explicit
  control over data movement and avoids runtime migration overhead, but
  requires proxy types for host access. The shared memory model is
  simpler to use but can suffer from unpredictable migration costs and
  may not perform well on all hardware. Which model should oneDPL's
  `device_vector` adopt? Should both be supported via allocator
  selection? *(Address at proposed stage.)*

- **How should queue association work?** The proposed direction is a
  defaulted `sycl::queue` parameter on constructors (falling back to a
  global default queue) and a `get_queue()` member function to retrieve
  it. This balances ease of use (no queue needed for simple cases) with
  flexibility (explicit queue for multi-device or multi-queue scenarios).
  Open sub-questions: What should the default queue be? Should the queue
  be changeable after construction? Should operations like `resize` and
  `push_back` use the stored queue for any required transfers?
  *(Address at proposed stage.)*

- **Should there be a buffer-backed mode?** SYCLomatic's dual-mode
  approach adds complexity. If the oneDPL `device_vector` is USM-only,
  it simplifies the design but limits use with buffer-based code.
  *(Address at proposed stage.)*

- **What constraints should be placed on `T`?** DR constrains its
  `device_ref` and `device_ptr` to `std::is_trivially_copyable_v<T>`.
  Thrust does not constrain at the type level but CUDA's `cudaMemcpy`
  effectively requires it. For a SYCL-based `device_vector`, data
  transfers use `sycl::queue::memcpy` and elements must be usable on
  the device, so the SYCL *device copyability* rules are the relevant
  constraint (similar to but not identical to `trivially_copyable`).
  Should this be enforced at compile time (like DR) or left as a
  precondition? *(Address at proposed stage.)*

- **Should `device_reference` support compound assignment operators?**
  Thrust provides `+=`, `-=`, etc. on its proxy reference. DR's
  `device_ref` does not. Each compound op implies a read-modify-write
  round trip. *(Address at proposed stage.)*
