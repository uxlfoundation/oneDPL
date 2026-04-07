# `device_vector` for oneDPL

## Introduction

This RFC proposes adding a `device_vector` container to oneDPL that provides
a `std::vector`-like interface for managing device memory. A `device_vector`
is a widely expected companion to parallel/device algorithm libraries, giving
users a familiar, RAII-managed container for data that lives on an accelerator.

### Motivation

- **Migration from CUDA/Thrust** - Thrust's `device_vector` is heavily used
  in CUDA codebases. Providing an equivalent in oneDPL lowers the barrier
  for porting to SYCL backends. SYCLomatic already generates code targeting
  a `dpct::device_vector` compatibility shim, and having an official oneDPL
  type would give that migration a stable target, in a repository which is
  actively maintained.
- **Ease of use** - Users currently must manually manage USM allocations or
  SYCL buffers and pair them with raw pointers or iterators. A
  `device_vector` encapsulates allocation, sizing, and lifetime in a
  single object and integrates directly with oneDPL algorithms.

## Comparison of Existing Implementations

| Implementation | Source |
|---|---|
| **Thrust** (`thrust::device_vector`) | [NVIDIA/cccl - device_vector.h](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/device_vector.h) |
| **SYCLomatic** (`dpct::device_vector`) | [SYCLomatic - vector.h](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/clang/runtime/dpct-rt/include/dpct/dpl_extras/vector.h) |
| **Distributed Ranges** (`dr::sp::device_vector`) | [distributed-ranges - device_vector.hpp](https://github.com/oneapi-src/distributed-ranges/blob/main/include/dr/sp/device_vector.hpp) |
| **Boost.Compute** (`boost::compute::vector`) | [boostorg/compute - vector.hpp](https://github.com/boostorg/compute/blob/master/include/boost/compute/container/vector.hpp) |

### 1. How They Differ

| Aspect | Thrust | SYCLomatic | Distributed Ranges | Boost.Compute |
|---|---|---|---|---|
| **Language/Runtime** | CUDA | SYCL (via DPC++) | SYCL (via DPC++) | OpenCL |
| **Inheritance** | Inherits `detail::vector_base<T,Alloc>` (all logic in base) | Standalone class (no inheritance) | Inherits `dr::sp::vector<T,Alloc>` (thin wrapper) | Standalone class |
| **Default Allocator** | `thrust::device_allocator<T>` (CUDA `cudaMalloc`) | USM: `sycl::usm_allocator<T, shared>` / Buffer: `__buffer_allocator<T>` | None (must be specified; typically `device_allocator<T>`) | `buffer_allocator<T>` (OpenCL buffers) |
| **Memory Model** | **Device memory** -- data lives on device; host access triggers explicit transfers | **Shared/managed memory** -- runtime manages data placement and host-device migration (both USM shared mode and buffer mode) | **Device memory** -- data lives on device; host access triggers explicit transfers | **Device memory** -- data lives in OpenCL buffer; host access via buffer map/unmap |
| **Backing Mechanism** | CUDA device memory (`cudaMalloc`) | USM `sycl::usm::alloc::shared` OR SYCL buffer/accessor (compile-time `#ifdef DPCT_USM_LEVEL_NONE`) | USM device (`sycl::malloc_device`) | OpenCL `cl::Buffer` |
| **Host Element Access** | Via `device_reference` proxy (explicit device-to-host copy) | Via `device_reference` proxy (runtime-managed migration) | Via `device_ref` proxy (explicit `queue.memcpy().wait()`) | Via `buffer_value<T>` proxy (OpenCL buffer read/write commands) |
| **std::vector Interop** | Copy constructors from/to `std::vector` | Copy/move + implicit `operator std::vector()` | No direct interop | No direct interop |
| **Multi-device** | No | No | Yes (`rank()` tracks owning device) | No |
| **Queue Association** | Implicit (CUDA stream) | Global default queue | Global default queue | Explicit `command_queue` parameter on constructors and operations |
| **Backend Dispatch** | Tag-based (`device_system_tag`) | Execution policy-based (oneDPL) | Direct SYCL calls | Direct OpenCL calls |
| **Uninitialized Construction** | `default_init_t`, `no_init_t` tags | Not supported | Not supported | Not supported |

### 2. Key Helper Types

#### Proxy Reference

Every implementation needs a proxy reference type to mediate host-side
access to elements that live in device memory. Thrust provides the most
full-featured version: `device_reference<T>` inherits a CRTP base and
supports all compound assignment and increment/decrement operators, making
it behave as close to a real `T&` as possible. SYCLomatic takes a similar
approach with its own `device_reference<T>`. Distributed Ranges goes
minimal -- `device_ref<T>` only provides `operator T()` and `operator=`,
with no compound assignment, and constrains `T` to trivially copyable
types. Boost.Compute's `buffer_value<T>` is comparable, proxying reads
and writes through OpenCL buffer commands.

#### Device Pointer

A device pointer wraps a raw device-side `T*` and provides pointer
semantics on the host. Thrust's `device_ptr<T>` inherits a CRTP pointer
base, carries a `device_system_tag` for backend dispatch, and offers a
`device_pointer_cast()` factory. SYCLomatic has a standalone
`device_pointer<T>`. Distributed Ranges uses `device_ptr<T>` which wraps
a raw pointer with a `get_raw_pointer()` accessor, again constrained to
trivially copyable types. Boost.Compute has no separate device pointer
type -- its `buffer_iterator` fills both roles.

#### Device Iterator

The device iterator is what algorithms operate on. In Thrust and
Distributed Ranges, the device pointer *is* the iterator -- `device_ptr`
models random access iterator directly, so no separate type is needed.
SYCLomatic introduces a distinct `device_iterator<T>` class. Boost.Compute
uses `buffer_iterator<T>`, a random access iterator that wraps a buffer
plus an index offset.

## Proposal

*This section is intentionally left as a rough skeleton for further design
discussion.*

## High Level Decisions

- **Use device memory as baseline, copy to/from host on demand when required**
   This matches semantics of all pre-existing implementations other than SYCLomatic
   where the runtime handles where memory lives. Shared memory has significantly
   worse performance than device memory, and if users want those semantics, they
   can directly use usm shared memory or sycl buffers. Considering the existing
   semantics of other offerings, let us define our device_vector to match; data
   lives on the device and can be accessed on host slowly.

- **Use USM, no support for buffer-backed device_vector**
  As mentioned above buffers provide an alternative but similar semantic. As learned
  in the experience with SYCLomatic, offering device_vector functionality with
  sycl buffer backing is awkward and breaks our decision to have the data live
  on the device. If users want a device_vector, they need USM support.

- **Users optionally provide a queue on construction**
  Users provide a queue or device for their device_vector memory to be associated
  with. If nothing is provided, a global default queue is used.  This matches
  how oneDPL device policies work. Among existing implementations,
  Boost.Compute's explicit `command_queue` parameter on constructors and
  operations is the closest precedent for this model.

- **Type T should only require device copyability**
  We should not need anything except device copyability (for copy to and from
  the device).

- **We don't need a tag system for dispatch to specific hardware**
  Execution policies dictate where algorithms are run. We don't intend to provide other flavors of vector / iterator which would have different tags, so this doesn't make much sense.

- **device_pointer should be device copyable and indirectly device accessible**
  The intent is for these to be directly usable in kernels / oneDPL algorithms so this is required.

- **No custom allocator template parameter; use `sycl::malloc_device` directly**
  The SYCL 2020 spec intentionally excludes `sycl::usm_allocator` for device
  memory because the C++ `Allocator` named requirement mandates host-accessible
  memory, which device USM is not. A device memory allocator cannot satisfy
  `Allocator`, so a custom allocator template parameter in the `std::vector`
  sense is not meaningful here. We use `sycl::malloc_device` / `sycl::free`
  directly, keeping the implementation simpler.

### Strawman API

```cpp
namespace oneapi::dpl::experimental {

template <typename T>
class device_vector {
public:
    // Standard container typedefs
    using value_type      = T;
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

} // namespace oneapi::dpl::experimental
```

### Usage Examples

```cpp
#include <oneapi/dpl/device_vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

// Basic construction and algorithm use
sycl::queue q;
oneapi::dpl::experimental::device_vector<float> d_vec(1024, q);  // 1024 elements on q's device

// Fill from host data
std::vector<float> host_data(1024, 3.14f);
oneapi::dpl::experimental::device_vector<float> d_vec2(host_data.begin(), host_data.end(), q);

// Use with oneDPL algorithms -- iterators work directly
auto policy = oneapi::dpl::execution::make_device_policy(q);
std::sort(policy, d_vec2.begin(), d_vec2.end());

// Host-side element access via proxy reference (synchronous, slow)
float val = d_vec2[0];       // device-to-host transfer
d_vec2[0] = 42.0f;           // host-to-device transfer

// Extract raw pointer for use in SYCL kernels
float* raw = d_vec2.data().get();  // or similar accessor
q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
    raw[i] *= 2.0f;
}).wait();

// Copy back to host
std::vector<float> result = static_cast<std::vector<float>>(d_vec2);
```

### Helper Types

A `device_vector` requires several supporting types (see comparison above):

- **`device_pointer<T>`** -- wraps a raw device pointer; models random
  access iterator; dereference returns `device_reference<T>`.
- **`device_reference<T>`** -- proxy reference for host-side element
  access; reads/writes trigger synchronous `memcpy` on the host path,
  direct dereference on the device path.

## Open Questions

- **Should we try to support multiple devices?**
  Distributed ranges work has been halted, but is this an important use case to
  preserve, or unnecessary complication for users?  We could always add another
  type distributed_device_vector in the future if it seems necessary.

- **Do we gain anything from having a separate device_iterator and device_pointer,
   or can we just only implement device_pointer?**