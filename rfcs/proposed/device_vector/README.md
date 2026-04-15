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
### 1. How They Differ

| Aspect | Thrust | SYCLomatic | Distributed Ranges | Proposed (oneDPL) |
|---|---|---|---|---|
| **Default Allocator** | `thrust::device_allocator<T>` (CUDA `cudaMalloc`) | USM: `sycl::usm_allocator<T, shared>` / Buffer: `__buffer_allocator<T>` | None (must be specified; typically `device_allocator<T>`) | N/A; Always uses `sycl::malloc_device` directly |
| **Memory Model** | **Device memory** via `cudaMalloc`; host access triggers explicit transfers | **Shared memory** via USM shared or SYCL buffer/accessor; runtime manages placement | **Device memory** via `sycl::malloc_device`; host access triggers explicit transfers | **Device memory** via `sycl::malloc_device`; host access triggers explicit transfers |
| **Host Element Access** | Via `device_reference` proxy (explicit device-to-host copy) | Via `device_reference` proxy (runtime-managed migration) | Via `device_ref` proxy (explicit `queue.memcpy().wait()`) | Via `device_reference` proxy (explicit device-to-host copy) |
| **std::vector Interop** | Copy constructors from/to `std::vector` | Copy/move + implicit `operator std::vector()` | No direct interop | Explicit constructor + `operator std::vector()` |
| **Multi-device** | No | No | Yes (`rank()` tracks owning device) | see [open question](#open-questions) |
| **Queue Association** | Implicit (CUDA stream) | Global default queue | Global default queue | Explicit `sycl::queue` parameter on constructors (see [open question](#open-questions)) |
| **Uninitialized Construction** | `default_init_t`, `no_init_t` tags | Not supported | Not supported | see [open question](#open-questions) |

### 2. Boost.Compute

[Boost.Compute](https://github.com/boostorg/compute/blob/master/include/boost/compute/container/vector.hpp)
(`boost::compute::vector`) is another variant, which is further removed from the rest, it is a OpenCL-based device vector built on `cl::Buffer`.
Its most relevant design choice is **explicit queue association**: constructors and
operations accept a `command_queue` parameter, making the queue relationship clear
rather than relying on a global default. This is the closest existing precedent
for our proposed explicit `sycl::queue` constructor parameter (see
[open question](#open-questions) on queue association).

## Proposal

The proposal consists of the high-level design decisions below and an API
skeleton that mirrors `std::vector` where applicable, adapted for device
memory semantics.


### High Level Decisions

- **Use device memory as baseline, copy to/from host on demand when required**
   This matches semantics of all pre-existing implementations other than SYCLomatic
   where the runtime handles where memory lives. Shared memory has significantly
   worse performance than device memory, and if users want those semantics, they
   can directly use usm shared memory or sycl buffers.

- **Use USM, no support for buffer-backed device_vector**
  As mentioned above, buffers provide an alternative but similar semantic to
  `device_vector`. We learned from the SYCLomatic implementation that offering
  `device_vector` functionality with sycl buffer backing is awkward and breaks
  our decision to have the data live on the device. If users want a device_vector,
  they need USM support.

- **Type T should only require device copyability**
  We should not need anything except device copyability (for copy to and from
  the device).

- **We don't need a tag system for dispatch to specific hardware**
  Execution policies dictate where algorithms are run. We don't intend to
  provide other flavors of vector / iterator which would have different tags,
  which would be required to dispatch based upon tag.

- **device_pointer should be [device copyable](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable), indirectly device accessible and usable with good performance on the device**
  The intent is for these to be directly usable in sycl kernels and oneDPL algorithms.
  Ideally, no hard coded support will be required for `device_vector` within oneDPL's
  input processing code. The indirectly device accessible trait should provide everything
  we need here, including for it to be composable with custom iterator adapters.

- **`device_reference` supports all compound assignment and increment/decrement operators**
  Following Thrust's convention, `device_reference<T>` will support all compound
  assignment operators (`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`)
  and increment/decrement (`++`, `--`). This makes host-side element access behave
  as close to a real `T&` as possible and eases migration from Thrust codebases
  where users expect expressions like `d_vec[i] += 1` to work. Each such operation
  implies a synchronous round-trip (read-modify-write) to device memory.

- **No custom allocator template parameter; use `sycl::malloc_device` directly**
  The SYCL 2020 spec intentionally excludes `sycl::usm_allocator` for device
  memory because the C++ `Allocator` named requirement mandates host-accessible
  memory, which device USM is not. A device memory allocator cannot satisfy
  `Allocator`, so a custom allocator template parameter in the `std::vector`
  sense is not meaningful here. We use `sycl::malloc_device` / `sycl::free`
  directly, keeping the implementation simpler.

### API Skeleton

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
    // All constructors accept an optional sycl::queue argument currently,
    // but this is an open question.
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
    device_vector(const device_vector&); // also copies the queue
    device_vector(device_vector&&) noexcept;
    device_vector& operator=(const device_vector&);
    device_vector& operator=(device_vector&&) noexcept;
    ~device_vector();

    // Interop with std::vector
    explicit device_vector(const std::vector<T>&,
                           sycl::queue q = /* default queue */);
    explicit operator std::vector<T>() const;

    // Element access (proxy references, implies host-device transfer)
    reference       operator[](size_type pos);
    const_reference operator[](size_type pos) const;
    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;
    pointer         data() noexcept;
    const_pointer   data() const noexcept;

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
    void assign(size_type count, const T& value);
    template <typename InputIt>
    void assign(InputIt first, InputIt last);
    void assign(std::initializer_list<T> ilist);
    void clear() noexcept;
    void push_back(const T& value);
    void pop_back();
    iterator insert(const_iterator pos, const T& value);
    iterator insert(const_iterator pos, size_type count, const T& value);
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last);
    iterator erase(const_iterator pos);
    iterator erase(const_iterator first, const_iterator last);
    void resize(size_type count);
    void resize(size_type count, const T& value);
    void swap(device_vector& other) noexcept;
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

// Fill from host data (interop constructor)
std::vector<float> host_data(1024, 3.14f);
oneapi::dpl::experimental::device_vector<float> d_vec2(host_data, q);

// Use with oneDPL algorithms -- iterators work directly
auto policy = oneapi::dpl::execution::make_device_policy(q);
std::sort(policy, d_vec2.begin(), d_vec2.end());

// Host-side element access via proxy reference (synchronous, slow)
float val = d_vec2[0];       // device-to-host transfer
d_vec2[0] = 42.0f;           // host-to-device transfer

// Extract device pointer for use in SYCL kernels
oneapi::dpl::experimental::device_pointer<float> d_ptr = d_vec2.data();

// device_pointer or possibly device_iterator (see open question)
auto d_iter = d_vec.begin();
q.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> i) {
    // Device-side side access (direct USM pointer dereference, fast)
    d_ptr[i] *= 2.0f;
    d_iter[i] += 3.0f;
}).wait();

// Copy back to host
std::vector<float> result = static_cast<std::vector<float>>(d_vec2);

// Read the element at d_vec[5] on the host
d_iter += 5;
std::cout << *d_iter;
```

### Helper Types

A `device_vector` requires two supporting types:

- **`device_pointer<T>`** -- wraps a raw device pointer; models random
  access iterator; dereference returns `device_reference<T>`.
- **`device_reference<T>`** -- proxy reference for host-side element
  access; reads/writes trigger synchronous `memcpy` on the host path,
  direct dereference on the device path. Existing implementations range
  from full-featured (Thrust: all compound assignment and
  increment/decrement operators) to minimal (Distributed Ranges: only
  `operator T()` and `operator=`). Our proposal follows Thrust's
  full-featured approach (see design decisions above).

## Open Questions

- **Should we try to support multiple devices?**
  Distributed ranges work has been halted, but is this an important use case to
  preserve, or unnecessary complication for users?  We could always add another
  type to support such use cases in the future.

- **Queue association model and its impact on device_pointer**
  We want `device_pointer` to be device copyable so it can be used directly
  in kernels and oneDPL algorithms. However, host-side dereference (returning
  a proxy reference that does `memcpy`) requires a `sycl::queue`, and
  `sycl::queue` is not device copyable. This creates a tension with several
  possible resolutions, here are the leading 2:

  1. **Store a raw `sycl::queue*` in device_pointer.** `device_pointer`
     holds `T*` + `sycl::queue*` -- a struct of two raw pointers is trivially
     copyable and therefore device copyable. On the host, the queue pointer
     is dereferenced to obtain the queue for `memcpy`. On the device, the
     queue pointer is unused dead bits. This preserves both device copyability
     and host-side dereference with the correct queue. The queue must
     outlive the pointer, but this is fine because it lives in the vector
     the same as the memory itself, which will be freed if the vector leaves
     scope.
  2. **Use a global default queue only.** `device_vector` does not accept a
     queue on construction. `device_pointer` holds only a raw `T*`, is device
     copyable, and host-side dereference uses the default queue. Simple, but
     limits users to a single device/queue.

  Among existing implementations, Boost.Compute's explicit `command_queue`
  parameter on constructors is the closest precedent for queue association.
  CUDA-based Thrust avoids the problem entirely since `cudaMemcpy` is a
  global function that doesn't require a queue object.

- **Should we support uninitialized / default-initialized construction?**
  Thrust provides `default_init_t` and `no_init_t` tags that let users skip
  value-initialization when constructing a `device_vector`.  No other existing
  implementation supports this. It fits nicely with uninitialized_* APIs.
  Should we support similar tags?

- **Should we use device_pointer as the device iterator?**
  It seems there is no use case for a separate device_iterator, but it's
  worth considering.
