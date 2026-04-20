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
- **Real-world usage patterns** - A [detailed survey](usage_pattern_study.md)
  of `thrust::device_vector` (~2,520 files on GitHub), `dpct::device_vector`
  (~111 code results across ~18 repos), and alternative implementations
  (FAISS `DeviceVector`, RMM `device_uvector`) reveals consistent patterns.
  Key findings:

  1. **Construction + bulk transfer + raw pointer extraction** are the core
     operations across all domains. `device_vector` is primarily used as an
     RAII device memory manager and host-device data shuttle.
  2. **`begin()`/`end()` integration with parallel algorithms** is the
     second-most critical capability.
  3. **Some popular AI/ML projects** (FAISS, cuDF, cuML) have **moved away from 
     `thrust::device_vector`** due to unwanted value initialization, lack of
     stream parameters, and header bloat — then built alternatives that
     prioritize explicit async control and uninitialized allocation. Other HPC
     and ML projects remain heavy users.
  4. **Full `std::vector`-like modifiers** (`push_back`, `insert`, `erase`)
     are rarely used in real workloads.
  5. **Raw pointer extraction** (`data().get()` / `raw_pointer_cast`) is
     ubiquitous for passing to kernels and library APIs.
  6. **`device_vector` as class member** is a major pattern in HPC
     (simulation state, sparse matrix storage, MPI buffers).

## Comparison of Existing Implementations

| Implementation | Source |
|---|---|
| **Thrust** (`thrust::device_vector`) | [NVIDIA/cccl - device_vector.h](https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/device_vector.h) |
| **SYCLomatic** (`dpct::device_vector`) | [SYCLomatic - vector.h](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/clang/runtime/dpct-rt/include/dpct/dpl_extras/vector.h) |
| **Distributed Ranges** (`dr::sp::device_vector`) | [distributed-ranges - device_vector.hpp](https://github.com/oneapi-src/distributed-ranges/blob/main/include/dr/sp/device_vector.hpp) |
| **sycl-thrust** (`thrust::device_vector`) | [SparseBLAS/sycl-thrust - device_vector.h](https://github.com/SparseBLAS/sycl-thrust/blob/main/include/thrust/device_vector.h) |
### 1. How They Differ

| Aspect | Thrust | SYCLomatic | Distributed Ranges | sycl-thrust | Proposed (oneDPL) |
|---|---|---|---|---|---|
| **Default Allocator** | `thrust::device_allocator<T>` (CUDA `cudaMalloc`) | USM: `sycl::usm_allocator<T, shared>` / Buffer: `__buffer_allocator<T>` | None (must be specified; typically `device_allocator<T>`) | `device_allocator<T>` (`sycl::malloc_device`); supports alignment template parameter | N/A; Always uses `sycl::malloc_device` directly |
| **Memory Model** | **Device memory** via `cudaMalloc`; host access triggers explicit transfers | **Shared memory** via USM shared or SYCL buffer/accessor; runtime manages placement | **Device memory** via `sycl::malloc_device`; host access triggers explicit transfers | **Device memory** via `sycl::malloc_device`; explicit transfers | **Device memory** via `sycl::malloc_device`; host access triggers explicit transfers |
| **Host Element Access** | Via `device_reference` proxy (explicit device-to-host copy) | Via `device_reference` proxy (runtime-managed migration) | Via `device_ref` proxy (explicit `queue.memcpy().wait()`) | Via `device_reference` proxy (`__SYCL_DEVICE_ONLY__` bifurcation) | Via `device_reference` proxy (explicit device-to-host copy) |
| **std::vector Interop** | Copy constructors from/to `std::vector` | Copy/move + implicit `operator std::vector()` | No direct interop | Constructor from `std::vector` | Explicit constructor + `operator std::vector()` |
| **Queue Association** | Implicit (CUDA stream) | Global default queue | Global default queue | Allocator stores `device` + `context`; queue resolved at runtime via pointer introspection | see [open question](#open-questions) |
| **Uninitialized Construction** | `default_init_t`, `no_init_t` tags | Not supported | Not supported | Not supported | `no_init_t` tag for construction and resize |

### 2. sycl-thrust

[sycl-thrust](https://github.com/SparseBLAS/sycl-thrust) is a SYCL-native
reimplementation of Thrust's `device_vector` API from the SparseBLAS project.
It uses `sycl::malloc_device`, a `device_ptr<T>` iterator wrapping a raw
`T*`, and a `device_reference<T>` proxy with `__SYCL_DEVICE_ONLY__`
bifurcation.

**Device & context vs queue.** sycl-thrust's allocator stores
`sycl::device` + `sycl::context` rather than a `sycl::queue`. This is
sufficient for allocation (`sycl::malloc_device` only requires device and
context) and means the allocator doesn't tie the vector to a particular
queue. When host-side `device_reference` needs a queue for memcpy, it
resolves one at runtime via `sycl::get_pointer_device`, looping through a
set of contexts saved in a inline global vector of all available device contexts.
This configuration allows `device_ptr` and `device_ref` to be a single raw pointer
(8 bytes) rather than carrying a queue or anything else in addition. It optimizes
the hot path (device code), and requires extra steps to search for a context
and create a queue for each host side operation. See [open question on queue association](#open-questions)
for further discussion of this tradeoff.

**Aligned allocation.** sycl-thrust's `device_allocator<T, Alignment>`
supports `sycl::aligned_alloc_device` via a template parameter, allowing
users to request specific alignment (e.g. 128 or 256 bytes for coalesced
memory access) see [open question on aligned allocation](#open-questions).

### 3. Alternatives Built by Projects That Rejected `thrust::device_vector`

Two notable alternatives — FAISS's `DeviceVector<T>` and RAPIDS's
`rmm::device_uvector<T>` — were built by high-performance projects that
found `thrust::device_vector` insufficient. Both prioritize explicit stream
control, uninitialized allocation, and stripped-down interfaces over STL
compatibility. See the [detailed analysis](usage_pattern_study.md#alternatives-built-by-projects-that-rejected-thrustdevice_vector)
for full design breakdowns.

While these alternatives are a separate category, it seems to support the argument
to provide uninitialized creation and resizing. Also, association with a queue
can provide some explicit synchronization via in-order queues. However, adding
explicit queue parameters to each operation does not seem appropriate.

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
  implies a synchronous round-trip (read-modify-write) to device memory. All
  mutating operators (`operator=`, compound assignment, increment/decrement) are
  **const-qualified** because `device_reference` is a proxy: `const` applies to
  the handle, not the underlying device data. This is required by
  `std::indirectly_writable` in C++20 (see [Range Support](#range-support)).

- **Support `no_init` tag for uninitialized construction and resizing**
  A `no_init_t` tag type allows users to skip value-initialization when
  constructing or resizing a `device_vector`. This avoids the cost of a
  device memset/kernel launch for temporary buffers that will be immediately
  overwritten.

- **No custom allocator template parameter; use `sycl::malloc_device` directly**
  The SYCL 2020 spec intentionally excludes `sycl::usm_allocator` for device
  memory because the C++ `Allocator` named requirement mandates host-accessible
  memory, which device USM is not. A device memory allocator cannot satisfy
  `Allocator`, so a custom allocator template parameter in the `std::vector`
  sense is not meaningful here. We use `sycl::malloc_device` / `sycl::free`
  directly, keeping the implementation simpler.

### API Skeleton

This skeleton is provided as a sample, and cannot be finalized until some of the
open questions have been resolved.

```cpp
namespace oneapi::dpl::experimental {

// Tag type for uninitialized construction / resize
struct no_init_t { /* ... */ };
inline constexpr no_init_t no_init{};

// =========================================================================
// device_pointer<T>
// =========================================================================

template <typename T>
class device_pointer {
public:
    // Iterator traits
    using iterator_concept  = std::random_access_iterator_tag;
    using value_type        = std::remove_cv_t<T>;
    using difference_type   = std::ptrdiff_t;
    using reference         = device_reference<T>;

    device_pointer() = default;
    explicit device_pointer(T* ptr);

    // Raw pointer access
    T* get() const;

    // Dereference -- returns a proxy reference
    reference operator*() const;
    reference operator[](difference_type n) const;

    // Arithmetic
    device_pointer& operator++();
    device_pointer  operator++(int);
    device_pointer& operator--();
    device_pointer  operator--(int);
    device_pointer& operator+=(difference_type n);
    device_pointer& operator-=(difference_type n);
    friend device_pointer operator+(device_pointer p, difference_type n);
    friend device_pointer operator+(difference_type n, device_pointer p);
    friend device_pointer operator-(device_pointer p, difference_type n);
    friend difference_type operator-(device_pointer a, device_pointer b);

    // Comparison
    friend bool operator==(device_pointer a, device_pointer b);
    friend auto operator<=>(device_pointer a, device_pointer b);
};

// =========================================================================
// device_reference<T>
// =========================================================================

template <typename T>
class device_reference {
public:
    // Implicit conversion to T (device-to-host read)
    operator T() const;
    T read(sycl::queue q) const;              // explicit queue overload

    // Assignment (host-to-device write)
    const device_reference& operator=(const T& val) const;
    const device_reference& operator=(const device_reference&) const;
    void write(const T& val, sycl::queue q) const;  // explicit queue overload

    // Compound assignment -- each is a synchronous read-modify-write.
    const device_reference& operator+=(const T&) const;
    // etc.  -=, *=, /= ...

    // Increment / decrement
    const device_reference& operator++() const;
    T operator++(int) const;
    const device_reference& operator--() const;
    T operator--(int) const;

    // Swap
    friend void swap(const device_reference& a, const device_reference& b);

    // Address-of returns device_pointer
    device_pointer<T> operator&() const;
};

// =========================================================================
// device_vector<T>
// =========================================================================

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
    // Three overload sets:
    //   1. Queue overloads -- the queue is used only to determine the target
    //      device and context; it is not retained or used for synchronization.
    //   2. Context + device overloads -- direct specification.
    //   3. No-arg overloads -- selects sycl::device{sycl::gpu_selector_v}
    //      and creates a context from it. Throws if no GPU is available.

    // --- No-arg overloads (default device: gpu_selector_v) ---
    device_vector();
    explicit device_vector(size_type count);
    device_vector(size_type count, const T& value);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last);
    device_vector(std::initializer_list<T> init);

    // --- Queue overloads (queue used only to extract context + device) ---
    explicit device_vector(sycl::queue q);
    device_vector(size_type count, sycl::queue q);
    device_vector(size_type count, const T& value, sycl::queue q);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last, sycl::queue q);
    device_vector(std::initializer_list<T> init, sycl::queue q);

    // --- Context + device overloads ---
    device_vector(sycl::context ctx, sycl::device dev);
    explicit device_vector(size_type count,
                           sycl::context ctx, sycl::device dev);
    device_vector(size_type count, const T& value,
                  sycl::context ctx, sycl::device dev);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last,
                  sycl::context ctx, sycl::device dev);
    device_vector(std::initializer_list<T> init,
                  sycl::context ctx, sycl::device dev);

    // --- Copy / move / destructor ---
    device_vector(const device_vector&);
    device_vector(device_vector&&) noexcept;
    device_vector& operator=(const device_vector&);
    device_vector& operator=(device_vector&&) noexcept;
    ~device_vector();

    // Uninitialized construction -- allocates without memset/kernel launch
    explicit device_vector(size_type count, no_init_t);
    explicit device_vector(size_type count, no_init_t, sycl::queue q);
    explicit device_vector(size_type count, no_init_t,
                           sycl::context ctx, sycl::device dev);

    // Interop with std::vector
    explicit device_vector(const std::vector<T>&);
    explicit device_vector(const std::vector<T>&, sycl::queue q);
    explicit device_vector(const std::vector<T>&,
                           sycl::context ctx, sycl::device dev);
    explicit operator std::vector<T>() const;
    std::vector<T> to_vector(sycl::queue q) const;   // explicit queue

    // Element access (proxy references -- host-device transfer on host use)
    reference       operator[](size_type pos);
    const_reference operator[](size_type pos) const;
    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;
    pointer         data();
    const_pointer   data() const;

    // Iterators
    iterator begin();
    iterator end();
    // + const/reverse variants

    // Views
    device_view<T> view();
    device_view<const T> view() const;

    // Context / device access (queue is never stored)
    sycl::context get_context() const;
    sycl::device get_device() const;

    // Capacity
    bool      empty()    const;
    size_type size()     const;
    size_type capacity() const;
    void reserve(size_type new_cap);
    void shrink_to_fit();

    // Modifiers -- operations that transfer data between host and device
    // accept an optional sycl::queue. When provided, the transfer is
    // submitted to that queue, enabling explicit synchronization (e.g.
    // via an in-order queue shared with kernel submissions). When omitted,
    // an internal queue is created from the stored context + device.
    //
    // Context constraint: any user-provided queue must share the same
    // sycl::context as the vector's allocation (q.get_context() ==
    // get_context()). Passing a queue with a different context is
    // undefined behavior for USM operations; implementations throw
    // sycl::exception if a context mismatch is detected.
    void assign(size_type count, const T& value);
    template <typename InputIt>
    void assign(InputIt first, InputIt last);
    template <typename InputIt>
    void assign(InputIt first, InputIt last, sycl::queue q);
    void assign(std::initializer_list<T> ilist);
    void assign(std::initializer_list<T> ilist, sycl::queue q);
    void clear();
    void clear(sycl::queue q);
    void push_back(const T& value);
    void push_back(const T& value, sycl::queue q);
    void pop_back();
    void pop_back(sycl::queue q);
    iterator insert(const_iterator pos, const T& value);
    iterator insert(const_iterator pos, const T& value, sycl::queue q);
    iterator insert(const_iterator pos, size_type count, const T& value);
    iterator insert(const_iterator pos, size_type count, const T& value,
                    sycl::queue q);
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last);
    template <typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last,
                    sycl::queue q);
    iterator erase(const_iterator pos);
    iterator erase(const_iterator pos, sycl::queue q);
    iterator erase(const_iterator first, const_iterator last);
    iterator erase(const_iterator first, const_iterator last, sycl::queue q);
    void resize(size_type count);
    void resize(size_type count, sycl::queue q);
    void resize(size_type count, const T& value);
    void resize(size_type count, const T& value, sycl::queue q);
    void resize(size_type count, no_init_t);
    void resize(size_type count, no_init_t, sycl::queue q);
    void swap(device_vector& other);
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

// Extract raw pointer for use in SYCL kernels
float* d_ptr = d_vec2.data().get();

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

// Uninitialized construction -- no memset, useful for output buffers
oneapi::dpl::experimental::device_vector<float> d_output(1024, dpl::no_init, q);
std::transform(policy, d_vec2.begin(), d_vec2.end(), d_output.begin(),
               [](float x) { return x * 2.0f; });

// Uninitialized resize -- grow without zeroing new elements
output.resize(2048, dpl::no_init);

std::vector<float> transform_out = static_cast<std::vector<float>>(d_output);

```

### Helper Types

A `device_vector` requires two supporting types, both shown in the API
skeleton above:

- **`device_pointer<T>`** -- wraps a raw device pointer; models
  `std::random_access_iterator`; dereference returns `device_reference<T>`.
  Allows raw pointer extraction via `.get()`.
- **`device_reference<T>`** -- proxy reference for host-side element
  access; reads/writes trigger synchronous `memcpy` on the host path,
  direct dereference on the device path.
  Our proposal follows Thrust's full-featured approach where all compound
  assignment and increment/decrement operators are provided.

### Range Support

**Range support requires C++20.** The C++20 iterator concepts
(`std::random_access_iterator`), range concepts (`std::ranges::random_access_range`,
`std::ranges::sized_range`), and `std::basic_common_reference` are all required
to make `device_view` usable with oneDPL's productized range algorithms and
standard range adaptors. oneDPL's productized ranges API is C++20-only, so this
is a natural fit. The core `device_vector`, `device_pointer`, and
`device_reference` types work with C++17; range support is additive.

The primary range interface for `device_vector` is through `device_view<T>`,
a lightweight, device-copyable view. `device_vector` itself is not intended
to be used directly as a range. Host-side iteration through proxy references
is a synchronous memcpy per element, making host-policy range algorithms
over a `device_vector` impractical. Instead, users obtain a device copyable
`device_view` and pass it to oneDPL range algorithms with a device policy.

#### Requirements on `device_reference` and `device_pointer`

For `device_pointer<T>` to model `std::random_access_iterator` (and therefore
for `device_view` to model `std::ranges::random_access_range`), several
requirements must be met:

##### `basic_common_reference`

The proxy reference type must satisfy the `common_reference_with` requirements.
This requires specializations of `std::basic_common_reference` for
`device_reference<T>`.

**Important:** The common reference type must be a **value type** (e.g.
`std::remove_cv_t<T>`), not a reference type (e.g. `T&`). Because
`device_reference` is a proxy, converting `device_reference<T>&&` to `T&`
would create a dangling reference to a temporary. The `indirectly_readable`
concept checks `convertible_to<device_reference<T>&&, common_reference_type>`,
which fails if the common reference is `T&` since `is_convertible_v<device_reference<T>&&, T&>`
is false.

##### Const-qualified operators on `device_reference`

The `std::indirectly_writable` concept tests assignment through
`const_cast<const iter_reference_t<I>&&>(*it) = value`. This means
`device_reference::operator=` (and all compound assignment / increment /
decrement operators) must be **const-qualified**. This is correct for a proxy
type: `const` applies to the proxy handle itself, not the underlying device
data. The proxy's pointer and queue members are unchanged by assignment; only
the pointed-to device memory is modified.

##### Iterator traits

`device_pointer<T>` must also expose the correct iterator traits:
- `iterator_concept = std::random_access_iterator_tag`
- `value_type = T`
- `reference = device_reference<T>`

#### `device_view<T>`

`device_vector` itself is not device copyable, it owns a `sycl::queue`
and manages device memory lifetime. `device_view<T>` is a lightweight,
device-copyable view that models `std::ranges::random_access_range` and
`std::ranges::sized_range`:

```cpp
template <typename T>
class device_view {
    device_pointer<T> __begin;
    std::size_t __size;
public:
    auto begin() const { return __begin; }
    auto end()   const { return __begin + __size; }
    auto size()  const { return __size; }
    auto operator[](std::size_t i) const { return __begin[i]; }
    bool empty() const { return __size == 0; }
};
```

`device_view` is trivially copyable (and therefore device copyable) because
it contains only a `device_pointer<T>` and a `size_t`. It must also opt into
`std::ranges::enable_borrowed_range` since it is a non-owning view:

```cpp
template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<
    oneapi::dpl::experimental::device_view<T>> = true;
```

It is obtained via a member function:

```cpp
auto view = dv.view();  // returns device_view<T>

// Use with oneDPL range algorithms:
oneapi::dpl::ranges::sort(policy, dv.view());
oneapi::dpl::ranges::for_each(policy, dv.view(), f);

// Compose with standard range adaptors:
auto pipeline = dv.view() | std::views::take(100) | std::views::transform(f);
oneapi::dpl::ranges::for_each(policy, pipeline, g);

// Capture into a kernel:
auto v = dv.view() | std::views::take(512);
q.parallel_for(sycl::range<1>(512), [=](sycl::id<1> i) {
    v[i] *= 2.0f;
});
```

**Note on lambdas with proxy references:** When using range algorithms with
`device_view`, lambdas should accept `auto&&` rather than `T&`, because the
range element type is `device_reference<T>`, not `T&`:

```cpp
// Correct:
oneapi::dpl::ranges::for_each(policy, dv.view(), [](auto&& x) { x *= 2; });

// Will not compile -- device_reference<int> is not int&:
oneapi::dpl::ranges::for_each(policy, dv.view(), [](int& x) { x *= 2; });
```

Separating usage as a range from `device_vector` itself allows us to keep the
most of the convenience functionality modelling `std::vector` on the host side
only, while allowing a simplified, lightweight, device_copyable view to enable
range support on the device.

## Open Questions

- **What should `device_vector` store: context + device, or queue?**
  `sycl::malloc_device` requires only a context and device, not a queue. Storing
  a queue would tie the vector to a particular queue and imply synchronization
  semantics (see synchronization question below). Storing context + device is
  sufficient for allocation and deallocation, with queues provided per-operation
  when needed.

  The API skeleton currently assumes context + device storage. Constructors
  that accept a `sycl::queue` use it only to extract the context and device;
  the queue is not retained. No-arg constructors default to
  `sycl::device{sycl::gpu_selector_v}` and create a context from it. Note that
  users who later want to provide a queue for explicit synchronization must
  ensure the queue shares the same context as the allocation; if using a no-arg
  constructor, they will need `get_context()` to create a compatible queue.

- **How should `device_pointer` / `device_reference` associate with a context?**
  On the device, `device_pointer` dereferences directly as a raw pointer — no
  context or queue is needed. On the host, dereferencing requires a queue for
  memcpy. The question is how `device_pointer` / `device_reference` obtains one.

  Options:
  1. **Store a pointer to the context** (or to the owning vector) alongside
     the raw pointer. Adds 8 bytes to `device_pointer` footprint but provides
     direct access to the context for creating a queue.
  2. **Store only the raw pointer** (sycl-thrust approach). Minimal footprint
     (8 bytes), but host-side dereference must search a global registry of
     contexts to find the one matching the pointer. Optimizes the device hot
     path at the cost of host-path complexity and a global data structure.
  3. **Store a pointer back to the owning `device_vector`**. Similar to
     option 1 in footprint, but couples pointer lifetime to the vector and
     may complicate standalone use of `device_pointer`. If we decide to store
     a queue in the vector, this could allow use of that queue for implicit
     synchronization.

- **Synchronization model for host-side operations**
  Thrust `device_vector` implicitly synchronizes with the whole device on every
  host operation. For at least some, this seems to be an inconvenience rather than
  a feature, see [usage study](usage_pattern_study.md#alternatives-built-by-projects-that-rejected-thrustdevice_vector).
  Also, sycl has no official specified way to synchronize with a full device, only
  with a queue via dependent events or via an in-order queue.  There is a
  [proposed extension](https://github.com/intel/llvm/blob/a27b442be0f3e72245846073b4ca254fe83246ca/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_wait.asciidoc)
  for this purpose. We must decide how to deal with synchronization of host side
  actions of the `device_vector` and helper classes.

  For now the proposal in the API skeleton is as follows:

  All host-side operations that transfer data (element access via
  `device_reference`, `assign`, `resize`, `insert`, etc.) are **blocking** —
  the call does not return until the data transfer is complete, regardless of
  whether a queue is provided.

  **Without an explicit queue:** the operation creates a temporary queue from the
  stored context + device, submits the transfer, and blocks until complete.

  **With an explicit queue:** the operation submits to the provided queue and
  blocks until complete. The provided queue must share the same context
  as the vector's allocation (`q.get_context() == dv.get_context()`);
  implementations throw on mismatch. The queue overload is useful for avoiding
  repeated temporary queue creation, or for ensuring operations go through a
  specific queue for profiling/debugging purposes.

  **Important:** These operations do **not** implicitly synchronize with
  other work on the device or queue. The user is responsible for ensuring
  that any kernels or operations modifying the same memory have completed
  before calling host-side `device_vector` operations. This can be achieved
  by using an in-order queue shared with kernel submissions, or by explicitly
  waiting on events from prior work.

  A possible future extension could add async overloads that return
  `sycl::event` for users who need explicit dependency management, but this
  is not included in the current proposal.


- **Should we support aligned allocation or a non-C++ allocator?**
  sycl-thrust's `device_allocator<T, Alignment>` supports
  `sycl::aligned_alloc_device` for types with specific alignment requirements.
  Some GPU hardware benefits from aligned allocations (e.g. 128-byte or
  256-byte alignment for coalesced memory access or cache line optimization).
  We currently use `sycl::malloc_device` which has implementation-defined
  alignment. More broadly, RMM's `device_uvector` demonstrates the value
  of pluggable memory resources (`device_async_resource_ref`) for pool
  allocation, arena allocation, etc. While a C++ `Allocator` in the
  `std::vector` sense is not viable for device memory (see design decisions),
  a non-C++ allocator concept — e.g. a type-erased memory resource similar
  to `std::pmr::memory_resource` or RMM's `resource_ref` — could provide
  both alignment control and pluggable allocation strategies without
  requiring host-accessible memory semantics.

- **Should we use device_pointer as the device iterator?**
  It seems there is no use case for a separate device_iterator, but it's
  worth considering.

- **Should we include lesser-used features for host side usage (push_back/insert/erase)?**
  These are significant implementation complexity and not used in the field, but
  provide more complete migration from thrust, and also closer alignment with `std::vector`.
