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
- **Real-world usage patterns** - A survey of `thrust::device_vector` and
  `dpct::device_vector` usage across open-source projects reveals consistent
  patterns that validate our design priorities. The sections below detail
  findings from two categories: native Thrust usage (CUDA projects) and
  migrated dpct usage (SYCLomatic output).

### Observed Usage Patterns

#### Thrust (`thrust::device_vector`) — Native CUDA Projects

[spblas-reference](https://github.com/SparseBLAS/spblas-reference) (sparse
BLAS standard reference implementation) demonstrates the minimal-but-dominant
pattern: `device_vector` as an **RAII device memory manager and host-device
data shuttle**:

1. **Constructing from `std::vector`** (~90% of uses) — bulk host-to-device
   transfer at setup time.
2. **Allocating output buffers by size** — e.g. after a symbolic phase
   computes output NNZ, a `device_vector` is constructed with just a count.
3. **Extracting raw device pointers** via `.data().get()` — every
   `device_vector` is ultimately unwrapped to a raw pointer for passing
   to library APIs (`csr_view`, `std::span`).
4. **Copying results back to host** via `thrust::copy(d.begin(), d.end(),
   host.begin())` — used in every test for verification.

Notably absent from spblas: element-level access (`operator[]`), `resize()`,
`push_back()`, `insert()`/`erase()`, or device-side algorithms on iterators.

#### dpct (`dpct::device_vector`) — Migrated CUDA-to-SYCL Projects

A broader survey of `dpct::device_vector` usage across ~18 repositories
(111 code results on GitHub) shows additional patterns beyond the
spblas-minimal case:

**Projects surveyed include:**
- [HeCBench](https://github.com/ORNL/HeCBench) (ORNL, 285+ stars) — HPC
  benchmark suite
- [oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) (Intel,
  1139+ stars) — radix sort migration samples
- [SYCLomatic-test](https://github.com/oneapi-src/SYCLomatic-test) —
  official compatibility test suite
- [OP-PIC](https://github.com/OP-DSL/OP-PIC) — particle-in-cell framework
- Various sparse matrix, radio astronomy, and optimization codes

**Consolidated operation frequency:**

| Operation | Frequency | Example |
|---|---|---|
| Construction from size | Very high | `dpct::device_vector<T> v(N)` |
| Assignment from `std::vector` | Very high | `d_vec = h_vec` (H2D) |
| `begin()`/`end()` for algorithms | Very high | `sort(policy, dv.begin(), dv.end())` |
| `data()` + raw pointer extraction | Very high | `get_raw_pointer(dv.data())` for kernels |
| Copy back to host | High | `std::copy(policy, dv.begin(), dv.end(), h.begin())` or `h = d` |
| Construction from `std::vector` | High | `dpct::device_vector<T> dv(host_vec)` |
| `size()` | Medium | For bounds checks, kernel launch args |
| `operator[]` | Medium | Host-side element access |
| `resize()` | Medium | Output buffer sizing |
| As class/struct member | Medium | Sparse matrices, particle systems |
| `clear()`, `push_back()`, `insert()`, `erase()` | Low | Mostly in tests, not real workloads |

**Notable real-world patterns:**
- **Persistent class members** — OP-PIC stores `device_vector`s in `std::map`
  keyed by MPI rank for inter-process communication buffers. Sparse matrix
  libraries store CSR components (`data`, `row_ptr`, `col_ind`) as
  `device_vector` struct members.
- **Default-construct then assign** — oneAPI-samples shows a "reset and
  re-sort" loop: `dpct::device_vector<T> d_keys;` then `d_keys = h_keys;`
  each iteration.
- **Raw pointer extraction is ubiquitous** — nearly every project that passes
  data to SYCL kernels unwraps `device_vector` to a raw pointer. This
  validates our `device_pointer::get()` and the importance of making
  `device_pointer` device-copyable for direct kernel capture.

**Pain points observed in the ecosystem:**
- **`resize()` not filling new elements** — heimdall-astro (radio astronomy
  pipeline) had to wrap `dpct::device_vector` with an explicit fill after
  resize.
- **Projects abandoning `device_vector`** — sycl-streamlines switched
  entirely to raw `sycl::malloc_device` + `q.memcpy()` because the
  `device_vector` + algorithm path was too problematic.
- **`operator[]` lifetime/synchronization issues** — SYCLomatic's own test
  suite has TODO comments about double-free errors with element access after
  assignment, suggesting the proxy reference + shared memory model has
  subtle lifetime issues.

#### Design Validation

These patterns validate our design priorities:
1. **Construction + bulk transfer + raw pointer extraction** are the core
   operations — get these right first.
2. **`begin()`/`end()` integration with oneDPL algorithms** is the
   second-most critical capability.
3. **Device memory (not shared)** is the right default — projects that need
   performance use device memory and explicit transfers.
4. **Full `std::vector`-like modifier API** (`push_back`, `insert`, `erase`)
   is a secondary convenience, rarely used in real workloads.
5. Our choice to have **`device_pointer::get()` return a raw `T*`** directly
   addresses the most common friction point in migrated code (`get_raw_pointer`
   indirection).

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
  implies a synchronous round-trip (read-modify-write) to device memory. All
  mutating operators (`operator=`, compound assignment, increment/decrement) are
  **const-qualified** because `device_reference` is a proxy: `const` applies to
  the handle, not the underlying device data. This is required by
  `std::indirectly_writable` in C++20 (see [Range Support](#range-support)).

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
    pointer         data() ;
    const_pointer   data() const;

    // Iterators
    iterator begin();
    iterator end();
    // + const/reverse variants
    
    // Views
    device_view<T> view();
    device_view<const T> view() const;

    // Queue access
    sycl::queue get_queue();

    // Capacity
    bool      empty()    const;
    size_type size()     const;
    size_type capacity() const;
    void reserve(size_type new_cap);
    void shrink_to_fit();

    // Modifiers
    void assign(size_type count, const T& value);
    template <typename InputIt>
    void assign(InputIt first, InputIt last);
    void assign(std::initializer_list<T> ilist);
    void clear();
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

```cpp
// device_reference<T> vs U (where T and U are the same unqualified type)
template <typename T, typename U, template<class> class TQual, template<class> class UQual>
    requires std::same_as<std::remove_cv_t<T>, std::remove_cv_t<U>>
struct std::basic_common_reference<
    oneapi::dpl::experimental::device_reference<T>, U, TQual, UQual> {
    using type = std::remove_cv_t<T>;
};

// U vs device_reference<T> (symmetric)
template <typename T, typename U, template<class> class TQual, template<class> class UQual>
    requires std::same_as<std::remove_cv_t<T>, std::remove_cv_t<U>>
struct std::basic_common_reference<
    U, oneapi::dpl::experimental::device_reference<T>, TQual, UQual> {
    using type = std::remove_cv_t<T>;
};

// device_reference<T> vs device_reference<U>
template <typename T, typename U, template<class> class TQual, template<class> class UQual>
struct std::basic_common_reference<
    oneapi::dpl::experimental::device_reference<T>,
    oneapi::dpl::experimental::device_reference<U>, TQual, UQual> {
    using type = std::common_reference_t<T&, U&>;
};
```

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

## Implementation Notes from POC

A proof-of-concept implementation validated the design and uncovered the
following points that should be considered during productization:

- **`device_reference` host/device bifurcation.** `device_reference::operator T()`
  and `operator=` must use different code paths depending on whether they are
  compiled for host or device. On the host, they use `sycl::queue::memcpy`
  (synchronous). On the device, they dereference the raw USM pointer directly.
  The `__SYCL_DEVICE_ONLY__` macro selects the correct path. This keeps the
  API consistent across compilation passes (same type, same member functions)
  while the implementation adapts.

- **`operator std::vector<T>()` should be non-const.** `sycl::queue::memcpy`
  is a non-const member function, so `operator std::vector<T>() const` does
  not compile without `mutable` on the queue. Making the conversion operator
  non-const is the simpler option, but const element access (`operator[] const`,
  `begin() const`, etc.) still requires `mutable sycl::queue` since they
  construct `device_reference`/`device_pointer` objects that hold a non-const
  queue pointer.

- **`sycl::queue::memcpy` requires `void*` casts.** The `sycl::queue::memcpy`
  overload set includes template overloads for `device_global` types. Passing
  typed pointers (e.g. `T*`) can be ambiguous. Explicit
  `static_cast<void*>` / `static_cast<const void*>` on the source and
  destination pointers resolves the ambiguity.

- **oneDPL `__brick_fill` / `__brick_fill_n` incompatibility with proxy
  references.** The hetero specializations of `__brick_fill` take their
  target parameter by lvalue reference (`_TargetT& __target`). When the
  iterator's `operator[]` returns a proxy reference prvalue (as
  `device_pointer` does), the prvalue cannot bind to the lvalue reference.
  Changing the parameter to a forwarding reference (`_TargetT&& __target`)
  fixes this for all proxy reference types. This is a pre-existing oneDPL
  bug, not specific to `device_vector`.

- **POC location.** The POC header is at
  `include/oneapi/dpl/experimental/device_vector.h` with tests at
  `test/parallel_api/experimental/device_vector.pass.cpp`.

## Open Questions

- **Should we try to support multiple devices?**
  Distributed ranges work has been halted, but is this an important use case to
  preserve, or unnecessary complication for users?  We could always add another
  type to support such use cases in the future.

- **Queue association model**
  - Impact on `device_pointer`:
  We want `device_pointer` to be device copyable so it can be used directly
  in kernels and oneDPL algorithms. However, host-side dereference (returning
  a proxy reference that does `memcpy`) requires a `sycl::queue`, and
  `sycl::queue` is not device copyable. This creates a tension with several
  possible resolutions, here are the leading 2:
  - Device & context instead?:
  All that is required to allocate is device & context, which can be dictated by a queue, but it is a many-to-one releationship in that many queues may exist on top of a device & context pair. However, If we want implicit synchronization between an in-order queue and host-side accesses of `device_vector`, we need the association with a queue. 
  
  Thrust has implicit synchronization with its default stream, which sychronizes its host access to either all streams or the default per-thread stream based upon compiliation settings.

  There is a [proposed extension](https://github.com/intel/llvm/blob/a27b442be0f3e72245846073b4ca254fe83246ca/sycl/doc/extensions/proposed/sycl_ext_oneapi_device_wait.asciidoc) to allow synchronizing with a whole device. Otherwise, we could make the synchronization the user's responsibility or tie it only to an associated in-order queue.  

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


## TODOs
