# `device_array<T>` — Simplified Device Memory Container

This is a stripped-down alternative to the `device_vector` proposal. It drops
proxy types, helper classes, and rarely-used modifiers in favor of a minimal
container focused on: RAII device allocation, explicit host-device transfers,
and direct use of raw USM pointers as iterators.

See the [device_vector RFC](../../../rfcs/proposed/device_vector/README.md) for
full motivation, usage study, and comparison of existing implementations. This
document only describes where `device_array` diverges.

## Key Simplifications vs `device_vector`

- **No `device_pointer<T>`** — `begin()`/`end()` return raw `T*`. USM device
  pointers already satisfy random access iterator requirements, are device
  copyable, and work directly in oneDPL algorithms and `parallel_for`.
- **No `device_reference<T>`** — no proxy reference type, no compound
  assignment round-trips, no `basic_common_reference` specializations, no
  `__SYCL_DEVICE_ONLY__` bifurcation. Host-side element access is through
  explicit `read()` / `write()` functions.
- **No `push_back`, `insert`, `erase`** — rarely used in practice (see usage
  study), high implementation complexity.
- **No default-device constructors** — a `sycl::context` + `sycl::device` (or
  a queue to extract them from) is always required.
- **Stores `sycl::context` + `sycl::device`**, not a queue. Operations that
  need a queue accept one optionally; when omitted, a temporary queue is
  created internally.
- **Uninitialized by default** — sized construction and `resize()` do not
  value-initialize new elements (no kernel launch or memset). Pass an explicit
  value to opt in: `device_array(1024, T{}, q)`. This encourages good practices
  when dealing with device memory, and  matches the behavior chosen by FAISS and
  RMM, which moved away from Thrust partially because of unwanted initialization
  overhead.

## Allocator

`device_array` accepts an optional allocator template parameter for device
memory allocation. The default allocator wraps `sycl::malloc_device` /
`sycl::free`.

### Allocator Requirements

A type `Alloc` satisfies `DeviceAllocator` for type `T` if, given an instance
`a` of type `Alloc`, a pointer `p` of type `T*`, a `std::size_t n`, a
`sycl::context ctx`, and a `sycl::device dev`, the following expressions are
valid:

| Expression | Return type | Semantics |
|---|---|---|
| `a.allocate(n, ctx, dev)` | `T*` | Allocate device memory for `n` objects of type `T` |
| `a.deallocate(p, n, ctx, dev)` | `void` | Free device memory previously allocated by `allocate` |

The allocator is not required to support `construct`, `destroy`, or any of the
`std::allocator` named requirements beyond `allocate`/`deallocate`. Device
memory is not host-accessible, so construction and destruction happen via
kernel launches or memcpy, managed by `device_array` itself.

The allocator must be copy-constructible and copy-assignable.

```cpp
// Default allocator
template <typename T>
struct device_allocator {
    T* allocate(std::size_t n, sycl::context ctx, sycl::device dev) {
        return sycl::malloc_device<T>(n, dev, ctx);
    }
    void deallocate(T* p, std::size_t n, sycl::context ctx, sycl::device dev) {
        sycl::free(p, ctx);
    }
};
```

### C++20 Concept (informational; enforced via SFINAE on C++17)

```cpp
template <typename Alloc, typename T>
concept DeviceAllocator = requires(Alloc a, T* p, std::size_t n,
                                   sycl::context ctx, sycl::device dev) {
    { a.allocate(n, ctx, dev) } -> std::same_as<T*>;
    { a.deallocate(p, n, ctx, dev) } -> std::same_as<void>;
};
```

On C++17, these requirements are enforced via `static_assert` or SFINAE on
the relevant expressions rather than a concept.

## API

```cpp
namespace oneapi::dpl::experimental {

template <typename T, typename Alloc = device_allocator<T>>
class device_array {
public:
    using value_type      = T;
    using allocator_type  = Alloc;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    // =====================================================================
    // Construction
    // =====================================================================

    // Allocate uninitialized device memory (default — no kernel launch or memset)
    // From queue (extracts context + device; queue is not retained)
    device_array(size_type count, sycl::queue q);
    // From context + device
    device_array(size_type count, sycl::context ctx, sycl::device dev);

    // Allocate and fill with value (requires kernel launch or memset)
    device_array(size_type count, const T& value, sycl::queue q);
    device_array(size_type count, const T& value,
                 sycl::context ctx, sycl::device dev);

    // Construct from host data (upload)
    template <typename InputIt>
    device_array(InputIt first, InputIt last, sycl::queue q);
    device_array(std::initializer_list<T> init, sycl::queue q);
    device_array(const std::vector<T>& src, sycl::queue q);

    template <typename InputIt>
    device_array(InputIt first, InputIt last, sycl::context ctx, sycl::device dev);
    device_array(std::initializer_list<T> init, sycl::context ctx, sycl::device dev);
    device_array(const std::vector<T>& src, sycl::context ctx, sycl::device dev);

    // Copy / move
    device_array(const device_array&);
    device_array(device_array&&);
    device_array& operator=(const device_array&);
    device_array& operator=(device_array&&);

    ~device_array();

    // =====================================================================
    // Host-device transfer
    // =====================================================================

    // Bulk download
    std::vector<T> to_vector() const;
    std::vector<T> to_vector(sycl::queue q) const;

    // Bulk upload (resizes to match)
    void assign(const T* first, const T* last);
    void assign(const T* first, const T* last, sycl::queue q);
    void assign(const std::vector<T>& src);
    void assign(const std::vector<T>& src, sycl::queue q);

    // Single-element host access (blocking, creates queue from context & device)
    T read(size_type pos) const;
    void write(size_type pos, const T& value);

    // Single-element host access (blocking, provided queue is used for copy submissions)
    T read(size_type pos, sycl::queue q) const;
    void write(size_type pos, const T& value, sycl::queue q);

    // Asynchronous single-element access, events allow synchronization with event driven workloads
    sycl::event async_read(size_type pos, T& out,
                     sycl::queue q,
                     const std::vector<sycl::event>& depends_on = {}) const;
    sycl::event async_write(size_type pos, const T& value,
                      sycl::queue q,
                      const std::vector<sycl::event>& depends_on = {});

    // Asynchronous bulk transfer
    sycl::event async_to_vector(std::vector<T>& out,
                          sycl::queue q,
                          const std::vector<sycl::event>& depends_on = {}) const;
    sycl::event async_assign(const T* first, const T* last,
                       sycl::queue q,
                       const std::vector<sycl::event>& depends_on = {});

    // =====================================================================
    // Device iteration — raw USM pointers
    // =====================================================================

    iterator       begin();
    const_iterator begin() const;
    iterator       end();
    const_iterator end() const;
    pointer        data();
    const_pointer  data() const;

    // =====================================================================
    // Capacity
    // =====================================================================

    size_type size()     const;
    size_type capacity() const;
    bool      empty()    const;

    // Resize — new elements are uninitialized by default
    void resize(size_type count);
    void resize(size_type count, sycl::queue q);
    // Resize — new elements filled with value
    void resize(size_type count, const T& value);
    void resize(size_type count, const T& value, sycl::queue q);

    void reserve(size_type new_cap);
    void clear();

    // =====================================================================
    // Views
    // =====================================================================

    device_span<T>       span();
    device_span<const T> span() const;

    // =====================================================================
    // Allocator access
    // =====================================================================

    allocator_type get_allocator() const;

    // =====================================================================
    // Context / device access
    // =====================================================================

    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

} // namespace oneapi::dpl::experimental
```

## `device_span<T>`

`device_array` is not device-copyable (it owns memory). For kernel capture,
non-owning views, and range composition, use `device_span<T>` via `.span()`.

`device_span` is guaranteed trivially copyable (and therefore device copyable),
has `enable_borrowed_range = true` and `enable_view = true`, and models
`contiguous_range` + `sized_range`. It replaces the `device_view` from the
`device_vector` proposal without proxy reference types or
`basic_common_reference` specializations.

### Definition

**C++23 and later:** `std::span` is guaranteed trivially copyable (per P2251R1),
so `device_span` is simply an alias:

```cpp
#if __cplusplus >= 202302L  // C++23

template <typename T>
using device_span = std::span<T>;

#else  // C++20

template <typename T>
class device_span {
    T* __ptr = nullptr;
    std::size_t __size = 0;
public:
    using element_type    = T;
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = T*;
    using reference       = T&;
    using iterator        = T*;

    device_span() = default;
    device_span(T* ptr, std::size_t size) : __ptr(ptr), __size(size) {}

    template <std::size_t N>
    device_span(T (&arr)[N]) : __ptr(arr), __size(N) {}

    // Implicit conversion from device_array
    device_span(device_array<T>& arr);

    T*          begin() const { return __ptr; }
    T*          end()   const { return __ptr + __size; }
    T*          data()  const { return __ptr; }
    std::size_t size()  const { return __size; }
    bool        empty() const { return __size == 0; }

    T& operator[](std::size_t i) const { return __ptr[i]; }
    T& front() const { return __ptr[0]; }
    T& back()  const { return __ptr[__size - 1]; }

    device_span first(std::size_t count) const { return {__ptr, count}; }
    device_span last(std::size_t count) const { return {__ptr + __size - count, count}; }
    device_span subspan(std::size_t offset, std::size_t count) const {
        return {__ptr + offset, count};
    }
};

template <typename T>
inline constexpr bool std::ranges::enable_borrowed_range<
    oneapi::dpl::experimental::device_span<T>> = true;

template <typename T>
inline constexpr bool std::ranges::enable_view<
    oneapi::dpl::experimental::device_span<T>> = true;

#endif
```

The C++20 `device_span` is a minimal subset of `std::span`'s interface — just
enough for device capture, range usage, and subspan operations. When C++23
becomes the baseline, `device_span` becomes a zero-cost alias and can
eventually be deprecated in favor of `std::span` directly.

## Range Support (C++20)

Since `begin()`/`end()` return raw `T*`, `device_array` is already a
`std::ranges::contiguous_range` and `std::ranges::sized_range` — no extra
machinery needed for use with oneDPL range algorithms.

For kernel capture or composition with range adaptors, use `device_span`
via `.span()`. Both `enable_borrowed_range` and `enable_view` must be
specialized for `device_span` — the latter is critical because without it,
`std::views::all` wraps lvalues in `std::ranges::ref_view`, which oneDPL
rejects in SYCL kernel code (the `ref_view` contains a host pointer that
cannot be captured by a device kernel). With `enable_view = true`,
`std::views::all` returns the `device_span` by copy, and view adaptor
pipelines (e.g., `span | std::views::take(n)`, `span | std::views::reverse`)
work correctly because the base range is the span itself, not a `ref_view`.

This matches `std::span`'s own trait specializations and means that
`device_span` and `std::span` (backed by USM pointers) behave identically
with oneDPL range algorithms and C++20 view adaptors.

## Usage Examples

```cpp
#include <oneapi/dpl/device_array>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

namespace dpl = oneapi::dpl::experimental;

sycl::queue q{sycl::property::queue::in_order{}};

// --- RAII allocation + upload from host ---
std::vector<float> host_data(1024, 3.14f);
dpl::device_array<float> d(host_data, q);

// --- Use with oneDPL algorithms (raw T* iterators) ---
auto policy = oneapi::dpl::execution::make_device_policy(q);
std::sort(policy, d.begin(), d.end());

// --- Use in a SYCL kernel ---
float* ptr = d.data();
q.parallel_for(sycl::range<1>(d.size()), [=](sycl::id<1> i) {
    ptr[i] *= 2.0f;
}).wait();

// --- Explicit single-element host access ---
float val = d.read(0, q);     // synchronous read
d.write(0, 42.0f, q);         // synchronous write

// --- Async transfer with dependency ---
float result;
sycl::event e = d.async_read(0, result, q, {some_prior_event});
e.wait();

// --- Bulk download ---
std::vector<float> out = d.to_vector(q);

// --- Output buffer (uninitialized by default — no memset) ---
dpl::device_array<float> output(1024, q);
std::transform(policy, d.begin(), d.end(), output.begin(),
               [](float x) { return x * 2.0f; });

// --- Zero-initialized allocation (opt-in) ---
dpl::device_array<float> zeroed(1024, 0.0f, q);

// --- Range usage (C++20) ---
// device_array itself works with range algorithms on the host side:
oneapi::dpl::ranges::sort(policy, d);

// For kernel capture or composition with range adaptors, use device_span:
auto s = d.span();  // returns device_span<float>
auto pipeline = s | std::views::take(100);
oneapi::dpl::ranges::for_each(policy, pipeline, [](float& x) { x += 1.0f; });

// Capture a device_span into a kernel:
auto s2 = d.span();
q.parallel_for(sycl::range<1>(s2.size()), [=](sycl::id<1> i) {
    s2[i] *= 2.0f;
}).wait();
```

## Open Questions

- **Should async overloads be in the initial proposal or deferred?** The
  `depends_on` + return-event pattern adds API surface. Could ship the
  synchronous forms first and add async later, since users who need async
  control can use `sycl::memcpy` directly on `data()`.
