# `device_array<T>` — Simplified Device Memory Container

`device_array` provides RAII device allocation, explicit host-device transfers,
and direct use of raw USM pointers as iterators. It emphasizes convenience of
use while making host-side use explicit. It provides control over synchronization
of host side operations.

This focus provides support for the main usage pattern for users of `device_vector`,
and fits nicely within SYCL while avoiding much of the complexity of `device_vector`.

See the [device_vector RFC](../../../rfcs/proposed/device_vector/README.md) for
full motivation, usage study, and comparison of existing implementations. This
document only describes `device_array`.

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

    // Construction

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

    // Deleted Copy — use copy_from for explicit device-to-device copies)
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;

    // Move
    device_array(device_array&&);
    device_array& operator=(device_array&&);

    ~device_array();

    // Device-to-device copy (allocates on the provided context+device)
    // Supports cross-device copies: source and destination may be on different devices
    static device_array copy_from(const device_array& src, sycl::queue q);
    static device_array copy_from(const device_array& src,
                                  size_type offset, size_type count, sycl::queue q);
    static device_array copy_from(const device_array& src,
                                  sycl::context ctx, sycl::device dev);
    static device_array copy_from(const device_array& src,
                                  size_type offset, size_type count,
                                  sycl::context ctx, sycl::device dev);

    // Host-device transfer

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

    // Device iteration — raw USM pointers
    iterator       begin();
    const_iterator begin() const;
    iterator       end();
    const_iterator end() const;
    pointer        data();
    const_pointer  data() const;

    // Capacity
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
    void swap(device_array& other);

    // Views
    sycl::span<T>       span();
    sycl::span<const T> span() const;

    // Allocator access
    allocator_type get_allocator() const;

    // Context / device access
    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

} // namespace oneapi::dpl::experimental
```

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

## Use within kernels

`device_array` is not device-copyable (it owns memory). For kernel capture,
non-owning views, and range composition, use `sycl::span<T>` via `.span()`.

`sycl::span` is guaranteed to be present with sycl 2020 and device copyable,
conforming to c++20 `std::span` even when compiled with c++17.

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

// For kernel capture or composition with range adaptors, use sycl::span:
auto s = d.span();  // returns sycl::span<float>
auto pipeline = s | std::views::take(100);
oneapi::dpl::ranges::for_each(policy, pipeline, [](float& x) { x += 1.0f; });

// Capture a sycl::span into a kernel:
auto s2 = d.span();
q.parallel_for(sycl::range<1>(s2.size()), [=](sycl::id<1> i) {
    s2[i] *= 2.0f;
}).wait();
```

## Open Questions

- **Should async overloads be in the initial proposal or deferred?** 
    This provides more control over synchronization than merely an in-order queue,
    but it is unclear whether users who are wanting this would just want to work
    with USM memory and memcpy directly.
