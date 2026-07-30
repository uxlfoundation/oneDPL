# `device_array<T>` — Simplified Device Memory Container

`device_array` provides RAII device allocation, explicit host-device transfers,
and direct use of spans over raw USM pointers. It emphasizes convenience of
use while making host-side use explicit.

It is intended to be minimal in its design: no allocator, no resizing, no
asynchronous access.

This focus provides support for the main usage pattern for users of `device_vector`,
and fits nicely within SYCL while avoiding much of the complexity of `device_vector`.

See the [device_vector RFC](../../../rfcs/proposed/device_vector/README.md) for
full motivation, usage study, and comparison of existing implementations. This
document only describes `device_array`.

## Relationship to the shared base

`device_array` and `compat::device_vector` share their implementation through a
non-public base, `oneapi::dpl::experimental::internal::__device_storage_base<T, Alloc>`,
which owns the device allocation and its lifetime, the size, the associated
`sycl::context` / `sycl::device`, the allocator instance, resizing, and the
host-device transfer helpers.

`device_array` **privately inherits** from
`__device_storage_base<T, device_allocator<T>>` and re-exposes (via `using`
declarations) only the fixed-size subset of that interface. The allocator type
is fixed to the default `device_allocator<T>` and is not part of
`device_array`'s public API; resizing, `reserve`, `capacity`, and `clear` are
not re-exposed. This lets `device_array` present a simplified interface while
`compat::device_vector` reuses the same base to offer a resizable,
allocator-aware container.

## API

```cpp
namespace oneapi::dpl::experimental {

template <typename T>
class device_array : private internal::__device_storage_base<T, device_allocator<T>> {
public:
    using value_type      = T;
    using size_type       = std::size_t;

    // Construction

    // Allocate uninitialized device memory (default — no kernel launch or memset)
    // From queue (extracts context + device; queue is not retained)
    device_array(size_type count, sycl::queue q);
    // From context + device
    device_array(size_type count, sycl::context ctx, sycl::device dev);

    // Allocate and fill with value
    device_array(size_type count, const T& value, sycl::queue q);
    device_array(size_type count, const T& value,
                 sycl::context ctx, sycl::device dev);

    device_array(sycl::span<const T> src, sycl::queue q);
    device_array(sycl::span<const T> src, sycl::context ctx, sycl::device dev);

    // explicit copy construction only (avoids accidental deep copies)
    explicit device_array(const device_array&);
    explicit device_array(const device_array&, sycl::queue q);

    device_array& operator=(const device_array&) = delete;

    // Move (shallow move, device memory remains where it is)
    device_array(device_array&&);
    device_array& operator=(device_array&&);

    // deallocates device memory
    ~device_array();

    // Host-device transfer
    // Bulk transfer from device (dst may be host memory or USM on this context)
    void copy_to(sycl::span<T> dst) const;
    void copy_to(sycl::span<T> dst, sycl::queue q) const;

    // Convenience download into a fresh host vector
    std::vector<T> to_vector() const;
    std::vector<T> to_vector(sycl::queue q) const;

    // Bulk transfer to device (src may be host memory or USM on this context)
    void copy_from(sycl::span<const T> src);
    void copy_from(sycl::span<const T> src, sycl::queue q);

    // Single-element host access (blocking, creates queue from context & device)
    T host_read(size_type pos) const;
    void host_write(size_type pos, const T& value);

    // Single-element host access (blocking, provided queue is used for copy submissions)
    T host_read(size_type pos, sycl::queue q) const;
    void host_write(size_type pos, const T& value, sycl::queue q);

    // Capacity (fixed size — no resize / reserve / capacity / clear)
    size_type size()  const;
    bool      empty() const;

    void swap(device_array& other);

    // Views
    sycl::span<T>       span();
    sycl::span<const T> span() const;

    // Context / device access
    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

} // namespace oneapi::dpl::experimental
```

## Allocator

`device_array` fixes its allocator to the default `device_allocator<T>` (which
wraps `sycl::malloc_device` / `sycl::free`) and does not expose it. Pluggable
allocation via the `DeviceAllocator` concept is available on
[`compat::device_vector`](device_vector_compat.md#allocator).

## Use within kernels

`device_array` is not device-copyable (it owns memory). For kernel capture,
non-owning views, and range composition, use `sycl::span<T>` via `.span()`.

`sycl::span` is guaranteed to be present with sycl 2020 and device copyable,
conforms to c++20 `std::span` even when compiled with c++17.

## Usage Examples

```cpp
#include <oneapi/dpl/device_array>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

namespace dpl = oneapi::dpl::experimental;

sycl::queue q{sycl::property::queue::in_order{}};

// --- RAII allocation + upload from host ---
// host_data converts implicitly to sycl::span<const float>; size is taken from it
std::vector<float> host_data(1024, 3.14f);
dpl::device_array<float> d(host_data, q);

// --- Use with oneDPL algorithms ---
auto policy = oneapi::dpl::execution::make_device_policy(q);
oneapi::dpl::sort(policy, d.span().begin(), d.span().end());

// --- Use in a SYCL kernel ---

auto s = d.span();
q.parallel_for(sycl::range<1>(s.size()), [=](sycl::id<1> i) {
    s[i] *= 2.0f;
}).wait();

float* ptr = d.span().data();
q.parallel_for(sycl::range<1>(d.size()), [=](sycl::id<1> i) {
    ptr[i] *= 2.0f;
}).wait();

// --- Pass the span directly to the oneDPL range-based algorithms ---
oneapi::dpl::ranges::for_each(policy, d.span(), [](float& x) { x += 1.0f; });

oneapi::dpl::ranges::sort(policy, d.span());

oneapi::dpl::ranges::sort(policy, d.span().subspan(0, 100));

// --- Explicit single-element host access ---
float val = d.host_read(0, q);     // synchronous read
d.host_write(0, 42.0f, q);         // synchronous write

// --- Bulk download ---
std::vector<float> out = d.to_vector(q);   // fresh vector
d.copy_to(out, q);                         // or into existing storage (sizes must match)

// --- Bulk upload into an existing device_array (does not resize) ---
d.copy_from(host_data, q);

// --- Device-to-device: span() is USM, so it is a valid source ---
dpl::device_array<float> d2(d.size(), q);
d2.copy_from(d.span(), q);                 // span<float> -> span<const float>

// --- Subrange copy into a new, smaller device_array ---
dpl::device_array<float> head(d.span().subspan(0, 100), q);

// --- Output buffer (uninitialized by default — no memset) ---
dpl::device_array<float> output(1024, q);
oneapi::dpl::transform(policy, d.span().begin(), d.span().end(), output.span().begin(),
               [](float x) { return x * 2.0f; });
// or, range-based, with both sides as spans
oneapi::dpl::ranges::transform(policy, d.span(), output.span(),
                               [](float x) { return x * 2.0f; });

// --- Zero-initialized allocation (opt-in) ---
dpl::device_array<float> zeroed(1024, 0.0f, q);


## Resolved Questions

- **Should async overloads be in the initial proposal or deferred?** 
    No, while this provides more control over synchronization, it complicates the interface too much for the initial API.
