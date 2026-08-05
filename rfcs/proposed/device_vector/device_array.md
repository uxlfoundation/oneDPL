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

    device_array(oneapi::dpl::span<const T> src, sycl::queue q, sycl::event depends_on = {});
    device_array(oneapi::dpl::span<const T> src, sycl::context ctx, sycl::device dev);

    // No copy — avoids accidental deep copies. A deep copy is can be done 
    // explicitly with the span constructor above: device_array<T> b(a.span(), q);
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;

    // Move (shallow move, device memory remains where it is)
    device_array(device_array&&) noexcept;
    device_array& operator=(device_array&&)  noexcept;

    // deallocates device memory
    ~device_array();

    // Host-device transfer
    // Transfer from device (dst may be host memory or USM on this context)
    // copies min(dst.size(), size() - src_offset) elements
    void copy_to(oneapi::dpl::span<T> dst, size_type src_offset = 0) const;
    // overload to support queue with defaulted offset = 0
    void copy_to(oneapi::dpl::span<T> dst, sycl::queue q, sycl::event depends_on = {}) const;
    void copy_to(oneapi::dpl::span<T> dst, size_type src_offset, sycl::queue q, sycl::event depends_on = {}) const;

    // single element
    T read_at(size_type pos) const;
    T read_at(sycl::queue q, size_type pos, sycl::event depends_on = {}) const;

    // Convenience download into a fresh host vector
    std::vector<T> to_vector() const;
    std::vector<T> to_vector(sycl::queue q, sycl::event depends_on = {}) const;

    // Transfer to device (src may be host memory or USM on this context)
    // copies min(src.size(), size() - dst_offset) elements
    void copy_from(oneapi::dpl::span<const T> src, size_type dst_offset = 0);
    // overload to support queue with defaulted offset = 0
    void copy_from(oneapi::dpl::span<const T> src, sycl::queue q, sycl::event depends_on = {});

    void copy_from(oneapi::dpl::span<const T> src, size_type dst_offset, sycl::queue q, sycl::event depends_on = {});

    // single element
    void copy_from(const T& value, size_type dst_offset = 0);
    // overload to support queue with defaulted offset = 0
    void copy_from(const T& value, sycl::queue q, sycl::event depends_on = {});
    void copy_from(const T& value, size_type dst_offset, sycl::queue q, sycl::event depends_on = {});

    // Capacity (fixed size — no resize / reserve / capacity / clear)
    size_type size()  const;
    bool      empty() const;

    void swap(device_array& other) noexcept;

    // Views
    oneapi::dpl::span<T>       span();
    oneapi::dpl::span<const T> span() const;

    // Context / device access
    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

template <typename T> void swap(device_array<T>& a, device_array<T>& b) noexcept;

} // namespace oneapi::dpl::experimental

namespace oneapi::dpl {

// Note the namespace: these are in oneapi::dpl, not oneapi::dpl::experimental, so ADL
// does not find them and calls must be qualified. By reference rather than by value,
// since device_array is not copyable.
template <typename T> T*       begin(experimental::device_array<T>& d);
template <typename T> T*       end  (experimental::device_array<T>& d);
template <typename T> const T* begin(const experimental::device_array<T>& d);
template <typename T> const T* end  (const experimental::device_array<T>& d);

} // namespace oneapi::dpl
```

### Iterator access

Iterators come from the non-member `oneapi::dpl::begin` / `oneapi::dpl::end` overloads,
which return raw `T*` (and `const T*` for a const `device_array`).

Span iterators should not be passed to a oneDPL iterator API with a device policy. `std::span<T>::iterator` is an implementation defined iterator, which isnt guaranteed to be
`oneapi::dpl::is_indirectly_device_accessible`. What we want for iterator APIs with a device policy are pointers to USM memory (use `oneapi::dpl::begin/end`).

## Allocator

`device_array` fixes its allocator to the default `device_allocator<T>` and does not expose it. Pluggable
allocation via the `DeviceAllocator` concept is available on
[`compat::device_vector`](device_vector_compat.md#allocator). Allocation via
`sycl::malloc_device` during construction can result in a `sycl::exception`.

## `oneapi::dpl::span`

The span type used throughout this API is an alias which resolves to
`std::span` when the standard library provides it, and falls back to
`sycl::span` otherwise:

```cpp
// onedpl_config.h
#if _ONEDPL_STD_FEATURE_MACROS_PRESENT
#    define _ONEDPL_CPP20_SPAN_PRESENT (_ONEDPL___cplusplus >= 202002L && __cpp_lib_span >= 202002L)
#else
#    define _ONEDPL_CPP20_SPAN_PRESENT 0
#endif

namespace oneapi::dpl {

#if _ONEDPL_CPP20_SPAN_PRESENT
inline constexpr std::size_t dynamic_extent = std::dynamic_extent;

template <typename T, std::size_t Extent = dynamic_extent>
using span = std::span<T, Extent>;
#else
inline constexpr std::size_t dynamic_extent = sycl::dynamic_extent;

template <typename T, std::size_t Extent = dynamic_extent>
using span = sycl::span<T, Extent>;
#endif

} // namespace oneapi::dpl
```

Both alternatives are usable for the purposes of this API: `sycl::span` is
guaranteed to be present in SYCL 2020, pre-adopted from c++20 and is
device_copyable.

Preferring `std::span` where available keeps oneDPL's interface in terms of a
standard type rather than a SYCL-specific one, so spans obtained from
`device_array` compose with the rest of a user's C++20 code (and with
`std::ranges`) without a conversion step. It also avoids some issues in the
`sycl::span` when combined with some range features in the current
implementation.

## Use within kernels

`device_array` is not device-copyable (it owns memory). For kernel capture,
non-owning views, and range composition, use `oneapi::dpl::span<T>` via
`.span()`, which is device copyable.

## Usage Examples

```cpp
#include <oneapi/dpl/experimental/device_array>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

namespace dpl_exp = oneapi::dpl::experimental;

sycl::queue q{sycl::property::queue::in_order{}};

// --- RAII allocation + upload from host ---
// host_data converts implicitly to oneapi::dpl::span<const float>; size is taken from it
std::vector<float> host_data(1024, 3.14f);
dpl_exp::device_array<float> d(host_data, q);

// --- Use with oneDPL algorithms ---
auto policy = oneapi::dpl::execution::make_device_policy(q);
// iterators come from oneapi::dpl::begin/end (raw pointers), never from span()
oneapi::dpl::sort(policy, oneapi::dpl::begin(d), oneapi::dpl::end(d));

// --- Pass the span directly to the oneDPL range-based algorithms ---
oneapi::dpl::ranges::for_each(policy, d.span(), [](float& x) { x += 1.0f; });

oneapi::dpl::ranges::sort(policy, d.span());

oneapi::dpl::ranges::sort(policy, d.span().subspan(0, 100));

// --- Use in a SYCL kernel ---

auto s = d.span();
q.parallel_for(sycl::range<1>(s.size()), [=](sycl::id<1> i) {
    s[i] *= 2.0f;
}).wait();

float* ptr = d.span().data();
q.parallel_for(sycl::range<1>(d.size()), [=](sycl::id<1> i) {
    ptr[i] *= 2.0f;
}).wait();

// --- Explicit single-element host access ---
float val = d.read_at(q, 0);       // synchronous read
d.copy_from(q, 42.0f, 0);          // synchronous write

// --- Bulk download ---
std::vector<float> out = d.to_vector(q);   // fresh vector
d.copy_to(q, out);                         // or into existing storage
                                           // copies min(out.size(), d.size()) elements

// --- Bulk upload into an existing device_array (does not resize) ---
d.copy_from(q, host_data);

// --- Offset transfers: copy the tail of d into the front of out ---
d.copy_to(q, out, 100);                    // d[100 .. 100 + n) -> out[0 .. n)
                                           // n = min(out.size(), d.size() - 100)

// --- Offset upload: write host_data into d starting at element 100 ---
d.copy_from(q, host_data, 100);            // truncated at the end of d

// --- Device-to-device: span() is USM, so it is a valid source ---
dpl_exp::device_array<float> d2(d.size(), q);
d2.copy_from(q, d.span());                 // span<float> -> span<const float>

// --- Full deep copy: device_array is not copyable, do it with span() ---
dpl_exp::device_array<float> d_copy(d.span(), q);

// --- Subrange copy into a new, smaller device_array ---
dpl_exp::device_array<float> head(d.span().subspan(0, 100), q);

// --- Move transfers ownership; no allocation, no data movement ---
dpl_exp::device_array<float> d_moved = std::move(d_copy);

// --- Output buffer (uninitialized by default — no memset) ---
dpl_exp::device_array<float> output(1024, q);
oneapi::dpl::transform(policy, oneapi::dpl::begin(d), oneapi::dpl::end(d),
                       oneapi::dpl::begin(output),
                       [](float x) { return x * 2.0f; });

// --- Zero-initialized allocation (opt-in) ---
dpl_exp::device_array<float> zeroed(1024, 0.0f, q);

// --- Out-of-order queue: chain transfers onto an existing event ---
sycl::queue ooo_q;                             // out-of-order by default
dpl_exp::device_array<float> d3(1024, ooo_q);

// a kernel the user submitted themselves; nothing orders it against d3's
// transfers on an out-of-order queue
sycl::event e = ooo_q.parallel_for(sycl::range<1>(d3.size()),
                                   [s = d3.span()](sycl::id<1> i) { s[i] = i; });

// the copy waits on e before reading, then blocks until the copy completes
std::vector<float> ooo_out(d3.size());
d3.copy_to(ooo_q, ooo_out, 0, e);

// same for uploads and single-element access
d3.copy_from(ooo_q, host_data, 0, e);
float first = d3.read_at(ooo_q, 0, e);

// with an in-order queue the parameter is unnecessary — prior submissions on q
// already order against the transfer
d.copy_to(q, out);
```

## Resolved Questions

- **Should async overloads be in the initial proposal or deferred?** 
    No, while this provides more control over synchronization, it complicates the interface too much for the initial API.

## Open Questions
- ** Should member functions which include a `sycl::queue` for synchronization also include an optional `sycl::event depends_on` parameter for event based synchronization?
  - The idea here is for out-of-order queue synchronization with existing workflows, I've added this into the proposal for now.
