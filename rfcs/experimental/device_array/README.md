# `device_array<T>` — Simplified Device Memory Container

`device_array` provides RAII device allocation, explicit host-device transfers,
and direct use of spans over raw USM pointers. It emphasizes convenience of
use while making host-side use explicit.

It is intended to be minimal in its design: no allocator, no resizing, no
asynchronous access.

This focus provides support for the main usage pattern for users of `device_vector`,
and fits nicely within SYCL while avoiding much of the complexity of `device_vector`.

`device_array` and its default allocator, `device_allocator`, are released as
experimental features in `oneapi::dpl::experimental`. They are the first part of the
broader device container direction described by the
[device containers RFC](../../proposed/device_vector/README.md), which holds the full
motivation, the [usage study](../../proposed/device_vector/usage_pattern_study.md), the
comparison of existing implementations, and the design decisions shared with
[`compat::device_vector`](../../proposed/device_vector/device_vector_compat.md). That
companion type is still only *proposed* and is not implemented; this document covers
`device_array` and `device_allocator` only.

## Relationship to the shared base

`device_array` and the proposed `compat::device_vector` share their implementation through
a non-public base, `oneapi::dpl::__internal::__device_storage_base<T, Alloc>`,
which owns the device allocation and its lifetime, the size, the associated
`sycl::context` / `sycl::device`, the allocator instance, resizing, and the
host-device transfer helpers.

`device_array` **privately inherits** from
`__internal::__device_storage_base<T, device_allocator<T>>` and re-exposes (via `using`
declarations) only the fixed-size subset of that interface. The allocator type
is fixed to the default `device_allocator<T>` and is not part of
`device_array`'s public API; resizing, `reserve`, `capacity`, and `clear` are
not re-exposed. This lets `device_array` present a simplified interface while
`compat::device_vector` can later reuse the same base to offer a resizable,
allocator-aware container.

Only the parts of the base that `device_array` needs are implemented so far; the resizing
machinery listed above is part of the `device_vector` proposal, not of this feature.

## Availability

`device_array` is defined only when a SYCL backend is enabled
(`_ONEDPL_BACKEND_SYCL`) and a span type is available (`_ONEDPL_SPAN_PRESENT` — see
[`oneapi::dpl::span`](#oneapidplspan)). Without a span there is no way to express the
interface, so the class is not declared at all rather than declared and unusable.

## Requirements on `T`

The shared base static asserts both of these:

- `sycl::is_device_copyable_v<T>` — every transfer is a `sycl::queue::memcpy` or
  `sycl::queue::fill`.
- `T` is a non-`const`, non-reference, non-`void` object type.

`to_vector()` additionally requires `std::is_default_constructible_v<T>`, but only when
it is called; `read_at()` deliberately does not, so a
`device_array<NonDefaultConstructible>` remains fully usable minus that one convenience.

## API

```cpp
namespace oneapi::dpl::experimental {

template <typename T>
class device_array : private __internal::__device_storage_base<T, device_allocator<T>> {
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

    // No copy — avoids accidental deep copies. A deep copy can be done
    // explicitly with the span constructor above: device_array<T> b(a.span(), q);
    device_array(const device_array&) = delete;
    device_array& operator=(const device_array&) = delete;

    // Move (shallow move, device memory remains where it is, but changes ownership).
    // A moved-from device_array is empty: size() == 0 and empty() is true. It may be
    // destroyed or used as the target of a move assignment, but other usage is
    // undefined behavior.
    //
    // Conditionally noexcept: SYCL 2020 does not specify the move operations of
    // sycl::context or sycl::device as noexcept, so these are noexcept exactly when
    // those are.
    device_array(device_array&&) noexcept(/* see above */);
    device_array& operator=(device_array&&) noexcept(/* see above */);

    // deallocates device memory
    ~device_array();

    // Host-device transfer
    //
    // Argument order is uniform: what is transferred, then an optional offset into the
    // container, then an optional queue, then an optional event to depend on. Each
    // operation comes in three forms (two for the single-element writes, where the
    // offset is not optional):
    //
    //   (data, offset)              -- queue-less; uses a queue built on demand from the
    //                                 stored context and device
    //   (data, queue, depends_on)   -- offset defaults to 0
    //   (data, offset, queue, depends_on)
    //
    // All of them block until the transfer completes.
    //
    // Preconditions: for the bulk overloads, offset <= size(); an offset equal to
    // size() is well-formed and transfers zero elements. For the single-element
    // overloads, pos < size(). Violating either throws std::out_of_range rather
    // than reading or writing out of bounds.
    //
    // The bulk overloads return the number of elements actually transferred,
    // min(other.size(), size() - offset), which may be less than requested.

    // Transfer from device (dst may be host memory or USM on this context)
    size_type copy_to(oneapi::dpl::span<T> dst, size_type src_offset = 0) const;
    // overload to support queue with defaulted offset = 0
    size_type copy_to(oneapi::dpl::span<T> dst, sycl::queue q,
                      sycl::event depends_on = {}) const;
    size_type copy_to(oneapi::dpl::span<T> dst, size_type src_offset, sycl::queue q,
                      sycl::event depends_on = {}) const;

    // single element
    T read_at(size_type pos) const;
    T read_at(size_type pos, sycl::queue q, sycl::event depends_on = {}) const;

    // Convenience download into a fresh host vector.
    // Additionally requires std::is_default_constructible_v<T>. With a
    // non-default-constructible T, use copy_to() into storage the caller owns.
    std::vector<T> to_vector() const;
    std::vector<T> to_vector(sycl::queue q, sycl::event depends_on = {}) const;

    // Transfer to device (src may be host memory or USM on this context)
    size_type copy_from(oneapi::dpl::span<const T> src, size_type dst_offset = 0);
    // overload to support queue with defaulted offset = 0
    size_type copy_from(oneapi::dpl::span<const T> src, sycl::queue q,
                        sycl::event depends_on = {});
    size_type copy_from(oneapi::dpl::span<const T> src, size_type dst_offset, sycl::queue q,
                        sycl::event depends_on = {});

    // single element
    // The offset is a required, leading argument, dst_offset == size() throws.
    void write_at(size_type dst_offset, const T& value);
    void write_at(size_type dst_offset, const T& value, sycl::queue q,
                  sycl::event depends_on = {});

    // Capacity (fixed size — no resize / reserve / capacity / clear)
    size_type size()  const;
    bool      empty() const;

    // Note: there is deliberately no data() member; see "Iterator access" below.

    void swap(device_array& other);

    // Views
    oneapi::dpl::span<T>       span();
    oneapi::dpl::span<const T> span() const;

    // Context / device access
    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

template <typename T> void swap(device_array<T>& a, device_array<T>& b);

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

Span iterators should not be passed to a oneDPL iterator API with a device policy.
`std::span<T>::iterator` is an implementation defined iterator, which isn't guaranteed
to be `oneapi::dpl::is_indirectly_device_accessible`. What we want for iterator APIs
with a device policy are pointers to USM memory (use `oneapi::dpl::begin/end`).

There is deliberately **no** `data()` member. `sycl::span`'s C++17 container constructor
only needs `data()` and `size()` to be findable, so for a `T` from the standard library a
public `data()` would make an expression such as `d2.copy_from(d)` — passing a
`device_array` where a `span` is expected — compile under C++17 and fail under C++20,
where `std::span`'s corresponding constructor is constrained. A raw pointer is available
as `oneapi::dpl::begin(d)` or `d.span().data()`.

## `device_allocator`

`device_array` fixes its allocator to `device_allocator<T>` and does not expose it.
Pluggable allocation via the proposed `DeviceAllocator` concept is a `device_vector`
feature; see the
[device_vector allocator requirements](../../proposed/device_vector/device_vector_compat.md#allocator).
`device_allocator` itself, however, ships with `device_array` and is public, so it is
specified here.

Because the allocator is stateful and not default constructible, each `device_array`
constructor builds one from the same context and device it is given.

```cpp
namespace oneapi::dpl::experimental {

template <typename T, std::size_t Alignment = 0>
class device_allocator {
public:
    using value_type = T;
    using size_type  = std::size_t;


    explicit device_allocator(sycl::context ctx, sycl::device dev,
                              const sycl::property_list& prop_list = {});
    explicit device_allocator(sycl::queue q,
                              const sycl::property_list& prop_list = {});

    // Converting constructor; carries the allocation target over. Conditionally noexcept:
    // SYCL does not guarantee noexcept copies of context, device or property_list.
    template <typename U>
    device_allocator(const device_allocator<U, Alignment>& other) noexcept(/* see above */);

    // Alignment == 0 uses sycl::malloc_device; otherwise sycl::aligned_alloc_device,
    // which itself raises the alignment to max(Alignment, alignof(T)).
    T*   allocate(size_type count) const;
    void deallocate(T* ptr, size_type count) const;
};

// Two device allocators compare equal if they share an alignment, a context and a device,
// following the requirement SYCL 2020 section 4.8.3.1 places on sycl::usm_allocator. As
// with sycl::usm_allocator, the value type and the property list do not participate.
template <typename T, std::size_t AlignmentT, typename U, std::size_t AlignmentU>
bool operator==(const device_allocator<T, AlignmentT>&,
                const device_allocator<U, AlignmentU>&) noexcept;
template <typename T, std::size_t AlignmentT, typename U, std::size_t AlignmentU>
bool operator!=(const device_allocator<T, AlignmentT>&,
                const device_allocator<U, AlignmentU>&) noexcept;

} // namespace oneapi::dpl::experimental
```

The API deliberately mirrors `sycl::usm_allocator`: stateful, carrying the `sycl::context`,
`sycl::device` and `sycl::property_list` to allocate against, so `allocate()` takes only an
element count. `sycl::usm_allocator` itself cannot serve this role — it contains
`static_assert(AllocKind != sycl::usm::alloc::device)`, because device memory is not
host-accessible and so cannot satisfy the `std::allocator` named requirements that
`usm_allocator` is built to satisfy. `device_allocator` provides only
`allocate`/`deallocate`, imposes none of those requirements, and correspondingly cannot be
used with the standard containers.

`rebind` and the `propagate_on_container_*` members are **not** provided. Nothing in
`device_array` needs them, so they are deferred to when `device_vector` is implemented,
where a container-level allocator contract first becomes observable.

Other behavior:

- `allocate(0)` returns `nullptr` and allocates nothing. `sycl::malloc_device(0)` is
  unspecified, and some backends return a non-null pointer that cannot be freed.
- `deallocate(nullptr, n)` is a no-op. The count is accepted to match the allocator
  convention and is ignored, USM deallocation needs only the pointer and the context.
- Allocation failure surfaces as a `sycl::exception` carrying
  `sycl::errc::memory_allocation`. The exception is raised by `device_allocator`, not
  propagated: `sycl::malloc_device` and `sycl::aligned_alloc_device` return `nullptr` on
  failure rather than throwing, both when resources are insufficient and when the
  requested alignment is unsupported, so a null result is translated into the exception.
  It is deliberately not translated to `std::bad_alloc`, which would discard the
  backend's diagnostics.
- Deallocation cannot fail observably: the destructor and the move assignment 
  operator catch and discard any exception from `sycl::free`, which is what
  lets them be (conditionally) `noexcept`. See the [open question](#open-questions)
   on this.

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

Neither alternative is guaranteed. `std::span` requires C++20. `sycl::span` is required by
SYCL 2020 3.9.2 but is missing from some implementations, so under C++17 there may be no
span to alias at all. In that case, `device_array` is  undefined in that configuration.

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
float val = d.read_at(0, q);       // synchronous read
d.write_at(0, 42.0f, q);           // synchronous write; the offset is required here

// --- Bulk download ---
std::vector<float> out = d.to_vector(q);   // fresh vector
std::size_t n = d.copy_to(out, q);         // or into existing storage; returns the
                                           // count actually copied, which is
                                           // min(out.size(), d.size())

// --- Bulk upload into an existing device_array (does not resize) ---
d.copy_from(host_data, q);                 // return value may be discarded

// --- Offset transfers: copy the tail of d into the front of out ---
d.copy_to(out, 100, q);                    // d[100 .. 100 + n) -> out[0 .. n)
                                           // n = min(out.size(), d.size() - 100)
                                           // throws std::out_of_range if 100 > d.size()

// --- Offset upload: write host_data into d starting at element 100 ---
d.copy_from(host_data, 100, q);            // truncated at the end of d

// --- Device-to-device: span() is USM, so it is a valid source ---
dpl_exp::device_array<float> d2(d.size(), q);
d2.copy_from(d.span(), q);                 // span<float> -> span<const float>

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
d3.copy_to(ooo_out, ooo_q, e);             // offset defaults to 0

// same for uploads and single-element access
d3.copy_from(host_data, ooo_q, e);
float first = d3.read_at(0, ooo_q, e);
d3.write_at(0, 1.0f, ooo_q, e);

// with an in-order queue the parameter is unnecessary — prior submissions on q
// already order against the transfer
d.copy_to(out, q);

// --- No queue at all: a queue is built on demand from the stored context + device ---
std::vector<float> out2 = d.to_vector();
d.copy_from(host_data);
```

## Resolved Questions

- **Should async overloads be in the initial proposal or deferred?**
    No, while this provides more control over synchronization, it complicates the
    interface too much for the initial API.

- **Should member functions which take a `sycl::queue` for synchronization also take an
  optional `sycl::event depends_on` parameter?**
    Yes, and it is implemented that way on every queue-taking member. It is what makes
    the container usable with an out-of-order queue without forcing the user to insert
    their own barrier, and it costs nothing when unused: a default-constructed
    `sycl::event` is already complete, so the parameter is forwarded unconditionally with
    no branch and no special case.

## Open Questions

- **Should a failure to release device memory be silenced?**
  The current behavior is the baseline but still open for discussion.
  `__device_storage_base::__deallocate()` wraps the `sycl::free` call in a
  `try`/`catch(...)` that discards the exception, so a failed release becomes a silent
  resource leak. The pointer and size are reset before the allocator call, so the object
  is left consistent either way.

  This is what makes deallocation usable from the destructor and from the
  (conditionally) `noexcept` move assignment operator, and it matches
  `thrust::device_vector`, which likewise cannot report a failed release out of its
  destructor. The alternatives:
  1) Remove noexcept from move operator, allow the exception to result in a crash for destructor
  2) Catch and report the error, at least in debug mode, but continue on without interruption
  3) Add a finalize() / free() method. This defeats a main motivation for the container (convenient RAII
     allocation/deallocation) to handle what should be a rare occurance, and one which may not be
     otherwise recoverable, like a lost device in many cases.

## Exit Criteria

`device_array` and `device_allocator` should become fully supported if:

- The open question above is resolved, or has a justification recorded for keeping the
  current behavior.
- There is positive adoption feedback, in particular that the explicit transfer interface
  (`copy_to` / `copy_from` / `read_at` / `write_at` plus `span()`) covers the
  construct / bulk-transfer / raw-pointer pattern the
  [usage study](../../proposed/device_vector/usage_pattern_study.md) identified as
  dominant.
- The interface for `device_allocator` has held up against the
  [`compat::device_vector`](../../proposed/device_vector/device_vector_compat.md)
  implementation. Implementing it is a real test of whether the shared base
  and the `DeviceAllocator` contract. Changes needed there may change `device_allocator`
  and/or `device_array`.
