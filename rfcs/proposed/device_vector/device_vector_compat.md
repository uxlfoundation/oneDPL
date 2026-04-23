# `device_vector<T>` Compatibility Layer

A Thrust-compatible wrapper layered on `device_array<T, Alloc>`, living in a
compatibility namespace. Adds `device_pointer`, `device_reference`, and
implicit host-access semantics on top of `device_array`'s explicit API.

See the [usage study](usage_pattern_study.md) for evidence on which Thrust APIs
are actually used, and [device_array](device_array.md) for the underlying
container.

## Relationship to `device_array`

`compat::device_vector<T, Alloc>` contains a `device_array<T, Alloc>`. Like
`device_array`, it stores `sycl::context` + `sycl::device`, not a queue. It adds:

- **`device_pointer<T>`** — wraps the raw `T*` from `device_array::begin()/end()`
  to provide a typed pointer that distinguishes device memory from host memory.
  Stores a pointer to the owning context (8 bytes overhead), which together with
  the USM pointer is sufficient to look up the device and create a temporary
  queue for host-side transfers.
- **`device_reference<T>`** — proxy returned by `operator[]` and
  `device_pointer` dereference, enabling Thrust-style `d[i] += 1` on the host
- **Host/device pointer dispatch** — `device_pointer` in range constructors
  enables automatic detection of device-to-device vs host-to-device copies

The raw `T*` iterators from the underlying `device_array` are wrapped into
`device_pointer<T>` on their way out of `device_vector`.

## Namespace

```cpp
namespace oneapi::dpl::experimental::compat {

template <typename T, typename Alloc = device_allocator<T>>
class device_vector;

template <typename T>
class device_pointer;

template <typename T>
class device_reference;

// Pointer cast utilities
template <typename T>
T* raw_pointer_cast(device_pointer<T> ptr);

template <typename T>
device_pointer<T> device_pointer_cast(T* ptr);

} // namespace oneapi::dpl::experimental::compat
```

## Supported Thrust Patterns

| Thrust Pattern | Supported | Notes |
|---|---|---|
| `device_vector<T> d(N)` | Yes | Context + device (or queue) required |
| `device_vector<T> d(N, val)` | Yes | |
| `device_vector<T> d = host_vec` | Yes | |
| `device_vector<T> d(ptr, ptr+N)` | Yes | Host pointer range |
| `device_vector<T> d2(d1.begin(), d1.end())` | Yes | `device_pointer` distinguishes D2D copy |
| `d.begin()` / `d.end()` in algorithms | Yes | Returns `device_pointer<T>` |
| `d.data().get()` / `raw_pointer_cast` | Yes | `device_pointer::get()` returns raw `T*` |
| `d[i]` (host read) | Yes | `device_reference` proxy |
| `d[i] = val` (host write) | Yes | |
| `d[i] += val` (compound assign) | Yes | Synchronous read-modify-write |
| `d.resize(N)` | Yes | Forwards to `device_array::resize` |
| `d.size()` / `d.empty()` | Yes | Forwards directly |
| `d.clear()` | Yes | Forwards directly |
| `h_vec = d` / `d = h_vec` | Yes | Bulk transfer |
| Custom allocator | Yes | `Alloc` forwarded to `device_array<T, Alloc>` |
| `push_back` / `insert` / `erase` | **No** | Rarely used in practice |

## API

```cpp
namespace oneapi::dpl::experimental::compat {

// =========================================================================
// device_pointer<T>
// =========================================================================
// Wraps a raw T* from device_array. On the device, dereferences directly
// as a USM pointer. On the host, dereference produces a device_reference
// that uses the stored context pointer to look up the device (via
// sycl::get_pointer_device) and create a temporary queue for transfers.
// This adds 8 bytes over a raw pointer (the context pointer).

template <typename T>
class device_pointer {
    T* __ptr = nullptr;
    const sycl::context* __ctx = nullptr;  // non-owning, from device_vector

public:
    using iterator_concept  = std::random_access_iterator_tag;
    using value_type        = std::remove_cv_t<T>;
    using difference_type   = std::ptrdiff_t;
    using reference         = device_reference<T>;

    device_pointer() = default;
    explicit device_pointer(T* ptr, const sycl::context* ctx = nullptr);

    // Raw pointer access — unwraps back to the T* that device_array uses
    T* get() const;

    reference operator*() const;
    reference operator[](difference_type n) const;

    // Full random access iterator arithmetic + comparison
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

    friend bool operator==(device_pointer a, device_pointer b);
    friend auto operator<=>(device_pointer a, device_pointer b);
};

// =========================================================================
// device_reference<T>
// =========================================================================

template <typename T>
class device_reference {
public:
    operator T() const;                                       // read
    const device_reference& operator=(const T& val) const;    // write
    const device_reference& operator=(const device_reference&) const;

    // Compound assignment (each is a synchronous read-modify-write)
    const device_reference& operator+=(const T&) const;
    const device_reference& operator-=(const T&) const;
    const device_reference& operator*=(const T&) const;
    const device_reference& operator/=(const T&) const;
    const device_reference& operator%=(const T&) const;
    const device_reference& operator&=(const T&) const;
    const device_reference& operator|=(const T&) const;
    const device_reference& operator^=(const T&) const;
    const device_reference& operator<<=(const T&) const;
    const device_reference& operator>>=(const T&) const;

    const device_reference& operator++() const;
    T operator++(int) const;
    const device_reference& operator--() const;
    T operator--(int) const;

    device_pointer<T> operator&() const;

    friend void swap(const device_reference& a, const device_reference& b);
};

// =========================================================================
// device_vector<T, Alloc>
// =========================================================================

template <typename T, typename Alloc = device_allocator<T>>
class device_vector {
    device_array<T, Alloc> __impl;  // stores context + device + allocator

public:
    using value_type      = T;
    using allocator_type  = Alloc;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = device_reference<T>;
    using const_reference = device_reference<const T>;
    using pointer         = device_pointer<T>;
    using const_pointer   = device_pointer<const T>;
    using iterator        = device_pointer<T>;
    using const_iterator  = device_pointer<const T>;

    // --- Construction from queue (extracts context + device) ---
    explicit device_vector(sycl::queue q);
    device_vector(size_type count, sycl::queue q);
    device_vector(size_type count, const T& value, sycl::queue q);
    device_vector(size_type count, no_init_t, sycl::queue q);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last, sycl::queue q);
    device_vector(std::initializer_list<T> init, sycl::queue q);
    explicit device_vector(const std::vector<T>& src, sycl::queue q);

    // --- Construction from context + device ---
    explicit device_vector(sycl::context ctx, sycl::device dev);
    device_vector(size_type count, sycl::context ctx, sycl::device dev);
    device_vector(size_type count, const T& value,
                  sycl::context ctx, sycl::device dev);
    device_vector(size_type count, no_init_t,
                  sycl::context ctx, sycl::device dev);
    template <typename InputIt>
    device_vector(InputIt first, InputIt last,
                  sycl::context ctx, sycl::device dev);
    device_vector(std::initializer_list<T> init,
                  sycl::context ctx, sycl::device dev);
    explicit device_vector(const std::vector<T>& src,
                           sycl::context ctx, sycl::device dev);

    // Allocator-aware construction
    device_vector(size_type count, sycl::queue q, const Alloc& alloc);
    device_vector(size_type count, sycl::context ctx, sycl::device dev,
                  const Alloc& alloc);

    // Copy / move
    device_vector(const device_vector&);
    device_vector(device_vector&&) noexcept;
    device_vector& operator=(const device_vector&);
    device_vector& operator=(device_vector&&) noexcept;

    ~device_vector();

    // Assign from host vector (bulk upload)
    device_vector& operator=(const std::vector<T>& src);

    // Convert to host vector (bulk download)
    explicit operator std::vector<T>() const;

    // --- Element access (proxy references) ---
    reference       operator[](size_type pos);
    const_reference operator[](size_type pos) const;
    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;

    // --- Pointer access (device_pointer wrapping device_array's T*) ---
    pointer       data();
    const_pointer data() const;

    // --- Iterators (device_pointer wrapping device_array's T*) ---
    iterator       begin();
    const_iterator begin() const;
    iterator       end();
    const_iterator end() const;

    // --- Capacity (forwarded to device_array) ---
    size_type size()     const;
    size_type capacity() const;
    bool      empty()    const;

    void resize(size_type count);
    void resize(size_type count, const T& value);
    void resize(size_type count, no_init_t);
    void reserve(size_type new_cap);
    void clear();

    // --- Swap ---
    void swap(device_vector& other);

    // --- Access to underlying device_array ---
    device_array<T, Alloc>&       base();
    const device_array<T, Alloc>& base() const;

    // --- Allocator ---
    allocator_type get_allocator() const;

    // --- Context / device ---
    sycl::context get_context() const;
    sycl::device  get_device()  const;
};

} // namespace oneapi::dpl::experimental::compat
```

## Key Differences from Thrust

1. **Context + device (or queue) always required** — no implicit default device.
2. **No `push_back`, `insert`, `erase`** — rarely used, high complexity.
3. **No `host_vector` type** — use `std::vector<T>` directly.
4. **No tag dispatch** — execution policies determine where algorithms run.

## Design Notes

**Layering on `device_array`:**
`device_vector` delegates all memory management, capacity logic, and bulk
transfers to its contained `device_array`. The compat layer's job is limited
to:
- Wrapping `T*` → `device_pointer<T>` for begin/end/data
- Providing `operator[]` → `device_reference<T>` proxy
- Handling host vs device pointer dispatch in range constructors

This keeps the RAII and allocation logic in one place (`device_array`) and the
Thrust compatibility surface as a thin adapter.

**`device_pointer` context association for host dereference:**
`device_pointer` stores a non-owning pointer to the `sycl::context` from
the owning `device_vector` (via its `device_array`). On host-side dereference,
`device_reference` uses this context pointer along with the USM pointer to:
1. Look up the device via `sycl::get_pointer_device(ptr, ctx)`
2. Create a temporary queue from the context + device
3. Perform the memcpy transfer

This adds 8 bytes to `device_pointer` over a raw pointer. No queue is stored
anywhere — queues are created on demand for host-side transfers. This matches
the sycl-thrust approach (which also stores context, not queue) and aligns
with `device_array`'s design of not retaining a queue.

**Allocator forwarding:**
The `Alloc` template parameter is forwarded directly to `device_array<T, Alloc>`.
`device_vector` does not interact with the allocator itself — all allocation
goes through `device_array`. The same `DeviceAllocator` requirements apply
(see [device_array allocator section](device_array.md#allocator)).

**`base()` for incremental migration:**
Exposes the underlying `device_array` so users can mix proxy and explicit
access during migration:

```cpp
compat::device_vector<float> d(host_data, q);
// Thrust-style
float val = d[0];
// Explicit via device_array
float val2 = d.base().read(0, q);
```

## Usage Example

```cpp
#include <oneapi/dpl/device_vector>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

namespace compat = oneapi::dpl::experimental::compat;

sycl::queue q{sycl::property::queue::in_order{}};

// --- Thrust-like construction ---
std::vector<float> host_data(1024, 3.14f);
compat::device_vector<float> d(host_data, q);

// --- Thrust-like algorithm use ---
auto policy = oneapi::dpl::execution::make_device_policy(q);
std::sort(policy, d.begin(), d.end());

// --- Thrust-like element access (proxy, synchronous) ---
float val = d[0];          // implicit device-to-host
d[0] = 42.0f;              // implicit host-to-device
d[1] += 10.0f;             // read-modify-write round-trip

// --- Raw pointer for kernels ---
float* ptr = d.data().get();
q.parallel_for(sycl::range<1>(d.size()), [=](sycl::id<1> i) {
    ptr[i] *= 2.0f;
}).wait();

// --- Device-to-device copy via device_pointer ---
compat::device_vector<float> d2(d.begin(), d.end(), q);  // D2D copy

// --- Bulk copy back to host ---
std::vector<float> result = static_cast<std::vector<float>>(d);

// --- Construction from context + device (no queue needed) ---
sycl::context ctx = q.get_context();
sycl::device  dev = q.get_device();
compat::device_vector<float> d3(1024, ctx, dev);

// --- Gradual migration to device_array ---
auto& arr = d.base();
float v = arr.read(5, q);           // explicit, no proxy
arr.write(5, v * 2.0f, q);          // explicit
std::vector<float> out = arr.to_vector(q);
```

## Migration Path

```
thrust::device_vector<T>  →  compat::device_vector<T>  →  device_array<T>
         (CUDA)                (SYCL, Thrust-like API)     (SYCL, explicit API)
```

1. **Mechanical migration** — replace `thrust::device_vector` with
   `compat::device_vector`, add queue to constructors. `device_pointer`,
   `device_reference`, and `raw_pointer_cast` map directly. SYCLomatic could
   target this namespace.
2. **Incremental cleanup** — use `d.base()` to access explicit `read`/`write`,
   replace `device_pointer` iteration with raw `T*` via `.get()` or
   `d.base().begin()`.
3. **Full migration** — switch to `device_array<T>` for the clean, explicit
   API with no proxy overhead.
