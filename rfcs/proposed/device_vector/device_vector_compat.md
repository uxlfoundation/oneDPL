# `device_vector<T>` Compatibility Layer

A Thrust-compatible device-memory vector, living in a
compatibility namespace. Adds `device_pointer`, `device_reference`, and
implicit host-access semantics on top of `device_array`'s explicit API.

See the [usage study](usage_pattern_study.md) for evidence on which Thrust APIs
are actually used, and [device_array](device_array.md) for the underlying
container.

The target for this compatibility layer is to be as close to a 
drop-in replacement as we can have for `thrust::device_vector` for the
functionality that people actually use, and in a way that fits within SYCL.

## Relationship to `device_array`

`compat::device_vector<T, Alloc>` contains a `device_array<T, Alloc>`, adding features like and std::vector-like functions.

It uses an iterator/pointer type, `device_pointer`, as a wrapper for USM memory, and reference type, `device_reference`, as a reference proxy type to enable host-side usage with implicit memory transfers. These types hold a pointer to a `sycl::context` to facilitate creation of a queue for memcpy.

## Namespace

```cpp
namespace oneapi::dpl::experimental::compat {

template <typename T, typename Alloc = device_allocator<T>>
class device_vector;

template <typename T>
class device_pointer;

template <typename T>
class device_reference;


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
| `d.resize(N)` | Yes | |
| `d.size()` / `d.empty()` | Yes |  |
| `d.clear()` | Yes |  |
| `h_vec = d` / `d = h_vec` | Yes | Bulk transfer |
| Custom allocator | Yes | `Alloc` forwarded to `device_array<T, Alloc>` |
| `push_back` / `insert` / `erase` | **No** | Rarely used in practice |

## API

```cpp
namespace oneapi::dpl::experimental::compat {

// =========================================================================
// device_pointer<T>
// =========================================================================
// Wraps a raw T* from device_array. Dereference provides device_reference.

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

    /* Full random access iterator arithmetic + comparison*/

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
    /* all other compound assignments... */

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


    // construction from size
    device_vector(size_type count, sycl::context ctx, sycl::device dev);
    // construction from size and value
    device_vector(size_type count, const T& value,
                  sycl::context ctx, sycl::device dev);

    //construction from iterators
    template <typename InputIt>
    device_vector(InputIt first, InputIt last,
                  sycl::context ctx, sycl::device dev);
    //construction from initializer_list
    device_vector(std::initializer_list<T> init,
                  sycl::context ctx, sycl::device dev);
    construction from std::vector
    explicit device_vector(const std::vector<T>& src,
                           sycl::context ctx, sycl::device dev);

    /* Copy of all above constructors for `sycl::queue` (extracts context + device), using no_init_t to avoid initialization, and with explicit allocator */


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
