# Storage Utilities for SYCL Backend Patterns in oneDPL

## Overview

The storage utilities provide a structured way to manage device and host memory
for SYCL backend patterns in oneDPL. They abstract over two underlying memory
backends — SYCL Unified Shared Memory (USM) and `sycl::buffer` — and handle
allocation, kernel access, result retrieval, and lifetime management in a
uniform way.

The utilities are defined in `namespace oneapi::dpl::__par_backend_hetero`.
Internal implementation details live in the nested
`namespace oneapi::dpl::__par_backend_hetero::__internal`.

The central design principle is a clean separation between the storage objects
that are created and used within a backend pattern, and the holder object that
owns the memory after the pattern completes. The holder is created by a caller
and outlives the pattern, carrying the results back to the caller.

```mermaid
flowchart TB
    subgraph CallerStart["Algorithm"]
        S1["Construct holder\n__parallel_pattern_holder&lt;N, Ts...&gt; __holder(__q)"]
        S2["Call a backend pattern\n__parallel_pattern_name(..., __holder, ...)"]

        subgraph BackendPattern["Backend Pattern"]
            BP1["Defines a holder alias\nusing __parallel_pattern_holder =\n__storage_holder&lt;NScratch, ResultTypes...&gt;"]
            BP2["Construct storage\n__device_storage&lt;T&gt;,\n__result_storage&lt;T&gt; or\n__combined_storage&lt;T&gt;"]
            BP3["Submit kernel(s)"]

            subgraph KernelSubmitter["Kernel Submitter"]
                KS1["Submit kernel\n__q.submit([&](sycl::handler& __cgh){...})"]
                KS2["Acquire accessor(s)\n__get_accessor(__mode, __storage, __cgh)\n__get_result_accessor(__mode, __storage, __cgh)"]
                KS3["Use storage in kernel\nauto* __ptr = __acc.__data()"]
                KS1 --> KS2 --> KS3
            end

            BP4["Transfer storage ownership\n__holder.__store&lt;Idx&gt;(std::move(__storage))\n__holder.__store_scratch(std::move(__storage))"]
            BP5["Return sycl::event"]

            BP3 -- "__storage&" --> KS1
            KS3 -- "sycl::event" --> BP4
            BP2 --> BP3
            BP4 --> BP5
        end

        FA3["Package result into\n__future{__event, std::move(__holder).__extract()}"]
        FS3["__event.wait_and_throw()"]
        FS4["Read result\n__holder.__copy_result&lt;Idx&gt;(__dst, __n)\nor __get_results(__holder)"]

        S1 -.- |uses the alias| BP1
        S1 --> S2
        FS3 --> FS4
        S2 -- "__storage_holder&" --> BP2
        BP5 -- "if sync" --> FS3
        BP5 -- "if async" --> FA3
    end
```

## Storage Types

Three storage types are provided, all built on a common base. Each type
allocates memory at construction time and releases it when destroyed.
Usually though ownership of the memory is transferred to a `__storage_holder`.

### `__device_storage<T>`

The base storage type for temporary scratch data. On construction it attempts
to allocate device USM; if the device does not support device USM allocations,
it falls back to a `sycl::buffer<T, 1>`.

`__device_storage` is used when a backend pattern needs intermediate working
memory for its kernels. It is also the base class for `__result_storage` and
`__combined_storage`.

### `__result_storage<T>`

A storage type for a single result value or a small array returned to the host.
The result type `T` must satisfy `sycl::is_device_copyable_v<T>`.

On construction it first attempts to allocate host USM, which allows the result
to be read back to the host without an explicit copy. If host USM is not
available, it falls back to the device USM or `sycl::buffer` strategy of
`__device_storage`. 

After the kernel completes, the result can be retrieved via `__copy_result`:
```cpp
_T __value{};
__result.__copy_result(&__value, 1);
```

#### `__create_result_storage_opt` function

A helper function template that conditionally creates a `__result_storage<T>`
or a `__no_storage_tag` sentinel at compile time:
```cpp
auto __opt_result = __create_result_storage_opt<_Condition, _T>(__q, __n);
// __opt_result is __result_storage<_T> if _Condition, else __no_storage_tag
```

This is useful in backend patterns that are parameterized over whether a result
is needed or not, such as for iteration stop position if the output is bounded.
When `_Condition` is `false`, no allocation is performed. The returned
`__no_storage_tag` still can be passed to `__get_accessor`;
see [Acquiring Accessors](#acquiring-accessors) for important additional details.

### `__combined_storage<T>`

A storage type that holds both scratch data and a result of the same type.
The type `T` must satisfy `sycl::is_device_copyable_v<T>`.

When host USM is available, the result is placed in a host USM buffer
and the scratch data in a separate device-side allocation.
Otherwise, a single device USM or `sycl::buffer` allocation is used,
with the scratch region at the start and the result region immediately following.
To access the scratch and result regions, two distinct accessors are used
(see [Acquiring Accessors](#acquiring-accessors)).

After the kernel completes, the result can be retrieved via `__copy_result`
the same way as with `__result_storage`.

`__combined_storage` is preferred over a pair of `__device_storage` and
`__result_storage` when the scratch and result data have the same type,
as this allows to optimize storage allocation if host USM is not available.

### Acquiring Accessors

To access the data in a storage, an accessor is acquired inside the lambda
passed to `sycl_queue.submit`. Two free functions are provided (where `cgh`
is the `sycl::handler` instance within the lambda):

- `__get_accessor(access_mode_tag, storage, cgh, property_list)` —
  acquires an accessor to the data storage region, or to the scratch-only
  region for `__combined_storage`.
- `__get_result_accessor(access_mode_tag, storage, cgh, property_list)` —
  acquires an accessor to the result region of a `__combined_storage`.

Both functions return a `__combi_accessor<T, AccessMode>`, which provides a
uniform data access interface regardless of whether the underlying storage is USM
or `sycl::buffer`. Inside the kernel, the raw `T*` pointer to the data (`const T*`
if the storage is accessed only for read) is obtained by calling `__data()`
on the accessor. The result should be cached in a local variable at the start
of the kernel, as `__data()` may involve an accessor dereference.

```cpp
__q.submit([&](sycl::handler& __cgh) {
    auto __scratch_acc = __get_accessor(sycl::read_write, __scratch_and_result, __cgh);
    auto __result_acc  = __get_result_accessor(sycl::write_only, __scratch_and_result,
                                               __cgh, __dpl_sycl::__no_init{});
    __cgh.parallel_for<_KernelName>(__range, [=](sycl::nd_item<1> __item) {
        auto* __scratch_ptr = __scratch_acc.__data();
        auto* __result_ptr  = __result_acc.__data();
        // use __scratch_ptr and __result_ptr
    });
});
```

`__get_accessor` also accepts a `__no_storage_tag` in place of a storage object,
in which case it returns a `__no_storage_tag` without any side effects.
This allows backend patterns to be written generically, whether a result storage
is needed or not. Using an accessor to an optional result storage must be guarded
by the same compile-time condition used for storage creation or by
`__is_combi_accessor(accessor)`.

## `__storage_holder`

`__storage_holder<_NScratch, _ResultTypes...>` is the lifetime and ownership manager
for the allocations of an asynchronous backend pattern. It is constructed by an
algorithm before the backend pattern is called, passed by reference into the pattern,
and used to retrieve results after the kernels submitted by the pattern are complete.

The template parameters are:
- `_NScratch` — the number of scratch storage slots. This must equal the
  maximum number of `__device_storage` and `__combined_storage` objects
  simultaneously used by any code path through the backend pattern.
- `_ResultTypes...` — the value types of the result slots, one per result.

Note that a `__combined_storage` needs both a scratch slot and a result slot;
see [Depositing Storages](#depositing-storages) for more information.

### Holder Type Alias

A backend pattern defines a holder type alias that fixes the storage
parameters and exposes it to callers. The alias typically hides the number of
scratch slots (usually irrelevant to callers) and any compile-time variation
of result types behind a pattern-specific name.

For example, the scan pattern defines two aliases depending on whether the
output can be bounded (requiring a stop position result) or it cannot:

```cpp
template <bool _Bounded, typename _ValueType, typename _StopPosType>
using __transform_scan_storage_holder =
    std::conditional_t<_Bounded,
                       __storage_holder<2, _ValueType, _StopPosType>,
                       __storage_holder<2, _ValueType>>;

template <typename _ValueType>
using __transform_scan_storage_holder_simple = __storage_holder<2, _ValueType>;
```

It is recommended that the number and the order of template type parameters
of the alias match the holder result slots, for predictable result retrieval API.

A caller uses the appropriate alias based on its own template parameters
and constructs the holder without any knowledge of the scratch count
or the internal storages:
```cpp
__transform_scan_storage_holder<_Bounded, diff_t, _PositionType> __holder(__q);
```

Note that callers may need to know the conditions for a certain result to be
retrieved, such as dependence of `_PositionType` availability on `_Bounded`
in the above example.

### Depositing Storages

After submitting kernels, the backend pattern transfers ownership of its
storage objects into the holder before returning. The pattern is responsible
for using a correct slot for each storage object, depending on the data type
and storage type.

To save a result storage for subsequent data retrieval, use `__store<Idx>`
with the index of the result slot that matches the storage data type, e.g.:
```cpp
__holder.template __store<0>(std::move(__result));
```

Each result slot may only be written once; an assertion enforces this at
runtime.

When a `__combined_storage` is stored via `__store<Idx>`, its scratch allocation,
if separate, is automatically moved into the next available scratch slot;
no special call is needed. That is the reason for the number of scratch slots
to include the combined storage objects.

To only protect storage lifetime without possibility to get any data for it,
use `__store_scratch`:
```cpp
__holder.__store_scratch(std::move(__scratch));
```

It is primarily intended for a `__device_storage`, but it works with other
storage types which inherit that. Note that it is especially dangerous to use
with a `__combined_storage` - it may have a separate allocation for result data,
which is not transferred by `__store_scratch` and risks to be freed prematurely
when the storage is destroyed. Whether stricter usage limitations should be
set for `__store_scratch` is an open question.

The following pseudocode example illustrates how a backend pattern with two
implementation paths uses the holder. The small-size path produces only
a result; the other path requires scratch memory in addition:

```cpp
template<typename ValueType>
using __parallel_pattern_holder = __storage_holder<1, ValueType>;

template<typename ValueType>
sycl::event
__parallel_pattern_small_impl(sycl::queue& __q,
                              __parallel_pattern_holder<ValueType>& __holder, ...)
{
    __result_storage<ValueType> __result(__q, 1);
    sycl::event __event = __single_wg_submitter(__q, __result, ...);
    __holder.template __store<0>(std::move(__result));
    return __event;
}

template<typename ValueType>
sycl::event
__parallel_pattern_impl(sycl::queue& __q,
                        __parallel_pattern_holder<ValueType>& __holder, ...)
{
    __combined_storage<<ValueType> __scratch_and_result(__q, __scratch_n, 1);
    sycl::event __event = __nd_range_submitter(__q, __scratch_and_result, ...);
    __holder.template __store<0>(std::move(__scratch_and_result));
    return __event;
}
```

Note that `_NScratch` is set to 1; the small-size path stores no scratch, while 
the other path may store a scratch region from the `__combined_storage`.

### Safe Memory Management

Both the storage objects and `__storage_holder` manage allocations via RAII.
A storage object that is destroyed without being moved into the holder —
for example if an exception is thrown between construction and `__store` —
will free its memory. Once ownership is transferred via `__store` or
`__store_scratch`, the holder takes responsibility: its destructor frees any
USM memory that has not subsequently been transferred out via `__extract()`
(see [Asynchronous Algorithm](#asynchronous-algorithm)).

Care must be taken when a kernel is in flight and holds a reference to storage
memory: destroying a storage object or the holder before the kernel completes
will cause premature deallocation. The backend pattern is responsible for
ensuring that all in-flight kernels have completed or that ownership has been
transferred to the holder before any storage object goes out of scope.
