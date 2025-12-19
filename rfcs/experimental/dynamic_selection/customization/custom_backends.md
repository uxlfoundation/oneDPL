# Custom Backends for Dynamic Selection

This document provides detailed information about customizing backends for Dynamic Selection. 
For an overview of both backend and policy customization approaches, see the main
[README](README.md).

## Current Backend Design

As described in [the current design](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection),
type `T` satisfies the *Backend* contract if given,

- `b` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` is of type `S` and satisfies *Selection* and `is_same_v<resource_t<S>, resource_t<T>>` is `true`
- `f` a function object with signature `wait_t<T> f(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_type` | Type alias for the resource type. |
| `resource_t<T>` | Backend trait for the resource type. |
| `b.submit(s, f, args…)` | Invokes `f` with the resource from the *Selection* and the `args`. Does not wait for the return object returned by `f`. Returns an object that satisfies *Submission*. |
| *Submission* type | `b.submit(s, f, args…)` returns a type that must define two member functions, `wait()` and `unwrap`. |
| `b.get_resources()` | Returns a `std::vector<resource_t<T>>`. |
| `b.get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this `wait` function blocks until all previous submissions to this backend are complete. |
| `lazy_report_v<T>` | `true` if the backend requires a call to `lazy_report()` to trigger reporting of `execution_info` back to the policy |
| `b.lazy_report()` | An optional function, only needed if `lazy_report_v<T>` is `true`. Is invoked by the policy before making a selection. |

Currently, these functions and traits (except the `lazy_report` function) must be implemented
in each backend. The experimental backend for SYCL queues is a bit more than 250 lines of code.
With sensible defaults, this proposal aims to simplify backend writing to open up Dynamic Selection to more use cases.

## Proposed Design to Enable Easier Customization of Backends

This proposal presents a flexible backend system that can be used for most resource types. It is based on two template classes: `backend_base` and `core_resource_backend`. For resource backends supporting reporting requirements, explicit specialization of the `core_resource_backend` is necessary since reporting cannot be done generically. For simple types serving policies without reporting requirements, use of the generically written `core_resource_backend` may be possible.

### Key Components

1. **`backend_base<ResourceType>`**: A proposed base class template that implements common backend functionality to inherit from.
2. **`core_resource_backend<CoreResourceType, ResourceType, ResourceAdapter>`**: A proposed template that backend developers partially specialize to implement a backend for a specific `CoreResourceType`. The implementation provides instrumentation and resource management for that core resource type. A `ResourceAdapter` can be used to transform a `ResourceType` into a `CoreResourceType`, allowing reuse of an existing `core_resource_backend` specialization.

   The `CoreResourceType` is the fundamental resource type that a backend knows how to instrument and submit work to. For example, `sycl::queue` is the core resource type for the SYCL backend. When using adapters, the `ResourceType` may differ from the `CoreResourceType` (e.g., storing `sycl::queue*` while the backend operates on `sycl::queue`).

   Note: any partial specialization of `core_resource_backend` that targets a particular `CoreResourceType` must be declared in the namespace `oneapi::dpl::experimental`.

3. **`default_backend<ResourceType, ResourceAdapter>`**: A wrapper template that **users instantiate** when creating policies. It automatically determines the `CoreResourceType` by applying the `ResourceAdapter` to the `ResourceType`, then inherits from the appropriate `core_resource_backend` specialization.
4. **A SYCL specialization of core_resource_backend**: A specialized implementation for `sycl::queue` resources that handles SYCL-specific event management and profiling. Using an adapter, it is possible to reuse this for other types that can be adapted into a `sycl::queue`, such as a `sycl::queue *` or a struct that contains a `sycl::queue`.

### Reporting Requirements and Scratch Space Contract

Backends must accept a (possibly empty) variadic list of reporting requirements describing the execution information the Policy will need. These reporting requirements are the same `execution_info` tag types used elsewhere in the Dynamic Selection API (for example `execution_info::task_time_t`, `execution_info::task_submission_t`, `execution_info::task_completion_t`).

#### Requirements for Backend Implementers

- Constructor contract: backend constructors (both the default and the one accepting a universe of resources) must accept an trailing variadic pack of reporting requirement types. For example:

```cpp
template <typename... ReportingReqs>
core_resource_backend(ReportingReqs... reqs);
template <typename... ReportingReqs>
core_resource_backend(const std::vector<ResourceType>& u, ResourceAdapter adapter, ReportingReqs... reqs);
```

- Compile-time checks: The backend implementation should `static_assert` if any type in `ReportingReqs...` is not a supported reporting requirement for that backend. This makes unsupported combinations a hard error at compile time.

- Resource filtering: Some reporting requirements imply properties of the underlying resource or device (for example, timing via `task_time_t` may require specific device support to enable profiling of work). Backends must examine the provided resources (or query devices when default-initializing resources) and filter out any resources that do not support all requested reporting requirements. Any special resource properties required to implement a reporting requirement must be checked here (for instance in the SYCL backend, checking `device.has(sycl::aspect::ext_oneapi_queue_profiling_tag)` in addition to creating queues with `sycl::property::queue::enable_profiling()` when `task_time_t` is requested). If after filtering the set of candidate resources there are no resources left that satisfy all requested reporting requirements, the backend must throw a `std::runtime_error` documenting that the requested reporting requirements cannot be satisfied on the available resources.

- Scratch space requirement: Backends need storage space within [selection handles](custom_policies.md#selection-handles) to implement instrumentation and fulfill reporting requirements. Backends must provide a nested template struct that allocates whatever per-selection scratch space is necessary for the requested reporting requirements. The required name and form are:

```cpp
template <typename... ReportingReqs>
struct scratch_space_t {
    // members required to implement reporting for ReportingReqs...
};
```

When a policy tracks execution information (like task timing or completion), the backend may need to store temporary data with each selection. For example, the SYCL backend's `scratch_space_t<execution_info::task_time_t>` includes an extra `sycl::event` to store the "start" profiling tag needed to measure elapsed time. For policies without reporting requirements, `scratch_space_t<>` should be empty (or inherit from `no_scratch_t<>`), adding no overhead.

Policy [selection handles](custom_policies.md#selection-handles) must declare a member named `scratch_space` of the appropriate type when they require execution information reporting:

```cpp
template <typename Policy, typename Backend>
class custom_selection_handle_t {
    // Required scratch space for the backend to implement instrumentation
    // with submission and completion reporting requirements
    using scratch_space_t =
        typename backend_traits<Backend>::template selection_scratch_t<
            execution_info::task_submission_t, execution_info::task_completion_t>;
    scratch_space_t scratch_space;

    // Note: this is not a full implementation of a selection handle
};
```

The backend populates and uses this scratch space during work submission and reporting.

- Wait type (optional): Backends may define a `wait_type` type alias to specify the exact type that user-submitted functions must return. If not defined, user functions may return any *waitable-type* (a type with a `wait()` member function). Defining a `wait_type` is typically necessary when the backend needs to instrument or track asynchronous operations. For example, the SYCL backend defines `using wait_type = sycl::event` to properly track dependencies and collect profiling information.


### Implementation Details

The `backend_base` class provides common functionality that can be customized by derived classes:

```cpp
template<typename ResourceType>
class backend_base
{
  public:
    using resource_type = ResourceType;
    using execution_resource_t = resource_type;
    using resource_container_t = std::vector<ResourceType>;

    template <typename SelectionHandle, typename Function, typename... Args>
    auto submit(SelectionHandle s, Function&& f, Args&&... args);

    auto get_submission_group();
    auto get_resources();

  protected:
    resource_container_t resources_;
    // Implementation methods that can be overridden
};
```

The `core_resource_backend` class extends `backend_base` with adapter support but requires the
`CoreResourceType` to be known:

```cpp
template< typename CoreResourceType, typename ResourceType, typename ResourceAdapter >
class core_resource_backend : public backend_base<ResourceType> {
public:
    using resource_type = ResourceType;
    using my_base = backend_base<ResourceType>;

    template <typename... ReportingReqs>
    core_resource_backend(ReportingReqs... reqs) : my_base(reqs...) {}
    template <typename... ReportingReqs>
    core_resource_backend(const std::vector<ResourceType>& u, ResourceAdapter adapter_, ReportingReqs... reqs)
       : my_base(u, reqs...), adapter(adapter_) {}

  private:
    ResourceAdapter adapter;
};
```

The class `default_backend` determines the `CoreResourceType` from the `ResourceType` and the type
returned by `Adapter`.

```cpp
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity>
class default_backend :
  public core_resource_backend<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>,
         ResourceType, ResourceAdapter>
{
  public:
    using base_t = core_resource_backend<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>, ResourceType, ResourceAdapter>;

    template <typename... ReportingReqs>
    default_backend(ReportingReqs... reqs) : base_t(reqs...)
    {
    }
    template <typename... ReportingReqs>
    default_backend(const std::vector<ResourceType>& r, ResourceAdapter adapt = {}, ReportingReqs... reqs) : base_t(r, adapt, reqs...)
    {
    }
};
```
The `core_resource_backend` class is partially specialized to create a specific backend. Adapters allow backends
to be reused by providing an adapter to transform a custom resource type `ResourceType` into a known
resource type `CoreResourceType` with an already existing `core_resource_backend`. 

### Default Implementation Details

The `backend_base` provides **minimal** default implementations for the common backend methods. Importantly:
- `backend_base` does **not** support any reporting requirements by default
- The default `scratch_t` template is empty (provides `no_scratch_t`)
- Resource-specific features (timing, profiling, etc.) require specializing `core_resource_backend` for your `CoreResourceType`

To create a backend with reporting support, specialize `core_resource_backend` for your specific `CoreResourceType` (e.g., `sycl::queue`).

#### `submit` Implementation
The default `submit` method calls the user-provided function with the unwrapped resource and returns a waitable submission type which wraps the return from the user-provided function.

```cpp
template <typename T>
class default_submission
{
    T t;
    void wait();
    T unwrap();
};

template <typename SelectionHandle, typename Function, typename... Args>
auto
submit(SelectionHandle s, Function&& f, Args&&... args)
{
    return default_submission{std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...)};
}
```
**Assumptions**: The user function must be callable with a `ResourceType` resource and the `Function f` 
returns an object that can be waited on. So if an adapter is provided, the user function is still called
with the unadapted resource type. For example, if an adapter `[](auto pointer){ return *pointer; }` is used to
reuse a `sycl::queue` with `sycl::queue *` resources, the user's function is still called with a 
`sycl::queue *`.

#### Instrumentation Support
Where instrumentation is required, the customizer will need to override `submit` with code that performs instrumentation for the `CoreResourceType`. The overload must call the user function and return an object which supports `wait()` (blocks until the job completes) and `unwrap()` (returns the user's returned object).

Each resource type has different characteristics for timing, profiling, and performance measurement, making a general instrumentation mechanism impossible.

#### `get_resources()` Implementation
Returns the vector of resources stored during construction. Resources must be copyable/movable into a `std::vector` and remain valid throughout the backend's lifetime.

#### `get_submission_group()` Implementation
Returns a group object that waits on all resources. The `default_submission_group` attempts to call `wait()` on the `CoreResourceType` by applying `adapter` to the `ResourceType` object. The `CoreResourceType` must provide a `wait()` method that blocks until all work on that resource is complete. Note that the default implementation waits on each resource, not each submission (works for SYCL queues, oneTBB `task_group` objects, etc.). Adapters can enable types without a `wait()` method; for example, `[](auto pointer){ return *pointer; }` allows `sycl::queue*` to work by adapting to `sycl::queue`.

## Support for Custom Resource Types

A primary goal of this proposal is to enable easy use of custom resource types with Dynamic Selection. The default backend works with many resource types, making it straightforward to integrate new compute resources without complex backend code. If the defaults are insufficient, specialize `core_resource_backend`. 

### Custom Resource Example: TBB Task Groups and Arenas

Consider a custom resource type that combines a `tbb::task_arena` and `tbb::task_group`:

```cpp
namespace numa {
    class ArenaAndGroup {
        tbb::task_arena *a_;
        tbb::task_group *tg_;
    public:
        ArenaAndGroup(tbb::task_arena *a, tbb::task_group *tg) : a_(a), tg_(tg) {}
      
        template<typename F>
        auto run(F&& f) {
            a_->enqueue(tg_->defer([&]() { std::forward<F>(f)(); }));
            return *this;
        }

        void wait() { 
            a_->execute([this]() { tg_->wait(); }); 
        }

        void clear() { delete a_; delete tg_; }
    };
}
```

This custom resource can be used directly with Dynamic Selection policies that do
not require instrumentation:

```cpp
ex::round_robin_policy<numa::ArenaAndGroup> rr{ /* resources */ };
for (auto i : numa_nodes) {
    ex::submit(rr, 
        [](numa::ArenaAndGroup ag) { 
            ag.run([]() { std::printf("o\n"); });
            return ag; 
        }
    );
}
ex::wait(rr.get_submission_group());
```

If `ArenaAndGroup` will be used with policies that require instrumentation, specialize `core_resource_backend` to provide an instrumented `submit` method.

## Adapter Support for Resource Transformation

Adapters allow backends to work with resources that require transformation before use. An adapter is a callable object that transforms a resource from the stored `ResourceType` to the `CoreResourceType` expected by backend functions.

**Adapter Requirements**:
- Callable with `ResourceType` and returns `CoreResourceType`
- Stored in the backend and applied during submission operations
- User functions still receive the **unadapted** `ResourceType` (not the `CoreResourceType`)
- Default adapter is `oneapi::dpl::identity` (no transformation)

For example, with `[](auto* p) { return *p; }`:
- Resources stored as `sycl::queue*` (`ResourceType`)
- Backend works with `sycl::queue` (`CoreResourceType`)
- User functions receive `sycl::queue*`

### Example: Pointer Dereferencing

```cpp
auto deref_op = [](auto pointer){ return *pointer; };

using policy_pointer_t = oneapi::dpl::experimental::round_robin_policy<
    sycl::queue*,
    decltype(deref_op),
    oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>
>;

std::vector<sycl::queue*> u_ptrs;
policy_pointer_t p(u_ptrs, deref_op);
```

### Adapter Usage Patterns

1. **Pointer Resources**: Store/copy pointers but reuse backend implementation for the decayed type
2. **Wrapper Types**: Unwrap resource containers or smart pointers
3. **Ownership Management**: Pair context (memory space, side information) with a core resource while reusing core resource backend

## Testing
Testing for these changes should include:
 * Test of SYCL backend using a `sycl::queue*` as the execution resource with a dereferencing resource adapter function.
 * Test of automatic backend selection by providing a universe of resources to construction which are used to deduce the backend.
 * Test of a policy using `default_backend` (which uses the default `backend_base` implementation) for a simple resource type.
 * Test of a custom backend created by partially specializing `core_resource_backend` with minimally overridden `submit()` for a simple resource type.

## Explored Alternatives

### Extra Resource (alternative to resource adapter)
We explored adding an optional "extra resource" universe paired 1-to-1 with execution resources and passed to user workloads alongside the execution resource. While slightly more straightforward to use, this approach has more complex implementation, copying overhead, and less freedom in stored execution resource type.

## Open questions
* What other backends would make sense as examples / descriptive tests for dynamic selection?
* Should users have a way to get / query the resource adapter function from their submitted workloads?

