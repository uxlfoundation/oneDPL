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
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_t<T>` | Backend trait for the resource type. |
| `wait_t<T>` | Backend trait for type that is expected to be returned by the user function |
| `b.submit(s, f, args…)` | Invokes `f` with the resource from the *Selection* and the `args`. Does not wait for the `wait_t<T>` object returned by `f`. Returns an object that satisfies *Submission*. |
| *Submission* type | `b.submit(s, f, args…)` returns a type that must define two member functions, `wait()` and `unwrap`. |
| `b.get_resources()` | Returns a `std::vector<resource_t<T>>`. |
| `b.get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this `wait` function blocks until all previous submissions to this backend are complete. |
| `lazy_report_v<T>` | `true` if the backend requires a call to `lazy_report()` to trigger reporting of `execution_info` back to the policy |
| `b.lazy_report()` | An optional function, only needed if `lazy_report_v<T>` is `true`. Is invoked by the policy before making a selection. |

Currently, these functions and traits (except the `lazy_report` function) must be implemented 
in each backend. The experimental backend for SYCL queues is a bit more than 250 lines of code. 
With sensible defaults, this proposal aims to simplify backend writing to open up Dynamic Selection to more use cases.

## Proposed Design to Enable Easier Customization of Backends

This proposal presents a flexible backend system based on a `backend_base` template class and a `default_backend_impl` template that can be used for most resource types. The design uses CRTP (Curiously Recurring Template Pattern) to allow customization while providing sensible defaults.

### Key Components

1. **`backend_base<ResourceType, Backend>`**: A proposed base class template that implements the core backend functionality using CRTP.
2. **`default_backend_impl<BaseResourceType, ResourceType, ResourceAdapter>`**: A proposed template that provides a complete backend implementation for any resource type, with optional adapter support. A developer creates a partial specialization of `default_backend_impl` to create a specific backend for the `BaseResourceType`. Adapters can be used with `default_backend` to reuse `default_backend_impl` for a `ResourceType` that is adapted to a `BaseResourceType`.
3. **`default_backend<ResourceType, ResourceAdapter>`**: A proposed template that determines the `BaseResourceType` from the `ResourceType` and `ResourceAdapter`.
4. **A SYCL specialization of default_backend_impl**: A specialized implementation for `sycl::queue` resources that handles SYCL-specific event management and profiling. Using an adapter, it is possible to reuse this for other types that can be adapted into a `sycl::queue`, such as a `sycl::queue *` or a struct that contains a `sycl::queue`.

### Core Features

- **Resource Management**: Backends store resources in a vector and provide `get_resources()` to access them
- **Submission System**: The `submit()` method invokes user functions with selected resources and returns submission objects
- **Instrumentation**: Functions `instrument_before` and `instrument_after` can be optionally overridden to provide reporting for policies that require reporting.
- **Group Operations**: `get_submission_group()` returns an object that can wait for all submissions to complete
- **Trait Support**: Type traits for `resource_t<T>`, `wait_t<T>`, and lazy reporting detection
- **Scratch Space**: Optional scratch space allocation for backend-specific needs via traits

### Implementation Details

The `backend_base` class provides core functionality that can be customized by derived classes:

```cpp
template<typename ResourceType, typename Backend>
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

The `default_backend_impl` class extends `backend_base` with adapter support but requires the
`BaseResourceType` to be known:

```cpp
template< typename BaseResourceType, typename ResourceType, typename ResourceAdapter >
class default_backend_impl : 
  public backend_base<ResourceType, default_backend_impl<BaseResourceType, ResourceType, ResourceAdapter>> {
public:
    using resource_type = ResourceType;
    using my_base = backend_base<ResourceType, default_backend_impl<BaseResourceType, ResourceType, ResourceAdapter>>;

    default_backend_impl() : my_base() {}
    default_backend_impl(const std::vector<ResourceType>& u, ResourceAdapter adapter_) 
       : my_base(u), adapter(adapter_) {}
};
```

The class `default_backend` determines the `BaseResourceType` from the `ResourceType` and the type
returned by `Adapter`.

```cpp
template <typename ResourceType, typename ResourceAdapter = oneapi::dpl::identity>
class default_backend : 
  public default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>())>,
         ResourceType, ResourceAdapter>
{
  public:
    using base_t = default_backend_impl<std::decay_t<decltype(std::declval<ResourceAdapter>()(std::declval<ResourceType>()))>, ResourceType, ResourceAdapter>;
    using wait_type = typename base_t::wait_type;

    default_backend()
    {
    }
    default_backend(const std::vector<ResourceType>& r, ResourceAdapter adapt = {}) : base_t(r, adapt)
    {
    }
};
```
The `default_backend_impl` class is partially specialized to create a specific backend. Adapters allow backends
to be reused by providing an adapter to transform a custom resource type `ResourceType` into a known
resource type `BaseResourceType` with an already existing `default_backend_impl`. 

### Default Implementation Details

The `backend_base` provides default implementations for the core backend methods:

#### `submit_impl` Implementation
The default `submit_impl` method calls `instrument_before`, then the user-provided function with the 
unwrapped resource, and finally returns what is returned by a call to `instrument_after`:

```cpp
template <typename SelectionHandle, typename Function, typename... Args>
auto
submit_impl(SelectionHandle s, Function&& f, Args&&... args)
{
    static_cast<Backend*>(this)->instrument_before_impl(s);
    auto w = std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), std::forward<Args>(args)...);
    return static_cast<Backend*>(this)->instrument_after_impl(s, w);
}
```
**Assumptions**: The user function must be callable with a `ResourceType` resource and the `Function f` 
returns an object that can be waited on. So if an adapter is provided, the user function is still called
with the unadapted resource type. For example, if an adapter `[](auto pointer){ return *pointer; }` is used to
reuse a `sycl::queue` with `sycl::queue *` resources, the user's function is still called with a 
`sycl::queue *`.

#### Instrumentation Support
The default backend provides `instrument_before` and `instrument_after` functions that are essentially empty implementations:

```cpp
template <typename SelectionHandle>
void instrument_before(SelectionHandle s) { 
    // Empty implementation - no general instrumentation possible
}

template <typename SelectionHandle, typename WaitType>
auto instrument_after(SelectionHandle s, WaitType w) {
    return default_submission{w}; // Simply wrap the result
}
```

**Rationale**: It is not possible to define a general mechanism for instrumenting code execution for arbitrary custom resource types. Each resource type has different characteristics for timing, profiling, and performance measurement. Specialized backends (like the SYCL backend) can override these methods to provide resource-specific instrumentation.

#### `get_resources()` Implementation
Returns the vector of resources stored during construction:
```cpp
auto get_resources() const noexcept {
    return resources_;
}
```
**Assumptions**: Resources can be copied or moved into a `std::vector` and remain valid throughout the backend's lifetime.

#### `get_submission_group()` Implementation
Returns a group object that can wait on all resources if they provide a `wait()` method:
```cpp
auto get_submission_group() {
    return default_submission_group{resources_, adapter};
}
```
The `default_submission_group` attempts to call `wait()` on the `BaseResourceType` by applying
`adapter` to the `ResourceType` object.
:
```cpp
class default_submission_group {
    void wait() {
        if constexpr (has_wait_method_v<BaseResourceType>) {
            for (auto& resource : resources_) {
                adapter(resource).wait();
            }
        } else {
            throw std::logic_error("wait() not supported by resource type");
        }
    }
};
```
**Assumptions**: For group waiting to work, the `BaseResourecType` must provide a `wait()` method
that blocks until all work on that resource is complete. Note that the default implementation does
not wait on each submission, but instead waits on each resource. This works for some resource types,
such as SYCL queues or oneTBB `task_group` objects, but may not be applicable to all types. Using
an adapter may allow types that do not provide a `wait` function to be used if they have a `wait` function
when adapted.  For example, an adapter `[](auto pointer){ return *pointer; }`, would allow `sycl::queue *`
or `tbb::task_group *` to be used as a `ResourceType`, because when adapted to their `BaseResourceType`
of `sycl::queue` or `tbb::task_group`, they define `wait` methods.

For SYCL resources, the proposed specialization provides:
- Event-based waiting with `sycl::event` as the wait type
- Profiling support for performance reporting
- Asynchronous submission handling
- SYCL-specific error handling

## Support for Custom Resource Types

A primary goal of this proposal is to enable easy use of custom resource types with Dynamic Selection. The default backend can work many resource types, making it straightforward to integrate new kinds of compute resources without writing complex backend code. If the defaults are not sufficient, a custom backend can be written by 
partially specializing `default_backend_impl`. 

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

If `ArenaAndGroup` will be used with policies that require instrumentation, then
a custom backend that provides `instrument_before_impl` and `instrument_after_impl`
will be needed. This can be done by partially specializing `default_backend_impl`.

## Adapter Support for Resource Transformation

For some cases a backend may be reused if a custom resource `ResourceType` can be transformed
into a resource `BaseResourceType` that already has a well defined backed. This proposal
includes support for **resource adapters**. Adapters allow backends to work with resources that
require transformation before use.

### Adapter Concept

An adapter is a callable object that transforms a resource from the stored type to the type
expected by the backend functions. The default adapter is `oneapi::dpl::identity`, which performs
no transformation.

### Example: Pointer Dereferencing

A common use case is working with pointers to resources:

```cpp
// Adapter to dereference a pointer
auto deref_op = [](auto pointer){ return *pointer; };

// Policy using pointer resources with dereferencing adapter
using policy_pointer_t = oneapi::dpl::experimental::round_robin_policy<
    sycl::queue*, 
    decltype(deref_op), 
    oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>
>;

std::vector<sycl::queue*> u_ptrs;
policy_pointer_t p(u_ptrs, deref_op);
```

### Adapter Usage Patterns

Adapters enable several useful patterns:

1. **Pointer Resources**: Store pointers but work with references/values
2. **Wrapper Types**: Unwrap resource containers or smart pointers  
3. **Type Conversion**: Convert between compatible resource types
4. **Ownership Management**: Pair a context (memory space, side information, etc.) with a core resource, but rely on the implementation of the core resource without extra backend implementation.

## Testing
Testing for these changes should include:
 * Test of SYCL backend using a ``sycl::queue*`` as the execution resource with a dereferencing resource adapter function.
 * Test of automatic backend selection by providing a universe of resources to construction which are used to deduce the backend.
 * Test of backend using default backend for a simple resource type.
 * Test of backend using backend with minimally overridden instrument_before and instrument_after for a simple resource type.

## Explored Alternatives

### Extra Resource (alternative to resource adapter)
 As an alternative to the resource adapter idea, we explored adding an optional "extra resource" universe which would
 be paired 1-to-1 with execution resources if provided, and if specified, passed to user-submitted workloads alongside the
 execution resource. This extra resource would be user-defined and would exist to provide freedom to the user to attach
 other information to a resource while still relying upon a defined backend since the execution resource would not be changed.
    Advantages:
    * Slightly more straightforward to use than the resource adapter idea
    Disadvantages:
    * Much more complex impact on dynamic selection code (less elegant)
    * More overhead in copying around extra resources and/or requiring users to provide extra resources which can be copied around with minimal overhead
    * Less freedom in stored execution resource type

## Open questions
* What other backends would make sense as examples / descriptive tests for dynamic selection?
* Should users have a way to get / query the resource adapter function from their submitted workloads?

## Conclusion

This proposal presents a simplified approach to backend customization for Dynamic Selection. Key benefits include:

1. **Reduced Complexity**: The default backend eliminates the need to write backend code for most use cases
2. **Custom Resource Integration**: Direct support for any resource type without transformation overhead
3. **Optional Adapter Pattern**: Enables flexible resource management when transformation enable reuse of an existing backend.
4. **SYCL Optimization**: Specialized backend for SYCL resources maintains performance while providing additional features
5. **Extensibility**: Easy to add new backend specializations

