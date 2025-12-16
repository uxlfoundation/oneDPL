# Dynamic Device Selection API

## Introduction

Dynamic Selection is a Technology Preview feature 
[documented in the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-selection-api.html).
Dynamic selection provides functions for choosing a resource using a *selection policy*. 
By default, the *resources* selected via Dynamic Selection APIs are SYCL queues. Since the
API itself is documented in the oneDPL Developer Guide that information will not
be repeated here excepted where necessary. It is assumed that readers of this document
are familiar with the API documentation.

This document contains the following:
- An overview of the current architecture and design.
- Exit criteria for moving from experimental to fully supported or, if these goals are not achieved in
a timely manner, for justifying removal of the feature.

## Overview of Architecture and Execution Flow

The key components of the Dynamic Selection API are shown below, including the
[Free Functions](#free_functions_id) (such as `submit`, `wait`, etc), a
[Policy](#policy_req_id) object (such as [fixed_resource_policy](#concrete_policies_id),
[round_robin_policy](#concrete_policies_id), [dynamic_load_policy](#concrete_policies_id) and
[auto_tune_policy](#concrete_policies_id)) and a [Backend](#backend_req_id) object (currently only
`sycl_backend`). Users interact with Dynamic Selection through the [Free Functions](#free_functions_id)
and their chosen [Policy](#policy_req_id). The [Free Functions](#free_functions_id) have
default implementations that depend on a limited set of required functions in the
[Policy](#policy_req_id). Optional functions may be defined by a [Policy](#policy_req_id)
to customize some of the Free Functions, such as `submit_and_wait` that would, by default,
depend on multiple basis functions. Resource specific instrumentation and types are defined
in the [Backend](#backend_req_id).

<img src="architecture.png" width=800>

The following code example shows some of the key aspects of the API. The use of any empty
`single_task` is for syntactic demonstration purposes only; any valid command group or series
of command groups can be submitted to the selected queue.

```cpp
  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {

    // (1) create a policy object
    ex::dynamic_load_policy p{ { sycl::queue{ sycl::cpu_selector_v },
                                 sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 6; ++i) {

      // (2) call one of the dynamic selection functions
      //     -- pass the policy to the API function
      //     -- provide a function to be called with a selected queue
      //        -- the user function must receive a sycl queue
      //        -- the user function must return a sycl event
      auto done = ex::submit(p,
                            // (3) use the selected queue in user function
                            [=](sycl::queue q) {
                              std::cout << "submit task to "
                                        << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                              return q.single_task([]() { /* work here */ });
                            });

      // (4) each submission can be waited on using the returned object
      ex::wait(done);
    }

    // (5) and/or all submissions can be waited on as a group
    //     in a typical case one would use (4) or (5) but not both.
    ex::wait(p.get_submission_group());
  }
```

The execution flow of a call to the function ``ex::submit`` is shown below.  

<img src="execution_flow.png" width=800>

The free function `submit` receives a [Policy](#policy_req_id) object `p` and
a function object `f` and returns an object `w` that can be waited on.
A valid [Policy](#policy_req_id) must define a `submit` member function and a
[Backend](#backend_req_id) must also define a `submit` member function. The
requirements on [Policy](#policy_req_id) and [Backend](#backend_req_id) types
are discussed later in this proposal. In the figure, the Policy's `submit` function
selects a resource `r` and then passes `r`, an execution info handle `h`, and `f`
to the Backend. The handle `h` is the mechanism for the Backend to report runtime
information required by the Policy logic for making future selections, such as
the execution time of a task when run on a specific resource.

The [Backend](#backend_req_id) `submit` function invokes `f` with the selected
resource as an argument and is also responsible for collecting and reporting
any information required by the [Policy](#policy_req_id). The
[Execution Info](#execution_info_id) section of this document describes how
to use traits to determine what information needs to be reported. 

## Named Requirements

<a id="policy_req_id"></a>
### Policy

A Policy is an object with a valid dynamic selection heuristic.

The type `T` satisfies *Policy* if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`, where `wait_t<T>` is the wait type for the policy. If the backend defines an explicit `wait_type`, user functions must return that specific type. If the backend does not define an explicit `wait_type`, user functions may return any *waitable-type* (a type with a `wait()` member function that blocks until the submitted work is complete).

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `T::backend_type`     | Type alias for the backend type.  |
| `T::resource_type`    | Type alias for the resource type. |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| *At least one must be well-formed* | Description |
| --------------------- | ----------- |
| `p.try_submit(f, args…)` | Returns `std::optional<submission_t<T>>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. Returns empty `std::optional` if no resource is available for selection. |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits for the job to complete. |

**Note:** Policies do not expose a public selection API (e.g., `select()` or `try_select()`). Selection is always implicit within the submission methods. Policy implementers may use `try_select()` as an internal/protected method to implement their selection logic, which is then called by the submission methods provided by `policy_base` or by custom submission implementations.

| Policy Traits | Description |
| ------- | ----------- |
| `policy_traits<T>::backend_type`, `backend_t<T>` | The backend type associated with this policy. |
| `policy_traits<T>::resource_type`, `resource_t<T>` | The backend-defined resource type that is passed to the user function object. |
| `policy_traits<T>::has_wait_type_v`, `has_wait_type_v<T>` | Boolean which determines if explicit wait type has been provided by the backend associated with this policy. Derived from the backend's `backend_traits<Backend>::has_wait_type_v`. |
| `policy_traits<T>::wait_type`, `wait_t<T>` | If `has_wait_type_v<T>` is `true`, contains the type that must be returned by the user function object for this policy, otherwise `void`. Derived from the backend's `backend_traits<Backend>::wait_type`. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns an object of type `wait_t<T>`. |

The default implementation of these traits depends on types and traits defined in the Policy and its Backend:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using backend_type = typename std::decay_t<Policy>::backend_type;
      using resource_type = typename std::decay_t<Policy>::resource_type;
      static constexpr bool has_wait_type_v =
          oneapi::dpl::experimental::backend_traits<backend_type>::has_wait_type_v;
      using wait_type =
          typename oneapi::dpl::experimental::backend_traits<backend_type>::wait_type;
  };

  template <typename Policy>
  using backend_t = typename policy_traits<Policy>::backend_type;

  template <typename Policy>
  using resource_t = typename policy_traits<Policy>::resource_type;

  template <typename Policy>
  inline constexpr bool has_wait_type_v = policy_traits<Policy>::has_wait_type_v;

  template <typename Policy>
  using wait_t = typename policy_traits<Policy>::wait_type;
```

<a id="selection_req_id"></a>
### Selection

The type `T` satisfies *Selection* for a given [Policy](#policy_req_id) `p` if given,

- `s` an arbitrary identifier of type `T`
- `i` an object of type `Info` where `execution_info_v<Info>` is `true`
- `v` an object of type `Info::value_type`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `s.unwrap()` | Returns `resource_t<T>` that should represent one of the resources returned by `p.get_resources()` for the Policy `p` that generated `s`. |
| `s.get_policy()` | Returns the Policy `p` that was used to make the selection. |
| `s.report(i)` | Returns `void`. Notifies policy that an execution info event has occurred. |
| `s.report(i, v)` | Returns `void`. Notifies policy of a new value `v` for execution info event `i`. |
| `report_execution_info<T, Info>::value`, `report_execution_info_v<T,Info>` | `true` if this selection needs the backend to report the Info. `false` otherwise. |

<a id="submission_req_id"></a>
### Submission

The type `T` satisfies *Submission* for a given [Policy](#policy_req_id) `p` if given,

- `s` an arbitrary identifier of type `T`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `s.wait()` | Blocks until the submission has completed. |
| `s.unwrap()` | Returns the underlying backend type value. This type may be void, may represent the backend’s synchronization type, or may represent a return value from the submission. |

<a id="concrete_policies_id"></a>
## Provided Concrete Policies 

The following concrete policies are provided in the experimental implementation. Their details can be found in
[the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-selection-api.html).

| Available Policies |
| ------------------------|
| [`fixed_resource_policy`](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/fixed-resource-policy.html) |
| [`round_robin_policy`](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/round-robin-policy.html) |
| [`dynamic_load_policy`](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-load-policy.html) |
| [`auto_tune_policy`](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/auto-tune-policy.html) |

<a id="backend_req_id"></a>
## Backends

Backends allow generic policies to be implemented. Application developers do not directly
interact with a backend, except to choose the backend if they opt-out of the default SYCL backend.
Custom policy writers that wish to implement a generic policy that can accept backends
should follow the backend contract. Developers that wish to provide a backend that can
be used with the existing concrete policies should also follow the backend contract.

### Named Requirements

The type `T` satisfies the *Backend* contract if given,

- `b` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` is of type `S` and satisfies *Selection* and `is_same_v<resource_t<S>, resource_t<T>>` is `true`
- `f` a function object with signature `backend_traits<T>::wait_type fun(resource_t<T>, Args…)`. If the backend defines an explicit `wait_type`, user functions must return that specific type. If the backend does not define an explicit `wait_type`, user functions may return any *waitable-type* (a type with a `wait()` member function that blocks until the submitted work is complete).

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `resource_type` | Type alias for the resource type. |
| `b.submit(s, f, args…)` | Returns an object that satisfies *Submission*. The function invokes `f` but does not wait on the type returned by it for the job to complete. |
| `b.get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this wait function blocks until all previous submissions to this backend are complete. |
| `b.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| *Optional* | Description |
| --------------------- | ----------- |
| `wait_type` | Type alias specifying the exact type user functions must return. If not defined, user functions may return any *waitable-type* (a type with a `wait()` member function, which waits for submitted work to complete). |
| `void lazy_report()` | If defined by a backend, this function must be called by a policy before each new selection. It triggers reporting of the necessary execution info back to the policy. |

### Resource Adapters

Resource adapters allow backends to work with resources that require transformation before use. An adapter is a callable object that transforms a resource from the stored type (`ResourceType`) to the type expected by the backend's core operations (`CoreResourceType`). The default adapter is `oneapi::dpl::identity`, which performs no transformation.

Common use cases for resource adapters include storing resources as pointers (e.g., `sycl::queue*`) while the backend operates on the dereferenced type (e.g., `sycl::queue`), unwrapping resource containers or smart pointers, and pairing a context or side information with a core resource while relying on existing backend implementations. Custom adapters enable reuse of existing backend implementations for new resource types without duplicating backend code.

The following example demonstrates using a pointer-dereferencing adapter to reuse the SYCL backend with `sycl::queue*` resources:

```cpp
auto deref_op = [](auto* pointer) { return *pointer; };

using policy_t = oneapi::dpl::experimental::round_robin_policy<
    sycl::queue*,
    decltype(deref_op),
    oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>
>;

std::vector<sycl::queue*> queue_ptrs = { /* ... */ };
policy_t p(queue_ptrs, deref_op);
```

Note that user-provided functions receive the unadapted `ResourceType` (e.g., `sycl::queue*`), not the `CoreResourceType`. The adapter is only used internally by the backend.

For detailed information on resource adapters, including implementation requirements and additional examples, see the [Custom Backends](customization/custom_backends.md#adapter-support-for-resource-transformation) documentation.

<a id="free_functions_id"></a>
## Free Functions

| Signature | Description |
| --------- | ----------- |
| `vector<typename policy_traits<P>::resource_type> get_resources(P&& p);` | Returns the resources associated with the Policy `p`. |
| `template<Policy P, typename F, typename... Args> auto try_submit(P&& p, F&& f, Args&&... args);` | Attempts to select a resource. If successful, invokes `f` with the selected resource and `args`. Implements any instrumentation necessary for the backend to report necessary execution information. Returns a `std::optional` of the submission type if successful, or an empty `std::optional` if unable to select a resource. |
| `template<Policy P, typename F, typename... Args> auto submit(P&& p, F&& f, Args&&... args);` | Selects a resource and invokes `f` with the selected resource and `args`. Implements any instrumentation necessary for the backend to report necessary execution information. If the policy provides `try_submit()` but not `submit()`, this function will retry with backoff until a resource becomes available. |
| `template<Policy P, typename F, typename... Args> auto submit_and_wait(P&& p, F&& f, Args&&... args);` | Selects a resource, invokes `f` with the selected resource and `args`, and then waits on the object returned by `f`. |
| `template<typename P> auto get_submission_group(P&& p);` | Returns an object that has a member function `void wait()`. Calling this wait function blocks until all previous submissions to this policy are complete. |
| `template<typename W> void unwrap(W&& w) noexcept;` | Returns `w.unwrap()` if available, otherwise returns `w`. |
| `template<typename W> void wait(W&& w);` | Calls `w.wait()`. |
| `template <typename S, typename Info> void report(S&& s, const Info& i);` | `S` is a *Selection*. Reports that event `i` has occurred if `s.report(i)` is available. |
| `template <typename S, typename Info, typename Value> void report(S&& s, const Info& i, const Value& v); ` | `S` is a *Selection*. Reports a new value `v` for event `i` if `s.report(i, v)` is available. |

| Backend Traits | Description |
| ------- | ----------- |
| `backend_traits<Backend>::has_wait_type_v` | Boolean indicating whether the backend defines an explicit wait type. `true` if the backend requires that user-submitted functions return a specific type. `false` if the backend allows any *waitable-type* (a type with a `wait()` member function). |
| `backend_traits<Backend>::wait_type` | If `has_wait_type_v` is `true`, contains the explicit type that user-submitted functions must return. If `has_wait_type_v` is `false`, this is `void`. |
| `backend_traits<Backend>::lazy_report_v` | Boolean indicating whether the backend uses lazy reporting. `true` if the backend requires that a policy calls `lazy_report()` before each selection to update execution information. `false` otherwise. |
| `backend_traits<Backend>::template has_scratch_space_v<Reqs...>` | Boolean indicating whether the backend requires scratch space for the given set of execution info reporting requirements `Reqs...`. `true` if scratch space is needed for instrumentation, `false` otherwise. |
| `backend_traits<Backend>::template selection_scratch_t<Reqs...>` | Type of the scratch space storage needed by the backend for instrumentation and fulfilling the reporting requirements `Reqs...`. If `has_scratch_space_v<Reqs...>` is `false`, this is `no_scratch_t<Reqs...>` (an empty type). |

### Policy Member Functions vs Free Functions

Policies may provide submission operations either as member functions or rely on the free function implementations:
- **Free functions** (`oneapi::dpl::experimental::submit`, `try_submit`, `submit_and_wait`) are the primary user-facing API
- **Policy member functions** of the same name are optional customization points that policies can implement
- When using `policy_base`, the base class provides member function implementations that delegate to fallback mechanisms
- Custom policies that don't inherit from `policy_base` can implement member functions directly to override default behavior

Both APIs are valid - use free functions when working with policies, and implement member functions when creating custom policy behavior.

### Deferred Initialization

A call to `get_resources`, `submit` or `submit_and_wait` may initialize
underlying state variables, including dynamic allocation. Initialization may throw `std::bad_alloc`.
If `p` is a policy constructed with deferred initialization, calling these functions before 
calling `initialize` will throw `std::logic_error`.

<a id="execution_info_id"></a>
## Execution Info

Policies are informed of key events through the reporting of Execution Info.
Most commonly, this reporting is done by a backend and is not visible to the
end user.  However, developers that implement custom backends, or that
manage work submission without using the `try_submit`, `submit` or `submit_and_wait` functions,
will need to report Execution Info to allow policies to work properly.

Execution information is specified using tag types and tag objects from the `oneapi::dpl::experimental::execution_info` namespace:

| Tag Type | Tag Object | Value Type | Description |
| ----------------- | ---------- | ---------- | ----------- |
| `task_submission_t` | `task_submission` | void       | Signals when a task is submitted. |
| `task_completion_t` | `task_completion` | void       | Signals when a task completes. |
| `task_time_t`     | `task_time`       | `uint64_t` | Clock ticks that elapsed between task submission and completion. |

The following table shows which reporting requirements are needed by each provided policy:

| Policy | Reporting Requirements |
| ------ | ---------------------- |
| `fixed_resource_policy` | None |
| `round_robin_policy` | None |
| `dynamic_load_policy` | `task_submission_t`, `task_completion_t` |
| `auto_tune_policy` | `task_time_t` |

Policies with no reporting requirements can work with any backend, including backends without specialized instrumentation support. Policies with reporting requirements need a backend that supports those specific types of execution information.

| Info Traits | Description |
| ------- | ----------- |
| `report_info<S,Info>::value`, `report_info_v<S,Info>` | `true` if the *Selection* requires the event type to be reported, `false` otherwise. |
| `report_value<S,Info,V>::value`, `report_info_v<S,Info,V>` | `true` if the *Selection* requires the event value to be reported, `false` otherwise. |

Backends must accept a (possibly empty) variadic list of execution-info reporting requirements in their constructors. At construction the backend should validate and, where possible, filter the available resources to remove those that cannot satisfy the requested reporting features (for example, missing device aspects or queue profiling properties). If filtering leaves no usable resources, the backend must throw a clear runtime error.

Backends must also provide a template `template <typename... ReportingReqs> struct scratch_space_t` which supplies per-selection scratch storage tailored to the requested reporting requirements.

Backend traits can be used to determine what events are need by the *Policy* that provided a *Selection*.
For example, below is code a function the receives a *Selection* and uses traits to determine if the
`task_submission_t` must be reported. If so, it is reported using the `report` free function.

```cpp
    template <typename SelectionHandle, typename Function, typename... Args>
    auto
    submit(SelectionHandle s, Function&& f, Args&&... args)
    {
        constexpr bool report_task_submission = report_info_v<SelectionHandle, execution_info::task_submission_t>;
        auto q = unwrap(s);
        if constexpr (report_task_submission)
            report(s, execution_info::task_submission);

        // ... the remainder of the implementation

    }
```

### Lazy Reporting

Some execution information (like `task_completion` or `task_time`) cannot be reported immediately when a task is submitted, as the task may still be executing asynchronously. Backends may choose to defer reporting of such events until they are needed by the policy.

A backend that uses lazy reporting should:
1. Store submission handles/events internally when `submit()` is called
2. Define a `lazy_report()` member function that checks for completed tasks and reports their execution information
3. The backend trait `backend_traits<Backend>::lazy_report_v` will be `true` for such backends

Policies that use backends with lazy reporting should call `backend.lazy_report()` before making selection decisions to ensure they have the most recent execution information.


The example below shows how a policy's selection function might check this trait and call `lazy_report()` before it
applies its selection logic.

```cpp
    template <typename Function, typename... Args>
    selection_type
    select(Function&& f, Args&&... args)
    {
        if constexpr (backend_traits<Backend>::lazy_report_v)
            backend_->lazy_report();

        // rest of selection logic ....

    }
```

<a id="exit_criteria_id"></a>
## Exit Criteria

- Demonstrate use cases where dynamic selection provides significant improvements.
- Address open questions
  - Is the current API sufficient, performant and user-friendly?
  - Does there need to be support to associate a selected resource with related application data, for example, a device-allocated buffer?
  - Are custom policies needed, and if so, is customization support sufficient and effective?
  - Are custom backends needed, and if so, is customization support sufficient and effective?
  - Should the oneDPL algorithms work with selection policies?
  - What is the proper namespace for the dynamic selection functionality?
  - Do we need a formal concept and/or type trait to check that a type is a selection policy?
  - What is the minimally required C++ standard version (if different from C++17)?
- After open questions are settled, the oneDPL specification must be updated and accepted.