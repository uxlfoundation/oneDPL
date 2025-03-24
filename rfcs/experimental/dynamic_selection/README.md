# Dynamic Device Selection API

## Introduction

Dynamic Selection is a Technology Preview feature 
[documented in the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-7/dynamic-selection-api.html).
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
[Free Functions](#free_functions_id) (such as `submit`, `select`, `wait`, etc), a
[Policy](#policy_req_id) object (such as [fixed_resource_policy](#fixed_resource_id),
[round_robin_policy](#round_robin_id), [dynamic_load_policy](#dynamic_load_id) and
[auto_tune_policy](#auto_tune_id)) and a [Backend](#backend_req_id) object (currently only
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
- `s` a selection of a type `selection_t<T>` , which satisfies [Selection](#selection_req_id), and was made by `p`.
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`
- `r` a container of resources of type `std::vector<resource_t<T>`.

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. calling this function before `initialize` throws a `std::runtime_exception`. |
| `p.select(args…)` | Returns `selection_t<T>` that satisfies [Selection](#selection_req_id). The selected resource must be within the set of resources returned by `p.get_resources()`. |
| `p.submit(s, f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` but does not wait for the `wait_t<T>` object returned by it. |

| Policy Traits* | Description |
| ------- | ----------- |
| `selection<T>::type`, `selection_t<T>` | The wrapped select type returned by `T`. Must satisfy [Selection](#selection_req_id). |
| `resource<T>::type`, `resource_t<T>` | The backend defined resource type that is passed to the user function object. Calling `unwrap` an object of type `selection_t<T>` returns an object of type `resource_t<T>`. |
| `wait_type<T>::type`, `wait_type_t<T>` | The backend type that is returned by the user function object. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns on object of type `wait_type_t<T>`. |

*Policy traits are defined in `include/oneapi/dpl/internal/dynamic_selection_impl/policy_traits.h`.

The current implementation of these traits depends on types defined in the Policy:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using selection_type = typename std::decay_t<Policy>::selection_type;
      using resource_type = typename std::decay_t<Policy>::resource_type;
      using wait_type = typename std::decay_t<Policy>::wait_type;
  };
```
| *Optional* | Description |
| --------------------- | ----------- |
| `p.submit_and_wait(s, f, args…)` | Returns `void`. The function invokes `f` and waits for the `wait_t<T>` it returns to complete. |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` but does not wait for the `wait_t<T>` it returns to complete. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function invokes `f` and waits for the `wait_t<T>` it returns to complete. |

<a id="selection_req_id"></a>
### Selection

The type `T` satisfies *Selection* for a given [Policy](#policy_req_id) `p` if given,

- `s` an arbitrary identifier of type `T`
- `i` an object of type `Info` where `execution_info_v<Info>` is `true`
- 'v' an object of type `V` and `is_same_v<Info::value_type, V>` is `true`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `s.unwrap()` | Returns `resource_t<T>` that should represent one of the resources returned by `p.get_resources()` for the Policy `p` that generated `s`. |
| `s.policy()` | Returns the Policy `p` that was used to make the selection. |
| `s.report(i)` | Returns `void`. Notifies policy that an execution info event has occurred. |
| `report_execution_info<T, Info>::value`, `report_execution_info_v<T,Info>` | `true` if this selection needs the backend to report the Info. `false` otherwise. |

<a id="submission_req_id"></a>
### Submission

The type `T` satisfies *Submission* for a given [Policy](#policy_req_id) `p` if given,

- `s` an arbitrary identifier of type `T`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `s.get_policy()` | Returns an object that satisfies [Policy](#policy_req_id) and corresponds to the Policy that made the selection. |
| `s.wait()` | Blocks until the submission has completed. |
| `s.unwrap()` | Returns the underlying backend type value. This type may be void, may represent the backend’s synchronization type, or may represent a return value from the submission. |

## Provided Concrete Policies 

<a id="fixed_resource_id"></a>
### `fixed_resource_policy`

```cpp
template<typename Backend=sycl_backend> 
class fixed_resource_policy;
```

| Constructors and Initialization |
| -----------------------------------------------------------|
| `fixed_resource_policy(deferred_initialization_t); // (1)` |
| `fixed_resource_policy(size_t offset=0); // (2)` |
| `fixed_resource_policy(const std::vector<resource_t<Backend>>& resources, size_t offset=0); // (3)` |
| `void initialize(size_t offset=0); // (4)` |
| `void initialize(const std::vector<resource_t<Backend>>& resources, size_t offset=0); // (5)` |

1. Defers initialization and requires a later call to `initialize`.
2. Always selects index `offset` in the default set of resources.
3. Uses the provided set of resources and always selects index `offset` from that set.
4. Always selects index `offset` in the default set of resources.
5. Uses the provided set of resources and always selects index `offset` from that set.

`initialize` should only be called for policies that are constructed with `deferred_initialization_t`.

#### select heuristic (expository)

```cpp
  template<typename ...Args>
  selection_type fixed_resource_policy::select(Args&&...) {
    if (initialized_) {
      return selection_type{*this, resources_[fixed_offset_]};
    } else {
      throw std::logic_error(“select called before initialize”);
    }
  }
```

#### execution info reporting requirements

none

#### Exceptions

Constructor or initialize may throw `std::bad_alloc` or `std::logic_error`.

<a id="round_robin_id"></a>
### `round_robin_policy`

```cpp
template<typename Backend=sycl_backend> 
class round_robin_policy;
```

| Constructors and Initialization |
| -----------------------------------------------------------|
| `round_robin_policy(deferred_initialization_t); // (1)` |
| `round_robin_policy(size_t offset=0); // (2)` |
| `round_robin_policy(const std::vector<resource_t<Backend>>& resources); // (3)` |
| `void initialize(size_t offset=0); // (4)` |
| `void initialize(const std::vector<resource_t<Backend>>& resources); // (5)` |

1. Defers initialization and requires a later call to `initialize`.
2. Rotates through the default set of resources at each call to `select` beginning with `offset`.
3. Uses the provided set of resources and rotates through the default set of resources at each call to `select`.
4. Rotates through the default set of resources at each call to `select`.
5. Uses the provided set of resources and rotates through the default set of resources at each call to `select`.

`initialize` should only be called for policies that are constructed with `deferred_initialization_t`.

#### select heuristic (expository)

```cpp
  template<typename ...Args>
  selection_type select(Args&&...) {
    if (initialized_) {
      resources_size_t i = 0;
      {
        std::lock_guard<mutex_type> l(mutex_);
        if (next_context_ == MAX_VALUE) {
          next_context_ = MAX_VALUE%num_contexts_;
        }
        i = next_context_++ % num_contexts_;
      }
      auto &r = resources_[i];
      return selection_type{*this, r};
    } else {
      throw std::logic_error(“select called before initialization”);
    }
  }
```

#### execution info reporting requirements

none

#### Exceptions

Constructor or initialize may throw `std::bad_alloc` or `std::logic_error`.

<a id="dynamic_load_id"></a>
### `dynamic_load_policy`

```cpp
template<typename Backend=sycl_backend> 
class round_robin_policy;
```

| Constructors and Initialization |
| -----------------------------------------------------------|
| `dynamic_load_policy(deferred_initialization_t); // (1)` |
| `dynamic_load_policy(size_t offset=0); // (2)` |
| `dynamic_load_policy(const std::vector<resource_t<Backend>>& resources); // (3)` |
| `void initialize(size_t offset=0); // (4)` |
| `void initialize(const std::vector<resource_t<Backend>>& resources); // (5)` |

1. Defers initialization and requires a later call to `initialize`.
2. Selects the least loaded resource from the default set of resources at each call to `select`.
3. Uses the provided set of resources and then selects the least loaded resource at each call to `select`.
4. Selects the least loaded resource from the default set of resources at each call to `select`.
5. Uses the provided set of resources and then selects the least loaded resource at each call to `select`.

`initialize` should only be called for policies that are constructed with `deferred_initialization_t`.

#### select heuristic (expository)

```cpp
  template<typename ...Args>
  selection_type select(Args&&...) {
    if (initialized_) {
      std::shared_ptr<resource> resource_ptr = nullptr;
      int least_load = std::numeric_limits<load_t>::max();
      for (auto& r : resources_) {
        load_t v = r->load_.load();
        if (resource_ptr == nullptr || v < least_load) {
          least_load = v;
          resource_ptr = r;
        }
      }
      return selection_type {*this, resource_ptr};
    } else {
      throw std::logic_error(“select called before initialization”);
    }
  }
```

#### execution info reporting requirements

```cpp
  void report(task_submission_t) { resource_ptr->load_.fetch_add(1); }
  void report(task_completion_t) { resource_ptr->load_.fetch_sub(1); }
```

#### Exceptions

Constructor or initialize may throw `std::bad_alloc` or `std::logic_error`.

<a id="auto_tune_id"></a>
### `auto_tune_policy`

```cpp
template<typename Backend=sycl_backend> 
class auto_tune_policy;
```

| Constructors and Initialization |
| -----------------------------------------------------------|
| `auto_tune_policy(deferred_initialization_t); // (1)` |
| `auto_tune_policy(double resample_time=never_resample); // (2)` |
| `auto_tune_policy(const std::vector<resource_t<Backend>>& resources, double resample_time=never_resample); // (3)` |
| `void initialize(double resample_time=never_resample); // (4)` |
| `void initialize(const std::vector<resource_t<Backend>>& resources, double resample_time=never_resample); // (5)` |

1. Defers initialization and requires a later call to `initialize`.
2. Profiles each set of unique `f` and `args` for each resource in the default set and then uses the best.
3. Uses the provided set of resources and profiles each set of unique `f` and `args` for each resource uses the best.
4. Profiles each set of unique `f` and `args` for each resource in the default set and then uses the best.
5. Uses the provided set of resources and profiles each set of unique `f` and `args` for each resource uses the best.

`initialize` should only be called for policies that are constructed with `deferred_initialization_t`.

#### select heuristic (expository)

```cpp
  template<typename Function, typename ...Args>
  selection_type select(Function&& f, Args&&...args) {
    if (initialized_) {
      auto k = make_task_key(f, args...);
      auto tuner = get_tuner(k);
      auto offset = tuner->get_resource_to_profile();
      if (offset == use_best) {
        return selection_type {*this, tuner->best_resource_, tuner}; 
      } else {
        auto r = resources_[offset];
        return selection{*this, r, tuner}; 
      }
    } else {
      throw std::logic_error(“select called before initialization”);
    } 
  }
```

#### execution info reporting requirements

```cpp
  void report(task_execution_time_t, timing_t v) { 
    tuner_->add_timing(offset_, v); 
  }
```

#### Exceptions

Constructor or initialize may throw `std::bad_alloc` or `std::logic_error`.

<a id="backend_req_id"></a>
## Backends

Backends allow generic policies to be implemented. End-users do not directly interact
with a backend, except to choose the backend if they opt-out of the default SYCL backend.
Custom policy writers that wish to implement a generic policy that can accept backends
should follow the backend contract. Developers that wish to provide a backend that can
be used with the existing concrete policies should also follow the backend contract.

### Named Requirements

The type `T` satisfies the *Backend* contract if given,

- `b` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` is of type `S` and satisfies *Selection* and `is_same_v<resource_t<S>, resource_t<T>>` is `true`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`
- `r` a container of resources of type `std::vector<resource_t<T>`.

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `b.submit(s, f, args…)` | Returns an object that satisfies *Submission*. The function invokes `f` but does not wait for the `wait_t<T>` object returned by it. |
| `b.get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this wait function blocks until all previous submissions to this backend are complete. |
| `b.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| *Optional* | Description |
| --------------------- | ----------- |
| `void lazy_report()` | If defined by a backend, this function must be called before each new selection. It triggers reporting of the necessary execution info back to the policy. |

<a id="free_functions_id"></a>
## Free Functions

| Signature | Description |
| --------- | ----------- |
| `vector<typename policy_traits<P>::resource_type> get_resources(P&& p);` | Returns the resources associated with the Policy `p`. |
| `template<typename P, typename... Args> selection_t<P> select(P&& p, Args&&... args);` | Applies the policy `p` and returns a *Selection*. |
| `template<Selection S, tyepname F, typename... Args> auto submit(Selection s, F&& f, Args&&... args);` | Invokes `f` with the unwrapped resource from selection `s` and `args`. Implements any instrumentation necessary for the backend to report necessary execution information. May be implemented as `s.get_policy().submit(s, f, args…)`. |
| `template<Policy P, tyepname F, typename... Args> auto submit(P&& p, F&& f, Args&&... args);` | Invokes `f` with the unwrapped resource returned by `select(p, f, args…)` and `args`. Implements any instrumentation necessary for the backend to report necessary execution information. May be implemented as `p.submit(p.select(p, f, args…), f, args…)`. |
| `template<Selection S, tyepname F, typename... Args> auto submit_and_wait(Selection s, F&& f, Args&&... args);` | Invokes `f` with the unwrapped resource from selection `s` and `args`. And then waits on object returned by the `f`.  May be implemented as `wait(s.get_policy().submit(s, f, args…))`. |
| `template<Policy P, tyepname F, typename... Args> auto submit_and_wait(P&& p, F&& f, Args&&... args);` |  Invokes `f` with the unwrapped resource returned by `select(p, f, args…)` and `args`.And then waits on object returned by the `f`. May be implemented as `wait(p.submit(p.select(f, args…),f,args…))`. |
| `template<typename P> auto get_submission_group(P&& p);` | Returns an object that has a member function `void wait()`. Calling this wait function blocks until all previous submissions to this policy are complete. |
| `template<typename W> void unwrap(W&& w) noexcept;` | Returns `w.unwrap()` if available, otherwise returns `w`. |
| `template<typename W> void wait(W&& w);` | Calls `w.wait()` if available. |
| `template <typename S, typename Info> void report(S&& s, const Info& i);` | Reports that event `i` has occurred. |
| `template <typename S, typename Info, typename Value> void report(S&& s, const Info& i, const Value& v); ` | Reports a new value `v` for event `i`. |

### Deferred Initialization

A call to `get_resources`, `select`, `submit` or `submit_and_wait` may initialize
underlying state variables, including dynamic allocation. Initialization may throw `std::bad_alloc`.
If `p` is a policy constructed with deferred initialization, calling these functions before 
calling `initialize` will throw `std::logic_error`.

<a id="execution_info_id"></a>
## Execution Info

Policies are informed of key events through the reporting of Execution Info.
Most commonly, this reporting is done by a backend and is not visible to the
end user.  However, developers that implement custom backends, or that
manage work submission without using the `submit` or `submit_and_wait` functions,
will need to report Execution Info to allow policies to work properly. There
are currently three kinds of Execution Info that may be required by a Policy:

| Execution Info    | Value Type | Description |
| ----------------- | ---------- | ----------- |
| `task_time`       | `uint64_t` | Clock ticks that elapsed between task submission and completion. |
| `task_submission` | void       | The task has been submitted. |
| `task_completion` | void       | The task is complete |

| Info Traits* | Description |
| ------- | ----------- |
| `report_info<S,Info>::value`, `report_info_v<S,Info>` | 'true' if the *Selection* requires the event type to be reported |
| `report_value<S,Info,V>::value`, `report_info_v<S,Info,V>` | `true` if the *Selection* requires the event value to be reported |

* These traits are defined in `include/oneapi/dpl/internal/dynamic_selection_traits.h`

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

A backend may choose not to actively report events and instead lazily report them on demand by defining the
optional `lazy_report` member function. A backend trait is provided to determine if calls to `lazy_report`
are required.

| Backend Traits* | Description |
| ------- | ----------- |
| `lazy_report`<B>::value`, `report_report_v<B>` | `true` if a *Backend* requires that a *Policy* calls to `lazy_report` before making a selection. |

The example below shows how a `select` function might check this trait and call the function before it 
applies its selection logic.

```cpp
    template <typename Function, typename... Args>
    selection_type
    select(Function&& f, Args&&... args)
    {
        if constexpr (backend_traits::lazy_report_v<Backend>)
            backend_->lazy_report();

        // rest of selection logic ....

    }
```

<a id="exit_criteria_id"></a>
## Exit Criteria

- Demonstrate use cases where dynamic selection provides significant improvements.
- Address open questions
  - Is the current API sufficient, performant and user-friendly
  - Are custom policies needed, and if so, is customization support sufficient and effective.
  - Are custom backends needed, and if so, is customization support sufficient and effective.