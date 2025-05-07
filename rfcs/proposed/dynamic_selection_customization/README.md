# Simplified Customization of Backends for Dynamic Selection

Dynamic Selection is a Technology Preview feature 
[documented in the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-selection-api.html)
and its current design is described by 
[an experimental RFC](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection).
When applying Dynamic Selection to use cases, there is often the desire to add a new
backend or new selection policy. While it is possible to add new policies or backends
by following the contract described in the current design, the process is non-trivial
and requires some (unnecessarily) verbose code.

## Current backend design

A backend defines resource types and implements reporting that is required by a policy.

As described in [the current design](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection),
type `T` satisfies the *Backend* contract if given,

- `b` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` is of type `S` and satisfies *Selection* and `is_same_v<resource_t<S>, resource_t<T>>` is `true`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_t<T>` | Backend trait for the resource type. |
| `wait_t<T>`     | Backend trait for type that is expected to be returned by the user function |
| `b.submit(s, f, args…)` | Invokes `f` with the resource from the *Selection* and the `args`. Does not wait for the `wait_t<T>` object returned by `f`. Returns an object that satisfies *Submission*. |
| *Submission* type   | `b.submit(s, f, args…)` returns a type that must define two member functions, `wait()` and `unwrap`. |
| `b.get_resources()` | Returns a `std::vector<resource_t<T>>`. |
| `b.get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this `wait` function blocks until all previous submissions to this backend are complete. |
| `lazy_report_v<T>` | `true` if the backend requires a call to `lazy_report` to trigger reporting of `execution_info` back to the policy |
| `b.lazy_report()`  | An optional function, only needed if `lazy_report_v<T>` is `true`. Is invoked by the policy before making a selection. |

Currently, these functions and traits (except the `lazy_report` function) must be implemented in each backend. The experimental backend
for SYCL queues is a bit more than 250 lines of code. With sensible defaults, this proposal aims to simplify backend writing to
open up Dynamic Selection to more use cases.

## An overview of the proposal

This document proposes that there is a default backend that can be used with any resource type to get
sensible default functionality. In addition, custom backends can be created when needed by mixing in
default functionality with specialized functionality. 

### Proposed Defaults

In the following subsections, we describe each
of the functions and traits described in the [Current backend design](#current-backend-design) and
describe the proposed default implementation and the implications of the defaults.

#### Default `resource_t<T>` 

Policies contain a backend and current policies are either default constructed/initialized or
constructed/initialized with a `std::vector<Resource>` of resources. When a vector is passed
to the constructor, the type of the resource can be deduced and used as a template argument
when constructing the backend. When a policy is default constructed, the resource type can be made
a manditory template argument and this can be used to set the resource type in the backend.

It therefore becomes unnecessary to explicitly provide the resource type in the default backend.
For example, in the code below, a `round_robin_policy` is constructed by passing a vector of
pointers to `tbb::task_group`. The type of the resource `tbb::task_group *` can be deduced and
passed as a template argument to the default backend. 

```cpp
    tbb::task_group t1, t2;
    ex::round_robin_policy p{ { &t1, &t2 } };
```

If the policy is default constructed or constructed for deferred initialization, the
resource type must be given explicitly.

```cpp
    ex::round_robin_policy<tbb::task_group*> p1{ };
    ex::round_robin_policy<tbb::task_group*> p2{ deferred_initialization_t };
```
#### Default `wait_t<T>`

The `wait_t<T>` can only be deduced at the time a user function is passed to submit since
it is the type returned by the user's function. There is no way to generally determine a
`wait_t<T>` from an arbitrary `resource_t<T>` and therefore the default backend cannot
provide a meaningful `wait_t<T>`.

We propose removing the `wait_t<T>` from policy traits. We do not think it is useful for
generic code anyway.

#### Default `b.submit(s, f, args…)`

The main role that `submit` plays in a backend is to add instrumentation around the call to `f`.
Experience has shown us that most backends perform four basic steps in their implementation
of the `submit` function:

1. Do any setup needed for implementing reporting before calling `f`.
2. Call the function `f` and capture the return value.
3. Do any setup needed for implementing reporting after calling `f`, perhaps using what was returned by `f`.
4. Construct and return the *Submission* object, which typically wraps what is returned by `f`.

It is not possible for a default backend to properly instrument execution for an unknown resource type to provide
reporting of `task_time`, `task_submission`, and `task_completion`. Therefore, a default implementation
cannot provide useful implementations for steps 1 and 3. However, it can provide more fine-grained hooks
that can be overridden to provide steps 1 and 3, without requiring a custom backend to reimplement the entire
four step pattern.

We propose that the default backend provide a `submit` function that implements the four step pattern
but also calls `instrument_before_impl` and `instrument_after_impl` functions that can be individually overridden
to add instrumentation by backends that need it.

```cpp
    template <typename SelectionHandle, typename Function, typename... Args>
    auto submit(SelectionHandle s, Function&& f, Args&&... args) {
        instrument_before_impl(s); // do instrumentation before calling `f`, step 1
        auto w = std::forward<Function>(f)(unwrap(s), std::forward<Args>(args)...); // step 2
        return instrument_after_impl( s, w ); // steps 3 & 4
    }
```

#### Default *Submission* type

A *Submission* object must support `s.wait()` and `s.unwrap`. A default backend cannot meaningfully
implement `wait` on an arbitrary type, but it can wrap a type, such as `sycl::queue` or `tbb::task_group`
that provides a `wait` member function itself and call that `wait` member function if it exists.
`s.unwrap` can return what was returned by the user's function `f`.

A possible default implementation of a *Submission* object is shown below:

```cpp
    template<typename UserWaitType>
    class default_submission
    {
        UserWaitType w_;
    public:
        default_submission(const UserWaitType& w) : w_{w} {}
        void wait() { 
            if constexpr (has_wait_v<UserWaitType>) {
                w_.wait();
            } 
        }
        UserWaitType unwrap() { return w_; }
    };
```

Uses such a default would support some useful use cases.

```cpp
    tbb::task_group t1, t2;

    // constructs backend with std::vector<tbb::task_group*>{ &t1, &t2 }
    ex::round_robin_policy p{ { &t1, &t2 } };

    auto s = ex::submit(p, 
                        [](tbb::task_group* t) {
                          struct WaitType {
                            task_group *t;
                            void wait() { t->wait(); }
                            task_group* unwrap() { return t; }
                          };

                          t->run(/* something */);

                          return WaitType(t);
                        });
    ex::wait(s);
```

In the above code, no special backend is written for `tbb::task_group*`, but
a policy can be created with a given set of resources that is can be used to submit
and wait on a submission.

#### Default `b.get_resources()`

The `get_resources` function returns the set of resources managed by the backend. If a policy is
constructed with a `std::vector<T>`, then `resource_t<Backend>` is `T` and the function
`get_resources` can return the vector that was passed to the policy constructor, which in turn,
is used to construct the backend.

```cpp
    tbb::task_group t1, t2;

    // constructs backend with std::vector<tbb::task_group*>{ &t1, &t2 }
    ex::round_robin_policy p{ { &t1, &t2 } };

    // returns a std::vector<tbb::task_group*>{ &t1, &t2 }
    // because backend returns the vector it was constructed with
     auto v = p.get_resources();
```

However, a default constructed policy has no information about what set makes sense for
an arbitrary resource type. And so `get_resources` will return an empty vector.

```cpp
    ex::round_robin_policy<tbb::task_group*> p1{ };

    // v is empty
     auto v = p.get_resources();
```

#### Default `b.get_submission_group()`

The function `get_submission_group()` is used by a policy's `get_submission_group()` function
to return an object that has a member function `void wait()`. A call to `wait` blocks until all
incomplete submissions are done.

There is no wait for a default backend to implement a type that waits for all submission to a set
of arbitrary resource types. There are several possible alternatives including:

* Return a submission group that calls `wait` on all of the resources in the backend if `r.wait()` is valid and throws an exception otherwise.
* Return a submission group that has an empty `wait` function.
* Return a submission group that throws an exception if it is waited on.
* Calls to `get_submission_group` fail to compile when the default backend is used.

This proposal recommends the first suggestion.

With that approach, the following example throws an exception:

```cpp
    tbb::task_group t1, t2;

    ex::round_robin_policy<tbb::task_group*> p{ { &t1, &t2 } };

    auto g = p.get_submission_group();
    try {
        ex::wait(g);
    } catch (std::logic_error& e) {
        std::cout << "Failed as expected: " << e.what() << "\n";
    }
```

But the next example completes without an exception since the resource type, tpw,
provides a wait function.

```cpp
    struct tpw {
        tbb::task_group *tg;
        void wait() { tg->wait(); }
    };

    ex::round_robin_policy<tpw> p2{ { tpw{&t1}, tpw{&t2} } };
    auto g2 = p2.get_submission_group();
    ex::wait(g2);
    std::printf("Ok\n");
```

```cpp
    class default_submission_group
    {
    std::vector<ResourceType>& r_;
    
    public:
        default_submission_group(std::vector<ResourceType>& r) : r_(r) {}

        void
        wait()
        {
        if constexpr (has_wait<ResourceType>::value) {
            for (auto& r : r_) 
                r.wait();
        } else {
            throw std::logic_error("wait called on unsupported submission_group.");
        }
        }
    };
```

#### Default `lazy_report_v<T>`

The default backend, which does not provide any reporting support at all, will
have `lazy_report_v<T>` as `false`.

## How to customize for specific resource types

We proposed using a CRTP (Curiously Recurring Template Pattern) to define backends since this
will allow developers to flexibly mix-in functionality from the base. We decided against free
functions or customization points, because we found that most backends require some kind of state
to be maintained (such as the universe) and accessing that state is easiest from within a backend
object. We also think that static polymorphism is sufficient.

The base will provide the common-sense default behaviours described [above](#default-backend) but
then functionality can be selectively customized using static polymorphism. The base class will
provide default implementations of `get_resources` (returning the vector passed to the constructor),
`get_submission_group` (but will fail to compile if invoked) and `submit`. There will be protected
`_impl` functions that can be overridden in the derived classes to change the default behaviors.

There is also a struct `scratch_t` that can be defined by backends to provide space in a policies
*Selection* object needed by the backend for reporting. For example, space to store a beginning time
in `instrument_before_impl` and ending event in `instrument_after_impl` and so on. `scratch_t` is a
template receives `execution_info` parameters and so can be specialized by a backend to only include
the space needed for the reporting requested by the policy. For example, a backend can choose to only
include space for a beginning time if the policy only requires reporting of `execution_info::task_time`.

Show below is a sketch of a `backend_base`:

```cpp
    template<typename ResourceType, typename Backend>
    class backend_base
    {
    public:
        using resource_type = ResourceType;

        // space needed by the backend in the Selection type
        // for implementing reporting, by default an empty struct
        template<typename ...Info> struct scratch_t;

        // constructors
        backend_base() {}
        backend_base(const std::vector<resource_type>& u) : resources_{v} {}

        template <typename SelectionHandle, typename Function, typename... Args>
        auto submit(SelectionHandle s, Function&& f, Args&&... args) {
            return static_cast<Backend*>(this)->submit_impl(s, f, std::forward<Args>(args)...);
        }

        std::vector<resource_type> get_resources() {
            return static_cast<Backend*>(this)->get_resources_impl();
        }

        auto get_submission_group() {
            return static_cast<Backend*>(this)->get_submission_group();
        }

    protected:

        //
        // The default functionality that can be overridden in derived classes
        //

        std::vector<resource_type> resources_;

        template <typename SelectionHandle, typename Function, typename... Args>
        auto submit_impl(SelectionHandle s, Function&& f, Args&&... args) {
        {
            static_cast<Backend*>(this)->instrument_before_impl(s);
            return static_cast<Backend*>(this)->instrument_after_impl(
                s, 
                std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), 
                                          std::forward<Args>(args)...)
            );
        }

        template <typename SelectionHandle>
        void instrument_before_impl(SelectionHandle s) { }

        template <typename SelectionHandle, typename WaitType>
        auto instrument_after_impl(SelectionHandle s, WaitType w) {
            return default_submission{w};
        }

        std::vector<resource_type> get_resources_impl() const noexcept {
            return resources_;
        }

        auto
        get_submission_group_impl() {
            return default_submission_group{resources_};
        }

    };
```

We also propose to provide a catch-all [default backend](#default-backend) for uncustomized resource
types:

```cpp
    // A general default backend for ResourceType
    template< typename ResourceType >
    class default_backend : public backend_base<ResourceType, default_backend<ResourceType>> {
    public:
        using resource_type = ResourceType;
    };
```

A backend that is in the `oneapi::dpl` namespace that relies on the defaults but overrides some functionality
can specialize the default backend for the type. For example, the existing SYCL backend can be implemented as:

```cpp
template< >
class default_backend<sycl::queue> : public backend_base<sycl::queue, default_backend<sycl::queue>>
{
  //* override as needed
};
```

And the use of this SYCL backend by a policy, such as `round_robin_policy` is implemented by a policy
as:

```cpp
template <typename ResourceType = sycl::queue, typename Backend = default_backend<ResourceType>>
struct round_robin_policy
{
    // policy implementation not shown here
};
```

The result is that round-robin policy still default to the SYCL backend, which is implemented using
a specialization of the default backend:

```cpp
    // uses ResourceType = sycl::queue and Backend = default_backend<sycl::queue>
    // and default_backend<sycl::queue> inherits from backend_base<sycl::queue, default_backend<sycl::queue>>
    round_robin_policy p;
```

A third-party can create a custom backend by using the `backend_base`:

```cpp
namespace third_party {
    class custom_resource {
        // implementation
    };

    class custom_backend : public oneapi::dpl::experimental::backend_base<custom_resource, custom_backend> {
        // override _impl functions as desired
    };
}
```

A round-robin policy using such a custom resource and policy would be define as shown:

```cpp
    round_robin_policy<third_party::custom_resource, third_party::custom_backend> p;
```

We propose adding a trait `backend_for_resource` that is used by policies to find the
correct backend for a given resource type, as shown below:

```cpp
namespace oneapi {
namespace dpl {
namespace experimental {

template <typename ResourceType>
struct backend_for_resource
{
    using backend_t = default_backend<ResourceType>;
};


template <typename ResourceType = sycl::queue, typename Backend = backend_for_resource<ResourceType>::backend_t>
struct round_robin_policy
{
    // policy implementation not shown here
};
}
}
}
```

There would be three ways then that a custom backend can be found:

1. The backend writer can add a specialized `backend_for_resource` in the `experimental::dpl` namespace.
2. The backend writer can specialized `default_backend` in the `experimental::dpl` namespace.
3. The backend can be provide explicitly to the policy.

The code below demonstrates each of these approaches:

```cpp
namespace third_party {

    // 1. specialize backend_for_resource in oneapi::dpl::experimental
    class custom_resource_1 {
        // implementation
    };

    class custom_backend_1 : public oneapi::dpl::experimental::backend_base<custom_resource_1, custom_backend_1> {
        // override _impl functions as desired
    };

    // 2. specialize default_backend in oneapi::dpl::experimental
    class custom_resource_2 {
        // implementation
    };

    // 3. no specializations, only available explicitly
    class custom_resource_3 {
        // implementation
    };

    class custom_backend_3 : public oneapi::dpl::experimental::backend_base<custom_resource_3, custom_backend_3> {
        // override _impl functions as desired
    };

}

namespace oneapi {
namespace dpl {
namespace experimental {

    // 1. specialize backend_for_resource in oneapi::dpl::experimental
    template <>
    struct backend_for_resource<third_party::custom_resource_1>
    {
        using backend_t = third_party::custom_backend;
    };

    // 2. specialize default_backend in oneapi::dpl::experimental
    template< >
    class default_backend<third_party::custom_resource_2> 
        : public backend_base<third_party::custom_resource_2, default_backend<third_party::custom_resource_2>> {
    public:
        using resource_type = third_party::custom_resource_2;
        // override _impl functions as desired
    };
}
}
}

int f() {
    // 1. specialize backend_for_resource in oneapi::dpl::experimental
    round_robin_policy<third_party::custom_resource_1> p1;

    // 2. specialize default_backend in oneapi::dpl::experimental
    round_robin_policy<third_party::custom_resource_2> p2;

    // 3. no specializations, only available explicitly
    round_robin_policy<third_party::custom_resource_3,
                       third_party::custom_backend_3> p3;
}
```

## Backend Examples

### Using the Default Backend with a new Resource Type

In the following example, the resource type is `std::pair<tbb::task_arena *, tbb::task_group *>`.
A `tbb::task_arena` represents something like a thread pool, a place where work can be submitted
and executed by threads. A `tbb::task_group` represents a group of tasks that can be waited on
as a group. Below, only the default backend is used, there is no additional backend that is
customized for `std::pair<tbb::task_arena *, tbb::task_group *>`:

```cpp
    namespace ex = oneapi::dpl::experimental;

    using pair_t = std::pair<tbb::task_arena *, tbb::task_group *>;

    // create pairs of arenas and task_groups, one per numa node
    std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();
    std::vector<pair_t> pairs;

    for (int i = 0; i < numa_nodes.size(); i++) {
        pairs.emplace_back(pair_t(new tbb::task_arena{tbb::task_arena::constraints(numa_nodes[i]), 0},
                               new tbb::task_group{}) );
    }
    // end creating default arenas and groups

    ex::round_robin_policy<pair_t> rr{ pairs };

    // helper struct for waiting on the work in the pair
    struct WaitType {
        pair_t pair;
        void wait() { pair.first->execute([this]() { pair.second->wait(); }); }
    };
    std::vector<WaitType> submissions;

    for (auto i : numa_nodes) {
        auto w = ex::submit( rr, 
                             [](pair_t p) {
                                p.first->enqueue(p.second->defer([]() { std::printf("o\n"); }));
                                return WaitType{ p };
                             }
                            );
        submissions.emplace_back(w.unwrap());
    }

    for (auto& s : submissions)
        ex::wait(s);
}
```

In this example, we can see that the user has to manual construct a vector of
resources and pass it to the policy (which in turn passes it to the default
backend). There is no obvious way to wait on this pair of pointers and so
the user also needs to define a `WaitType` in their code and return that
from the user body passed to `submit`.  The default backend also cannot create
a reasonable `submission_group` for this type and so the code manually collects
each submission in a vector and then iterates over this vector to wait on each
submission.

Even those this code is verbose, it required zero lines of backend code to be
written and still allowed the logic of round-robin policy to be used for
selection and submission.

### An Example Custom NUMA Resource and Backend (no instrumentation yet)

The code below shows a custom resource type `ArenaAndGroup` that combines a
`tbb::task_arena` and `tbb::task_group`, providing functions to submit work
 to the pair, `run`, and a function to wait on the pair, `wait`.

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

We can create a `numa_backend` for this type outside of the
`oneapi::dpl::experimental` namespace by inheriting from `backend_base`
and provide a default constructor:

```cpp
namespace numa {
    class numa_backend : public ex::backend_base<ArenaAndGroup, numa_backend> {
    public:
        using resource_type = ArenaAndGroup;
        using my_base = backend_base<ArenaAndGroup, numa_backend>;
        numa_backend() : my_base(), owns_groups_(true) { 
            std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();
            for (int i = 0; i < numa_nodes.size(); i++) {
                resources_.emplace_back( ArenaAndGroup(new tbb::task_arena{tbb::task_arena::constraints(numa_nodes[i]), 0},
                                                       new tbb::task_group{}) );
            }
        }

        numa_backend(const std::vector<ArenaAndGroup>& u) : my_base(u) {  }

        ~numa_backend() {
            if (owns_groups_)
                for (auto& r : resources_) 
                    r.clear();
        }

    private:
        bool owns_groups_ = false;
    };
}
```

With less than 25 lines of code, we now have a backend that simplifies the use of the `ArenaAndGroup` resource:

```cpp
    std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();

    ex::round_robin_policy<numa::ArenaAndGroup, numa::numa_backend> rr{ };
    for (auto i : numa_nodes) {
        ex::submit(rr, 
            [](numa::ArenaAndGroup ag) { 
                ag.run([]() { std::printf("o\n"); });
                return ag; }
        );
    }
    ex::wait(rr.get_submission_group());
```

