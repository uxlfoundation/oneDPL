# Simplified Customization of Backends and Policies for Dynamic Selection

Dynamic Selection is a Technology Preview feature 
[documented in the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-selection-api.html)
and its current design is described by 
[an experimental RFC](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection).
When applying Dynamic Selection to use cases, there is often the desire to add a new
backend or new selection policy. While it is possible to add new policies or backends
by following the contract described in the current design, the process is non-trivial
and requires some (unecessarily) verbose code.

## Current backend design

A backend defines resource types and implements reporting that is requireed by
a policy.

As described in [the current design](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection),
type `T` satisfies the *Backend* contract if given,

- `b` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` is of type `S` and satisfies *Selection* and `is_same_v<resource_t<S>, resource_t<T>>` is `true`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Required Member Function | Description |
| --------------------- | ----------- |
| `submit(s, f, args…)` | Returns an object that satisfies *Submission*. The function invokes `f` but does not wait for the `wait_t<T>` object returned by it. |
| `get_resources()` | Returns a `std::vector<resource_t<T>>`. |
| `get_submission_group()` | Returns an object that has a member function `void wait()`. Calling this `wait` function blocks until all previous submissions to this backend are complete. |

## Use Cases to Improve for Backends

| Use Case | Description |
| --------------------- | -------- |
| Default Backend | User only provides universe, no custom backend at all. |
| Uninstrumented Backend | User defines small part of backend: `wait_type`, `submission_group`. Better waiting support. |
| Instrumented Backend | User provides instrumentation for reporting |

We remind readers of this proposal that the requirements of *Submission* and *Selection* objects are provided in
[the current design description](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection).

### Default Backend

For policies that require no reporting of `execution_info`, it should be simple to use a new resource type,
perhaps even without writing a custom backend. For example, what if we want to round-robin through pointers to
queues instead of queues. 

```cpp
    sycl::queue q1, q2;
    ex::round_robin_policy p{ { &q1, &q2 } };

    auto s1 = onedpl::dpl::experimental::submit(p, [](sycl::queue *qp) -> sycl::queue {
        qp->submit(/*...*/);
        return *q;
    }

    auto s2 = onedpl::dpl::experimental::submit(p, [](sycl::queue *qp) -> sycl::queue {
        qp->submit(/*...*/);
        return *q;
    }

    wait(s1);
    wait(s2);
```

#### submit

The main role that `submit` plays is to add instrumentation around the call to `f`.
Experience has shown us that most backends perform four basic steps in their implementation
of the `submit` function:

1. Do any setup needed for implementing reporting before calling `f`.
2. The function `f` is called.
3. Do any setup needed for implementing reporting after calling `f`, perhaps using what was returned by `f`.
4. The *Submission* object, which typically wraps what is returned by `f`, is constructed and returned.

If a backend writer doesn't care if their backend works with policies that require reporting, then
they can use a default implementation of submit that simply calls `f` and returns the backends *Submission*
object. A *Submission* object must support `s.wait()` and `s.unwrap`. A default backend cannot meaningfully
implement `wait` on an arbitrary type, but could easily wrap a type, such as `sycl::queue` or `tbb::task_group`
that provides a `wait` member function and call that `wait` member function if it finds it, and in other cases
provide an empty `wrap` function. `s.unwrap` can return whatever was returned by the user's function. Our simple
example above returns a `sycl::queue` from its function and so a default backend can easily synthesize a meaningful
*Submission* type from it, since `q.wait()` is valid.

#### get_resources

The `get_resources` function can default to returning the vector that is passed to the policy, and
then down to the backend's constructor. A policy that is created with a default constructor will
have no resources in this case.

#### get_submission_group

The `get_submission_group` is not easily implemented in a meaningful way. It must return a type that
defines a member function `wait`. But since there is no way to know how to wait on all previous submissions
for an arbitrary backend resource, this function will likely need to return a dummy type or 
`get_submission_group` be undefined for the default backend.

### Uninstrumented Backend

It may be possible that a backend writer wants to do a little bit of work, so that `get_submission_group` works
properly, and/or that there is a meaningful *Submission* type defined, and/or there is a default set of resources
defined even when called with no explicit universe. But, they may not want to go through the effort of
implementing the reporting mechanism. In that case, we should define a way to customize the default
backend for the resource type in a way to opt-in to only those parts that should be customized.

For example, we may round-robin through pointers to queues, have a default set of resources
and/or wait on the submission group:

```cpp
    ex::round_robin_policy<sycl::queue*> p;

    auto r = p.get_resources(); // returns some meaningful default set

    auto s1 = onedpl::dpl::experimental::submit(p, [](sycl::queue *qp) -> sycl::queue {
        qp->submit(/*...*/);
        return *q;
    }

    auto s2 = onedpl::dpl::experimental::submit(p, [](sycl::queue *qp) -> sycl::queue {
        qp->submit(/*...*/);
        return *q;
    }

    wait(p.get_submission_group());
```

#### submit

See [Default Backend](#default-backend).

#### get_resources

Can opt-in to defining a `get_resources` function that will determine a useful universe of resources even if
the backend was default constructed. For example, a SYCL backend might use `get_devices`.

#### get_submission_group

Can opt-in to defining a `get_submission_group` function that will return a non dummy type.

### Instrumented Backend

As desribed in the [Default backend](#default-backend) section, the `submit` function generally
performs four steps, two of which are related to instrumentation. We proposed adding two new
functions `instrument_before` and `instrument_after` that can be used to opt-in to instrumentation
in a backend. A backend writer will not need to fully define `submit` but instead only write the
code to set up and reporting needed for the backend.

For example, we may using auto-tune for pointers to queues:

```cpp
    sycl::queue q1, q2;
    ex::auto_tune_policy p{ { &q1, &q2 } };

   for (int i = 0; i < 100; ++i) {
        auto s = onedpl::dpl::experimental::submit(p, [](sycl::queue *qp) -> sycl::queue {
            qp->submit(/*...*/);
            return *q;
        }
        wait(s);
   }
```

#### submit

```cpp
    template <typename SelectionHandle, typename Function, typename... Args>
    auto submit(SelectionHandle s, Function&& f, Args&&... args) {
        instrument_before(s); // do insrumentation before calling `f`

        return instrument_after( // do insrumentation before calling `f`, and create Submission type
            s,
            std::forward<Function>(f)(unwrap(s), std::forward<Args>(args)...) // call f
        );
    }
```

#### get_resources and get_submission_group

An instrumented backend, can independently opt-in to customizing other functions to better
support `get_submission_group`, `get_resources` and to define the *Submission* type.
See [Uninstrumented Backend](#uninstrumented-backend).

## Backend Customization Approach

We proposed using a CRTP (Curiously Recurring Template Pattern) to define backends since this
will allow developers to flexibly mix-in functionaility from the base. We decided against free
functions or customization points, because we found that most backends require some kind of state
to be maintained (such as the universe) and accessing that state is easiest from within a backend
object. We also think that static polymorphism is sufficient.

The base will provide common-sense default behaviours but a backend can selectively customization behaviours
via static polymorphism. The base class will provide default implementations of `get_resources` (returning
the vector passed to the constructor), `get_submission_group` (returning a dummy type) and `submit`. 
There will be protected `_impl` functions that can be overridden in the derived classes to change the 
default behaviors.

There will also be a struct `scratch_t` that can be used to provide space in a *Selection* needed by
the backend for reporting. For example, space to store a beginning time in `instrument_before` and
ending event in `instrument_after` and so on. `scratch_t` is a template that will receives
`execution_info` parameters and so can be specialized for a backend to only include the space needed
by the policy that uses it. For example, a backend can choose to only include space for a
beginning time if the policy requires reporting of `execution_info::task_time`.

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

        template <typename SelectionHandle, typename Function, typename... Args>
        auto submit(SelectionHandle s, Function&& f, Args&&... args) {
            return static_cast<Backend*>(this)->submit_impl(s, f, std::forward<Args>(args)...);
        }

    protected:

        //
        // The default functionality that can be overriden in derived classes
        //

        std::vector<resource_type> resources_;

        template <typename SelectionHandle, typename Function, typename... Args>
        auto submit_impl(SelectionHandle s, Function&& f, Args&&... args);

        template <typename SelectionHandle>
        void instrument_before_impl(SelectionHandle s);

        template <typename SelectionHandle, typename WaitType>
        auto instrument_after_impl(SelectionHandle s, WaitType w);

        std::vector<resource_type> get_resources_impl() const noexcept;

        auto get_submission_group_impl();

    };
```

We expect to provide a catch-all [default backend](#default-backend) for uncustomized resource types:

```cpp
    // A general default backend for ResourceType
    template< typename ResourceType >
    class default_backend : public backend_base<ResourceType, default_backend<ResourceType>> {
    public:
        using resource_type = ResourceType;
    };
```

#### The Default submit

The default implementation therefore eliminates the need to implement `submit` for backends that
follow the usual four step pattern. Instead, backends can optionally implement `instrument_before` 
and `instrument_after`.  The code snippet below shows how this pattern might be implemented in the
`submit_impl` by calling `instrument_before_impl` and `instrument_after_impl`.  The `async_waiter`
type in the base class is the *Submission* type and wraps a type that must provide a `wait` function
(for example `sycl::queue` or `tbb::task_group`).

```cpp
    template<typename ResourceType, typename Backend>
    class backend_base
    {
        // ...

    protected:

        // default submit implements the usual pattern
        template <typename SelectionHandle, typename Function, typename... Args>
        auto submit_impl(SelectionHandle s, Function&& f, Args&&... args)
        {
            static_cast<Backend*>(this)->instrument_before_impl(s);
            return static_cast<Backend*>(this)->instrument_after_impl(
                s, 
                std::forward<Function>(f)(oneapi::dpl::experimental::unwrap(s), 
                                          std::forward<Args>(args)...)
            );
        }

        // by default no instrumentation before the call
        template <typename SelectionHandle>
        void instrument_before_impl(SelectionHandle /*s*/) { }

        // by default no instrumentation after the call
        // But it returns the wait_type 
        template <typename SelectionHandle, typename WaitType>
        auto instrument_after_impl(SelectionHandle /*s*/, WaitType w)
        {
            return async_waiter{w}; // async_waiter is described below
        }
    
    private:

        class async_waiter
        {
            wait_type w_;
        public:
            async_waiter(wait_type w) : w_{w} {}
            void wait() { w_.wait(); }
            wait_type unwrap() { return w_; }
        };

        // ...
    };
```

#### The Default get_resources

For many cases, backends hold the set of resources that are passed to their
constructors and then return that set when `get_resouces` is called. The
default implementation in the base class therefore takes that approach:

```cpp
    template<typename ResourceType, typename Backend>
    class backend_base
    {
        // ...

    protected:
        // default returns the vector of resources
        resource_container_t get_resources_impl() const noexcept {
            return resources_;
        }

        std::vector<resource_type> resources_;
    };
```

If a backend wants to provide a set of resources for a policy/backend that
uses default construction, they can override this function in their derived
type to determine that default set.

#### Customizing get_submission_group

As mentioned earlier there is not a generally helpful default implementation of `get_submission_group`,
but since some backends may want to skip this functionality, we propose a default implementation
that proves an empty implementation.

```cpp
    template<typename ResourceType, typename Backend>
    class backend_base
    {
        // ...

    protected:
        // default returns an empty submission group
        auto get_submission_group_impl()
        {
            return submission_group{};
        }

    private:

        class submission_group
        {
        public:
            void wait() {
            }
        };
    };
```

One possible alternative is to iterate over the resources and call `wait` on
each resource. This might work for some cases such as `sycl::queue` and
`tbb::task_group`.  Another alternative is to capture all *Submission* objects
returned by `submit` and iterate over those. 

#### Default scratch space

In the base class, there is a struct `scratch_t`. To implement reporting, a backend
may need per-selection scratch space to store values. For example, a beginning time,
a `sycl::event`, etc. These will often be set during `instrument_before` or
`instrument_after` and then read later, perhaps in `lazy_report`. The `scratch_t`
defined by a backend is added to the *Selection* type created by the policy for 
use by the backend.

Policies include the space using a backend trait only for the `execution_info` types
that they need:

```cpp
// within oneapi::dpl::experimental
namespace backend_traits
{
    template <typename Backend, typename ...Req>
    inline constexpr bool scratch_space_v = internal::has_scratch_space<Backend, Req...>::value;

    template <typename Backend, typename ...Req>
    using selection_scratch_t = typename scratch_trait_t_impl<Backend, backend_traits::scratch_space_v<Backend, Req...>,Req...>::type;
} //namespace backend_traits
```

## Backend Examples

### Default Backend for NUMA Nodes

```cpp
    std::vector<tbb::numa_node_id> numa_nodes = tbb::info::numa_nodes();

    std::vector<tbb::task_arena> arenas(numa_nodes.size());
    std::vector<tbb::task_group> groups(numa_nodes.size());
    using pair_t = std::pair<tbb::task_arena*, tbb::task_group*>
    struct wait_type {
        tbb::task_group &tg_;
        void wait() { tg_.wait(); }
    };
    std::vector<pair_t> pairs(numa_nodes.size());

    for (int i = 0; i < numa_nodes.size(); i++) {
        arenas[i].initialize(tbb::task_arena::constraints(numa_nodes[i]), 0);
        pairs.emplace_back(&arena[i], &group[i]);
    }

    ex::round_robin_policy<pair_t> rr{ pairs };

    std::vector<wait_type> submissions;

    for (auto i : numa_nodes)
        w.emplace_back(
            onedpl::dpl::experimental::submit(rr, [](pair_t p) {
                p.first->enqueue(
                    p.second->defer([] { 
                        tbb::parallel_for( /*... */);
                    })
                );
                return wait_type{ *p.second };
            });
        );

    for (auto& s : submissions)
        s.wait(); // not waiting in correct arena though...
```

### Uninstrumented Custom Backend for NUMA Nodes

```cpp
    using pair_t = std::pair<tbb::task_arena*, tbb::task_group*>
    ex::round_robin_policy<pair_t> rr{ };

    for (int i = 0; i < 100; ++i) {
        onedpl::dpl::experimental::submit(rr, [](pair_t p) {
            p.first->enqueue(
                p.second->defer([] { 
                    tbb::parallel_for( /*... */);
                })
            );
            return p;
        });
    }

    wait(rr.get_submission_group());
```

### Instrumented Custom Backend for NUMA Nodes

```cpp
    using pair_t = std::pair<tbb::task_arena*, tbb::task_group*>
    ex::auto_tune_policy<pair_t> rr{ };

    for (int i = 0; i < 100; ++i) {
        onedpl::dpl::experimental::submit(rr, [](pair_t p) {
            p.first->enqueue(
                p.second->defer([] { 
                    tbb::parallel_for( /*... */);
                })
            );
            return p;
        });
    }

    wait(rr.get_submission_group());
```