# Custom Policies for Dynamic Selection

This document provides detailed information about customizing selection policies for
Dynamic Selection. For an overview of both backend and policy customization approaches,
see the main [README](README.md).

## Current Policy Design

As described in [the current design](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection),
type `T` satisfies the *Policy* contract if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `/*ret_type*/ fun(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_t<T>` | Policy trait for the resource type. |
| `p.try_select(args…)` | Returns selection within `std::optional` if available. The selected resource must be within the set of resources returned by `p.get_resources()`, or returns empty `std::optional`. |
| `p.select_impl(args...)` | Loops calling `try_select(args...)` until a selection is returned. |
| `p.try_submit(f, args...)` |  Selects a resource and invokes `f` with the selected resource and `args...`, returning a `std::optional` holding the submission object. Returns empty `std::optional` if no resource is available for selection. |
| `p.submit(f, args…)` | Calls `select()` then `submit(s, f, args…)` |
| `p.submit_and_wait(f, args…)` | Calls `select()` then `submit_and_wait(s, f, args…)` |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. Delegates to backend. |
| `p.get_submission_group()` | Returns an object that can wait for all submissions. Delegates to backend. |

Currently, these functions must be implemented in each policy, along with proper resource management,
backend integration, and selection logic. This proposal aims to simplify policy writing by providing
a base class that handles the common functionality.

## Proposed Changes to Policy Contract

### Removal of Public Selection API
We propose to remove the public selection API from policies. In the previous design, users could call `select()` to get a selection handle, then separately call `submit()` with that handle. This API has been removed in favor of a simplified model where **selection is always implicit within submission**.

The following functions and traits are removed from the public policy contract:

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `p.select(args…)` | Returns `selection_t<T>` that satisfies [Selection](#selection_req_id). The selected resource must be within the set of resources returned by `p.get_resources()`. |
| `p.submit(s, f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` with the selected resource `s` and the arguments `args...`. |

| *Optional* | Description |
| ---------- | ----------- |
| `p.submit_and_wait(s, f, args…)` | Returns `void`. The function invokes `f` with `s` and `args...` and waits for the `wait_t<T>` it returns to complete. |

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` a selection of a type `selection_t<T>` , which satisfies [Selection](#selection_req_id), and was made by `p`.
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Policy Traits* | Description |
| -------------- | ----------- |
| `policy_traits<T>::selection_type`, `selection_t<T>` | The wrapped select type returned by `T`. Must satisfy [Selection](#selection_req_id). |

#### New Public API Contract

Instead of exposing `select()` publicly, policies now provide **only submission methods**. Users interact exclusively through the following free functions (which delegate to policy member functions):

- `ex::submit(policy, function, args...)` - Submit work with implicit selection
- `ex::submit_and_wait(policy, function, args...)` - Submit and wait with implicit selection
- `ex::try_submit(policy, function, args...)` - Attempt submission, returns null if no resource available

Policies must implement at least one of the following member functions to support these APIs:

| at least one *Must* be well-formed | Description |
| ---------------------------------- | ----------- |
| `p.try_submit(f, args…)` | Returns `std::optional<submission_t<T>>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. Returns empty `std::optional` if no resource is available for selection |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits on the return value of the submission to complete. |

This results in a greatly simplified contract for policies:

A Policy is an object with a valid dynamic selection heuristic.

The type `T` satisfies *Policy* if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `T::backend_type` | Type alias for the backend type used by the policy. |
| `T::resource_type` | Type alias for the resource type used by the policy. |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| One of the following *must* be well-formed | Description |
| ------------------------------------------ | ----------- |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits on the return value of the submission to complete. |

| Policy Traits* | Description |
| -------------- | ----------- |
| `policy_traits<T>::backed_type`, `backend_t<T>` | The backend type associated with this policy. |
| `policy_traits<T>::resource_type`, `resource_t<T>` | The backend-defined resource type that is passed to the user function object. |
| `policy_traits<T>::has_wait_type_v`, `has_wait_type_v<T>` | Boolean which determines if explicit wait type has been provided by the backend associated with this policy. 
| `policy_traits<T>::wait_type`, `wait_type_t<T>` | If `has_wait_type_v<T>` is `true`, contains the type that must returned by the user function object for this policy, otherwise `void`. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns an object of type `wait_type_t<T>`. |

The default implementation of these traits depends on types defined in the Policy:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using resource_type = typename std::decay_t<Policy>::resource_type;
  };
```

**Note:** Policies inheriting from `policy_base` automatically have `backend_type` and `resource_type` type aliases provided by the base class, satisfying these requirements without additional code.

With this contract, if `p.submit(f, args…)` is well-formed, a generic implementation of `submit_and_wait` that uses `submit` is available and waits on the result unless overridden. If `p.try_submit(f,args...)` is well-formed, then a generic implementation of `submit` which uses `try_submit` is available unless overridden. Therefore, providing `try_submit` is enough to have implementations for all three submit variants automatically.

#### Internal Selection Primitive for Policy Authors

While the **public API** no longer exposes `select()`, policy authors still need a way to implement selection logic. The `policy_base` class provides generic implementations of the submission methods, but delegates the actual selection decision to a protected method that derived policies must implement:

**`try_select(args...)`** - Protected method that policy authors implement to encode their selection strategy. Returns `std::optional<selection_type>` containing the selected resource, or an empty `std::optional` if no resource is available.

This separation means:
- **Users** only call `ex::submit()`, `ex::submit_and_wait()`, or `ex::try_submit()` (never `select()`)
- **Policy authors** implement `try_select()` as the internal primitive
- **`policy_base`** provides generic submission methods that call `try_select()` internally and forward to the backend

This would be a breaking change, but since dynamic selection is an experimental API, we can modify the API in this way. However, we will want to consider this fully and perhaps investigate if there is any usage that we may break with these changes.

Beyond simplifying the public interface and requirements, these changes may provide inherent benefits for existing policies by enforcing a specific usage pattern. With the removal of select interfaces, implicit selections and submissions must be paired 1-to-1, and implicit selection will occur very close to submission time. For Dynamic Load Policy and Autotune Policy, which dynamically use statistics about resource load and job performance, this means the implicit selection will be more accurate and up-to-date for the submission.

### Removal of Wait Type
For further simplification of the contract, we also propose to remove the wait_type trait. It is not necessary and can always be obtained with `decltype`, and `submit` with a valid user function. Of course, policies can provide a `wait_type` if they so choose, but it is not required. Requiring policies to provide this wait type increases complexity of customization and we have not seen a use for the public trait in use cases thus far.

## Proposed Design to Enable Easier Customization of Policies

This proposal presents a flexible policy system based on a `policy_base` template class that can be used 
for most selection strategies. The design uses CRTP (Curiously Recurring Template Pattern) to allow 
customization while providing sensible defaults for resource management and backend integration.

### Key Components

1. **`policy_base<Policy, ResourceAdapter, Backend, ReportReqs...>`**: A proposed base class template that implements the core policy functionality using CRTP.
2. **Selection Strategy Implementation**: Derived policies only need to implement `try_select()` and `initialize_state()` methods.
3. **Backend Integration**: The base class handles all backend interactions, resource management, and submission delegation.

### Implementation Details

The proposed `policy_base` class provides core functionality that derived policies can customize:

```cpp
template <typename Policy, typename ResourceAdapter, typename Backend, typename... ReportReqs>
class policy_base
{
  protected:
    using backend_t = Backend;
    using selection_type = basic_selection_handle_t<Policy, resource_type>;

  public:
    // Required type aliases for policy contract
    using backend_type = Backend;
    using resource_type = typename backend_type::resource_type;

  protected:
    std::shared_ptr<backend_t> backend_;

  public:
    // Resource management
    auto get_resources() const;

    // Initialization support
    void initialize();
    template <typename... Args>
    void initialize(const std::vector<resource_type>& u, Args... args);
    
    // Submission delegation(s)
    auto try_submit(Function&& f, Args&&... args);

    template <typename Function, typename... Args>
    auto submit(Function&& f, Args&&... args);

    template <typename Function, typename... Args>
    void submit_and_wait(Function&& f, Args&&... args);
    
    // Group operations
    auto get_submission_group();
};
```

### Policy Implementation Pattern

Derived policies need only implement two key methods:

#### `initialize_state()` Implementation
Handles policy-specific initialization after the backend is set up:

```cpp
void initialize_state() {
    // Policy-specific initialization logic
    // Backend and resources are already available via base_t::get_resources()
}
```

#### `try_select()` Implementation
Implements the policy's resource selection strategy:

```cpp
template <typename... Args>
std::optional<selection_type> try_select(Args&&... args) {
    // Implement selection logic here
    // If a resource is available:
    //   return std::make_optional<selection_type>(selection_type{*this, selected_resource});
    // If no resource is available:
    //   return std::nullopt; // empty optional
}
```
Providing `try_select()` results in `try_submit()`, `submit()`, and `submit_and_wait()` functions to be supported
via generic implementations depending only on the selection logic.

#### Task Reporting Requirements

Policies must also specify what reporting requirements the background is required to support to serve this policy. These
requirements are specified as objects of the following structs: `task_time_t`, `task_submission_t`, `task_completion_t`
in the namespace `oneapi::dpl::experimental::execution_info` and passed to the `policy_base` constructor. They will then
be passed to the backend constructor when initialization occurs, and devices will be filtered based upon the
availability of features required for these reporting requirements.

`auto_tune_policy` requires `task_time_t`, and `dynamic_load_policy` requires `task_submission_t` and `task_completion_t`.

When creating a selection handle for your policy, you must include a member variable `scratch_space` of type `backend_traits<Backend>::template selection_scratch_t<reqs...>` where `Backend` is your backend and `reqs` is a variadic pack of reporting requirements needed for your policy. This allows the backend to have the storage it needs allocated alongside each selection handle to implement instrumentation for the reporting requirements.

## Examples of Policy Implementation

### Round Robin Policy

The `round_robin_policy` demonstrates a simple stateful selection strategy:

```cpp
template <typename ResourceType, typename ResourceAdapter, typename Backend>
class round_robin_policy : public policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>
{
  protected:
    using base_t = policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>;
    using resource_container_size_t = typename std::vector<resource_type>::size_type;
    
    struct selector_t {
        std::vector<std::shared_ptr<resource_t>> resources_;
        resource_container_size_t num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
    };
    std::shared_ptr<selector_t> selector_;
    using selection_type = base_t::selection_type;

  public:
    // Required type aliases (provided by policy_base)
    using backend_type = typename base_t::backend_type;
    using resource_type = typename base_t::resource_type;

    // Constructors
    round_robin_policy() { base_t::initialize(); }
    round_robin_policy(deferred_initialization_t) {}
    round_robin_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {}) { 
        base_t::initialize(u, adapter); 
    }

    // Policy-specific initialization
    void initialize_state() {
        if (!selector_) {
            selector_ = std::make_shared<selector_t>();
        }
        auto u = base_t::get_resources();
        selector_->resources_ = u;
        selector_->num_contexts_ = u.size();
        selector_->next_context_ = 0;
    }

    // Round-robin selection strategy
    template <typename... Args>
    std::optional<selection_type> try_select(Args&&...) {
        if (selector_) {
            resource_container_size_t current;
            // Atomic round-robin selection
            while (true) {
                current = selector_->next_context_.load();
                auto next = (current + 1) % selector_->num_contexts_;
                if (selector_->next_context_.compare_exchange_strong(current, next)) 
                    break;
            }
            return std::make_optional<selection_type>(*this, selector_->resources_[current]);
        } else {
            throw std::logic_error("select called before initialization");
        }
    }
};
```

### Dynamic Load Policy

The `dynamic_load_policy` demonstrates a more complex selection strategy with load tracking:

```cpp
template <typename ResourceType, typename ResourceAdapter, typename Backend>
class dynamic_load_policy : public policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend,
                                               execution_info::task_submission_t, execution_info::task_completion_t>
{
  protected:
    using base_t = policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend,
                               execution_info::task_submission_t, execution_info::task_completion_t>;
    using load_t = int;

    // Resource wrapper with load tracking
    struct resource_t {
        resource_type e_;
        std::atomic<load_t> load_;
        resource_t(resource_type e) : e_(e), load_(0) {}
    };
    using resource_container_t = std::vector<std::shared_ptr<resource_t>>;

    // Custom selection handle with load reporting
    template <typename Policy>
    class dl_selection_handle_t {
        Policy policy_;
        std::shared_ptr<resource_t> resource_;
        
      public:
        dl_selection_handle_t(const Policy& p, std::shared_ptr<resource_t> r) 
            : policy_(p), resource_(std::move(r)) {}

        auto unwrap() {
            return ::oneapi::dpl::experimental::unwrap(resource_->e_);
        }

        void report(const execution_info::task_submission_t&) const {
            resource_->load_.fetch_add(1);
        }

        void report(const execution_info::task_completion_t&) const {
            resource_->load_.fetch_sub(1);
        }
    };

    resource_container_t resources_;

    using selection_type = dl_selection_handle_t<dynamic_load_policy>;
  public:

    // Initialization with load tracking setup
    void initialize_state() {
        auto u = base_t::get_resources();
        resources_.clear();
        resources_.reserve(u.size());
        for (auto& resource : u) {
            resources_.emplace_back(std::make_shared<resource_t>(resource));
        }
    }

    // Load-based selection strategy
    template <typename... Args>
    std::optional<selection_type> try_select(Args&&...) {
        if (!resources_.empty()) {
            // Find resource with minimum load
            auto min_resource = std::min_element(resources_.begin(), resources_.end(),
                [](const auto& a, const auto& b) {
                    return a->load_.load() < b->load_.load();
                });
            return std::make_optional<selection_type>(*this, *min_resource);
        } else {
            throw std::logic_error("select called before initialization");
        }
    }
};
```

## Benefits of the Proposed Design

The main benefit of the proposed design is simplification of the policy customization experience, and reduction of
individual policy implementations to only their unique elements, selection and initialization. This reduces redundancy
and boiler-plate code, and improves maintainability.  It also makes it easier on customizers to implement new policy
ideas.

## Implementation Requirements

To implement a custom policy using this design:

1. **Inherit from `policy_base`**: Use CRTP pattern with appropriate template parameters
2. **Implement `initialize_state()`**: Set up any policy-specific state after backend initialization
3. **Implement `select_impl()`**: Implement the core selection strategy
4. **Optional: Custom selection handle**: For policies requiring specialized reporting or state tracking
5. **Constructors**: Provide constructors for immediate and deferred initialization

### Minimal Custom Policy Example

Here's a minimal example of a custom policy that selects resources randomly:

```cpp
template <typename ResourceType, typename ResourceAdapter, typename Backend>
class random_policy : public policy_base<random_policy<ResourceType, ResourceAdapter, Backend>, ResourceAdapter, Backend>
{
  protected:
    using base_t = policy_base<random_policy<ResourceType,  ResourceAdapter, Backend>, ResourceAdapter, Backend>;

  public:
    // Required type aliases (provided by policy_base)
    using backend_type = typename base_t::backend_type;
    using resource_type = typename base_t::resource_type;

  protected:
    std::vector<resource_type> resources_;
    std::random_device rd_;
    std::mt19937 gen_;
    using typename base_t::selection_type;

  public:

    random_policy() : gen_(rd_()) { base_t::initialize(); }
    random_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {}) 
        : gen_(rd_()) { base_t::initialize(u, adapter); }

    void initialize_state() {
        resources_ = base_t::get_resources();
    }

    template <typename... Args>
    std::optional<selection_type> select_impl(Args&&...) {
        if (!resources_.empty()) {
            std::uniform_int_distribution<> dis(0, resources_.size() - 1);
            auto index = dis(gen_);
            return std::make_optional<selection_type>(*this, resources_[index]);
        } else {
            throw std::logic_error("select called before initialization");
        }
    }
};
```

## Testing
Testing for these changes should include:
 * Test of policy providing a universe of resources to construction which are used to deduce the resource/backend.
 * Much of the proof of the benefit of these changes will be demonstrated through the reduction of redundancy in
   implementations of provided policies. They should take advantage of the ``policy_base`` and only implement what is
   necessary.

## Open Questions
 * The generic `submit` and `submit_and_wait` design has an infinite loop in its generic implementation if a resource is
   never available from the policy's `try_select`. If a resource is never available, this is generally an issue with
   the backend or the environment. However, there is some question as to whether there should be some external programmatic
   way to break out of this loop if it is taking too long; currently there is none. This is possible future work to
   investigate and possibly add a way to abort submission. TBB's `concurrent_queue` has a possible system we could use
   as a model within its `pop` implementation.
