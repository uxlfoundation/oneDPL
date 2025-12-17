# Custom Policies for Dynamic Selection

This document provides detailed information about customizing selection policies for
Dynamic Selection. For an overview of both backend and policy customization approaches,
see the main [README](README.md).

## Previous Policy Design

Previous to this proposal, the following contract existed for policies.
Type `T` satisfies the *Policy* contract if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `/*ret_type*/ fun(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_t<T>` | Policy trait for the resource type. |
| `p.select(args…)` | Returns `selection_t<T>` that satisfies [Selection](#selection_req_id). The selected resource must be within the set of resources returned by `p.get_resources()`. |
| `p.submit(s, f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` with the selected resource `s` and the arguments `args...`. |
| `p.submit(f, args…)` | Calls `select()` then `submit(s, f, args…)` |
| `p.submit_and_wait(f, args…)` | Calls `select()` then `submit_and_wait(s, f, args…)` |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. Delegates to backend. |
| `p.get_submission_group()` | Returns an object that can wait for all submissions. Delegates to backend. |

These functions had to be implemented in each policy, along with proper resource management,
backend integration, and selection logic. This proposal aims to simplify policy writing by providing
a base class that handles the common functionality.

## Proposed Changes to Policy Contract

### Removal of Public Selection API
We propose removing the public selection API from policies. Previously, users could call `select()` to get a selection handle, then separately call `submit()` with that handle. This is replaced with a simplified model where **selection is always implicit within submission**.

The following are removed from the public policy contract:
- `p.select(args…)` - Explicit selection
- `p.submit(s, f, args…)` - Submission with explicit selection handle
- `p.submit_and_wait(s, f, args…)` - Wait with explicit selection handle
- `selection_t<T>` trait

#### New Public API Contract

Users interact exclusively through free functions that perform implicit selection:
- `ex::submit(policy, function, args...)` - Submit work with implicit selection
- `ex::submit_and_wait(policy, function, args...)` - Submit and wait with implicit selection
- `ex::try_submit(policy, function, args...)` - Attempt submission, returns null if no resource available

This results in a simplified contract for policies:

A Policy is an object with a valid dynamic selection heuristic.

The type `T` satisfies *Policy* if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `backend_type` | Type alias for the backend type used by the policy. |
| `resource_type` | Type alias for the resource type used by the policy. |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| One of the following *must* be well-formed | Description |
| ------------------------------------------ | ----------- |
| `p.try_submit(f, args…)` | Returns `std::optional<submission_t<T>>`. Selects a resource and invokes `f`, or returns empty if no resource is available. |
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

**Note:** Policies inheriting from `policy_base` automatically have `backend_type` and `resource_type` type aliases provided. Generic implementations exist: `submit_and_wait` uses `submit` if available, and `submit` uses `try_submit` if available. Therefore, implementing `try_submit` provides all three variants automatically.

#### Internal Selection Primitive for Policy Authors

Policy authors implement a protected **`try_select(args...)`** method that encodes their selection strategy. It returns `std::optional<selection_type>` containing the selected resource, or empty if no resource is available. The `policy_base` class provides generic submission methods that call `try_select()` internally and forward to the backend.

Since dynamic selection is an experimental API, we can modify the API in this way, but this is a breaking change for users using the public selection functions.

An additional benefit of pairing selection and submission 1-to-1 with implicit selection is that selection must occur close to submission time, which improves accuracy for policies like Dynamic Load and Autotune that use runtime statistics.

## Proposed Design to Enable Easier Customization of Policies

This proposal presents a flexible policy system based on a `policy_base` template class that can be used 
for most selection strategies. The design uses CRTP (Curiously Recurring Template Pattern) to allow 
customization while providing sensible defaults for resource management and backend integration.


### Implementation Details

The proposed `policy_base` class provides core functionality using CRTP that derived policies can customize:

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

Policies specify reporting requirements (`task_time_t`, `task_submission_t`, `task_completion_t` from `oneapi::dpl::experimental::execution_info`) via the `policy_base` constructor. These are passed to the backend constructor, which filters devices based on feature availability. Examples: `auto_tune_policy` requires `task_time_t`; `dynamic_load_policy` requires `task_submission_t` and `task_completion_t`.

Selection handles must include a `scratch_space` member of type `backend_traits<Backend>::template selection_scratch_t<reqs...>` to provide backend storage for instrumentation.

## Examples of Policy Implementation

### Round Robin Policy

The `round_robin_policy` demonstrates a simple stateful selection strategy. See the full implementation at:

https://github.com/uxlfoundation/oneDPL/blob/38c94b0bf58b4cde2431085180893bc957e6d07c/include/oneapi/dpl/internal/dynamic_selection_impl/round_robin_policy.h#L36-L100

### Dynamic Load Policy

The `dynamic_load_policy` demonstrates a more complex selection strategy with load tracking. See the full implementation at:

https://github.com/uxlfoundation/oneDPL/blob/38c94b0bf58b4cde2431085180893bc957e6d07c/include/oneapi/dpl/internal/dynamic_selection_impl/dynamic_load_policy.h#L37-L152

## Implementation Requirements

To implement a custom policy:

1. Inherit from `policy_base` using CRTP pattern
2. Implement `initialize_state()` for policy-specific initialization
3. Implement `try_select()` for selection strategy
4. Optionally provide custom selection handle for specialized reporting
5. Provide constructors for immediate and deferred initialization

These changes reduce policy implementations to only their unique elements (selection and initialization), eliminating redundancy and boilerplate while improving maintainability.

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
    std::optional<selection_type> try_select(Args&&...) {
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
