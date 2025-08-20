# Custom Policies for Dynamic Selection

This document provides detailed information about customizing selection policies for
Dynamic Selection. For an overview of both backend and policy customization approaches,
see the main [README](README.md).

## Current Policy Design

As described in [the current design](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection),
type `T` satisfies the *Policy* contract if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Functions and Traits  | Description |
| --------------------- | ----------- |
| `resource_t<T>` | Policy trait for the resource type. |
| `wait_t<T>` | Policy trait for type that is expected to be returned by the user function |
| `selection_t<T>` | Policy trait for the selection handle type returned by `select` |
| `p.select(args…)` | Returns a selection handle that satisfies *Selection* |
| `p.submit(s, f, args…)` | Submits function `f` using selection `s`. Delegates to backend. |
| `p.submit(f, args…)` | Calls `select()` then `submit(s, f, args…)` |
| `p.submit_and_wait(s, f, args…)` | Submits and waits for completion |
| `p.submit_and_wait(f, args…)` | Calls `select()` then `submit_and_wait(s, f, args…)` |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. Delegates to backend. |
| `p.get_submission_group()` | Returns an object that can wait for all submissions. Delegates to backend. |

Currently, these functions must be implemented in each policy, along with proper resource management,
backend integration, and selection logic. This proposal aims to simplify policy writing by providing
a base class that handles the common functionality.

## Proposed Design to Enable Easier Customization of Policies

This proposal presents a flexible policy system based on a `policy_base` template class that can be used 
for most selection strategies. The design uses CRTP (Curiously Recurring Template Pattern) to allow 
customization while providing sensible defaults for resource management and backend integration.

### Key Components

1. **`policy_base<Policy, ResourceType, Backend>`**: A proposed base class template that implements the core policy functionality using CRTP.
2. **Selection Strategy Implementation**: Derived policies only need to implement `select_impl()` and `initialize_impl()` methods.
3. **Backend Integration**: The base class handles all backend interactions, resource management, and submission delegation.

### Core Features

- **Resource Management**: Policies store and manage resources through the backend
- **Backend Integration**: All backend operations are handled by the base class
- **Initialization Support**: Both immediate and deferred initialization patterns
- **Selection Delegation**: The base class delegates selection to the derived policy's `select_impl()` method
- **Submission Handling**: All submission operations are forwarded to the backend
- **Error Handling**: Proper error checking for uninitialized policies

### Implementation Details

The proposed `policy_base` class provides core functionality that derived policies can customize:

```cpp
template <typename Policy, typename ResourceType, typename Backend>
class policy_base 
{
  protected:
    using backend_t = Backend;
    using resource_container_t = typename backend_t::resource_container_t;
    using execution_resource_t = typename backend_t::execution_resource_t;

  public:
    using resource_type = decltype(unwrap(std::declval<execution_resource_t>()));
    using selection_type = basic_selection_handle_t<Policy, execution_resource_t>;

  protected:
    std::shared_ptr<backend_t> backend_;

  public:
    // Resource management
    auto get_resources() const;
    
    // Initialization support
    void initialize();
    template <typename... Args>
    void initialize(const std::vector<resource_type>& u, Args... args);
    
    // Selection delegation
    template <typename... Args>
    auto select(Args&&... args);
    
    // Submission delegation
    template <typename selection_type, typename Function, typename... Args>
    auto submit(selection_type e, Function&& f, Args&&... args);
    
    // Group operations
    auto get_submission_group();
};
```

### Policy Implementation Pattern

Derived policies need only implement two key methods:

#### `initialize_impl()` Implementation
Handles policy-specific initialization after the backend is set up:

```cpp
void initialize_impl() {
    // Policy-specific initialization logic
    // Backend and resources are already available via base_t::get_resources()
}
```

#### `select_impl()` Implementation  
Implements the policy's resource selection strategy:

```cpp
template <typename... Args>
selection_type select_impl(Args&&... args) {
    // Implement selection logic here
    // Return selection_type{*this, selected_resource}
}
```

## Examples of Policy Implementation

### Round Robin Policy

The `round_robin_policy` demonstrates a simple stateful selection strategy:

```cpp
template <typename ResourceType, typename ResourceAdapter, typename Backend>
class round_robin_policy : public policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend> 
{
  protected:
    using base_t = policy_base<round_robin_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>;
    
    struct selector_t {
        typename base_t::resource_container_t resources_;
        resource_container_size_t num_contexts_;
        std::atomic<resource_container_size_t> next_context_;
    };
    std::shared_ptr<selector_t> selector_;

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::selection_type;

    // Constructors
    round_robin_policy() { base_t::initialize(); }
    round_robin_policy(deferred_initialization_t) {}
    round_robin_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {}) { 
        base_t::initialize(u, adapter); 
    }

    // Policy-specific initialization
    void initialize_impl() {
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
    selection_type select_impl(Args&&...) {
        if (selector_) {
            resource_container_size_t current;
            // Atomic round-robin selection
            while (true) {
                current = selector_->next_context_.load();
                auto next = (current + 1) % selector_->num_contexts_;
                if (selector_->next_context_.compare_exchange_strong(current, next)) 
                    break;
            }
            return selection_type{*this, selector_->resources_[current]};
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
class dynamic_load_policy : public policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>
{
  protected:
    using base_t = policy_base<dynamic_load_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>;
    using load_t = int;

    // Resource wrapper with load tracking
    struct resource_t {
        execution_resource_t e_;
        std::atomic<load_t> load_;
        resource_t(execution_resource_t e) : e_(e), load_(0) {}
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

  public:
    using selection_type = dl_selection_handle_t<dynamic_load_policy>;

    // Initialization with load tracking setup
    void initialize_impl() {
        auto u = base_t::get_resources();
        resources_.clear();
        resources_.reserve(u.size());
        for (auto& resource : u) {
            resources_.emplace_back(std::make_shared<resource_t>(resource));
        }
    }

    // Load-based selection strategy
    template <typename... Args>
    selection_type select_impl(Args&&...) {
        if (!resources_.empty()) {
            // Find resource with minimum load
            auto min_resource = std::min_element(resources_.begin(), resources_.end(),
                [](const auto& a, const auto& b) {
                    return a->load_.load() < b->load_.load();
                });
            return selection_type{*this, *min_resource};
        } else {
            throw std::logic_error("select called before initialization");
        }
    }
};
```

## Benefits of the Proposed Design

The proposed design provides several advantages:

1. **Reduced Complexity**: Policy writers only need to implement selection logic, not resource management
2. **Backend Integration**: All backend operations are handled automatically by the base class
3. **Error Handling**: Proper initialization checking and error reporting built-in
4. **Flexibility**: Support for both immediate and deferred initialization patterns
5. **Performance**: Template-based design with compile-time polymorphism
6. **Consistency**: All policies follow the same patterns for resource management and submission
7. **Extensibility**: Easy to add new selection strategies by focusing only on the selection logic

## Implementation Requirements

To implement a custom policy using this design:

1. **Inherit from `policy_base`**: Use CRTP pattern with appropriate template parameters
2. **Implement `initialize_impl()`**: Set up any policy-specific state after backend initialization
3. **Implement `select_impl()`**: Implement the core selection strategy
4. **Optional: Custom selection handle**: For policies requiring specialized reporting or state tracking
5. **Constructor support**: Provide constructors for immediate and deferred initialization

### Minimal Custom Policy Example

Here's a minimal example of a custom policy that selects resources randomly:

```cpp
template <typename ResourceType, typename ResourceAdapter, typename Backend>
class random_policy : public policy_base<random_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend> 
{
  protected:
    using base_t = policy_base<random_policy<ResourceType, ResourceAdapter, Backend>, ResourceType, Backend>;
    typename base_t::resource_container_t resources_;
    std::random_device rd_;
    std::mt19937 gen_;

  public:
    using resource_type = typename base_t::resource_type;
    using typename base_t::selection_type;

    random_policy() : gen_(rd_()) { base_t::initialize(); }
    random_policy(const std::vector<resource_type>& u, ResourceAdapter adapter = {}) 
        : gen_(rd_()) { base_t::initialize(u, adapter); }

    void initialize_impl() {
        resources_ = base_t::get_resources();
    }

    template <typename... Args>
    selection_type select_impl(Args&&...) {
        if (!resources_.empty()) {
            std::uniform_int_distribution<> dis(0, resources_.size() - 1);
            auto index = dis(gen_);
            return selection_type{*this, resources_[index]};
        } else {
            throw std::logic_error("select called before initialization");
        }
    }
};
```

## Conclusion

This proposal presents a simplified approach to policy customization for Dynamic Selection. Key benefits include:

1. **Focused Development**: Policy writers can focus solely on selection strategy implementation
2. **Automatic Backend Integration**: All resource management and backend operations handled automatically  
3. **Consistent Interface**: All policies follow the same patterns and provide the same baseline functionality with their own selection criteria.
4. **Reduced Boilerplate**: Eliminates the need to implement resource management, submission delegation, and error handling
5. **Backward Compatibility**: Maintains the existing Dynamic Selection API contract
6. **Extensibility**: Easy to add new selection strategies with minimal code

This design opens Dynamic Selection policy development to broader use cases while maintaining performance and providing a consistent developer experience.
