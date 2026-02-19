# Simplified Customization for Dynamic Selection

Dynamic Selection is a Technology Preview feature 
[documented in the oneDPL Developer Guide](https://www.intel.com/content/www/us/en/docs/onedpl/developer-guide/2022-8/dynamic-selection-api.html)
and its current design is described by 
[an experimental RFC](https://github.com/uxlfoundation/oneDPL/tree/main/rfcs/experimental/dynamic_selection).
When applying Dynamic Selection to use cases, there is often the desire to add new
backends or new selection policies. While it is possible to add new policies or backends
by following the contract described in the current design, the process is non-trivial
and requires some (unnecessarily) verbose code.

## Proposed Customization Solutions

This RFC presents two complementary proposals to simplify Dynamic Selection customization:

### Backend Customization

When applying Dynamic Selection, there is often the desire to add a new backend for
different resource types. With sensible defaults, this proposal aims to simplify
backend writing to open up Dynamic Selection to more use cases. The approach presents a flexible
backend system based on a `backend_base` template class and a `core_resource_backend` template that
can be used for simple resource types with policies which don't have reporting requirements.
For more complex resources or to serve policies with reporting requirements, specialization of
`core_resource_backend` is required.

For detailed information about backend customization, see [Custom Backends](custom_backends.md).

### Policy Customization

There is also the expectation that new selection policies will be implemented. This proposal
aims to simplify policy writing by providing a base class that handles the common functionality.
The approach presents a flexible policy system based on a `policy_base` template class that can
be used for most selection strategies.

For detailed information about policy customization, see [Custom Policies](custom_policies.md).

### High Level Class Diagram

The following diagram shows the relationships of the helpers for customization, and the entry points for the user and for the customizer of policies and backends. This diagram shows `round_robin_policy` specifically, but the same relationship exists for other policies.

```mermaid

classDiagram
    direction BT

    %% Backend Layer
    class backend_base~ResourceType~ {
        #resources_: vector~ResourceType~
        +backend_base(ReportReqs...)
        +backend_base(vector~ResourceType~, ReportReqs...)
        +get_resources()
        +get_submission_group()
        +submit(SelectionHandle, f, args...)
    }


    %% Public API Layer
    class FreeFunctions["Free Functions (dynamic_selection_traits.h)"] {
        <<namespace>>
        +try_submit(policy, f, args...)
        +submit(policy, f, args...)
        +submit_and_wait(policy, f, args...)
        +wait(wait_object)
        +get_resources(policy)
        +unwrap(value)
    }

    class InternalFallbacks["internal namespace"] {
        <<namespace>>
        +submit_fallback(policy, f, args...)
        +submit_and_wait_fallback(policy, f, args...)
    }

    %% Policy Layer
    class policy_base~Policy, ResourceAdapter, Backend, ReportReqs~ {
        <<CRTP Base>>
        #backend_: shared_ptr~Backend~
        +initialize()
        +initialize(vector~resource_type~)
        +initialize(vector~resource_type~, adapter)
        +try_submit(f, args...) shared_ptr~WaitType~
        +submit(f, args...)
        +submit_and_wait(f, args...)
        +get_resources()
        +get_submission_group()
    }

    class round_robin_policy~ResourceType, ResourceAdapter, Backend~ {
        #selector_: shared_ptr~selector_t~
        +round_robin_policy()
        +round_robin_policy(vector~ResourceType~)
        +initialize_state()
        +try_select(args...) shared_ptr~selection_type~
    }

    class core_resource_backend~CoreResourceType, ResourceType, ResourceAdapter~ {
        -adapter: ResourceAdapter
        +core_resource_backend(ReportReqs...)
        +core_resource_backend(vector~ResourceType~, adapter, ReportReqs...)
        +submit(s, f, args...)
    }

    class default_backend~ResourceType, ResourceAdapter~ {
        +default_backend(ReportReqs...)
        +default_backend(vector~ResourceType~, adapter, ReportReqs...)
    }

    %% Relationships - Inheritance
    round_robin_policy --|> policy_base : inherits (CRTP)
    core_resource_backend --|> backend_base : inherits
    default_backend --|> core_resource_backend : inherits

    %% Relationships - Composition
    policy_base *-- default_backend : backend_


    %% Relationships - Dependencies
    FreeFunctions ..> round_robin_policy : calls methods
    FreeFunctions ..> InternalFallbacks : delegates
    InternalFallbacks ..> FreeFunctions : calls
    policy_base ..> InternalFallbacks : delegates submit/submit_and_wait
    policy_base ..> default_backend : submit(selection, f, args)
    round_robin_policy ..> default_backend : default template param



    %% Notes
    note for FreeFunctions "Entry points for users in the form of free functions for submission"
    note for round_robin_policy "Entry points for users in the form of member functions for submission"
    note for core_resource_backend "Customize by partially specializing for specific CoreResourceType inheriting from backend_base"
    note for round_robin_policy "Customize at this level with minimal effort inheriting from policy_base"

```

