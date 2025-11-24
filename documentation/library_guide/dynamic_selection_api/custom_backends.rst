Custom Backends
###############

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. While a SYCL backend is provided by default, you can
create custom backends to support different resource types or provide
specialized instrumentation.

A backend manages resources and handles submission of work to those resources.
Backends can optionally provide instrumentation for execution information
like task timing and completion events.

Backend Architecture
--------------------

Custom backends are created by specializing ``core_resource_backend`` for
your resource type:

.. code:: cpp

  namespace oneapi::dpl::experimental {

    // Base class providing minimal default implementations
    template<typename ResourceType>
    class backend_base { /* ... */ };

    // Specialize this for your resource type
    template<typename CoreResourceType, typename ResourceType,
             typename ResourceAdapter>
    class core_resource_backend : public backend_base<ResourceType> {
      // Override submit() and provide scratch_t for reporting
    };

    // Convenience class that determines CoreResourceType
    template<typename ResourceType,
             typename ResourceAdapter = oneapi::dpl::identity>
    class default_backend : public core_resource_backend<...> { /* ... */ };

  }

The ``backend_base`` provides minimal default implementations but does **not**
support reporting requirements. To add reporting support, specialize
``core_resource_backend`` for your specific base resource type.

Backend with Reporting Support
-------------------------------

To support execution information reporting (like timing), your backend
specialization must:

1. Accept reporting requirements in constructors
2. Filter resources based on capabilities
3. Provide ``scratch_t`` template with appropriate storage
4. Override ``submit()`` to instrument execution

Resource Adapters
-----------------

Resource adapters allow reusing backend specializations for related types.
For example, to use the SYCL backend with ``sycl::queue*`` instead of
``sycl::queue``:

.. code:: cpp

  // Adapter converts pointer to reference
  auto adapter = [](sycl::queue* qp) -> sycl::queue& { return *qp; };

  std::vector<sycl::queue*> queue_ptrs = get_queue_pointers();

  // Uses SYCL backend specialization via adapter
  ex::round_robin_policy<sycl::queue*, decltype(adapter)> p{queue_ptrs, adapter};

The adapter is applied internally by the backend when accessing the resource,
but the user's function still receives the original ``sycl::queue*``.

Backend Contract
----------------

Required Members
^^^^^^^^^^^^^^^^

Your backend specialization must provide:

.. list-table:: Required Backend Members
  :widths: 50 50
  :header-rows: 1

  * - Member
    - Description
  * - ``resource_type``
    - The type of resources managed
  * - ``scratch_t<ReportReqs...>``
    - Template for per-selection scratch storage
  * - Constructors with ``ReportReqs...``
    - Accept reporting requirements as trailing variadic pack
  * - ``submit(selection, f, args...)``
    - Execute user function and return submission object
  * - ``get_resources()``
    - Return ``std::vector<resource_type>``
  * - ``get_submission_group()``
    - Return object with ``wait()`` for all submissions

Optional Members
^^^^^^^^^^^^^^^^

.. list-table:: Optional Backend Members
  :widths: 50 50
  :header-rows: 1

  * - Member
    - Description
  * - ``lazy_report()``
    - Check for completed tasks and report execution info
  * - ``lazy_report_v<Backend>``
    - Trait indicating if ``lazy_report()`` is needed

Reporting Requirements
----------------------

Backends can support various execution information reporting requirements.
When a policy requests reporting, the backend must confirm that it can
satisfy those requirements, and filter out resources that cannot satisfy them.

See the :ref:`Execution Information <execution-information>` section of the backends
page for more information about the specific reporting requirements available.


In your constructor, validate and filter:

.. code:: cpp

  template<typename... ReportReqs>
  my_backend(const std::vector<ResourceType>& u, ReportReqs... reqs) {
    // 1. Validate: static_assert for unsupported requirements

    // 2. Filter: remove resources that can't satisfy requirements
    for (auto& r : u) {
      if (can_satisfy_requirements(r)) {
        resources_.push_back(r);
      }
    }

    // 3. Error if none left
    if (resources_.empty()) {
      throw std::runtime_error("No resources support requested reporting");
    }
  }

Lazy Reporting
--------------

For asynchronous execution, backends may defer reporting until the policy
requests it. To enable lazy reporting:

1. Store submission handles/events internally
2. Implement ``lazy_report()`` to check and report completed tasks
3. The trait ``backend_traits::lazy_report_v<Backend>`` will be true

.. code:: cpp

  class my_backend {
    std::vector<async_handle> pending_tasks_;

  public:
    void lazy_report() {
      // Check pending tasks for completion
      for (auto it = pending_tasks_.begin(); it != pending_tasks_.end();) {
        if (it->is_complete()) {
          it->report_execution_info();
          it = pending_tasks_.erase(it);
        } else {
          ++it;
        }
      }
    }
  };

Policies using backends with lazy reporting will automatically call
``lazy_report()`` before selection.

See Also
--------

- :doc:`custom_policies` - Creating custom selection policies
- :doc:`policies` - Overview of the policy concept
- :doc:`functions` - Free functions for working with backends and policies
