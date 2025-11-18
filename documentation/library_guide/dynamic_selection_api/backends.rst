Backends
########

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. A *backend* is responsible for managing a set of resources
and handling the submission of work to those resources.

Backends are typically not directly visible to application developers - they work
behind the scenes to enable policies to function.

Role of Backends
----------------

A backend serves several key purposes:

1. **Resource Management** - Stores and provides access to a set of resources
2. **Work Submission** - Handles the execution of user functions on selected resources
3. **Instrumentation** - Optionally reports execution information (timing, completion) to policies
4. **Synchronization** - Provides mechanisms to wait for submitted work to complete

The dynamic selection API separates the *selection logic* (in policies) from the
*resource management and submission mechanics* (in backends), allowing policies
to work with different resource types.

Default SYCL Backend
--------------------

By default, policies use the SYCL backend which manages ``sycl::queue`` resources.
The SYCL backend provides:

- Default initialization from available SYCL devices
- Event-based submission and waiting
- Optional profiling and timing instrumentation

When you construct a policy without specifying resources, it uses the SYCL
backend's default initialization:

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>

  namespace ex = oneapi::dpl::experimental;

  // Uses SYCL backend with default device initialization
  ex::round_robin_policy<> policy;

  // Explicitly specifying SYCL queues
  std::vector<sycl::queue> my_queues = { /* ... */ };
  ex::round_robin_policy<sycl::queue> policy2{my_queues};

Backend Template Parameters
---------------------------

Policies accept a backend template parameter to control the resource type:

.. code:: cpp

  template<typename ResourceType = sycl::queue,
           typename ResourceAdapter = oneapi::dpl::identity,
           typename Backend = default_backend<ResourceType, ResourceAdapter>>
  class round_robin_policy { /* ... */ };

- **ResourceType** - The type of resources managed (default: ``sycl::queue``)
- **ResourceAdapter** - Adapter to transform resources (default: ``oneapi::dpl::identity``)
- **Backend** - The backend implementation (default: ``default_backend``)

Most users only need to specify ``ResourceType`` if using non-SYCL resources.

Resource Adapters
-----------------

Resource adapters allow a backend designed for one resource type to work with
related types. For example, using pointers instead of values:

.. code:: cpp

  // Adapter converts sycl::queue* to sycl::queue&
  auto adapter = [](sycl::queue* qp) -> sycl::queue& { return *qp; };

  std::vector<sycl::queue*> queue_ptrs = get_queue_pointers();

  // Policy uses SYCL backend via the adapter
  ex::round_robin_policy<sycl::queue*, decltype(adapter)> p{queue_ptrs, adapter};

The adapter is applied internally when the backend needs to access backend-specific
features, but user functions still receive the original resource type (``sycl::queue*``
in this example).

Execution Information
---------------------

Backends can provide execution information to policies that need it for making
informed selection decisions. The SYCL backend supports:

.. list-table:: Execution Information Types
  :widths: 30 30 40
  :header-rows: 1

  * - Information Type
    - Value Type
    - Description
  * - ``task_submission``
    - void
    - Signals when a task is submitted
  * - ``task_completion``
    - void
    - Signals when a task completes
  * - ``task_time``
    - ``std::chrono::milliseconds``
    - Elapsed time from submission to completion

Policies that require execution information must specify reporting requirements.
The backend will only provide resources capable of satisfying those requirements.

Built-In Policy Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table shows the reporting requirements for each built-in policy:

.. list-table:: Policy Reporting Requirements
  :widths: 40 60
  :header-rows: 1

  * - Policy
    - Reporting Requirements
  * - ``fixed_resource_policy``
    - None
  * - ``round_robin_policy``
    - None
  * - ``dynamic_load_policy``
    - ``task_submission``, ``task_completion``
  * - ``auto_tune_policy``
    - ``task_submission``, ``task_completion``, ``task_time``

Policies with no reporting requirements can work with any backend, including
the minimal ``default_backend``. Policies with reporting requirements need
a backend that supports those specific types of execution information.

For example, the ``auto_tune_policy`` requires timing information:

.. code:: cpp

  // This policy requires task_time_t reporting
  ex::auto_tune_policy<sycl::queue> p{queues};

If no devices support profiling, the backend will throw an error during construction.

Lazy Reporting
--------------

For asynchronous execution, backends may use *lazy reporting* where execution
information is not immediately available. The SYCL backend uses lazy reporting
for ``task_completion`` and ``task_time``.

Policies that use execution information automatically call the backend's
``lazy_report()`` function before making selections, ensuring they have
up-to-date information about completed tasks.

This is transparent to application developers - the policy handles the details.

Backend Traits
--------------

Backend traits provide compile-time information about backend capabilities:

.. code:: cpp

  namespace oneapi::dpl::experimental::backend_traits {

    // True if backend requires lazy_report() to be called
    template<typename Backend>
    inline constexpr bool lazy_report_v = /* ... */;

    // Scratch space type for selection handles
    template<typename Backend, typename... ReportReqs>
    using selection_scratch_t = /* ... */;

  }

These traits are primarily used by policy implementers, not application developers.

Common Backend Patterns
-----------------------

Checking Available Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All backends provide ``get_resources()`` to query available resources:

.. code:: cpp

  ex::round_robin_policy<> p;
  auto resources = ex::get_resources(p);

  std::cout << "Number of available queues: " << resources.size() << "\n";
  for (auto& q : resources) {
    auto dev = q.get_device();
    std::cout << "  - " << dev.get_info<sycl::info::device::name>() << "\n";
  }

Waiting for All Submissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Backends provide a submission group for bulk synchronization:

.. code:: cpp

  ex::round_robin_policy<> p;

  // Submit multiple tasks
  for (int i = 0; i < 10; ++i) {
    ex::submit(p, [](sycl::queue q) {
      return q.submit([](sycl::handler& h) { /* kernel */ });
    });
  }

  // Wait for all submissions to complete
  ex::wait(ex::get_submission_group(p));

Deferred Initialization
^^^^^^^^^^^^^^^^^^^^^^^

Backends support deferred initialization, which defers resource allocation
until explicitly initialized:

.. code:: cpp

  // Construct without initializing backend
  ex::round_robin_policy<> p{ex::deferred_initialization};

  // ... later, initialize with specific resources
  p.initialize(my_queues);

This is useful when resource availability is not known at construction time.

Custom Backends
---------------

For advanced use cases, you can create custom backends to support new resource
types or provide specialized instrumentation. Custom backends are created by
specializing ``default_backend_impl`` for your resource type.

See :doc:`custom_backends` for detailed information on creating custom backends.

See Also
--------

- :doc:`custom_backends` - Creating custom backends for new resource types
- :doc:`policies` - Overview of selection policies
- :doc:`functions` - Free functions for working with backends and policies
