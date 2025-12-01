Backends
########

The dynamic selection API is an experimental feature in the |onedpl_long|  
(|onedpl_short|) that selects an *execution resource* based on a chosen  
*selection policy*. For the policies to work with different resource types.  
the resource management and work submission mechanics are handled  
separately by resource-specific *backends*.

A backend is responsible for:  

1. **Resource Management** - storing and providing access to a set of resources  
2. **Work Submission** - executing user functions on selected resources  
3. **Synchronization** - waiting for submitted work to complete  
4. **Instrumentation** - optionally reporting execution information (timing, completion)  

Backends are typically not directly visible to application developers - they work  
behind the scenes to enable policies to function.

SYCL Backend
------------

A sycl backend is provided to manage ``sycl::queue`` core resources, and ``sycl::queue``
is the default resource if none is provided or deduce-able from resource arguments.
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

The SYCL backend supports the reporting for :ref:`Execution Information <execution-information>`
of ``task_submission``, ``task_completion``, and ``task_time``.

Backend Architecture
--------------------

The backend system uses a layered design to support multiple resource types:

**default_backend**: When you create a policy, ``default_backend`` automatically
determines the core resource type by applying any
:ref:`resource adapter <resource-adapters>` to your resource type. It then 
delegates to the appropriate ``core_resource_backend`` specialization.

**core_resource_backend**: Specialized backend implementations exist for specific core
resource types (like ``sycl::queue``). A generic implementation is provided
for core resources without an explicitly specialized implementation which provides
a minimal amount of functionality:

- Basic resource storage and retrieval
- Simple work submission without instrumentation (No execution information reporting)

To use a custom resource type with full instrumentation support, you must create a
``core_resource_backend`` specialization.

Lazy Reporting
--------------

.. _lazy_report:

For asynchronous execution, backends may use *lazy reporting* where
:ref:`Execution Information <execution-information>` is not immediately available.
 The SYCL backend uses lazy reporting for ``task_completion`` and ``task_time``.

Policies that use execution information always call the backend's
``lazy_report()`` function before making selections, ensuring they have
up-to-date information about completed tasks.

This is transparent to application developers - the policy handles the details.

Backend Traits
--------------

Backend traits provide compile-time information about backend capabilities:

.. code:: cpp

  namespace oneapi::dpl::experimental
  {
    template<typename Backend>
    struct backend_traits {

      // True if backend has explicit wait_type
      static constexpr bool has_wait_type_v = /* ... */;

      // If has_wait_type_v is True, specific type required from user functions.
      // If has_wait_type_v is False, void (user functions must return waitable-type)
      using wait_type = /* ... */;

      // True if backend requires lazy_report() to be called
      static constexpr bool lazy_report_v = /* ... */;

      // Scratch space type for selection handles
      template<typename... ReportReqs>
      using selection_scratch_t = /* ... */;
    };
  }

These traits are primarily used by policy implementers, not application developers.

Wait Type Requirements
^^^^^^^^^^^^^^^^^^^^^^

Backends specify return type requirements for user-submitted functions through the
``wait_type`` member:

**Explicit wait_type**: If a backend defines a ``wait_type`` alias (e.g., ``using wait_type = sycl::event;``),
user functions **must** return that specific type. This is typically required when the backend
needs to instrument or track asynchronous operations.

**No explicit wait_type**: If a backend does not define a ``wait_type`` alias, user functions
may return any *waitable-type*. A waitable-type is any type with a ``wait()`` member function
that can be called to synchronize with the operation's completion.

The SYCL backend defines ``wait_type = sycl::event``, requiring user functions to return
``sycl::event`` for proper instrumentation and synchronization.

Selection Scratch Space
^^^^^^^^^^^^^^^^^^^^^^^

.. _selection_scratch_space:

Backends need storage space within selection handles to implement instrumentation.
The ``selection_scratch_t`` trait specifies what additional data a backend requires
based on the policy's reporting needs.

When a policy tracks execution information (like task timing or completion), the backend
needs to store temporary data with each selection. For example, the SYCL backend stores
an extra ``sycl::event`` for start-time profiling tags when ``task_time`` reporting is requested.

The backend populates and uses this scratch space during work submission and reporting.
For policies without reporting requirements, ``selection_scratch_t<>`` is empty,
adding no overhead.

Custom Backends
---------------

For advanced use cases, you can create custom backends to support new resource
types or provide specialized instrumentation. Custom backends are created by
specializing ``core_resource_backend`` for your resource type. For an example of
how to do this, look at ``core_resource_backend<sycl::queue, ...>``.

See Also
--------

- :doc:`policies` - Overview of selection policies
- :doc:`functions` - Free functions for working with backends and policies
