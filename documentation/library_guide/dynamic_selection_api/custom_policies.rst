Custom Policies
###############

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. While several policies are provided out of the box,
you can create custom policies to implement application-specific selection
strategies.

Using policy_base
-----------------

The recommended approach for creating custom policies is to inherit from ``policy_base``,
which provides default implementations of submission and initialization logic.

``policy_base`` uses the Curiously Recurring Template Pattern (CRTP), where the derived
policy class passes itself as the first template parameter. This allows ``policy_base``
to call derived class methods (like ``try_select``) without virtual function overhead.

.. code:: cpp

  namespace oneapi::dpl::experimental {

    template<typename Policy, typename ResourceAdapter, typename Backend,
             typename... ReportReqs>
    class policy_base {
    public:
      using resource_type = /* backend resource type */;

      // Initialization
      void initialize();
      void initialize(const std::vector<resource_type>& u);
      void initialize(const std::vector<resource_type>& u,
                      ResourceAdapter adapter, Args... args);

      // Submission operations
      auto submit(Function&& f, Args&&... args);           // Retries until success
      void submit_and_wait(Function&& f, Args&&... args);  // Blocks until complete
      auto try_submit(Function&& f, Args&&... args);       // Returns std::optional

      // Queries
      auto get_resources() const;
      auto get_submission_group();

    protected:
      std::shared_ptr<Backend> backend_;
    };

  }

When using ``policy_base``, your custom policy must implement:

- ``try_select(Args...)`` - Returns ``std::optional<selection_type>``, empty if no resource available
- ``initialize_state(Args...)`` - Performs policy-specific initialization

The ``policy_base`` automatically provides the required ``backend_type`` and ``resource_type`` aliases.


Initialization
--------------

The ``initialize_state()`` function is called after the backend is initialized.
Use it to set up policy-specific state using resources from ``get_resources()``.

Selection Logic
---------------

The ``try_select()`` function implements your selection algorithm:

- Returns ``std::optional<selection_type>`` with selected resource
- Returns ``std::nullopt`` if no resource is currently available
- May accept additional arguments for selection hints

Selection Type
--------------

.. _selection-type:

The ``selection_type`` represents a selected resource and encapsulates the policy and
resource information. It must satisfy the *Selection* requirements:

- ``unwrap()`` - Returns the resource object (e.g., ``sycl::queue``) the selection represents
- ``get_policy()`` - Returns the policy that created the selection
- *optional* ``report(i)`` and ``report(i, v)`` - Report execution information back to the policy if required

For policies that do not require execution information reporting (such as simple
``round_robin_policy``), you may use the provided ``basic_selection_handle_t``:

.. code:: cpp

  template<typename Policy, typename Resource>
  class basic_selection_handle_t {
  public:
    explicit basic_selection_handle_t(const Policy& p, Resource e);
    Resource unwrap();
    Policy get_policy();
  };

For policies that need execution information (like ``dynamic_load_policy`` which tracks
task submissions and completions, or ``auto_tune_policy`` which measures task timing),
define a custom selection type with ``report()`` methods, as is shown in the following
example:

.. code:: cpp

  template<typename Policy, typename Backend>
  class custom_selection_handle_t {
    Policy policy_;
    resource_type resource_;

    using scratch_space_t =
        typename backend_traits<Backend>::template selection_scratch_t<
            execution_info::task_submission_t, execution_info::task_completion_t>;
    scratch_space_t scratch_space;

  public:
    custom_selection_handle_t(const Policy& p, resource_type r)
      : policy_(p), resource_(std::move(r)) {}

    auto unwrap() { return oneapi::dpl::experimental::unwrap(resource_); }
    Policy get_policy() { return policy_; }

    // Report execution events
    void report(const execution_info::task_submission_t&) const {
      // Handle task submission event
    }
    void report(const execution_info::task_completion_t&) const {
      // Handle task completion event
    }
  };

The backend will call the selection handle's ``report()`` methods when execution
events occur, allowing the policy to update its state accordingly.

As shown above, for policies that need execution information, the selection
handle must also include a member named ``scratch_space`` with type dictated by
``backend_traits<Backend>::template selection_scratch_t<Reqs...>`` where ``Reqs...``
is a variadic pack of all execution information requirements.  The backend will
use this ``scratch_space`` member to store temporary instrumentation data (like
profiling events) needed to satisfy the reporting requirement. For more
information, see :ref:`Selection Scratch Space <selection-scratch-space>`.

Reporting Requirements
----------------------

If your policy needs execution information (like task completion times), specify
reporting requirements as template parameters to ``policy_base``:

.. code:: cpp

  class timing_aware_policy
    : public ex::policy_base<timing_aware_policy,
                             oneapi::dpl::identity,
                             ex::default_backend<sycl::queue>,
                             ex::execution_info::task_time_t> {
    // Policy implementation that receives timing information
  };

Execution Information
---------------------

.. _execution-information:

Backends can provide execution information to policies for making informed selection
decisions. The ``oneapi::dpl::experimental::execution_info`` namespace contains
tag types and tag objects that describe the instrumentation information policies
require for their selection logic. Policies specify their requirements using these
tags during backend construction. Backends then call ``report`` with these tags
to provide the requested execution information to the policy via
:ref:`selection objects <selection-type>`.

The following execution information types are available:

.. list-table:: Execution Information Types
  :widths: 18 17 30 35
  :header-rows: 1

  * - Tag Type
    - Tag Object
    - Value Type
    - Description
  * - ``task_submission_t``
    - ``task_submission``
    - void
    - Signals when a task is submitted
  * - ``task_completion_t``
    - ``task_completion``
    - void
    - Signals when a task completes
  * - ``task_time_t``
    - ``task_time``
    - ``std::chrono::milliseconds``
    - Elapsed time from submission to completion


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
    - ``task_time``

Policies with no reporting requirements can work with any backend, including
the provided generic backend implementation which is used when no specialization
of ``core_resource_backend`` exists for the specific resource. Policies with
reporting requirements need a backend that supports those specific types of
execution information.

Policies with reporting requirements must call ``lazy_report()`` prior to selection,
if the backend supports it. :ref:`Lazy Reporting <lazy-report>` allows backends 
to update their execution information state before making selection decisions.
See ``dynamic_load_policy`` and ``auto_tune_policy`` for examples of this.

Policy State Reference Semantics
--------------------------------

Best practice is to make your policy's selection state stored in a ``shared_ptr`` to enable
common reference semantics - copies of your policy will share the same state, as set up by
``initialize_state`` calls.

.. code:: cpp

  struct selector_t {
    // Policy-specific selection state
  };
  std::shared_ptr<selector_t> selector_;

Examples
--------

For examples please look to the existing policies within oneDPL (``fixed_resource_policy``,
``round_robin_policy``, ``dynamic_load_policy``, ``auto_tune_policy``), which are
all written using ``policy_base`` and according to these best practices.


See Also
--------

- :doc:`backends` - Overview of the backend concept
- :doc:`policies` - Overview of the policy concept
- :doc:`functions` - Free functions for working with policies
