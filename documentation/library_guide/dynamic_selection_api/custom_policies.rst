Custom Policies
###############

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. While several policies are provided out of the box,
you can create custom policies to implement application-specific selection
strategies.

Creating a custom policy can be done in two ways:

1. **Using policy_base** - Inherit from ``policy_base`` to minimize boilerplate code
2. **Implementing from scratch** - Implement the policy contract directly for full control

Using policy_base
-----------------

The recommended approach for creating custom policies is to inherit from ``policy_base``,
which provides default implementations of submission and initialization logic.

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

      // Submission operations (provided by base)
      auto try_submit(Function&& f, Args&&... args);  // Returns std::optional
      auto submit(Function&& f, Args&&... args);      // Retries until success
      void submit_and_wait(Function&& f, Args&&... args);  // Blocks until complete

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

Example: Custom Random Policy
------------------------------

The following example demonstrates creating a custom policy that randomly
selects from available resources:

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <random>

  namespace ex = oneapi::dpl::experimental;

  template<typename ResourceType = sycl::queue,
           typename ResourceAdapter = oneapi::dpl::identity,
           typename Backend = ex::default_backend<ResourceType, ResourceAdapter>>
  class random_policy
    : public ex::policy_base<random_policy<ResourceType, ResourceAdapter, Backend>,
                             ResourceAdapter, Backend> {
  protected:
    using base_t = ex::policy_base<random_policy, ResourceAdapter, Backend>;
    friend base_t;

    using typename base_t::selection_type;
    using typename base_t::resource_container_t;

    struct selector_t {
      resource_container_t resources_;
      std::mt19937 gen_{std::random_device{}()};
    };

    std::shared_ptr<selector_t> selector_;

    // Required: Initialize policy-specific state
    void initialize_state() {
      if (!selector_) {
        selector_ = std::make_shared<selector_t>();
      }
      selector_->resources_ = base_t::get_resources();
    }

    // Required: Select a resource (returns empty std::optional if none available)
    template<typename... Args>
    std::optional<selection_type> try_select(Args&&...) {
      if (selector_ && !selector_->resources_.empty()) {
        std::uniform_int_distribution<> dist(0, selector_->resources_.size() - 1);
        auto idx = dist(selector_->gen_);
        return std::make_optional<selection_type>(*this, selector_->resources_[idx]);
      }
      return nullptr;
    }

  public:
    using resource_type = typename base_t::resource_type;

    // Constructors
    random_policy() { base_t::initialize(); }
    random_policy(ex::deferred_initialization_t) {}
    random_policy(const std::vector<ResourceType>& u, ResourceAdapter adapter = {}) {
      base_t::initialize(u, adapter);
    }
  };

This custom policy can be used like any built-in policy:

.. code:: cpp

  random_policy<sycl::queue> p{my_queues};

  ex::submit(p, [](sycl::queue q) {
    return q.submit([](sycl::handler& h) {
      // kernel code
    });
  });

Key Points for Custom Policies
-------------------------------

Selection State
^^^^^^^^^^^^^^^

Best practice is to make your policy's selection state stored in a ``shared_ptr`` to enable
common reference semantics - copies of your policy will share the same state.

.. code:: cpp

  struct selector_t {
    // Policy-specific selection state
  };
  std::shared_ptr<selector_t> selector_;

Initialization
^^^^^^^^^^^^^^

The ``initialize_state()`` function is called after the backend is initialized.
Use it to set up policy-specific state using resources from ``get_resources()``.

Selection Logic
^^^^^^^^^^^^^^^

The ``try_select()`` function implements your selection algorithm:

- Returns ``std::optional<selection_type>`` with selected resource
- Returns ``nullptr`` if no resource is currently available
- May accept additional arguments for selection hints

Reporting Requirements
^^^^^^^^^^^^^^^^^^^^^^

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

See the :ref:`Execution Information <execution-information>` section of the
backends page for more information about the specific reporting requirements
available, including ``task_time``.

Implementing from Scratch
--------------------------

For full control, you can implement the policy contract directly without
inheriting from ``policy_base``. Your policy must provide:

Required Members
^^^^^^^^^^^^^^^^

.. list-table:: Required Policy Members
  :widths: 50 50
  :header-rows: 1

  * - Member
    - Description
  * - ``resource_type``
    - Type of resources (e.g., ``sycl::queue``)
  * - ``get_resources()``
    - Returns ``std::vector<resource_type>``
  * - ``try_select(Args...)``
    - Returns ``std::optional<selection_type>``, empty if no resource available

Optional Members
^^^^^^^^^^^^^^^^

.. list-table:: Optional Policy Members
  :widths: 50 50
  :header-rows: 1

  * - Member
    - Description
  * - ``try_submit(f, args...)``
    - Returns ``std::optional<submission>``, empty if no resource available
  * - ``submit(f, args...)``
    - Returns submission, retrying with backoff if needed
  * - ``submit_and_wait(f, args...)``
    - Submits and blocks until complete
  * - ``get_submission_group()``
    - Returns object with ``wait()`` for all submissions

If optional members are not provided, free function fallbacks will be used
based on ``try_select()``.

Best Practices
--------------

1. Inherit from ``policy_base`` unless you have specific reasons not to
2. Use ``shared_ptr`` for state to enable common reference semantics
3. Make selection fast - avoid expensive operations in ``try_select()``
4. Handle empty resource sets - return ``nullptr`` when no resources available
5. Call backend ``lazy_report()`` if your policy uses execution information

See Also
--------

- :doc:`custom_backends` - Creating custom backends for new resource types
- :doc:`policies` - Overview of the policy concept
- :doc:`functions` - Free functions for working with policies
