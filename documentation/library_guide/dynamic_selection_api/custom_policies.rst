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

The ``policy_base`` automatically provides the required ``backend_type`` and ``resource_type`` aliases.

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

    struct selector_t {
      std::vector<resource_type> resources_;
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
      return std::nullopt;
    }

  public:
    using backend_type = Backend;
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
- Returns ``std::nullopt`` if no resource is currently available
- May accept additional arguments for selection hints

Selection Type
^^^^^^^^^^^^^^

The ``selection_type`` represents a selected resource and encapsulates the policy and
resource information. It must satisfy the *Selection* requirements:

- ``unwrap()`` - Returns the unwrapped resource (e.g., ``sycl::queue``)
- ``get_policy()`` - Returns the policy that created the selection
- *optional* ``report(i)`` and ``report(i, v)`` - Report execution information back to the policy if required

For policies that do not require execution information reporting (such as simple
``round_robin_policy``), you may use the provided ``basic_selection_handle_t``:

.. code:: cpp

  template<typename Policy, typename Resource>
  class basic_selection_handle_t {
  public:
    explicit basic_selection_handle_t(const Policy& p, Resource e);
    auto unwrap();
    Policy get_policy();
  };

For policies that need execution information (like ``dynamic_load_policy`` which tracks
task submissions and completions, or ``auto_tune_policy`` which measures task timing),
define a custom selection type with ``report()`` methods:

.. code:: cpp

  template<typename Policy>
  class custom_selection_handle_t {
    Policy policy_;
    resource_type resource_;

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


See Also
--------

- :doc:`backends` - Overview of the backend concept
- :doc:`policies` - Overview of the policy concept
- :doc:`functions` - Free functions for working with policies
