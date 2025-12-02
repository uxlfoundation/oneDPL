Policies
########

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. There are several policies provided as part
of the API. Policies encapsulate the logic and any associated state needed
to make a selection.

Policy Basics
-------------

A policy manages a collection of execution resources and applies a selection
strategy to choose among them. Internally, policies use a *backend* to handle
resource-specific operations such as work submission and synchronization.

All policies support three construction patterns:

**Initialization with explicit resources**: Construct the policy with an explicit
vector of resources to manage:

.. code:: cpp

  namespace ex = oneapi::dpl::experimental;
  std::vector<resource_type> r {/* resources */};
  ex::round_robin_policy policy{r};

**Default initialization**: Construct the policy without arguments, allowing the
backend to select default resources:

.. code:: cpp

  namespace ex = oneapi::dpl::experimental;
  ex::round_robin_policy policy{}; // uses backend's default resources

**Deferred initialization**: Construct the policy in an uninitialized state, then
initialize it later by calling ``initialize()``:

.. code:: cpp

  namespace ex = oneapi::dpl::experimental;
  ex::round_robin_policy policy{ex::deferred_initialization};
  // ... later, when resources are available ...
  policy.initialize(resources);

Deferred initialization is useful when the policy must be constructed before
execution resources are available, such as during early program setup or when
resources depend on runtime configuration.

Attempting to use a deferred-initialization policy before calling ``initialize()``
will throw ``std::logic_error``.

Once policies are initialized, work can be submitted via the submit
:doc:`free functions <functions>`: ``submit``, ``submit_and_wait``, or ``try_submit``,
which will select the appropriate execution resource for the work.

Policy Traits
-------------

.. _policy-traits:

Traits can be used to determine useful type information about policies. 

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Policy>
    struct policy_traits {
      // backend associated with this policy
      using backend_type = /*...*/;

      // resource type associated with this policy
      using resource_type = /* ... */;

      // True if explicit wait_type is required by associated backend, False otherwise
      static constexpr bool has_wait_type_v = /* ... */;

      // If has_wait_type_v is True, type required to be returned by user submitted functions.
      // If has_wait_type_v is False, void
      using wait_type = typename std::decay_t<Policy>::wait_type;
    };

    template <typename Policy>
    using backend_t = typename policy_traits<Policy>::backend_type;

    template <typename Policy>
    using resource_t = typename policy_traits<Policy>::resource_type;

    template <typename Policy>
    inline constexpr bool has_wait_type_v = typename policy_traits<Policy>::has_wait_type_v;

    template <typename Policy>
    using wait_t = typename policy_traits<Policy>::wait_type;
  }

When using the default SYCL backend, ``resource_t<Policy>`` is ``sycl::queue`` and ``wait_t<Policy>`` is
``sycl::event``.  

If ``has_wait_type_v<Policy>`` is ``true``, the user functions passed to submission functions are expected to have a
signature of:

.. code:: cpp

  wait_t<Policy> user_function(resource_t<Policy>, ...);

If ``has_wait_type_v<Policy>`` is ``false``, the user functions passed to submission functions are expected to have a
signature of:

.. code:: cpp

  T user_function(resource_t<Policy>, ...);

Where ``T`` is a *waitable-type*. A *waitable-type* is a type which has a member method ``wait()`` that can be called to
wait for completion of the submitted work.


Common Reference Semantics
--------------------------

If a policy maintains state, the state is maintained separately for each 
independent policy instance. So for example, two independently constructed 
instances of a ``round_robin_policy`` will operate independently of each other. 
However, policies provide *common reference semantics*, so copies of a
policy instance share state.

An example, demonstrating this difference, is shown below:

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>
  #include <string>
  
  namespace ex = oneapi::dpl::experimental;
  
  struct print_type{
    sycl::event operator()(sycl::queue q, const std::string &str) {
      std::cout << str << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
      return sycl::event{};
    }
  };
  
  int main() {
    ex::round_robin_policy p1{ { sycl::queue{ sycl::cpu_selector_v },  
                                 sycl::queue{ sycl::gpu_selector_v } } };
    ex::round_robin_policy p2{ { sycl::queue{ sycl::cpu_selector_v },  
                                 sycl::queue{ sycl::gpu_selector_v } } };
    ex::round_robin_policy p3 = p2; 
  
    print_type prnt{};

    std::cout << "independent instances operate independently\n";
    ex::submit_and_wait(p1, prnt, "p1 selection 1: ");
    ex::submit_and_wait(p2, prnt, "p2 selection 1: ");
    ex::submit_and_wait(p2, prnt, "p2 selection 2: ");
    ex::submit_and_wait(p1, prnt, "p1 selection 2: ");
  
    std::cout << "\ncopies provide common reference semantics\n";
    ex::submit_and_wait(p3, prnt, "p3 (copy of p2) selection 1: ");
    ex::submit_and_wait(p2, prnt, "p2 selection 3: ");
    ex::submit_and_wait(p3, prnt, "p3 (copy of p2) selection 2: ");
    ex::submit_and_wait(p3, prnt, "p3 (copy of p2) selection 3: ");
    ex::submit_and_wait(p2, prnt, "p2 selection 4: ");
  }

The output of this example is::

  p1 selection 1: cpu
  p2 selection 1: cpu
  p2 selection 2: gpu
  p1 selection 2: gpu
  
  copies provide common reference semantics
  p3 (copy of p2) selection 1: cpu
  p2 selection 3: gpu
  p3 (copy of p2) selection 2: cpu
  p3 (copy of p2) selection 3: gpu
  p2 selection 4: cpu


Available Policies
------------------

More detailed information about the built-in policies is provided in the following sections:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   fixed_resource_policy
   round_robin_policy
   dynamic_load_policy
   auto_tune_policy

Customization
-------------

Resource Adapters
^^^^^^^^^^^^^^^^^

.. _resource-adapters:


A *resource adapter* is a transformation function applied to convert a custom resource type
to the type the policy backend operates with.

Resource adapters let you use variations of a resource type when you need a different resource
storage format (like pointers or wrappers) or when additional information should be associated
with each resource.

All oneDPL selection policy types allow setting a resource adapter as the second template argument
when creating a policy. If not provided via function argument or specified as a template argument,
it defaults to ``oneapi::dpl::identity``.

The example code shows how you can use ``sycl::queue*`` with the SYCL backend
instead of ``sycl::queue`` by providing an adapter that converts between them.

.. code:: cpp

  // Adapter converts pointer to reference for backend use
  auto adapter = [](sycl::queue* qp) -> sycl::queue& { return *qp; };

  std::vector<sycl::queue*> queue_ptrs = get_queue_pointers();

  // Policy works with pointers, backend uses references internally
  ex::round_robin_policy<sycl::queue*, decltype(adapter)> p{queue_ptrs, adapter};

  ex::submit(p, [](sycl::queue* qp) {  // User function receives pointer
    return qp->submit(/* ... */);
  });

The backend applies the adapter internally, but your user functions always receive the 
original resource type (``sycl::queue*``).

Custom Policies
^^^^^^^^^^^^^^^

The dynamic selection API supports creating custom policies to extend the system
with new selection strategies or resource types. For details, see:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   custom_policies

See Also
--------

- :doc:`backends` - Overview of backends for managing resources and handling work submission
- :doc:`functions` - Free functions for working with backends and policies