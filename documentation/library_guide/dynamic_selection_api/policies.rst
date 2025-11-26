Policies
########

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several policies provided as part 
of the API. Policies encapsulate the logic and any associated state needed 
to make a selection. 

Policy Traits
-------------

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
  
  template<typename Selection>
  void print_type(const std::string &str, Selection s) {
    auto q = ex::unwrap(s);
    std::cout << str << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
  }
  
  int main() {
    ex::round_robin_policy p1{ { sycl::queue{ sycl::cpu_selector_v },  
                                 sycl::queue{ sycl::gpu_selector_v } } };
    ex::round_robin_policy p2{ { sycl::queue{ sycl::cpu_selector_v },  
                                 sycl::queue{ sycl::gpu_selector_v } } };
    ex::round_robin_policy p3 = p2; 
  
    std::cout << "independent instances operate independently\n";
    auto p1s1 = ex::select(p1);  
    print_type("p1 selection 1: ", p1s1);
    auto p2s1 = ex::select(p2);  
    print_type("p2 selection 1: ", p2s1);
    auto p2s2 = ex::select(p2);  
    print_type("p2 selection 2: ", p2s2);
    auto p1s2 = ex::select(p1);  
    print_type("p1 selection 2: ", p1s2);
  
    std::cout << "\ncopies provide common reference semantics\n";
    auto p3s1 = ex::select(p3);  
    print_type("p3 (copy of p2) selection 1: ", p3s1);
    auto p2s3 = ex::select(p2);  
    print_type("p2 selection 3: ", p2s3);
    auto p3s2 = ex::select(p3);  
    print_type("p3 (copy of p2) selection 2: ", p3s2);
    auto p3s3 = ex::select(p3);  
    print_type("p3 (copy of p2) selection 3: ", p3s3);
    auto p2s4 = ex::select(p2);  
    print_type("p2 selection 4: ", p2s4);
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

The dynamic selection API supports creating custom policies to extend the system
with new selection strategies or resource types:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   custom_policies

See Also
--------

- :doc:`backends` - Overview of backends for managing resources and handling work submission
- :doc:`functions` - Free functions for working with backends and policies