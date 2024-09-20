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
      using selection_type = typename std::decay_t<Policy>::selection_type;  
      using resource_type = typename std::decay_t<Policy>::resource_type;
      using wait_type = typename std::decay_t<Policy>::wait_type;   
    };
  
    template<typename Policy>
    using selection_t = typename policy_traits<Policy>::selection_type;
  
    template<typename Policy>
    using resource_t = typename policy_traits<Policy>::resource_type;
  
    template<typename Policy>
    using wait_t = typename policy_traits<Policy>::wait_type;
  
  }

``selection_t<Policy>`` is the type returned by calls to ``select`` when using policy of type ``Policy``. 
Calling ``unwrap`` on an object of type ``selection_t<Policy>`` returns an object of 
type ``resource_t<Policy>``. When using the default SYCL backend, ``resource_t<Policy>`` 
is ``sycl::queue`` and ``sycl::wait_t<Policy>`` is ``sycl::event``.  The user functions
passed to ``submit`` and ``submit_and_wait`` are expected to have a signature of:

.. code:: cpp

  wait_t<Policy> user_function(resource_t<Policy>, ...);

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

More detailed information about the API is provided in the following sections:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   fixed_resource_policy
   round_robin_policy
   dynamic_load_policy
   auto_tune_policy
   
