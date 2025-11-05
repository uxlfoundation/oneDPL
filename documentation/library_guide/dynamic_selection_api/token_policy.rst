Token Policy
############

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several policies provided as part 
of the API. Policies encapsulate the logic and any associated state needed 
to make a selection. 

The token policy limits the number of concurrent submissions to each resource using a token-based 
mechanism. Each resource has a capacity (number of tokens), and a resource can only be selected 
if it has available tokens. When a submission completes, its token is returned to the resource's 
token pool. ``token_policy`` is useful for controlling resource utilization and preventing 
oversubscription of devices, particularly when managing multiple concurrent submissions to 
resources with limited capacity or when trying to maintain a specific level of concurrency per device.

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Backend = sycl_backend> 
    class token_policy {
    public:
      // useful types
      using resource_type = typename Backend::resource_type;
      
      // constructors
      token_policy(deferred_initialization_t, const int& capacity = 1);
      token_policy(const int& capacity = 1);
      token_policy(const std::vector<resource_type>& u, const int& capacity = 1);
  
      // deferred initializer
      void initialize(const int& capacity = 1);
      void initialize(const std::vector<resource_type>& u, const int& capacity = 1);
                      
      // queries
      auto get_resources() const;
      auto get_submission_group();
      
      // other implementation defined functions...
    };
  
  }
  
This policy can be used with all the dynamic selection functions, such as ``submit``,
``try_submit``, and ``submit_and_wait``. It can also be used with ``policy_traits``.

Example
-------

The following example demonstrates how to use ``token_policy`` to limit concurrent 
submissions to each device. In this case, each resource has a capacity of 2, allowing 
up to 2 concurrent submissions per resource.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  const std::size_t N = 10000;
  namespace ex = oneapi::dpl::experimental;

  void f(sycl::handler& h, float* v);

  int token_example(std::vector<sycl::queue>& devices, 
                    std::vector<float*>& usm_data) {

    const int capacity_per_device = 2; // (1)
    ex::token_policy p{devices, capacity_per_device}; // (2)

    auto num_devices = p.get_resources().size();
    auto num_arrays = usm_data.size();

    std::cout << "Running with " << num_devices << " queues\n"
              << "             " << num_arrays  << " usm arrays\n"
              << "Capacity per device: " << capacity_per_device << "\n";

    for (int i = 0; i < 100; ++i) { // (3)
      auto submission = ex::try_submit(p, [&](sycl::queue q) { // (4)
        int idx = i % num_arrays;
        float* data = usm_data[idx];
        return q.submit([=](sycl::handler &h) { // (5)
          f(h, data);
        });
      }); 

      if (!submission.has_value()) { // (6)
        std::cout << "No available tokens, waiting...\n";
        --i; // retry this iteration
        std::this_thread::yield();
      }
    }   
    ex::wait(p.get_submission_group()); // (7)
    return 0;
  }

The key points in this example are:

#. The capacity per device is set to 2, allowing up to 2 concurrent submissions to each resource.
#. A ``token_policy`` is constructed with the capacity parameter, controlling the maximum number of concurrent submissions per resource.
#. The loop iterates 100 times, attempting to submit work.
#. ``try_submit`` is used to attempt selection and submission. It returns an empty optional if no resource has available tokens.
#. The queue is used in a function to perform an asynchronous offload. The SYCL event returned from the call to ``submit`` is returned. Returning an event is required for functions passed to ``submit`` and ``submit_and_wait``.
#. If no resource is available (all resources are at capacity), the submission is retried after yielding.
#. ``wait`` is called to block until all submissions complete.

Selection Algorithm
-------------------
 
The selection algorithm for ``token_policy`` iterates through the available 
resources and selects the first resource that has available capacity (available tokens). 
A resource has available capacity if its current number of active submissions is less 
than its capacity limit. The token is held by the selection handle and automatically 
returned when the submission completes.

Simplified, expository implementation of the selection algorithm:
 
.. code:: cpp

  template<typename... Args>
  std::optional<selection_type> token_policy::try_select_impl(Args&&...) {
    if (initialized_) {
      for (auto resource : resources_) {
        int current = resource->availability_.load();
        if (current < capacity) {
          // Try to atomically increment the availability counter
          if (resource->availability_.compare_exchange_weak(current, current + 1)) {
            auto token = create_token(resource); // token decrements on destruction
            return selection_type{*this, resource, token};
          }
        }
      }
      // No resource has available capacity
      return {};
    } else {
      throw std::logic_error("select called before initialization");
    }
  }

where ``resources_`` is a container of resources (such as ``std::vector`` of ``sycl::queue``), 
and ``capacity`` is the maximum number of concurrent submissions allowed per resource. The token 
automatically decrements the availability counter when it is destroyed (when the submission completes).

Constructors
------------

``token_policy`` provides three constructors.

.. list-table:: ``token_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``token_policy(deferred_initialization_t, const int& capacity = 1);``
    - Defers initialization. An ``initialize`` function must be called prior to use. The capacity parameter sets the maximum concurrent submissions per resource.
  * - ``token_policy(const int& capacity = 1);``
    - Initialized to use the default set of resources with the specified capacity per resource.
  * - ``token_policy(const std::vector<resource_type>& u, const int& capacity = 1);``
    - Overrides the default set of resources and sets the capacity per resource.

Deferred Initialization
-----------------------

A ``token_policy`` that was constructed with deferred initialization must be 
initialized by calling one of its ``initialize`` member functions before it can be used
to select or submit.

.. list-table:: ``token_policy`` initializers
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``initialize(const int& capacity = 1);``
    - Initialize to use the default set of resources with the specified capacity per resource.
  * - ``initialize(const std::vector<resource_type>& u, const int& capacity = 1);``
    - Overrides the default set of resources and sets the capacity per resource.

Queries
-------

A ``token_policy`` has ``get_resources`` and ``get_submission_group`` 
member functions.

.. list-table:: ``token_policy`` queries
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``std::vector<resource_type> get_resources();``
    - Returns the set of resources the policy is selecting from.
  * - ``auto get_submission_group();``
    - Returns an object that can be used to wait for all active submissions.
