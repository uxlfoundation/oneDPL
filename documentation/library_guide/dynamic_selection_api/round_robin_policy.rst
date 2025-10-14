Round-Robin Policy
##################

The dynamic selection API is an experimental feature in the |onedpl_long|
(|onedpl_short|) that selects an *execution resource* based on a chosen
*selection policy*. There are several policies provided as part
of the API. Policies encapsulate the logic and any associated state needed
to make a selection. 

The round-robin policy cycles through the set of resources at each selection. ``round_robin_policy``
is useful for offloading kernels of similar cost to devices of similar
capabilities. In those cases, a round-robin assignment of kernels to devices
will achieve a good load balancing.

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Backend = sycl_backend> 
    class round_robin_policy {
    public:
      // useful types
      using resource_type = typename Backend::resource_type;
      using wait_type = typename Backend::wait_type;
      
      class selection_type {
      public:
        round_robin_policy<Backend> get_policy() const;
        resource_type unwrap() const;
      };
      
      // constructors
      round_robin_policy(deferred_initialization_t);
      round_robin_policy();
      round_robin_policy(const std::vector<resource_type>& u);
  
      // deferred initializer
      void initialize();
      void initialize(const std::vector<resource_type>& u);
                      
      // queries
      auto get_resources() const;
      auto get_submission_group();
      
      // other implementation defined functions...
    };
  
  }
  
This policy can be used with all the dynamic selection functions, such as ``select``, ``submit``,
and ``submit_and_wait``. It can also be used with ``policy_traits``.

Example
-------

The following example demonstrates a simple approach to send work to each 
queue in a set of queues, and then wait for all devices to complete the work
before repeating the process. A ``round_robin_policy`` is used rotate through
the available devices.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  const std::size_t N = 10000;
  namespace ex = oneapi::dpl::experimental;

 void f(sycl::handler& h, float* v);


  int round_robin_example(std::vector<sycl::queue>& similar_devices,
                          std::vector<float*>& usm_data) {

    ex::round_robin_policy p{similar_devices}; // (1)

    auto num_devices = p.get_resources().size();
    auto num_arrays = usm_data.size();

    // (2)
    auto submission_group_size = (num_arrays < num_devices) ? num_arrays : num_devices;

    std::cout << "Running with " << num_devices << " queues\n"
              << "             " << num_arrays  << " usm arrays\n"
              << "Will perform " << submission_group_size << " concurrent offloads\n";

    for (int i = 0; i < 100; i += submission_group_size) { // (3)
      for (int j = 0; j < submission_group_size; ++j) {  // (4)
        ex::submit(p, [&](sycl::queue q) { // (5)
          float* data = usm_data[j];
          return q.submit([=](sycl::handler &h) { // (6)
            f(h, data);
          });
        }); 
      }   
      ex::wait(p.get_submission_group()); // (7)
    }
    return 0;
  }

The key points in this example are:

#. A ``round_robin_policy`` is constructed that rotates between the CPU and GPU queues.
#. The total number of concurrent offloads, ``submission_group_size``, will be limited to the number of USM arrays or the number of queues, whichever is smaller. 
#. The outer ``i``-loop iterates from 0 to 99, stepping by the ``submission_group_size``. This number of submissions will be offload concurrently.
#. The inner ``j``-loop iterates over ``submission_group_size`` submissions.
#. ``submit`` is used to select a queue and pass it to the user's function, but does not block until the event returned by that function completes. This provides the opportunity for concurrency across the submissions.
#. The queue is used in a function to perform an asynchronous offload. The SYCL event returned from the call to ``submit`` is returned. Returning an event is required for functions passed to ``submit`` and ``submit_and_wait``.
#. ``wait`` is called to block for all the concurrent ``submission_group_size`` submissions to complete.

Selection Algorithm
-------------------
 
The selection algorithm for ``round_robin_policy`` rotates through
the elements of the set of available resources. A simplified, expository 
implementation of the selection algorithm follows:
 
.. code:: cpp

  template<typename ...Args>
  selection_type round_robin_policy::select(Args&&...) {
    if (initialized_) {
      auto& r = resources_[next_context_++ % num_resources_];
      return selection_type{*this, r};
    } else {
      throw std::logic_error("selected called before initialization");
    }
  }

where ``resources_`` is a container of resources, such as 
``std::vector`` of ``sycl::queue``, ``next_context_`` is 
a counter that increments at each selection, and ``num_resources_``
is the size of the ``resources_`` vector.

Constructors
------------

``round_robin_policy`` provides three constructors.

.. list-table:: ``round_robin_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``round_round_policy(deferred_initialization_t);``
    - Defers initialization. An ``initialize`` function must be called prior to use.
  * - ``round_robin_policy();``
    - Initialized to use the default set of resources.
  * - ``round_robin_policy(const std::vector<resource_type>& u);``
    - Overrides the default set of resources.

Deferred Initialization
-----------------------

A ``round_robin_policy`` that was constructed with deferred initialization must be 
initialized by calling one its ``initialize`` member functions before it can be used
to select or submit.

.. list-table:: ``round_robin_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``initialize();``
    - Initialize to use the default set of resources.
  * - ``initialize(const std::vector<resource_type>& u);``
    - Overrides the default set of resources.

Queries
-------

A ``round_robin_policy`` has ``get_resources`` and ``get_submission_group`` 
member functions.

.. list-table:: ``round_robin_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``std::vector<resource_type> get_resources();``
    - Returns the set of resources the policy is selecting from.
  * - ``auto get_submission_group();``
    - Returns an object that can be used to wait for all active submissions.

Reporting Requirements
----------------------

If a resource returned by ``select`` is used directly without calling
``submit`` or ``submit_and_wait``, it may be necessary to call ``report``
to provide feedback to the policy. However, the ``round_robin_policy`` 
does not require any feedback about the system state or the behavior of 
the workload. Therefore, no explicit reporting of execution information 
is needed, as is summarized in the table below.

.. list-table:: ``round_robin_policy`` reporting requirements
  :widths: 50 50
  :header-rows: 1
  
  * - ``execution_info``
    - is reporting required?
  * - ``task_submission``
    - No
  * - ``task_completion``
    - No
  * - ``task_time``
    - No

In generic code, it is possible to perform compile-time checks to avoid
reporting overheads when reporting is not needed, while still writing 
code that will work with any policy, as demonstrated below:

.. code:: cpp

  auto s = select(my_policy);
  if constexpr (report_info_v<decltype(s), execution_info::task_submission_t>)
  {
    s.report(execution_info::task_submission);
  }
