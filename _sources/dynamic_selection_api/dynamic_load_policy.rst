Dynamic Load Policy
###################

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several policies provided as part 
of the API. Policies encapsulate the logic and any associated state needed 
to make a selection. 

The dynamic load policy tracks the number of submissions currently submitted but not yet completed on each 
resource and selects the resource that has the fewest unfinished submissions. 
``dynamic_load_policy`` is useful for offloading kernels of varying cost to devices 
of varying performance. A load-based assignment may achieve a good load balancing 
by submitting tasks to a resource that completes work faster.

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Backend = sycl_backend> 
    class dynamic_load_policy {
    public:
      // useful types
      using resource_type = typename Backend::resource_type;
      using wait_type = typename Backend::wait_type;
      
      class selection_type {
      public:
        dynamic_load_policy<Backend> get_policy() const;
        resource_type unwrap() const;
      };
      
      // constructors
      dynamic_load_policy(deferred_initialization_t);
      dynamic_load_policy();
      dynamic_load_policy(const std::vector<resource_type>& u);  
  
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

The following example demonstrates a simple approach to send work to more than
one queue concurrently using ``dynamic_load_policy``. The policy selects the
resource with the fewest number of unfinished submissions.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  const std::size_t N = 10000;
  namespace ex = oneapi::dpl::experimental;

  void f(sycl::handler& h, float* v);
  void do_cpu_work();

  int dynamic_load_example(std::vector<sycl::queue>& devices, 
                           std::vector<float *>& usm_data) {

    ex::dynamic_load_policy p{devices}; // (1)

    auto num_devices = p.get_resources().size();
    auto num_arrays = usm_data.size();
    // (2)
    auto submission_group_size = num_arrays;

    std::cout << "Running with " << num_devices << " queues\n"
              << "             " << num_arrays  << " usm arrays\n"
              << "Will perform " << submission_group_size << " concurrent offloads\n";


    for (int i = 0; i < 100; i+=submission_group_size) { // (3)
      for (int j = 0; j < submission_group_size; ++j) {  // (4)
        ex::submit(p, [&](sycl::queue q) { // (5)
          float *data = usm_data[j];
          return q.submit([=](sycl::handler &h) { // (6) 
            f(h, data);
          });
        }); 
        do_cpu_work(); // (7)
      }   
      ex::wait(p.get_submission_group()); // (8) 
    }
    return 0;
  }

The key points in this example are:

#. A ``dynamic_load_policy`` is constructed that selects from queues in the ``devices`` vector.
#. The total number of concurrent offloads, ``submission_group_size``, will be limited to the number of USM arrays. In this example, we allow multiple simultaneous offloads to the same queue. The only limitation is that there should be enough available vectors to support the concurrent executions.
#. The outer ``i``-loop iterates from 0 to 99, stepping by the ``submission_group_size``. This number of submissions will be offloaded concurrently.
#. The inner ``j``-loop iterates over ``submission_group_size`` submissions.
#. ``submit`` is used to select a queue and pass it to the user's function, but does not block until the event returned by that function completes. This provides the opportunity for concurrency across the submissions.
#. The queue is used in a function to perform an asynchronous offload. The SYCL event returned from the call to ``submit`` is returned. Returning an event is required for functions passed to ``submit`` and ``submit_and_wait``.
#. Some additional work is done between calls to ``submit``. ``dynamic_load_policy`` is most useful when there is time for work to complete on some devices before the next assignment is made. If all submissions are performed simultaneously, all devices will appear equally loaded, since the fast devices would have had no time to complete their work.
#. ``wait`` is called to block for all the concurrent ``submission_group_size`` submissions to complete.

Selection Algorithm
-------------------
 
The selection algorithm for ``dynamic_load_policy`` chooses the resource
that has the fewest number of unfinished offloads. The number of unfinished
offloads is the difference between the number of reported task submissions 
and then number of reported task completions. This value is tracked for each 
available resource.

Simplified, expository implementation of the selection algorithm:
 
.. code:: cpp

  template<typename... Args>
  selection_type dynamic_load_policy::select(Args&& ...) {
    if (initialized_) {
      auto least_loaded_resource = find_least_loaded(resources_);
      return selection_type{dynamic_load_policy<Backend>(*this), least_loaded};
    } else {
      throw std::logic_error("select called before initialization");
    }
  }

where ``resources_`` is a container of resources, such as 
``std::vector`` of ``sycl::queue``.  The function ``find_least_loaded``
iterates through the resources available to the policy and returns the
resource with the fewest number of unfinished offloads. 

Constructors
------------

``dynamic_load_policy`` provides three constructors.

.. list-table:: ``dynamic_load_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``dynamic_load_policy(deferred_initialization_t);``
    - Defers initialization. An ``initialize`` function must be called prior to use.
  * - ``dynamic_load_policy();``
    - Initialized to use the default set of resources.
  * - ``dynamic_load_policy(const std::vector<resource_type>& u);``
    - Overrides the default set of resources.

Deferred Initialization
-----------------------

A ``dynamic_load_policy`` that was constructed with deferred initialization must be 
initialized by calling one of its ``initialize`` member functions before it can be used
to select or submit.

.. list-table:: ``dynamic_load_policy`` constructors
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

A ``dynamic_load_policy`` has ``get_resources`` and ``get_submission_group`` 
member functions.

.. list-table:: ``dynamic_load_policy`` constructors
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
to provide feedback to the policy. The ``dynamic_load_policy`` tracks the
number of outstanding submissions on each device via callbacks that report
when a submission is started, and when it is completed. The instrumentation
to report these events is included in the implementations of 
``submit`` and ``submit_and_wait``.  However, if you use ``select`` and then
submit work directly to the selected resource, it is necessary to explicitly
report these events.

.. list-table:: ``dynamic_load_policy`` reporting requirements
  :widths: 50 50
  :header-rows: 1
  
  * - ``execution_info``
    - is reporting required?
  * - ``task_submission``
    - Yes
  * - ``task_completion``
    - Yes
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
