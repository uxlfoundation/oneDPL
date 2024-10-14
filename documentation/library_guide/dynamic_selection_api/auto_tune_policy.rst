Auto-Tune Policy
################

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several policies provided as part 
of the API. Policies encapsulate the logic and any associated state needed 
to make a selection. 

The auto-tune policy selects resources using runtime profiling. ``auto_tune_policy`` 
is useful for determining which resource performs best
for a given kernel. The choice is made based on runtime performance
history, so this policy is only useful for kernels that have stable
performance. Initially, this policy acts like ``round_robin_policy``,
rotating through each resource (one or more times). Then, once it has
determined which resource is performing best, it uses that resource
thereafter. Optionally, a resampling interval can be set to return to
the profiling phase periodically.

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Backend = sycl_backend> 
    class auto_tune_policy {
    public:
      // useful types
      using resource_type = typename Backend::resource_type;
      using wait_type = typename Backend::wait_type;
      
      class selection_type {
      public:
        auto_tune_policy<Backend> get_policy() const;
        resource_type unwrap() const;
      };
      
      // constructors
      auto_tune_policy(deferred_initialization_t);
      auto_tune_policy(uint64_t resample_interval_in_milliseconds = 0);
      auto_tune_policy(const std::vector<resource_type>& u,
                       uint64_t resample_interval_in_milliseconds = 0);
  
      // deferred initializer
      void initialize(uint64_t resample_interval_in_milliseconds = 0);
      void initialize(const std::vector<resource_type>& u,
                      uint64_t resample_interval_in_milliseconds = 0);
                      
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

In the following example, an ``auto_tune_policy`` is used to dynamically select between 
two queues, a CPU queue and a GPU queue. 

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    std::vector<sycl::queue> r { sycl::queue{sycl::cpu_selector_v},
                                 sycl::queue{sycl::gpu_selector_v} };

    const std::size_t N = 10000;
    std::vector<float> av(N, 0.0);
    std::vector<float> bv(N, 0.0);
    std::vector<float> cv(N, 0.0);
    for (int i = 0; i < N; ++i) {
      av[i] = bv[i] = i;
    }

    ex::auto_tune_policy p{r}; // (1)

    {
      sycl::buffer<float> a_b(av);
      sycl::buffer<float> b_b(bv);
      sycl::buffer<float> c_b(cv);


      for (int i = 0; i < 6; ++i) {
        ex::submit_and_wait(p, [&](sycl::queue q) { // (2)
          // (3)
          std::cout << (q.get_device().is_cpu() ? "using cpu\n" : "using gpu\n");
          return q.submit([&](sycl::handler &h) { // (4)
            sycl::accessor a_a(a_b, h, sycl::read_only);
            sycl::accessor b_a(b_b, h, sycl::read_only);
            sycl::accessor c_a(c_b, h, sycl::read_write);
            h.parallel_for(N, [=](auto i) { c_a[i] = a_a[i] + b_a[i]; }); 
          });
        }); 
      };  
    }

    for (int i = 0; i < N; ++i) {
      if (cv[i] != 2*i) {
         std::cout << "ERROR!\n";
      }   
    }
    std::cout << "Done.\n";
  }

The key points in this example are:

#. An ``auto_tune_policy`` is constructed to select between the CPU and GPU.
#. ``submit_and_wait`` is invoked with the policy as the first argument. The selected queue will be passed to the user-provided function.
#. For clarity when run, the type of device is displayed.
#. The queue is used in function to perform and asynchronous offload. The SYCL event returned from the call to ``submit`` is returned. Returning an event is required for functions passed to ``submit`` and ``submit_and_wait``.

Selection Algorithm
-------------------
 
The selection algorithm for ``auto_tune_policy`` uses runtime profiling
to choose the best resource for the given function. A simplified, expository 
implementation of the selection algorithm follows:
 
.. code:: cpp

  template<typename Function, typename ...Args>
  selection_type auto_tune_policy::select(Function&& f, Args&&...args) {
    if (initialized_) {
      auto k = make_task_key(f, args...);
      auto tuner = get_tuner(k);
      auto offset = tuner->get_resource_to_profile();
      if (offset == use_best) {
        return selection_type {*this, tuner->best_resource_, tuner}; 
      } else {
        auto r = resources_[offset];
        return selection{*this, r, tuner}; 
      }
    } else {
      throw std::logic_error("selected called before initialization");
    } 
  }

where ``make_task_key`` combines the inputs, including the function and its
arguments, into a key that uniquely identifies the user function that is being
profiled. ``tuner`` is the encapsulated logic for performing runtime profiling
and choosing the best option for a given key. When the call to ``get_resource_to_profile()``
return ``use_best``, the tuner is not in the profiling phase, and so the previously
determined best resource is used. Otherwise, the resource at index ``offset`` 
in the ``resources_`` vector is used and its resulting performance is profiled. 
When an ``auto_tune_policy`` is initialized with a non-zero resample interval,
the policy will periodically return to the profiling phase base on the provided
interval value.

Constructors
------------

``auto_tune_policy`` provides three constructors.

.. list-table:: ``auto_tune_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``auto_tune_policy(deferred_initialization_t);``
    - Defers initialization. An ``initialize`` function must be called prior to use.
  * - ``auto_tune_policy(uint64_t resample_interval_in_milliseconds = 0);``
    - Initialized to use the default set of resources. An optional resampling interval can be provided.
  * - ``auto_tune_policy(const std::vector<resource_type>& u, uint64_t resample_interval_in_milliseconds = 0);``
    - Overrides the default set of resources. An optional resampling interval can be provided.

.. Note::

   When initializing the ``auto_tune_policy`` with SYCL queues, constructing the queues with the
   ``sycl::property::queue::enable_profiling`` property allows a more accurate determination of the
   best-performing device to be made.

Deferred Initialization
-----------------------

A ``auto_tune_policy`` that was constructed with deferred initialization must be 
initialized by calling one its ``initialize`` member functions before it can be used
to select or submit.

.. list-table:: ``auto_tune_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``initialize(uint64_t resample_interval_in_milliseconds = 0);``
    - Initialize to use the default set of resources. An optional resampling interval can be provided.
  * - ``initialize(const std::vector<resource_type>& u, uint64_t resample_interval_in_milliseconds = 0);``
    - Overrides the default set of resources. An optional resampling interval can be provided.

.. Note::

   When initializing the ``auto_tune_policy`` with SYCL queues, constructing the queues with the
   ``sycl::property::queue::enable_profiling`` property allows a more accurate determination of the
   best-performing device to be made.

Queries
-------

A ``auto_tune_policy`` has ``get_resources`` and ``get_submission_group`` 
member functions.

.. list-table:: ``auto_tune_policy`` constructors
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
to provide feedback to the policy. The ``auto_tune_policy`` tracks the
performance of submissions on each device via callbacks that report
the execution time. The instrumentation to report these events is included 
in the implementations of ``submit`` and ``submit_and_wait``.  However, if you 
use ``select`` and then submit work directly to the selected resource, it 
is necessary to explicitly report these events.

.. list-table:: ``auto_tune_policy`` reporting requirements
  :widths: 50 50
  :header-rows: 1
  
  * - ``execution_info``
    - is reporting required?
  * - ``task_submission``
    - No
  * - ``task_completion``
    - No
  * - ``task_time``
    - Yes

In generic code, it is possible to perform compile-time checks to avoid
reporting overheads when reporting is not needed, while still writing 
code that will work with any policy, as demonstrated below:

.. code:: cpp

  auto s = select(my_policy);
  if constexpr (report_info_v<decltype(s), execution_info::task_submission_t>)
  {
    s.report(execution_info::task_submission);
  }
