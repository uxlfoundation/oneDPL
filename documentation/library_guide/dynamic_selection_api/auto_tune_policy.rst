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
  template <typename ResourceType = sycl::queue, typename ResourceAdapter = oneapi::dpl::identity,
          typename Backend = default_backend<ResourceType, ResourceAdapter>, typename... KeyArgs>
    class auto_tune_policy
      : public policy_base<auto_tune_policy<ResourceType, ResourceAdapter, Backend, KeyArgs...>,
                           ResourceAdapter, Backend, execution_info::task_time_t>
    {
      public:
        using resource_type = ResourceType;
        using backend_type = Backend;

        auto_tune_policy(deferred_initialization_t);
        auto_tune_policy(uint64_t resample_interval_milliseconds = 0);
        auto_tune_policy(const std::vector<ResourceType>& u, ResourceAdapter adapter = {},
                         uint64_t resample_interval_milliseconds = 0);

        // deferred initializer
        void initialize(uint64_t resample_interval_milliseconds = 0);
        void initialize(const std::vector<resource_type>& u,
                        uint64_t resample_interval_milliseconds = 0);
        // other implementation defined functions...
    };

  }

This policy can be used with all the dynamic selection :doc:`free functions <functions>`,
as well as with :ref:`policy traits <policy-traits>`.

Task Identification with KeyArgs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The template parameter pack ``KeyArgs`` allows the policy to track performance for submitted jobs.
By default (empty ``KeyArgs``), all invocations of the same function share performance history.
When ``KeyArgs`` are specified, the policy uses both the function pointer and the specified
arguments to create a unique key for tracking performance.

.. note::
The number of ``KeyArgs`` types must exactly match the number of extra arguments
passed to the user function beyond the resource. This requirement is enforced at compile-time.

For example, ``auto_tune_policy<sycl::queue, oneapi::dpl::identity, default_backend, std::size_t>``
will track performance separately for each distinct ``std::size_t`` argument value, useful
when performance varies with problem size. Here, one ``KeyArg`` type corresponds to one extra
argument passed to the function after the ``sycl::queue``.

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
  //not a public function, for exposition purposes only
  template<typename Function, typename ...Args>
  selection_type auto_tune_policy::select(Function&& f, Args&&...args) {
    if (initialized_) {
      auto k = make_task_key(f, args...);
      auto tuner = get_tuner(k);
      auto offset = tuner->get_resource_to_profile();
      if (offset == use_best) {
        return selection_type{*this, tuner->best_resource_, tuner}; 
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

.. list-table::
  :widths: 50 50
  :header-rows: 1

  * - Signature
    - Description
  * - ``auto_tune_policy(deferred_initialization_t);``
    - Defers initialization. An ``initialize`` function must be called prior to use.
  * - | ``auto_tune_policy(``
      |   ``uint64_t resample_interval_milliseconds = 0);``
    - Initialized to use the default set of resources. An optional resampling interval can be provided.
  * - | ``auto_tune_policy(``
      |   ``const std::vector<ResourceType>& u,``
      |   ``ResourceAdapter adapter = {},``
      |   ``uint64_t resample_interval_milliseconds = 0);``
    - Overrides the default set of resources with an optional resource adapter. An optional resampling interval can be provided.

.. Note::

   When initializing the ``auto_tune_policy`` with SYCL queues, constructing the queues with the
   ``sycl::property::queue::enable_profiling`` property allows a more accurate determination of the
   best-performing device to be made.

Deferred Initialization
-----------------------

An ``auto_tune_policy`` that was constructed with deferred initialization must be
initialized by calling one of its ``initialize`` member functions before it can be used
to select or submit.

.. list-table::
  :widths: 50 50
  :header-rows: 1

  * - Signature
    - Description
  * - | ``initialize(``
      |   ``uint64_t resample_interval_milliseconds = 0);``
    - Initialize to use the default set of resources. An optional resampling interval can be provided.
  * - | ``initialize(``
      |   ``const std::vector<resource_type>& u,``
      |   ``uint64_t resample_interval_milliseconds = 0);``
    - Overrides the default set of resources. An optional resampling interval can be provided.

.. Note::

   When initializing the ``auto_tune_policy`` with SYCL queues, constructing the queues with the
   ``sycl::property::queue::enable_profiling`` property allows a more accurate determination of the
   best-performing device to be made.

Queries
-------

An ``auto_tune_policy`` has ``get_resources`` and ``get_submission_group``
member functions.

.. list-table::
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
A ``auto_tune_policy`` requires the ``task_time`` reporting requirement. See the
:ref:`Execution Information <execution-information>` section for more information
about reporting requirements.
