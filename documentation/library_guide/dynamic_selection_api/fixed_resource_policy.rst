Fixed-Resource Policy
#####################

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several policies provided as part 
of the API. Policies encapsulate the logic and any associated state needed 
to make a selection. 

The fixed-resource policy always returns the same resource selection. 
``fixed_resource_policy`` is designed for two primary scenarios: 

#. debugging the use of dynamic selection
#. special casing a dynamic selection capable application for a specific resource when it is known to be best on that platform.

.. code:: cpp

  namespace oneapi::dpl::experimental {
  
    template<typename Backend = sycl_backend>
    class fixed_resource_policy {
    public:
      // useful types
      using resource_type = typename Backend::resource_type;
      using wait_type = typename Backend::wait_type;
      
      class selection_type {
      public:
        fixed_resource_policy<Backend> get_policy() const;
        resource_type unwrap() const;
      };
      
      // constructors
      fixed_resource_policy(deferred_initialization_t);
      fixed_resource_policy(std::size_t offset = 0);
      fixed_resource_policy(const std::vector<resource_type>& u,
                            std::size_t offset = 0);
  
      // deferred initializers
      void initialize(std::size_t offset = 0);
      void initialize(const std::vector<resource_type>& u,
                      std::size_t offset = 0);
                      
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

In the following example, a ``fixed_resource_policy`` is used when the code is
compiled so that it selects a specific device.  When ``USE_CPU`` is defined at 
compile-time, this example always uses the CPU queue. When ``USE_GPU`` is defined 
at compile-time, it always uses the GPU queue. Otherwise, it uses an 
``auto_tune_policy`` to dynamically select between these two queues. Such a scenario 
could be used for debugging or simply to maintain the dynamic selection code even if 
the best device to use is known for some subset of platforms.  

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

  #if USE_CPU
    ex::fixed_resource_policy p{r};    // (1) uses index 0 of r, the cpu
  #elif USE_GPU
    ex::fixed_resource_policy p{r, 1}; // (2) uses index 1 of r, the gpu
  #else 
    ex::auto_tune_policy p{r};
  #endif

    {
      sycl::buffer<float> a_b(av);
      sycl::buffer<float> b_b(bv);
      sycl::buffer<float> c_b(cv);


      for (int i = 0; i < 6; ++i) {
        ex::submit_and_wait(p, [&](sycl::queue q) { // (3)
          // (4)
          std::cout << (q.get_device().is_cpu() ? "using cpu\n" : "using gpu\n");
          return q.submit([&](sycl::handler &h) {   // (5)
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

#. If ``USE_CPU`` is defined, a ``fixed_resouce_policy`` is constructed that targets the CPU.
#. If ``USE_GPU`` is defined, a ``fixed_resouce_policy`` is constructed that targets the GPU.
#. ``submit_and_wait`` is invoked with the policy as the first argument. The selected queue will be passed to the user-provided function.
#. For clarity when run, the type of device is displayed.
#. The queue is used in a function to perform an asynchronous offload. The SYCL event returned from the call to ``submit`` is returned. Returning an event is required for functions passed to ``submit`` and ``submit_and_wait``.

Selection Algorithm
-------------------
 
The selection algorithm for ``fixed_resource_policy`` always returns 
the same specific resource from its set of resources. The index of the
resource is set during construction or deferred initialization.

Simplified, expository implementation of the selection algorithm:
 
.. code:: cpp

  template<typename... Args>
  selection_type fixed_resource_policy::select(Args&& ...) {
    if (initialized_) {
      return selection_type{*this, resources_[fixed_offset_]};
    } else {
      throw std::logic_error("select called before initialization");
    }
  }

where ``resources_`` is a container of resources, such as 
``std::vector`` of ``sycl::queue``, and ``fixed_offset_`` stores a
fixed integer offset. Both ``resources_`` and ``fixed_offset`` 
are set during construction or deferred initialization of the policy
and then remain constant. 

Constructors
------------

``fixed_resource_policy`` provides three constructors.

.. list-table:: ``fixed_resource_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``fixed_resource_policy(deferred_initialization_t);``
    - Defers initialization. An ``initialize`` function must be called prior to use.
  * - ``fixed_resource_policy(std::size_t offset = 0);``
    - Sets the index for the resource to be selected. Uses the default set of resources.
  * - ``fixed_resource_policy(const std::vector<resource_type>& u, std::size_t offset = 0);``
    - Overrides the default set of resources and optionally sets the index for the resource to be selected.

Deferred Initialization
-----------------------

A ``fixed_resource_policy`` that was constructed with deferred initialization must be 
initialized by calling one its ``initialize`` member functions before it can be used
to select or submit.

.. list-table:: ``fixed_resource_policy`` constructors
  :widths: 50 50
  :header-rows: 1
  
  * - Signature
    - Description
  * - ``initialize(std::size_t offset = 0);``
    - Sets the index for the resource to be selected. Uses the default set of resources.
  * - ``initialize(const std::vector<resource_type>& u, std::size_t offset = 0);``
    - Overrides the default set of resources and optionally sets the index for the resource to be selected.

Queries
-------

A ``fixed_resource_policy`` has ``get_resources`` and ``get_submission_group`` 
member functions. 

.. list-table:: ``fixed_resource_policy`` constructors
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
to provide feedback to the policy. However, the ``fixed_resource_policy`` 
does not require any feedback about the system state or the behavior of 
the workload. Therefore, no explicit reporting of execution information 
is needed, as is summarized in the table below.

.. list-table:: ``fixed_resource_policy`` reporting requirements
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
