Dynamic Selection API
#####################

The dynamic selection API is an experimental feature in the |onedpl_long| (|onedpl_short|).
Dynamic selection provides functions for choosing a *resource* using a 
*selection policy*.  By default, the resources selected via these APIs 
in |onedpl_short| are SYCL queues. There are several functions and selection 
policies provided as part of the API. 

The selection policies include: ``fixed_resource_policy`` that always selects a 
specified resource, ``round_robin_policy`` that rotates between resources,
``dynamic_load_policy`` that chooses the resource that has the fewest outstanding 
submissions, and ``auto_tune_policy`` that chooses the best resources based on runtime 
profiling information.  

Policy objects are used as arguments to the dynamic selection functions. The 
``select`` function picks and returns a resource based on a policy. The ``submit`` 
and ``submit_and_wait`` functions select a resource and then pass the chosen resource 
to a developer-provided function. 

The following code example shows some of the key aspects of the API. The use 
of any empty ``single_task`` is for syntactic demonstration purposes only;
any valid command group or series of command groups can be submitted to the 
selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {

    // (1) create a policy object
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 6; ++i) {

      // (2) call one of the dynamic selection functions
      //     -- pass the policy to the API function
      //     -- provide a function to be called with a selected queue
      //        -- the user function must receive a sycl queue
      //        -- the user function must return a sycl event
      auto done = ex::submit(p,  
                             // (3) use the selected queue in user function
                             [=](sycl::queue q) {
                              std::cout << "submit task to "
                                        << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                              return q.single_task([]() { /* work here */ }); 
                            });

      // (4) each submission can be waited on using the returned object
      ex::wait(done);
    }

    // (5) and/or all submissions can be waited on as a group
    ex::wait(p.get_submission_group());
  }

In the preceding example, the key points in the code include:

#. A policy object is created. In this example, the policy is a ``round_robin_policy`` that will rotate between a CPU and GPU SYCL queue.
#. The ``submit`` function is called in a loop. The arguments to the call include the policy object and user-provided function.
#. The user-provided function receives a SYCL queue (selected by the policy) and returns a SYCL event that represents the end of the chain of work that was submitted to the queue.
#. The submit function returns an object that can be waited on. Calling ``wait`` on the ``done`` object blocks the main thread until the work submitted to the queue by your function is complete.
#. The whole group of submissions made during the loop can be waited on. In this example, the call is redundant, since each submission was already waited for inside of the loop body.
 
The output from this example is::

  submit task to cpu
  submit task to gpu
  submit task to cpu
  submit task to gpu
  submit task to cpu
  submit task to gpu

And shows that the user function is passed alternating queues.

More detailed information about the API is provided in the following sections:

.. toctree::
   :maxdepth: 2
   :titlesonly:

   dynamic_selection_api/functions
   dynamic_selection_api/policies
