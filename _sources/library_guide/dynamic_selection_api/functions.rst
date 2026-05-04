Functions
#########

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several functions provided as part 
of the API.

Submit
------

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename F, typename... Args> 
    auto submit(Policy&& p, F&& f, Args&&... args);
  }

Chooses a resource using the policy ``p`` and 
then calls the user function ``f``, passing the selected resource
and ``args...`` as the arguments. It also implements the necessary
calls to report execution information for policies that
require reporting.

``submit`` returns a *submission* object. Passing the *submission* object to the 
``wait`` function will block the calling thread until the work offloaded by the
submission is complete.

When using the SYCL backend, user-submitted functions ``f`` must return a
``sycl::event``. Calling the ``wait`` function on the returned *submission*
object behaves as if calling ``sycl::event::wait`` on the ``sycl::event``
returned by ``f``.

The following example demonstrates the use of the function ``submit`` and the
function ``wait``. The use of ``single_task`` is for syntactic demonstration
purposes only; any valid command group or series of command groups can be
submitted to the selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 4; ++i) {
      auto done = ex::submit(/* policy object */ p,  
                             /* user function */
                             [](sycl::queue q, /* any additional args... */ int j) {
                                std::cout << "(j == " << j << "): submit to " 
                                          << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                                auto e = q.single_task([]() { /* do some work */ }); 
                                return e; /* MUST return sycl::event */
                             },
                             /* any additional args... */ i);  
      std::cout << "(i == " << i << "): async work on main thread\n";
      ex::wait(done);
      std::cout << "(i == " << i << "): submission done\n"; 
    }
  }

The output from this example is::

  (j == 0): submit to cpu
  (i == 0): async work on main thread
  (i == 0): submission done
  (j == 1): submit to gpu
  (i == 1): async work on main thread
  (i == 1): submission done
  (j == 2): submit to cpu
  (i == 2): async work on main thread
  (i == 2): submission done
  (j == 3): submit to gpu
  (i == 3): async work on main thread
  (i == 3): submission done

Wait
----

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename W> 
    void wait(W&& w);
  }
  
The function ``wait`` blocks the calling thread until the work associated with
object ``w`` is complete. The object returned from 
a call to ``submit`` can be passed to this function to wait for the completion of a specific submission or the
object returned from a call to ``get_submission_group`` to wait for all submissions
made using a policy.  Example code that demonstrates waiting for a specific 
submission can be seen in the section for ``submit``.  

The following is an example that demonstrates waiting for all submissions by passing
the object returned by ``get_submission_group()`` to ``wait``:

.. code::  cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>
  
  namespace ex = oneapi::dpl::experimental;
  
  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };
  
    for (int i = 0; i < 4; ++i) {
      auto done = ex::submit(/* policy object */ p,  
                             /* user function */
                             [](sycl::queue q, /* any additional args... */ int j) {
                                std::cout << "(j == " << j << "): submit to " 
                                          << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                                auto e = q.single_task([]() { /* do some work */ }); 
                                return e; /* MUST return sycl::event */
                             },
                             /* any additional args... */ i);  
      std::cout << "(i == " << i << "): async work on main thread\n";
    }
    ex::wait(p.get_submission_group());
    std::cout << "done waiting for all submissions\n";
  }
  
The output from this example is::

  (j == 0): submit to cpu
  (i == 0): async work on main thread
  (j == 1): submit to gpu
  (i == 1): async work on main thread
  (j == 2): submit to cpu
  (i == 2): async work on main thread
  (j == 3): submit to gpu
  (i == 3): async work on main thread
  done waiting for all submissions

Submit and Wait
---------------

The difference between ``submit_and_wait`` and ``submit`` is that 
``submit_and_wait`` blocks the calling thread until the work associated
with the submission is complete. This behavior is essentially a short-cut
for calling ``wait`` on the object returned by a call to ``submit``. 

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename F, typename... Args> 
    void submit_and_wait(Policy&& p, F&& f, Args&&... args);
  }

Chooses a resource using the policy ``p`` and 
then calls the user function ``f``, passing the selected resource
and ``args...`` as the arguments. It implements the necessary 
calls to report execution information for policies that 
require reporting. This function blocks the calling thread until 
the user function and any work that it submits to the selected resource
are complete.

When using the SYCL backend, user-submitted functions ``f`` must return a
``sycl::event``. ``submit_and_wait`` calls ``sycl::event::wait`` on the
``f`` returned event before returning.

The following example demonstrates the use of the function ``submit_and_wait``. 
The use of ``single_task`` is for syntactic demonstration 
purposes only; any valid command group or series of command groups can be 
submitted to the selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>
  
  namespace ex = oneapi::dpl::experimental;
  
  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };
  
    for (int i = 0; i < 4; ++i) {
      ex::submit_and_wait(/* policy object */ p,  
                          /* user function */
                          [](sycl::queue q, /* any additional args... */ int j) {
                             std::cout << "(j == " << j << "): submit to " 
                                       << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                             auto e = q.single_task([]() { /* do some work */ }); 
                             return e; /* MUST return sycl::event */
                          },
                          /* any additional args... */ i);  
      std::cout << "(i == " << i << "): submission done\n"; 
    }
  }

The output from this example is::

  (j == 0): submit to cpu
  (i == 0): submission done
  (j == 1): submit to gpu
  (i == 1): submission done
  (j == 2): submit to cpu
  (i == 2): submission done
  (j == 3): submit to gpu
  (i == 3): submission done

Try Submit
----------

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename F, typename... Args>
    auto try_submit(Policy&& p, F&& f, Args&&... args);
  }

Attempts to choose a resource using the policy ``p`` and, if successful,
calls the user function ``f``, passing the selected resource and ``args...``
as the arguments. Returns a ``std::optional`` to a submission object if
a resource was available, empty if no resource could be selected.

This function is useful when you want to handle the case where no resources
are immediately available without blocking.

When using the SYCL backend, user-submitted functions ``f`` must return a
``sycl::event``. If ``try_submit`` is successful, calling the ``wait`` function
on the *submission* object contained in the ``std::optional`` behaves as if
calling ``sycl::event::wait`` on the ``sycl::event`` returned by ``f``.

The following example demonstrates the use of ``try_submit``.
The use of ``single_task`` is for syntactic demonstration
purposes only; any valid command group or series of command groups can be
submitted to the selected queue. Note that the code checks whether the returned
``std::optional`` has a value before calling ``wait`` on it.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },
                                sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 4; ++i) {
      auto maybe_done = ex::try_submit(p,
                                       [](sycl::queue q, int j) {
                                         std::cout << "(j == " << j << "): submit to "
                                                   << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
                                         return q.single_task([]() { /* work */ });
                                       },
                                       i);

      if (maybe_done.has_value()) {
        std::cout << "(i == " << i << "): submission succeeded\n";
        ex::wait(maybe_done.value());
      } else {
        std::cout << "(i == " << i << "): no resource available, trying alternative work\n";
        // Could retry, do other work, or handle the failure differently
      }
    }
  }

.. Note::

   For most policies, like ``round_robin_policy``, resources are always
   available and ``try_submit`` will always succeed in submitting work.
   ``try_submit`` is most useful with policies that may temporarily have no
   available resources, such as a custom policy which has hard capacity limits
   on specific devices.

Policy Queries
--------------

Getting the Resource Options
++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename... Args> 
    std::vector<resource_t<Policy>> get_resources(Policy&& p);
  }
  
Returns a ``std::vector`` that contains the resources that a policy
selects from. The following example demonstrates the use of the function 
``get_resources``. 

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::round_robin_policy p_explicit{ { sycl::queue{ sycl::cpu_selector_v },
                                         sycl::queue{ sycl::gpu_selector_v } } };

    std::cout << "Resources in explicitly set policy\n";
    for (auto& q : ex::get_resources(p_explicit))
      std::cout << "queue is " << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");

    std::cout << "\nResources in default policy\n";
    ex::round_robin_policy p_default;
    for (auto& q : ex::get_resources(p_default))
      std::cout << "queue is " << ((q.get_device().is_gpu()) ? "gpu\n" : "not-gpu\n");
  }
  
The output from this example on a test machine is::

  Resources in explicitly set policy
  queue is cpu
  queue is gpu

  Resources in default policy
  queue is not-gpu
  queue is not-gpu
  queue is gpu
  queue is gpu
  
When passing queues to the policy, the results show that the policy uses those
resources, a single CPU queue and a single GPU queue.

The platform used to run this example has two GPU drivers installed, 
as well as an FPGA emulator. When no resources are explicitly provided to the 
policy constructor, the results show two non-GPU devices (the CPU and the FPGA 
emulator) and two drivers for the GPU.

Getting the Group of Submissions
++++++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy> 
    auto get_submission_group(Policy&& p);
  }
   
Returns an object that can be passed to ``wait`` to block the main
thread until all work submitted to queues managed by the policy are
complete. 

An example that demonstrates the use of this function can be found in
the section that describes the ``submit`` function.
