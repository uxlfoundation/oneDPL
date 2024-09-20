Functions
#########

The dynamic selection API is an experimental feature in the |onedpl_long| 
(|onedpl_short|) that selects an *execution resource* based on a chosen 
*selection policy*. There are several functions provided as part 
of the API.

Select
------

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename... Args> 
    selection_t<Policy> select(Policy&& p, Args&&... args);
  }
  
The function ``select`` chooses a resource (the *selection*) based on the 
policy ``p``. Whether any additional arguments are needed or considered 
depends on the policy.

An example that calls ``select`` using a ``round_robin_policy``:

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 6; ++i) {
      auto selection = ex::select(p);  
      auto q = ex::unwrap(selection);
      std::cout << "selected queue is " 
                << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");
    }
  }
  
The output of this example is::

  selected queue is cpu
  selected queue is gpu
  selected queue is cpu
  selected queue is gpu
  selected queue is cpu
  selected queue is gpu

The object returned by ``select`` is a *selection*. The exact type of the 
selection object depends on the policy. If it is necessary to know the exact 
type, it can be determined by using traits: 
``policy_trait<Policy>::selection_type`` or the helper trait ``selection_t<Policy>``.

Unwrapping a *selection* returns the underlying resource. For example, unwrapping
a selection when using SYCL (the default) results in a SYCL queue.

A selection can also be used to report *execution info*. More advanced policies,
such as ``dynamic_load_policy`` and ``auto_tune_policy`` require that runtime
execution information be reported back through the selection object when the
selection resource is used.

When possible, the selection should be passed to a ``submit`` or ``submit_and_wait`` function as the mechanism for submitting work to the resource. The ``submit`` and
``submit_and_wait`` functions implement the reporting of execution information 
needed by some policies, such as ``dynamic_load_policy`` and ``auto_tune_poliy``. 
If the selected resource is used directly, this reporting must be done explicitly 
(using the ``report`` functions).

Submit
------

``submit`` has two function signatures: 

#. the first argument is a *policy* object. 
#. the first argument is a *selection* object that was returned by a previous call to ``select``.

Submit Using a Policy
+++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename F, typename... Args> 
    submission_t<Policy> submit(Policy&& p, F&& f, Args&&... args);
  }

Chooses a resource using the policy ``p`` and 
then calls the user function ``f``, passing the unwrapped selection 
and ``args...`` as the arguments. It also implements the necessary 
calls to report execution information for policies that 
require reporting.

``submit`` returns a *submission* object. Passing the *submission* object to the 
``wait`` function will block the calling thread until the work offloaded by the
submission is complete. When using SYCL queues, this behaves as if calling
``sycl::event::wait`` on the SYCL event returned by the user function.

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

Submit Using a Selection
++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Selection, typename F, typename... Args> 
    auto submit(Selection s, F&& f, Args&&... args);
  }
  
Calls the user function ``f``, passing the unwrapped selection ``s`` and ``args...`` 
as the arguments. It also implements the necessary calls to report execution 
information for policies that require reporting.

``submit`` returns a *submission* object. Passing the *submission* object to the 
``wait`` function will block the calling thread until the work offloaded by the
submission is complete. When using SYCL queues, this behaves as if calling
``sycl::event::wait`` on the SYCL event returned by the user function.

The following example demonstrates the use of the function ``submit`` with an
object return by a call to select. The use of ``single_task`` is for 
syntactic demonstration purposes only; any valid command group or series of 
command groups can be submitted to the selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 4; ++i) {
      auto s = ex::select(p);
      auto done = ex::submit(/* selection object */ s,  
                             /* user function */
                             [](sycl::queue q, /* any additional args... */ int j) {
                                std::cout << "(j == " << j << "): submit to " 
                                          << ((q.get_device().is_gpu()) ? "gpu\n" :  "cpu\n");
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

Just like ``submit``, ``submit_and_wait`` has two function signatures: 

#. the first argument is a *policy* object. 
#. the first argument is a *selection* object that was returned by a previous call to ``select``.

The difference between ``submit_and_wait`` and ``submit`` is that 
``submit_and_wait`` blocks the calling thread until the work associated
with the submission is complete. This behavior is essentially a short-cut
for calling ``wait`` on the object returned by a call to ``submit``. 

Submit and Wait Using a Policy
++++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Policy, typename F, typename... Args> 
    void submit_and_wait(Policy&& p, F&& f, Args&&... args);
  }

Chooses a resource using the policy ``p`` and 
then calls the user function ``f``, passing the unwrapped selection 
and ``args...`` as the arguments. It implements the necessary 
calls to report execution information for policies that 
require reporting. This function blocks the calling thread until 
the user function and any work that it submits to the selected resource
are complete.

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


Submit and Wait Using a Selection
+++++++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Selection, typename F, typename... Args> 
    void submit_and_wait(Selection s, F&& f, Args&&... args);
  }
  
Calls the user function ``f``, passing the unwrapped selection ``s`` and ``args...`` 
as the arguments. It also implements the necessary calls to report execution 
information for policies that require reporting.

This function blocks the calling thread until 
the user function and any work that it submits to the resource
are complete.

The following example demonstrates the use of the function ``submit_and_wait``. 
The use of ``single_task`` is for syntactic demonstration 
purposes only; any valid command group or series of command groups can be 
submitted to the selected queue.

.. code::  cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <sycl/sycl.hpp>
  #include <iostream>
  
  namespace ex = oneapi::dpl::experimental;
  
  int main() {
    ex::round_robin_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                sycl::queue{ sycl::gpu_selector_v } } };
  
    for (int i = 0; i < 4; ++i) {
      auto s = ex::select(p);
      ex::submit_and_wait(/* selection object */ s,  
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
    for (auto& q : p_explicit.get_resources())
      std::cout << "queue is " << ((q.get_device().is_gpu()) ? "gpu\n" : "cpu\n");

    std::cout << "\nResources in default policy\n";
    ex::round_robin_policy p_default;
    for (auto& q : p_default.get_resources())
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

Report
------

Reporting Events with No Associated Values
++++++++++++++++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Selection, typename Info> 
    void report(Selection&& s, const Info& i);
  }

Reports an execution info event to the policy. What events must reported
is policy dependent. No reporting is necessary when using the ``submit`` or
``submit_and_wait`` functions, since these functions contain all necessary
instrumentation.

An example that uses reporting for the ``dynamic_load_policy`` is shown
below. This reporting is only necessary because ``select`` is used
but the resource is not passed to a ``submit`` or ``submit_and_wait`` function but
is instead used directly. The use of ``single_task`` is for syntactic demonstration 
purposes only; any valid command group or series of command groups can be 
submitted to the selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <chrono>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::dynamic_load_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                                 sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 6; ++i) {
      auto selection = ex::select(p);  
      auto q = ex::unwrap(selection);

      ex::report(selection, ex::execution_info::task_submission);
      q.single_task([]() { /* do work */ }).wait();
      ex::report(selection, ex::execution_info::task_completion);
    }
  }
  
Reporting Events with Associated Values
+++++++++++++++++++++++++++++++++++++++

.. code:: cpp

  namespace oneapi::dpl::experimental {
    template<typename Selection, typename Info, typename Value> 
    void report(Selection&& s, const Info& i, const Value& v);
  }
  
Reports an execution info event along with an associated value to the policy. 
What events must reported is policy dependent. No reporting is necessary 
if using the ``submit`` or ``submit_and_wait`` functions, since these functions contain 
all necessary instrumentation.

An example that uses reporting for the ``auto_tune_policy`` is shown
below. This reporting is only necessary in this case because ``select`` is used
but the resource is not passed to a ``submit`` or ``submit_and_wait`` function but
is instead used directly. The use of ``single_task`` is for syntactic demonstration 
purposes only; any valid command group or series of command groups can be 
submitted to the selected queue.

.. code:: cpp

  #include <oneapi/dpl/dynamic_selection>
  #include <chrono>
  #include <sycl/sycl.hpp>
  #include <iostream>

  namespace ex = oneapi::dpl::experimental;

  int main() {
    ex::auto_tune_policy p{ { sycl::queue{ sycl::cpu_selector_v },  
                              sycl::queue{ sycl::gpu_selector_v } } };

    for (int i = 0; i < 6; ++i) {
      auto f = []() {}; 
      auto selection = ex::select(p, f);  
      auto q = ex::unwrap(selection);

      auto before = std::chrono::steady_clock::now();
      q.single_task(f).wait();
      auto after = std::chrono::steady_clock::now();
      ex::report(selection, ex::execution_info::task_time, (after-before).count());
    }
  }
