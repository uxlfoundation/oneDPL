Parallel API
############

Parallel API is an implementation of the C++ standard libraries algorithms and execution
policies, as specified in the ISO/IEC 14882:2017 standard (commonly called C++17). The implementation
supports the unsequenced execution policy and the ``shift_left``/``shift_right`` algorithms, which are specified
in C++20. For more details see the `C++ Standard Execution Policies
<https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_.

|onedpl_long| (|onedpl_short|) provides specific versions of some algorithms, including:

* Segmented reduce
* Segmented scan
* Vectorized search algorithms
* Sorting of key-value pairs
* Conditional transform

Parallel API offers support for the parallel and vectorized execution of algorithms on Intel®
processors and heterogeneity support with a DPC++ based implementation for device execution policies.
For sequential execution, |onedpl_short| relies on an available implementation of the C++ standard library.

The utility API includes iterators and function object classes. The iterators implement
zip, transform, complete permutation operations on other iterators, and include a counting and discard iterator.
The function object classes provide minimum, maximum, and identity operations
that may be passed to algorithms such as reduce or transform.

|onedpl_short| also includes an experimental implementation of range-based algorithms with their
required ranges, and experimental asynchronous algorithms.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   parallel_api/execution_policies
   parallel_api/iterators
   parallel_api/async_api
   parallel_api/range_based_api
   parallel_api/additional_algorithms
   parallel_api/pass_data_algorithms
