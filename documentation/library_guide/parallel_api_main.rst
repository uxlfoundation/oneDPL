Parallel API
############

The Parallel API in |onedpl_long| (|onedpl_short|) is an implementation of the C++ standard algorithms
with `execution policies <https://en.cppreference.com/w/cpp/algorithm#Execution_policies>`_,
as specified in the ISO/IEC 14882:2017 standard (commonly called C++17), as well as those added in C++20.
It offers threaded and SIMD execution of these algorithms on Intel® processors implemented on top of OpenMP*
and |onetbb_short|, as well as data parallel execution on accelerators backed by SYCL* support in |dpcpp_cpp|.

In addition, |onedpl_short| provides specific variations of some algorithms, including:

* Segmented reduce
* Segmented scan
* Vectorized search algorithms
* Sorting of key-value pairs
* Conditional transform

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
