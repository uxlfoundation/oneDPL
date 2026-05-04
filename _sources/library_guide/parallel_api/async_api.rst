Asynchronous Algorithms
#######################

The functions defined in the STL ``<algorithm>`` or ``<numeric>`` headers are traditionally blocking. |onedpl_long| (|onedpl_short|)
extends the functionality of the C++17 parallel algorithms by providing asynchronous algorithms with non-blocking behavior.
This experimental feature enables you to express a concurrent control flow by building dependency chains, interleaving algorithm calls,
and interoperability with SYCL* kernels. 

The current implementation for async algorithms is limited to device execution policies.
All the functionality described below is available in the ``oneapi::dpl::experimental`` namespace.

The following async algorithms are currently supported:

* ``copy_async``
* ``fill_async``
* ``for_each_async``
* ``reduce_async``
* ``sort_async``
* ``inclusive_scan_async``
* ``exclusive_scan_async``
* ``transform_async``
* ``transform_reduce_async``
* ``transform_inclusive_scan_async``
* ``transform_exclusive_scan_async``

All the interfaces listed above are a subset of the C++17 STL algorithms,
where the suffix ``_async`` is added to the corresponding name (for example: ``reduce``, ``sort``, etc.).
The behavior and signatures are overlapping with the C++17 STL algorithm with the following changes:

* They do not block the execution.
* They take an arbitrary number of events (including 0) as last arguments to allow you to express input dependencies.
* They return a future-like object that allows you to use ``wait`` for completion and ``get`` for the result.

The type of the future-like object returned from an asynchronous algorithm is unspecified. The following member functions are present:

* ``get()`` returns the result.
* ``wait()`` waits for the result to become available.

If the returned object is the result of an algorithm with a device policy, it can be converted into a ``sycl::event``.
The lifetime of any resources the algorithm allocates (for example: temporary storage) is bound to the lifetime of
the returned object.

The following utility functions are available:

* ``wait_for_all(…)`` waits for an arbitrary number of objects that are convertible into ``sycl::event`` to become ready.


Example of Async API Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

    #include <oneapi/dpl/execution>
    #include <oneapi/dpl/async>
    #include <sycl/sycl.hpp>
    
    int main() {
        /* Build and compute a simple dependency chain: Fill buffer -> Transform -> Reduce */
        sycl::buffer<int> a{10};

        auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default, 
                                                  dpl::begin(a),dpl::end(a),7);
        
        auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [&](const int& x){return x + 1; },fut1);
        auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),fut1,fut2).get();
        return 0;
    }
