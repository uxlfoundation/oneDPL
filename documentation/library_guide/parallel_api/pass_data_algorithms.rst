Pass Data to Algorithms
#######################

When using the C++ standard execution policies, oneDPL supports data being passed to its algorithms as specified
in the ISO/IEC 14882:2017 standard (commonly called C++17). According to the standard, the calling code
must prevent data races when using algorithms with parallel execution policies.

Note: Implementations of ``std::vector<bool>`` are not required to avoid data races for concurrent modifications
of vector elements. Some implementations may optimize multiple ``bool`` elements into a bitfield, making it unsafe
for multithreading. For this reason, it is recommended to avoid ``std::vector<bool>`` for anything but a read-only
input with the C++ standard execution policies.

When using a device execution policy, you can use one of the following ways to pass data to an algorithm:

* ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` functions
* Unified shared memory (USM) pointers
* ``std::vector`` with or without a USM allocator

.. _use-buffer-wrappers:

Use oneapi::dpl::begin and oneapi::dpl::end Functions
-----------------------------------------------------

``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are special helper functions that
allow you to pass SYCL buffers to parallel algorithms. These functions accept
a SYCL buffer and return an object of an unspecified type that provides the following API:

* It satisfies ``CopyConstructible`` and ``CopyAssignable`` C++ named requirements and comparable with
  ``operator==`` and ``operator!=``.
* It gives the following valid expressions: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b``
  are objects of the type, and ``n`` is an integer value. The effect of those operations is the same as for the type
  that satisfies the ``LegacyRandomAccessIterator``, a C++ named requirement.
* It provides the ``get_buffer`` method, which returns the buffer passed to the ``begin`` and ``end`` functions.

The ``begin`` and ``end`` functions can take SYCL 2020 deduction tags and ``sycl::no_init`` as arguments
to explicitly mention which access mode should be applied to the buffer accessor when submitting a
SYCL kernel to a device. For example:

.. code:: cpp

  auto first1 = begin(buf, sycl::read_only);
  auto first2 = begin(buf, sycl::write_only, sycl::no_init);
  auto first3 = begin(buf, sycl::no_init);

The example above allows you to control the access mode for the particular buffer passing to a parallel algorithm.

To use the functions, add ``#include <oneapi/dpl/iterator>`` to your code. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/iterator>
  #include <random>
  #include <sycl/sycl.hpp>
  int main(){
    std::vector<int> vec(1000);
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    //create a buffer from host memory
    sycl::buffer<int> buf { vec.data(), vec.size() };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    std::sort(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end);
    return 0;
  }

.. _use-usm:

Use Unified Shared Memory
-------------------------

If you have USM-allocated memory, pass the pointers to the start and past the end
of the sequence to a parallel algorithm. Make sure that the execution policy and
the USM-allocated memory were created for the same queue. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <sycl/sycl.hpp>
  int main(){
    sycl::queue q;
    const int n = 1000;
    int* d_head = sycl::malloc_shared<int>(n, q);
    std::generate(d_head, d_head + n, std::minstd_rand{});

    std::sort(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n);

    sycl::free(d_head, q);
    return 0;
  }

When using device USM, such as allocated by ``malloc_device``, you are responsible for data
transfers to and from the device to ensure that input data is device accessible during oneDPL
algorithm execution and that the result is available to the subsequent operations.

Use std::vector
-----------------------------

The following examples demonstrate two ways to use the parallel algorithms with ``std::vector``:

* Host allocators
* USM allocators

You can use iterators to host allocated ``std::vector`` data
as shown in the following example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>
  int main(){
    std::vector<int> vec( 1000 );
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    std::sort(oneapi::dpl::execution::dpcpp_default, vec.begin(), vec.end());
    return 0;
  }

When using iterators to host allocated data, a temporary SYCL buffer is created, and the data
is copied to this buffer. After processing on a device is complete, the modified data is copied
from the temporary buffer back to the host container. While convenient, using host allocated
data can lead to unintended copying between host and device. We recommend working with SYCL buffers
or USM memory to reduce data copying between the host and device. 

You can also use ``std::vector`` with a USM allocator, as shown in the following example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <sycl/sycl.hpp>
  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    std::vector<int, decltype(alloc)> vec(n, alloc);
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    // Recommended to use USM pointers:
    std::sort(policy, vec.data(), vec.data() + vec.size());

    // Iterators for USM allocators might require extra copying - not recommended method
    // std::sort(policy, vec.begin(), vec.end());
    return 0;
  }

Make sure that the execution policy and the USM-allocated memory were created for the same queue.

For ``std::vector`` with a USM allocator we recommend to use ``std::vector::data()`` in
combination with ``std::vector::size()`` as shown in the example above, rather than iterators to
``std::vector``. That is because for some implementations of the C++ Standard Library it might not
be possible for |onedpl_short| to detect that iterators are pointing to USM-allocated data. In that
case the data will be treated as if it were host-allocated, with an extra copy made to a SYCL buffer.
Retrieving USM pointers from ``std::vector`` as shown guarantees no unintended copying.