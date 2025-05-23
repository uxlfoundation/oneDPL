.. _pass-data-algorithms:

Pass Data to Algorithms
#######################

For an algorithm to access data, it is important that the used execution policy matches the data storage type.
The following table shows which execution policies can be used with various data storage types.

================================================ ========================== =============
Data Storage                                     Device Policies            Host Policies
================================================ ========================== =============
`SYCL buffer`_                                   Yes                        No
Device-allocated `unified shared memory`_ (USM)  Yes                        No
Shared and host-allocated USM                    Yes                        Yes
``std::vector`` with ``sycl::usm_allocator``     Yes                        Yes
``std::vector`` with an ordinary allocator       See :ref:`use-std-vector`  Yes
Other data in host memory                        No                         Yes
================================================ ========================== =============

When using the standard-aligned (or *host*) execution policies, |onedpl_short| supports data being passed
to its algorithms as specified in the C++ standard (C++17 for algorithms working with iterators,
C++20 for parallel range algorithms), with :ref:`known restrictions and limitations <library-restrictions>`.

According to the standard, the calling code must prevent data races when using algorithms
with parallel execution policies.

.. note::
   Implementations of ``std::vector<bool>`` are not required to avoid data races for concurrent modifications
   of vector elements. Some implementations may optimize multiple ``bool`` elements into a bitfield, making it unsafe
   for multithreading. For this reason, it is recommended to avoid ``std::vector<bool>`` for anything but a read-only
   input with the standard-aligned execution policies.

The following subsections describe proper ways to pass data to an algorithm invoked with a device execution policy.

.. _indirectly-device-accessible:

Indirectly Device Accessible
----------------------------
Iterators and iterator-like types may or may not refer to content accessible within a
`SYCL <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_ kernel on a device. The term
*indirectly device accessible* refers to a type that represents content accessible on a device. An indirectly device
accessible iterator is a type that can also be dereferenced within a SYCL kernel.

When passed to |onedpl_short| algorithms with a device execution policy, indirectly device accessible types minimize
data movement and behave equivalently to using the type directly within a SYCL kernel.

.. _indirectly-device-accessible-trait:

Indirect Device Accessibility Type Trait
----------------------------------------
The following class template and variable template are defined in ``<oneapi/dpl/iterator>`` inside the namespace
``oneapi::dpl:``

.. code:: cpp
  template <typename T>
  struct is_indirectly_device_accessible{ /* see below */ };

  template <typename T>
  inline constexpr bool is_indirectly_device_accessible_v = is_indirectly_device_accessible<T>::value;

``template <typename T> oneapi::dpl::is_indirectly_device_accessible`` is a template which has the base characteristic
of ``std::true_type`` if ``T`` is indirectly device accessible. Otherwise, it has the base characteristic of
``std::false_type``.

.. _use-buffer-wrappers:

Use oneapi::dpl::begin and oneapi::dpl::end Functions
-----------------------------------------------------

``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are special helper functions that
allow you to pass SYCL buffers to parallel algorithms. These functions accept
a `SYCL buffer`_ and return an object of an unspecified type that provides the following API:

* It satisfies ``CopyConstructible`` and ``CopyAssignable`` C++ named requirements and comparable with
  ``operator==`` and ``operator!=``.
* It gives the following valid expressions: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b``
  are objects of the type, and ``n`` is an integer value. The effect of those operations is the same as for the type
  that satisfies the ``LegacyRandomAccessIterator``, a C++ named requirement.
* It provides the ``get_buffer`` method, which returns the buffer passed to the ``begin`` and ``end`` functions.

The ``begin`` and ``end`` functions can take SYCL 2020 deduction tags and ``sycl::no_init`` as arguments
to explicitly control which access mode should be applied to a particular buffer when submitting
a SYCL kernel to a device:

The return types of the ``begin`` and ``end`` functions are
`indirectly device accessible <indirectly-device-accessible>`_.

.. code:: cpp

  sycl::buffer<int> buf{/*...*/};
  auto first_ro = oneapi::dpl::begin(buf, sycl::read_only);
  auto first_wo = oneapi::dpl::begin(buf, sycl::write_only, sycl::no_init);
  auto first_ni = oneapi::dpl::begin(buf, sycl::no_init);

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

    sycl::buffer<int> buf{ vec.data(), vec.size() };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end);
    return 0;
  }

.. _use-usm:

Use Unified Shared Memory
-------------------------

If you have USM-allocated data, pass the pointers to the start and past the end
of the data sequence to a parallel algorithm. Make sure that the execution policy and
the USM allocation use the same SYCL queue. For example:

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

    oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n);

    sycl::free(d_head, q);
    return 0;
  }

.. note::
   Use of non-USM pointers is not supported for algorithms with device execution policies.

When using device USM, such as allocated by ``malloc_device``, you are responsible for data
transfers to and from the device to ensure that input data is device accessible during oneDPL
algorithm execution and that the result is available to the subsequent operations.

USM shared and device pointers are `indirectly device accessible <indirectly-device-accessible>`_.

.. _use-std-vector:

Use std::vector
---------------

You can use iterators to an ordinary ``std::vector`` with data in host memory, as shown in the following example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>

  int main(){
    std::vector<int> vec( 1000 );
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, vec.begin(), vec.end());
    return 0;
  }

In this case a temporary SYCL buffer is created, the data is copied to this buffer, and it is processed
according to the algorithm semantics. After processing on a device is complete, the modified data is copied
from the temporary buffer back to the host container.

.. note::
   For parallel range algorithms with device execution policies the use of ordinary ``std::vector``\s is not supported.

While convenient, direct use of an ordinary ``std::vector`` can lead to unintended copying between the host
and the device. We recommend working with SYCL buffers or with USM to reduce data copying.

.. note::
   For specialized memory algorithms that begin or end the lifetime of data objects, that is,
   ``uninitialized_*`` and ``destroy*`` families of functions, the data to initialize or destroy
   should be accessible on the device without extra copying. Therefore these algorithms may not use
   data storage on the host with device execution policies.

You can also use ``std::vector`` with a ``sycl::usm_allocator``, as shown in the following example.
Make sure that the allocator and the execution policy use the same SYCL queue:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>
  #include <sycl/sycl.hpp>

  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    std::vector<int, decltype(alloc)> vec(n, alloc);
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    // Recommended to use USM pointers:
    oneapi::dpl::sort(policy, vec.data(), vec.data() + vec.size());
  /*
    // Iterators for USM allocators might require extra copying - not a recommended method
    oneapi::dpl::sort(policy, vec.begin(), vec.end());
  */
    return 0;
  }

Iterators to ``std::vector`` are `indirectly device accessible <indirectly-device-accessible>`_ if and only if a 
``sycl::usm_allocator`` is used. For ``std::vector`` with a USM allocator we recommend to use ``std::vector::data()`` in
combination with ``std::vector::size()`` as shown in the example above, rather than iterators to
``std::vector``. That is because for some implementations of the C++ Standard Library it might not
be possible for |onedpl_short| to detect that iterators are pointing to USM-allocated data. In that
case the data will be treated as if it were in host memory, with an extra copy made to a SYCL buffer.
Retrieving USM pointers from ``std::vector`` as shown guarantees no unintended copying.

.. _use-iterators:

Use |onedpl_short| Iterators
----------------------------

|onedpl_short| provides a set of `iterators <iterators-details>`_ that can be used to pass data to algorithms, in
combination with the data storage types described above. To pass data to an algorithm with a device execution policy,
use iterators that are both `SYCL device-copyable`_ and `indirectly device accessible <indirectly-device-accessible>`_.
These properties of the |onedpl_short| iterators are described in `this table <iterator-properties-table>`_.

Use Custom Iterators
--------------------
If the provided iterators are not sufficient for your needs, you can create your own iterators that can be used as input
to |onedpl_short| algorithms. To pass data efficiently to an algorithm with a device execution policy, the custom
iterator must be `SYCL device-copyable`_ and indirectly device accessible. If a custom iterator is *not* defined to be
indirectly device accessible, the algorithm will create a temporary SYCL buffer to copy the data from the iterator to
the device. This may lead to performance degradation, and also requires that the data be accessible on the host.

You may customize your own iterator type ``T`` to be indirectly device accessible by defining a free function
``is_onedpl_indirectly_device_accessible(T)``, which returns a type with the base characteristic of ``std::true_type``
if ``T`` is indirectly device accessible. Otherwise, it returns a type with the base characteristic of
``std::false_type``. The function must be discoverable by argument-dependent lookup (ADL). It may be provided as a
forward declaration only, without defining a body.

The return type of ``is_onedpl_indirectly_device_accessible`` is examined at compile time to determine if ``T`` is
indirectly device accessible. The function overload to use must be selected with argument-dependent lookup.

.. note::
  Therefore, according to the rules in the C++ Standard, a derived type for which there is no function overload
  will match its most specific base type for which an overload exists.

Once ``is_onedpl_indirectly_device_accessible(T)`` is defined, the `public trait <indirectly-device-accessible-trait>`_
``template<typename T> oneapi::dpl::is_indirectly_device_accessible[_v]`` will return the appropriate value. This public
trait can also be used to define the return type of ``is_onedpl_indirectly_device_accessible(T)`` by applying it to any
source iterator component types.

The following example shows how to define a customization for the ``is_indirectly_device_accessible`` trait for a simple
user defined iterator. It also shows a more complex example where the customization is defined as a hidden friend of
the iterator class.

.. code:: cpp

  namespace usr
  {
      struct accessible_it
      {
          /* user definition of an indirectly device accessible iterator */
      };

      std::true_type
      is_onedpl_indirectly_device_accessible(accessible_it);

      struct inaccessible_it
      {
          /* user definition of an iterator which is not indirectly device accessible */
      };

      // The following could be omitted, as returning std::false_type matches the default behavior.
      std::false_type
      is_onedpl_indirectly_device_accessible(inaccessible_it);
  }

  static_assert(oneapi::dpl::is_indirectly_device_accessible<usr::accessible_it> == true);
  static_assert(oneapi::dpl::is_indirectly_device_accessible<usr::inaccessible_it> == false);

  // Example with base iterators and ADL overload as a hidden friend
  template <typename It1, typename It2>
  struct it_pair
   {
        It1 first;
        It2 second;
        friend auto
        is_onedpl_indirectly_device_accessible(it_pair) ->
            std::conjunction<oneapi::dpl::is_indirectly_device_accessible<It1>,
                             oneapi::dpl::is_indirectly_device_accessible<It2>>
        {
            return {};
        }
    };

  static_assert(oneapi::dpl::is_indirectly_device_accessible<
                                  it_pair<usr::accessible_it, usr::accessible_it>> == true);
  static_assert(oneapi::dpl::is_indirectly_device_accessible<
                                  it_pair<usr::accessible_it, usr::inaccessible_it>> == false);

.. _use-range-views:

Use Range Views
---------------

For :doc:`parallel range algorithms <parallel_range_algorithms>` with device execution policies,
place the data in USM or a USM-allocated ``std::vector``, and pass it to an algorithm
via a device-copyable range or view object such as ``std::ranges::subrange`` or ``std::span``.

.. note::
   Use of ``std::ranges::views::all`` is not supported for algorithms with device execution policies.

These data ranges as well as supported range adaptors and factories may be combined into
data transformation pipelines that also can be used with parallel range algorithms. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>
  #include <span>
  #include <ranges>
  #include <functional>
  #include <sycl/sycl.hpp>

  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::queue q = policy.queue();

    int* d_head = sycl::malloc_host<int>(n, q);
    std::generate(d_head, d_head + n, std::minstd_rand{});

    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(q);
    std::vector<int, decltype(alloc)> vec(n, alloc);

    oneapi::dpl::ranges::copy(policy,
        std::ranges::subrange(d_head, d_head + n) | std::views::transform(std::negate{}),
        std::span(vec));

    oneapi::dpl::ranges::sort(policy, std::span(vec));

    sycl::free(d_head, q);
    return 0;
  }

.. _`SYCL buffer`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:buffers
.. _`SYCL device-copyable`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable
.. _`unified shared memory`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm
