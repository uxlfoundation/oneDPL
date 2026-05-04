oneDPL 2022 Breaking Changes
############################

This page lists the incompatible changes that occurred between the 2021.7.1 and 2022.0 releases.

.. note::
   You may need to modify and/or rebuild your code when switching to oneDPL 2022.

Support for C++11 and C++14 has been discontinued. To use any functionality of oneDPL 2022,
a compiler that supports C++17 or newer version of the C++ language standard is required.

The following APIs are not supported in C++17 and have been removed from ``namespace oneapi::dpl``:

* In the ``<oneapi/dpl/functional>`` header:

  * ``binary_function``
  * ``unary_function``

The following APIs are deprecated in C++17 and not supported in C++20:

* In the ``<oneapi/dpl/functional>`` header:
  
  * ``binary_negate``
  * ``not1``
  * ``not2``
  * ``unary_negate``
  
* In the ``<oneapi/dpl/type_traits>`` header:

  * ``is_literal_type``
  * ``is_literal_type_v``
  * ``result_of``
  * ``result_of_t``

The size and the layout of the ``discard_block_engine`` class template were changed to align its 
implementation with the С++ standard proposal found at https://cplusplus.github.io/LWG/issue3561.
This change lets you utilize the full range of values for the template parameters of the engine.
