.. SPDX-FileCopyrightText: UXL Foundation Contributors
..
.. SPDX-License-Identifier: CC-BY-4.0

oneDPL Specification Version History
====================================

This document lists releases of the oneDPL specification.

See also `oneAPI Specification Releases <https://oneapi-spec.uxlfoundation.org/>`_.

Latest Revision
---------------

.. list-table::
  :widths: 25 75
  :header-rows: 1

  * - Revision
    - Changes
  * - 1.5-provisional rev 1 (`HTML <index.rst>`__)
    - - Added 56 more parallel range algorithms
      - Aligned parallel range algorithms with the C++26 standardization proposals
      - Require segmented and histogram algorithms to be defined in ``<oneapi/dpl/numeric>``
      - Clarified the rules of device data accessibility and working with SYCL buffers
      - Added customizable ``is_indirectly_device_accessible`` trait for iterators
      - Added Philox random number generation engine

Earlier Revisions
-----------------

.. list-table::
  :widths: 25 75
  :header-rows: 1

  * - Revision
    - Changes
  * - 1.4 rev 1 (`HTML <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.4-rev-1/>`__)
    - - Added a specification version macro
      - Added requirements for oneDPL header files
      - Added new algorithms: ``transform_if``, ``sort_by_key``, ``stable_sort_by_key``, ``histogram``
      - Added parallel range algorithms: 22 new algorithms in namespace ``oneapi::dpl::ranges``
      - Clarified the semantics and requirements of execution policies, iterators, and buffer wrappers
      - Improved C++ standard compliance of random number generators
  * - 1.3 rev 1 (`HTML <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/>`__)
    - - Added ``sycl::vec`` support to random number generators and distributions
  * - 1.2 rev 1 (`HTML <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.2-rev-1/>`__)
    - - Added ``namespace dpl`` alias
      - Added random number generation APIs
      - Added ``base()`` method to iterator adaptors
      - Explicitly specified the standard-aligned execution policies
      - Clarified data handling with device execution policies
  * - 1.1 rev 1 (`HTML <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/>`__)
    - - Removed a separate namespace for C++ standard library APIs
      - Clarified requirements for binary search algorithms
  * - 1.0 rev 3 (`HTML <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.0-rev-3/>`__)
    - - The initial release

Source Code
-----------

The specification source code is maintained in the
`oneDPL repository <https://github.com/uxlfoundation/oneDPL/tree/main/documentation/specification>`__.
For the versions prior to and including 1.5-provisional revision 1, it was located in the `oneAPI specification repository
<https://github.com/uxlfoundation/oneAPI-spec/tree/main/source/elements/oneDPL/source>`__, from where
the snapshot with the commit ID 0d9d8e99 has been copied.