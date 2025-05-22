# Indirectly Device Accessible Trait and Customization Point for User Defined Types

## Introduction
Iterators and iterator-like types may or may not refer to content accessible within a
`SYCL <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_ kernel on a device. The term
*indirectly device accessible* refers to a type that represents content accessible on a device. An indirectly device
accessible iterator is a type that can also be dereferenced within a SYCL kernel.

When passed to |onedpl_short| algorithms with a device execution policy, indirectly device accessible types minimize
data movement and behave equivalently to using the type directly within a SYCL kernel.

oneDPL provides a some tools to enable users to be able to customize their own user-defined iterators to be indirectly
device accessible, and therefore be usable performantly with oneDPL algorithms when using a device execution policy.

### Changes from proposed RFC
The original RFC as proposed focussed more on the implementation detail "is passed directly", where during discussion
of the specification, we shifted the semantic meaning of the public trait to only describe whether the type refers
to data which is accessible on the device. This separates its meaning from other requirements for direct usage within
a SYCL kernel like random access iterator or SYCL device copyable. The name indirectly device accessible was selected
to represent this trait.

### Public Trait

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

### ADL-based Customization Point for User-Defined Iterator Types
Users may customize their own iterator types ``T`` to be indirectly device accessible by defining a free function
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

```cpp
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
```

### Indirectly Device Accessible Trait Value for oneDPL Inputs

The following table summarizes oneDPL input types and whether they are Indirectly Device Accessible:

+-------------------------------------------+---------------------------------------------+
| Iterator                                  | Indirectly Device Accessible                |
+-------------------------------------------+---------------------------------------------+
| counting_iterator                         | Yes                                         |
+-------------------------------------------+---------------------------------------------+
| discard_iterator                          | Yes                                         |
+-------------------------------------------+---------------------------------------------+
| zip_iterator                              | If all source iterators are                 |
+-------------------------------------------+---------------------------------------------+
| transform_iterator                        | If the source iterator is                   |
+-------------------------------------------+---------------------------------------------+
| permutation_iterator                      | If both source iterator and index map are   |
+-------------------------------------------+---------------------------------------------+
| Return of                                 |                                             |
| oneapi::dpl::begin(),                     |                                             |
| oneapi::dpl::end()                        | Yes                                         |
+-------------------------------------------+---------------------------------------------+
| USM pointers                              | Yes                                         |
+-------------------------------------------+---------------------------------------------+
| std::reverse_iterator                     | If the source iterator is                   |
+-------------------------------------------+---------------------------------------------+
| std::vector::iterator                     | If the allocator is distinguishable in      |
| with a USM allocator                      | the vector iterator type (*not recommended) |
+-------------------------------------------+---------------------------------------------+
| Iterators containing                      |                                             |
| using is_passed_directly = std::true_type | Yes (*may be removed)                       |
+-------------------------------------------+---------------------------------------------+
| std::vector::iterator                     |                                             |
| with a host allocator                     | No                                          |
+-------------------------------------------+---------------------------------------------+


### Implementation Details

Internally, a trait `template <typename _Iter> oneapi::dpl::__ranges::__is_passed_directly_device_ready`
is defined as the conjunction of `oneapi::dpl::is_indirectly_device_accessible<_Iter>` and
`sycl::is_device_copyable<_Iter>`. Input types that are *passed directly device ready* are passed to
SYCL kernels without any additional processing for direct usage.

A static assertion is used to prevent incorrect usage of the ADL-based customization point
`is_onedpl_indirectly_device_accessible()` by returning a type which is not a `std::bool_constant`. This is meant to
alert users clearly of incorrect usage of the customization point.

All hidden friend functions for the `is_onedpl_indirectly_device_accessible()` customization point provide a minimal
body within oneDPL, as do our examples in the specification and documentation. This is because GCC 15.1 provides a
warning for hidden friend functions that are body-less, as such usage matches a common anti-pattern where a
non-template friend is declared but not defined within the class definition, and then cannot be defined externally
to the class with a matching function signature. There is technically no problem with a body-less hidden friend
function for our purposes, as we only need the function declaration for evaluating the return type in a `declval`
scope. However, it is better to avoid this warning and the appearance of any problems.

## Alternatives Considered 
Note: these are carried forward from the previous RFC, where the framing was more around the implementation detail of
whether a type "is passed directly" to a sycl kernel. We have since reframed that for the public API within the
specification. 

### Public Trait Struct Explicit Specialization
oneDPL could make public our internal structure `oneapi::dpl::__ranges::is_passed_directly` as
`oneapi::dpl::is_passed_directly` for users to specialize to define rules for their types. This would be a similar
mechanism to `sycl::is_device_copyable`. The implementation details of this option should avoid some complexity required
to properly implement the customization point.

However, as we have learned from experience within oneDPL, explicit specialization of a structure in another library's
namespace creates maintenance problems. It either requires lots of closing of nested namespaces, opening of the external
library's namespace for the specialization, or it requires separating these specializations to a separate location
removed from the types they are specializing for. oneDPL has chosen to use the latter, which can be seen in
`include/oneapi/dpl/pstl/hetero/dpcpp/sycl_traits.h`. This has led to several errors where changes to structures should
have included changes to sycl_traits but did not, and needed to be fixed later.

In an effort to avoid this same issue for our users, we propose a similar method but instead with a constexpr
customization point, allowing the user to override that customization point within their own namespace as a free
function.

### Require Specifically Named Typedef / Using in User's Type
oneDPL could make official our requirements for users' types to include a typedef or using statement to define if the
type is "passed directly" like `using is_passed_directly = std::true_type;`, where the absence of this would be
equivalent to a `std::false_type`.

However, this clutters the user type definitions with specifics of oneDPL. It also may not be as clear what this
signifies for maintenance of user code without appropriate comments describing the details of oneDPL and SYCL. Users
have expressed that this is undesirable.

### Wrapper Class
oneDPL could provide some wrapper iterator `direct_iterator` which wraps an arbitrary base iterator and marks it as
"passed directly". `direct_iterator` could utilize either of the above alternatives to accomplish this and signal that
the iterator should be "passed directly". It would need to pass through all operations to the wrapped base iterator and
make sure no overhead is added in its usage.

There is some complexity in adding such a wrapper iterator, and it would need to be considered carefully to make sure no
problems would be introduced. This wrapper class may obfuscate users' types and make them more unwieldy to use. It is
also less expressive than the other options in that it only has the ability to unilaterally mark a type as "passed
directly". There is no logic that can be used to express some iterator type which may be conditionally "passed
directly", other than to have logic to conditionally apply the wrapper in the first place. This option seems less clear
and has more opportunity to cause problems.

## Testing
`test/parallel_api/iterator/indirectly_device_accessible.pass.cpp` provides testing for a variety of types, and
recursions upon types and the value of their `is_indirectly_device_accessible` trait both in the negative and positive
result.

It would be good to extend `test/parallel_api/iterator/input_data_sweep.h` tests to include a custom iterator
adaptor which tests actual usage of a custom type within oneDPL using the trait in different scenarios. However, with
the tests of oneDPL's provided iterators, we do cover the functionality of this system because those types use the ADL
customization point to define themselves indirectly device accessible.
