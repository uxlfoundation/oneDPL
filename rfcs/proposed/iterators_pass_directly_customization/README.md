# Passed Directly Customization Point for User Defined Types

## Introduction
oneDPL processes data from some `std::vector::iterator` types automatically as input to its SYCL-based device backend by
copying data between host and device, as described
[here](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html). Other iterator types, like
Unified Shared Memory (USM) pointers, refer to data inherently accessible by the device, so no data transfers are
required when passed into oneDPL APIs with a device policy. Iterators that do not require such data transfers are said
to have the "passed directly" trait.

oneDPL also defines rules for its provided
[iterator types](https://uxlfoundation.github.io/oneDPL/parallel_api/iterators.html) to be "passed directly" under
certain conditions, often based on their base types. When iterators lacking the "passed directly" trait are passed into
oneDPL APIs with a device policy, additional overhead is incurred to transfer data to and from the device. Properly
marking iterator types as "passed directly" is essential to avoid this overhead.

Internally, these rules are defined using the trait `oneapi::dpl::__ranges::is_passed_directly<T>`, which evaluates to
`std::true_type` or `std::false_type` to indicate whether the iterator type `T` should be passed directly to SYCL
kernels, rather than copied to a SYCL buffer first to enable a transfer to and from the device. A legacy
`is_passed_directly` trait exists, allowing types to define `using is_passed_directly = std::true_type;`. This legacy
method is used for helper types in SYCLomatic compatibility headers (`device_pointer`, `device_iterator`,
`tagged_pointer`, `constant_iterator`, `iterator_adaptor`). However, there is no official public API for users to define
custom iterator types that can be passed directly to SYCL kernels.

This gap should be addressed with an official public API. Without it, users must rely on oneDPL's provided iterator
types or access implementation details outside the specified interface.

## Proposal
### High Level Proposal

We will create an Argument-Dependent Lookup (ADL) customization point which defines if iterators are "passed directly"
`is_passed_directly_in_onedpl_device_policies`. Users may define if their iterator types should be "passed directly" by
oneDPL by defining a function in the same namespace where the iterator is defined.
`is_passed_directly_in_onedpl_device_policies` must accept a const lvalue reference to the iterator type being
specialized and return `std::true_type{}` if the iterator type is "passed directly" and `std::false_type{}` otherwise.
oneDPL will not call the `is_passed_directly_in_onedpl_device_policies` function at runtime. It will merely capture its
return type based upon the argument iterator type passed.

Additionally, oneDPL will provide a public trait, `is_passed_directly_to_device[_v]` as follows:
```cpp
namespace oneapi
{
namespace dpl
{
template <typename T>
struct is_passed_directly_to_device; // must have the characteristics of std::true_type or of std::false_type

template <typename T>
inline constexpr bool is_passed_directly_to_device_v = oneapi::dpl::is_passed_directly_to_device::value;
} // dpl
} // oneapi
```
indicating if the iterator type `T` is "passed directly". This public trait is intended to be used to help define
`is_passed_directly_in_onedpl_device_policies` for wrapper iterator types which depend on the "passed directly" status
of their base iterator(s) using only the base iterator type(s) rather than a named instance. It may also be used to
confirm user iterator types "passed directly" traits are as intended to prevent unnecessary overhead in oneDPL calls.

### Additional Information

The default implementation of the customization point `is_passed_directly_in_onedpl_device_policies` will be used to
explicitly mark the following iterators as "passed directly":
* Pointers (assumes USM pointers)
* Iterators containing the legacy `using is_passed_directly = std::true_type` trait defined within the type definition
* Iterators to USM shared allocated `std::vector`-s (when knowable)
* `std::reverse_iterator<Iter>` when `Iter` is also "passed directly"

oneDPL will define the "passed directly" definitions of its custom iterators as follows:
* `zip_iterator` is "passed directly" when all base iterators are "passed directly"
* `counting_iterator` and `discard_iterator` are always "passed directly"
* `transform_iterator` is "passed directly" if its source iterator is "passed directly"
* `permutation_iterator` is "passed directly" if both its source iterator and its index map are "passed directly"

If a "passed directly" customization point is defined for a type, any derived type will also match the existing
customization point function unless explicitly overridden. Users can override this behavior by implementing a more
specific ADL customization point function for the derived class. This is particularly useful in cases of multiple
inheritance or ambiguous ADL matches, where the default behavior may not align with the intended design.

Users must be aware of this behavior. A user might incorrectly assume that the absence of a customization point
specifically for the derived class would cause the derived iterator type to use the default overload of
`is_passed_directly_in_onedpl_device_policies` (and return `std::false_type`).

### Implementation Details

When using device policies, oneDPL will run compile-time checks on argument iterator types by using `decltype` to
determine the return type of `is_passed_directly_in_onedpl_device_policies` when called with iterator as argument. If
`std::true_type{}` is returned, oneDPL will pass the iterator directly to SYCL kernels rather than copying the data into
`sycl::buffers` and using those buffers to transfer data to SYCL kernels.

The specification and implementation will be prepared once this RFC is accepted as "proposed". We do not intend to offer
this first as experimental. This RFC will target "Supported" once the specification and implementation are accepted.

### Examples

Below is a simple example of an iterator and ADL customization point definition which is always "passed directly".

```cpp
namespace user
{

    struct my_passed_directly_iterator
    {
        /* unspecified user definition */
    };

    std::true_type
    is_passed_directly_in_onedpl_device_policies(const my_passed_directly_iterator&)
    {
        return std::true_type{};
    }
} //namespace user
```

Users can use any compile-time logic based on their iterator type to determine if it can be "passed directly" to define
the ADL function's return type. Commonly, this will involve some check of base template types and their "passed
directly" trait status. Below is an example of a type that contains a pair of iterators and should be treated as "passed
directly" if and only if both base iterators are also "passed directly". Note that the use of the public trait enables
the use of the base iterator type alone without creating a named instance of the base iterator to pass into the ADL.

```cpp
namespace user
{
    template <typename It1, typename It2>
    struct iterator_pair
    {
        It1 first;
        It2 second;
    };

    template <typename It1, typename It2>
    auto is_passed_directly_in_onedpl_device_policies(const iterator_pair<It1, It2>&)
    {
        return std::conjunction<oneapi::dpl::is_passed_directly_to_device<It1>,
                                oneapi::dpl::is_passed_directly_to_device<It2>>{};
    }
} //namespace user
```

It is also possible to write overloads without a body. The following is an
alternative written in this way for the previous example:

```cpp
    template <typename It1, typename It2>

    auto is_passed_directly_in_onedpl_device_policies(const iterator_pair<It1, It2>&) ->
        std::conjunction<oneapi::dpl::is_passed_directly_to_device<It1>,
                         oneapi::dpl::is_passed_directly_to_device<It2>>>;
```
Finally, it is possible to write overloads using the "hidden friend" idiom as functions with or without a body inside
the scope of the iterator definition itself. This option may be preferred when a user wants to ensure that this "passed
directly" trait of their iterator is coupled tightly with the definition itself for maintenance. It also has the
advantage of only making available for ADL the combinations of template arguments for this iterator which are explicitly
instantiated in the code already.

```cpp
    template <typename It1, typename It2>
    struct iterator_pair
    {
        It1 first;
        It2 second;
        friend auto is_passed_directly_in_onedpl_device_policies(const iterator_pair&) ->
            std::conjunction<oneapi::dpl::is_passed_directly_to_device<It1>,
                             oneapi::dpl::is_passed_directly_to_device<It2>>;
    };
```

## Alternatives Considered
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
We will need a detailed test checking both positive and negative responses to
`oneapi::dpl::is_passed_directly_to_device_v` to ensure they have the expected result. This should
include tests with custom types, combinations of iterators, USM pointers, etc.

## Open Questions

* How should we name the (1) ADL customization point and (2) public trait and value in a way that is concise and
properly conveys the meaning to the users?
    * Other names proposed:
        * `oneapi::dpl::onedpl_is_iterator_device_ready`
        * `oneapi::dpl::is_passed_directly_to_sycl_backend`
        * `oneapi::dpl::requires_explicit_data_transfer_onedpl_device_policies` (inverted)
        * `oneapi::dpl::is_passed_directly_to_device`
        * `oneapi::dpl::is_passed_directly`
    * This RFC provides different names for the (1) and (2). (1) will be used in the user's namespace, so it needs to
      identify onedpl with its name, but (2) will be used from `oneapi::dpl` so including onedpl in the name is
      redundant. Additionally, separating the names of (1) and (2) allows us to provide both a struct and value version
      of the public trait, which is the norm. These decisions could be reconsidered in the specification PR.
* Where should this be located?
    * Possible options include `oneapi/dpl/iterator`, `oneapi/dpl/type_traits`.
* What possibilities for problems / advantages exist for ADL matching for derived-from types, etc.?
  * This can lead to ambiguity when deriving from multiple classes, but that can then be differentiated by implementing
    a more specific ADL customization point function for the derived class.
