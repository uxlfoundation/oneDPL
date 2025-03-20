# Passed Directly Customization Point for User Defined Types

## Introduction

oneDPL processes data from some `std::vector::iterator` types automatically as input to its device backend (SYCL-based)
by implicitly copying data between host and device as described
[here](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html). Other iterator types, like
Unified Shared Memory (USM) pointers, refer to data that is inherently device accessible, so no data transfers are
required from oneDPL to pass this data into oneDPL APIs with a device policy. When iterator types do not require oneDPL
to copy data between host and device as preparation for being passed into SYCL kernels, we refer to these iterator types
as having the trait "passed directly". oneDPL also defines some rules for its provided
[iterator types](https://uxlfoundation.github.io/oneDPL/parallel_api/iterators.html) to be "passed directly" to SYCL
under some circumstances (usually based on their base types). When iterators which are not "passed directly" are passed
in to oneDPL APIs with a device policy, there is an overhead involved to transfer data to and from the device.
Accurately marking iterator types which are "passed directly" is necessary to avoid this overhead.

Internally, these rules are currently defined with a trait `oneapi::dpl::__ranges::is_passed_directly<T>` which
evaluates to `std::true_type` or `std::false_type` to indicate whether the iterator type `T` should be passed directly
to SYCL kernels. There exists an unofficial legacy `is_passed_directly` trait which types can define like this:
`using is_passed_directly = std::true_type;` which is supported within oneDPL. This method is currently used for a
number of helper types within the SYCLomatic compatibility headers (`device_pointer`, `device_iterator`,
`tagged_pointer`, `constant_iterator`, `iterator_adaptor`). There is no official public API for users who want to
create their own iterator types that could be passed directly to SYCL kernels. This is a gap that should be filled 
with an official public API.

Without something like this, users are forced to rely only upon our provided iterator types or reach into implementation
details that are not part of oneDPL's specified interface.

## Proposal
### High Level Proposal

We will create an Argument-Dependent Lookup (ADL) customization point which defines if iterators are "passed directly"
`is_passed_directly_in_onedpl_device_policies`. Users may define if their iterator types should be "passed directly" by
oneDPL by defining a constexpr function in the same namespace where the iterator is defined.
`is_passed_directly_in_onedpl_device_policies` must accept a const lvalue reference to the iterator type being
specialized and return `std::true_type{}` if the iterator type is "passed directly" and `std::false_type{}` otherwise.

Additionally, oneDPL will provide a public trait,
`inline constexpr bool oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<T>`, indicating if the iterator type
`T` is "passed directly". This public trait is intended to be used to help define
`is_passed_directly_in_onedpl_device_policies` for wrapper iterator types which depend on the "passed directly" status
of their base iterator(s) using only the base iterator type(s) rather than a named instance. It may also be used to
confirm user iterator types "passed directly" traits are as intended to prevent unnecessary overhead in oneDPL calls.

### Additional Information

The default implementation of the customization point `is_passed_directly_in_onedpl_device_policies` will be used to
explicitly mark the following iterators as "passed directly":
* Pointers (assumes USM pointers)
* Iterators containing the legacy `using is_passed_directly = true` trait defined within the type definition
* Iterators to USM shared allocated `std::vectors` (when knowable)
* `std::reverse_iterator<Iter>` when `Iter` is also "passed directly"

oneDPL will define the "passed directly" definitions of its custom iterators as follows:
* `zip_iterator` is "passed directly" when all base iterators are "passed directly"
* `counting_iterator` and `discard_iterator` are always "passed directly"
* `transform_iterator` is "passed directly" if its source iterator is "passed directly"
* `permutation_iterator` is "passed directly" if both its source iterator and its index map are "passed directly"

### Implementation Details

When using device policies, oneDPL will run compile-time checks on argument iterator types by calling
`is_passed_directly_in_onedpl_device_policies`. If `std::true_type{}` is returned, oneDPL will pass the iterator
directly to SYCL kernels rather than copying the data into `sycl::buffers` and using those buffers to transfer data to
SYCL kernels.

The specification and implementation will be prepared once this RFC is accepted as "proposed". We do not intend to offer
this first as experimental. This RFC will target "Supported" once the specification and implementation are accepted.

### Examples

Below is a simple example of an iterator and ADL customization point definition which is always "passed directly".

```
namespace user
{

    struct my_passed_directly_iterator
    {
        /* unspecified user definition */
    };

    constexpr
    auto
    is_passed_directly_in_onedpl_device_policies(const my_passed_directly_iterator&)
    {
        return std::true_type{};
    }
} //namespace user
```

Users can use any `constexpr` logic based on their iterator to determine if the iterator can be passed directly into a
SYCL kernel without any processing. Below is an example of a type which contains a pair of iterators and should be 
treated as passed directly if and only if both base iterators are also passed directly. Note that the use of the public
trait enables the use of the base iterator type alone without creating a named instance of the base iterator to pass
into the ADL.

```
namespace user
{
    template <typename It1, typename It2>
    struct iterator_pair
    {
        It1 first;
        It2 second;
    };

    template <typename It1, typename It2>
    constexpr 
    auto is_passed_directly_in_onedpl_device_policies(const iterator_pair<It1, It2>&)
    {
        if constexpr (oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<It1> &&
                      oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<It2>)
            return std::true_type{};
        else 
            return std::false_type{}; 
    }
} //namespace user
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
`oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v` to ensure they have the expected result. This should
include tests with custom types, combinations of iterators, USM pointers, etc.

## Open Questions

* Is there a better, more concise name than `oneapi::dpl::is_passed_directly_in_onedpl_device_policies[_v]` that
properly conveys the meaning to the users?
    * Other names proposed:
        * `oneapi::dpl::onedpl_is_iterator_device_ready[_v]`
        * `oneapi::dpl::is_passed_directly_to_sycl_backend[_v]`
        * `oneapi::dpl::requires_explicit_data_transfer_onedpl_device_policies[_v]` (inverted)
* Where should this be located?
    * Possible options include `oneapi/dpl/iterator`, `oneapi/dpl/type_traits`.
