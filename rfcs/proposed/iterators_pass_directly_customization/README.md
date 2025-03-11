# Passed Directly Customization Point for User Defined Types

## Introduction

oneDPL handles some types of input data automatically as input to its dpcpp (sycl-based) backend as described
[here](https://uxlfoundation.github.io/oneDPL/parallel_api/pass_data_algorithms.html). Unified Shared Memory (USM)
pointers refer to data which is device accessible inherently, so no processing is required to pass this type of input
data to SYCL kernels, we refer to this trait as "passed directly". oneDPL also defines some rules for its provided
[iterator types](https://uxlfoundation.github.io/oneDPL/parallel_api/iterators.html) to be passed directly to SYCL
under some circumstances (based usually on their base types).

Internally, these rules are defined with a trait `oneapi::dpl::__ranges::is_passed_directly<T>` which evaluates to
`std::true_type` or `std::false_type` to indicate whether the iterator type `T` should be passed directly to sycl
kernels. There exists an unofficial legacy `is_passed_directly` trait which types can define like this:
`using is_passed_directly = std::true_type;` which is supported within oneDPL. This method is currently used for a
number of helper types within the SYCLomatic compatibility headers, (`device_pointer`, `device_iterator`,
`tagged_pointer`, `constant_iterator`, `iterator_adaptor`). There is no official public API for users who want to
create their own iterator types which could be passed directly to SYCL kernels, this is a gap that should be filled 
with an official public API.

Without something like this users are forced to only rely upon our provided iterator types, or reach into implementation
details which are not part of oneDPL's specified interface.

## Proposal

Create a customization point `is_passed_directly_in_onedpl_device_policies` which allows users to mark
their types as "passed directly" by returning either `std::true_type{}` or as "non passed directly" by returning
`std::false_type{}`. Also create also a public trait `oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<T>`
which is an `inline constexpr bool` indicating the "passed directly" status of the template type `T`.`oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<T>` is very useful in `std::enable_if` and other SFINAE
contexts, since we only require the typename rather than a named instance of the type to determine if the type is passed
directly. It is also useful in the context of defining customizations for user types which are conditionally passed
directly, as shown in an example below.

The default implementation of the customization point `is_passed_directly_in_onedpl_device_policies` will defer to the
existing `oneapi::dpl::__ranges::is_passed_directly<T>` structure as shown, which is already used internally to oneDPL.

```
template <typename T>
constexpr
auto
is_passed_directly_in_onedpl_device_policies(const T&)
{
	return oneapi::dpl::__ranges::is_passed_directly<T>{};
}
```

User who want to customize this function for their own type must override it in the same namespace as their defined
type.

`oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<T>` will rely upon the customization point
`is_passed_directly_in_onedpl_device_policies`, as well as `std::declval<T>()` and `decltype` to determine the passed
directly status of a type. oneDPL will use this customization point internally when determining how to handle incoming
data, picking up any user defined customizations.

When using device policies, oneDPL will run compile time checks on argument iterator types by calling
`is_passed_directly_in_onedpl_device_policies`. If `std::true_type{}` is returned, oneDPL will pass the iterator
directly to sycl kernels rather than copying the data into `sycl::buffers` and using those buffers to transfer data to
sycl kernels. Users may also use `oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v<T>` themselves to check
how the oneDPL internals will treat any iterator types. This may be useful to ensure that no extra overhead occurs in
device policy calls of oneDPL APIs.

This option can exist in concert with existing methods, the legacy `is_passed_directly` typedef in types, the internal
`oneapi::dpl::__ranges::is_passed_directly` trait specializations. It would be possible to simplify the internal
implementation away from explicit specializations and instead using the customization point, but that is not required
at first implementation.

`oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v` will be defined in `oneapi/dpl/type_traits`.

The specification and implementation will be prepared once this RFC is accepted as "proposed", we do not intend to offer
this first as experimental. This RFC will target "Supported" once we have implementation solidified its details.

### Examples

Below is a simple example of a type and customization point definition which is always passed directly.

```
namespace user
{

    struct my_passed_directly_type
    {
        /* unspecified user definition */
    };

    constexpr
    auto
    is_passed_directly_in_onedpl_device_policies(const my_passed_directly_type&)
    {
        return std::true_type{};
    }
} //namespace user
```

Users can use any `constexpr` logic based on their type to determine if the type can be passed directly into a SYCL
kernel without any processing. Below is an example of a type which contains a pair of iterators, and should be treated
as passed directly if and only if both base iterators are also passed directly. 

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
namespace makes for maintenance problems. It either requires lots of closing of nested namespaces, opening of the
external library's namespace for the specialization or it requires separating these specializations to a separate
location removed from the types they are specializing for. oneDPL has chosen to use the later, which can be seen in
`include/oneapi/dpl/pstl/hetero/dpcpp/sycl_traits.h`. This has made for several errors where changes to structures
should have included changes to sycl_traits, but did not, and needed to be fixed later.

In an effort to avoid this same issue for our users, we propose a similar method but instead with a constexpr
customization point, allowing the user to override that customization point within their own namespace as a free
function.

### Require Specifically Named Typedef / Using in User's Type
oneDPL could make official our requirements for user's types to include a typedef or using statement to define if the
type is passed directly like `using is_passed_directly = std::true_type;`, where the absence of this would be equivalent
to a `std::false_type`. 

However, this clutters the user type definitions with specifics of oneDPL. It also may not be as clear what this
signifies for maintenance of user code without appropriate comments describing the details of oneDPL and SYCL. Users
have expressed that this is undesirable.

### Wrapper Class
oneDPL could provide some wrapper iterator `direct_iterator` which wraps an arbitrary base iterator and marks it as
passed directly. `direct_iterator` could utilize either of the above alternatives to accomplish this, and signal
that the iterator should be passed directly. It would need to pass through all operations to the wrapped base iterator,
and make sure no overhead is added in its usage.

There is some complexity in adding such a wrapper iterator, and it would need to be considered carefully to make sure no
problems would be introduced. This wrapper class may obfuscate users types, and make them more unwieldy to use. It is
also less expressive than the other options in that it only has the ability to unilaterally mark a type as passed
directly.  There is no logic that can be used to express some iterator type which may be conditionally passed directly,
other than to have logic to conditionally apply the wrapper in the first place. This option seems less clear and has
more opportunity to cause problems.

## Testing
We will need a detailed test checking both positive and negative responses to
`oneapi::dpl::is_passed_directly_in_onedpl_device_policies_v` have the expected result, with custom types and
combinations of iterators, usm pointers etc.

## Open Questions

* Is there a better, more concise name than `oneapi::dpl::is_passed_directly_in_onedpl_device_policies[_v]` that
properly conveys the meaning to the users?
  * Other names proposed:
    * `oneapi::dpl::onedpl_is_iterator_device_ready[_v]`
    * `oneapi::dpl::is_passed_directly_to_sycl_backend[_v]`
    * `oneapi::dpl::requires_explicit_data_transfer_onedpl_device_policies[_v]` (inverted)
* Where should this be located?  
  * Possible options include `oneapi/dpl/iterator`, `oneapi/dpl/type_traits`.