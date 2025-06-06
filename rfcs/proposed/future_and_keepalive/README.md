# Future Sycl Keep-alives and asynchrony

## Introduction

We currently depends upon `__future` to provide three things, an event, return value and and to extend the lifetime
of temporary data which is required for SYCL kernels launched asynchronously.

We often use a type `__result_and_scratch_storage` within oneDPL which combines temporary allocation with a result value
with the goal of performance. `__result_and_scratch_storage` is commonly included in a future to achieve a combination
of keep-alive and sometimes also asynchronous return. The combining of the temp storage and result complicates future,
and just using future as a vehicle to extend the lifetime of temporaries is an abuse of the semantics of future as
currently written. 

From @akukanov, the future of asynchrony within oneDPL may look something like the following:

* Experimental async algorithms will be deprecated and removed.
* Kernel template algorithms are SYCL specific and return an event, not a future. Any computed values, whether one or 
  many, are communicated back via function parameters.
* Deferred waiting - for algorithms that can semantically support it – is a first-class feature.

We can summarize in a table how these are (or should be) used in different APIs:

| Algorithm Type         | Event Handling           | Return Value                                         | Keepalive Handling                                 |
|-----------------------|-------------------------|------------------------------------------------------|----------------------------------------------------|
| Synchronous           | Wait for completion     | Pass to the caller                                   | Destroy                                            |
| Deferred waiting      | Destroy without waiting | N/A (algorithms with return values cannot defer wait)| Delegate destruction upon kernel completion        |
| Asynchronous          | Pass to the caller      | Pass to the caller as a future                       | Delegate destruction / implicitly pass to the caller|
| Kernel templates      | Pass to the caller      | N/A (computed values returned via parameters)        | Delegate destruction / Require caller to provide storage |

Using future for keepalives, especially with combined result and scratch storage causes a some issues:
1) Return types are unweildy and cause issues.  Especially when different algorithms may be used internally due to
   runtime branches, we must match return type. When algorithms differ in temporary storage requirements, `__future`
   return types will not match.
2) It spills implementation details of the individual device backend out into at least the "hetero" section of the
   code if not further, outside the device backend.
3) Calling `get()` on `__future` with a mixture of return and keepalive data is complicated, requires some hacky code
   to handle.

## Proposal
We must take steps to improve our usage of future, and specifically to handle keep-alives for sycl algorithms in a
robust and maintainable way. This will help us achieve our longer term vision of asynchrony in oneDPL.
Lets follow these steps:
1) Break up `__result_and_scratch_storage` into a pair of utility structures: `__kernel_result` and 
   `__kernel_scratch_storage`. For cases which have access to host USM memory and device USM memory, we already break
   these allocations up separately. We may see a slight performance regression for hardware which only supports device
   USM but not host USM, but that is acceptible to remove the complexity of the combined type.  For this change, lets
   keep the `__kernel_scratch_storage` in the `__future` type, so we can do this step by step. In this process, we must
   improve the system for `__future::get()` to handle `__kernel_result` type appropriately.  It may include adding in 
   the ability to specify the number of return values being requested via a template argument.

2) Find and implement alternative method for preserving keep-alives, remove them from `__future`.

3) Use new feature to implement a more robust deferred waiting feature, and possibly support asynchronous API

## Open Questions
What is the best alternative for extending lifetimes of keepalives for step (2) above?
    The following options have been raised:
        1) Create a type `event_with_keepalive` similar to `__future` where `get()` is not defined which can be used as
            the `event` in a `__future`.  This allows the us to fix the semantic problem with `__future`, by controlling
            explicitly what is a return value and what is a keepalive, while still relying upon future for the
            functional keep-alive behavior.
        2) Use the experimental sycl feature for asynchronous memory allocation and free to schedule freeing of
           temporary storage after a kernel completes. This is a nice option, but requires a fallback, as it will not
           always be available in all environments.
        3) Use a globally allocated storage system where keep-alives can be registered to be stored. Use host_task
           scheduled in the sycl queue to mark the temporary storage for deletion when the kernels complete. This also
           requires a separate thread to run cleanup after the host_task marks it as OK, because the deallocation step
           should not be launched directly from a host_task due to restrictions about initiating an L0 call from an L0
           callback.  host_task will be L0 callbacks in the future.
        4) Use some other location, like a component of the execution policy to store keepalives.  A type for this
           purpose can be extracted from the policy within the backend and passed explicitly. For deferred waiting, we
           can provide tools for the user to clear temporary storage once the event has been waited on.

