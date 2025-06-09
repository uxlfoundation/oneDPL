# Future SYCL Keep-alives and Asynchrony

## Introduction

We currently depend on `__future` to provide three things: an event, a return value, and to extend the lifetime of temporary data required for SYCL kernels launched asynchronously.

Within oneDPL, we often use a type called `__result_and_scratch_storage`, which combines temporary allocation with a result value for performance reasons. `__result_and_scratch_storage` is commonly included in a future to achieve both keep-alive and, sometimes, asynchronous return. Combining temporary storage and result complicates the future, and using future solely to extend the lifetime of temporaries is an abuse of the semantics of future as currently written.

According to @akukanov, the future of asynchrony within oneDPL may look like the following:

* Experimental async algorithms will be deprecated and removed.
* Kernel template algorithms are SYCL-specific and return an event, not a future. Any computed values, whether one or many, are communicated back via function parameters.
* Deferred waiting,  for algorithms that can semantically support it, is a first-class feature.

We can summarize in a table how these are (or should be) used in different APIs:

| Algorithm Type         | Event Handling           | Return Value                                         | Keepalive Handling                                         |
|-----------------------|-------------------------|------------------------------------------------------|------------------------------------------------------------|
| Synchronous           | Wait for completion     | Pass to the caller                                   | Destroy                                                    |
| Deferred waiting      | Destroy without waiting | N/A (algorithms with return values cannot defer wait)| Delegate destruction upon kernel completion                |
| Asynchronous          | Pass to the caller      | Pass to the caller as a future                       | Delegate destruction / implicitly pass to the caller        |
| Kernel templates      | Pass to the caller      | N/A (computed values returned via parameters)        | Delegate destruction / Require caller to provide storage    |

Using future for keepalives, especially with combined result and scratch storage, causes several issues:
1) Return types are unwieldy and problematic. Especially when different algorithms may be used internally due to runtime branches, we must match return types. When algorithms differ in temporary storage requirements, `__future` return types will not match.
2) It exposes implementation details of the individual device backend to at least the "hetero" section of the code, if not further, outside the device backend.
3) Calling `get()` on `__future` with a mixture of return and keepalive data is complicated and requires hacky code to handle.

## Proposal
We must take steps to improve our usage of future, specifically to handle keepalives for SYCL algorithms in a robust and maintainable way. This will help us achieve our longer-term vision of asynchrony in oneDPL.
Let's follow these steps:
1) Break up `__result_and_scratch_storage` into a pair of utility structures: `__kernel_result` and `__kernel_scratch_storage`. For cases with access to both host USM memory and device USM memory, we already separate these allocations. We may see a slight performance regression for hardware that only supports device USM but not host USM, but that is acceptable to remove the complexity of the combined type. For this change, let's keep the `__kernel_scratch_storage` in the `__future` type, so we can do this step-by-step.

2) Improve the system for `__future::get()` to handle the `__kernel_result` type and number of elements appropriately. Allow the caller to specify the number of arguments to return (default to either 1 or to a tuple of all elements in the future).

3) Find and implement an alternative method for preserving keepalives, and remove them from `__future`.

What is the best alternative for extending the lifetimes of keepalives?
The following options have been raised:
  a) Use the experimental SYCL feature for asynchronous memory allocation and free to schedule freeing of temporary storage after a kernel completes. This is a good option, but requires a fallback, as it will not always be available in all environments.
  b) Use another location, such as a component of the execution policy, to store keepalives. A type for this purpose can be extracted from the policy within the backend and passed explicitly. For deferred waiting, we can provide tools for the user to clear temporary storage once the event has been waited on.
  c) Use a globally allocated storage system where keepalives can be registered. Use a host_task scheduled in the SYCL queue to mark the temporary storage for deletion when the kernels complete. This also requires a separate thread to run cleanup after the host_task marks it as OK, because the deallocation step should not be launched directly from a host_task due to restrictions about initiating an L0 call from an L0 callback. host_task will be L0 callbacks in the future.
  d) Create a type `event_with_keepalive` similar to `__future` where `get()` is not defined, which can be used as the `event` in a `__future`. This allows us to fix the semantic problem with `__future` by explicitly controlling what is a return value and what is a keepalive, while still relying on future for the functional keepalive behavior.

The following table describes the options presented, whether they resolve the remaining issues, and their biggest downside.

| Option              | Return Type         | Implementation Details                        | Biggest Downside                                         |
|---------------------|--------------------|-----------------------------------------------|----------------------------------------------------------|
| Async free          | Resolved* (when available) | Resolved* (when available)                  | Not available everywhere, needs fallback                 |
| Execution policy    | Resolved           | Somewhat fixed, requires generic interface    | Possibly not compatible with async free                  |
| Global storage      | Resolved           | Resolved                                      | Very complex, requires extra thread, global memory, complexity |
| Event with keepalive| Not fixed, still an issue | Mostly fixed, with separation of actual future from keepalive | Does not fix return type issue                           |

This infrastructure allows us to proceed with a more robust deferred waiting feature and better support for the existing asynchronous API.

## Open Questions
1) What should the default be for get():  single element, or tuple of all elements in the future
2) What option should we take to resolve the lifetime extension issue?



