# Token Policy and Changes to Policy Contract

## Introduction
The key distinguishing feature between different dynamic selection policies is the method by which they select resources to submit jobs to. Currently, `select` is a standalone public API that can be called to obtain a resource, but it can also be invoked by calling `submit` or `submit_and_wait` directly. In this RFC, we propose to remove the standalone `select` API and introduce a new policy called TokenPolicy. TokenPolicy provides users with a way to specify hard concurrency limits on resources.

We have not found much utility in standalone `select` functionality. Relying on direct submission with implicit selection greatly simplifies Token Policy and also makes usage of DynamicLoadPolicy and AutotunePolicy more likely to be up-to-date with their instrumented metrics to make good selections.

## Proposal

### Removal of select routine
We propose to remove the following functions and traits from the public contract for a policy:

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `p.select(args…)` | Returns `selection_t<T>` that satisfies [Selection](#selection_req_id). The selected resource must be within the set of resources returned by `p.get_resources()`. |
| `p.submit(s, f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` with the selected resource `s` and the arguments `args...`. |
| *Optional* | Description |
| --------------------- | ----------- |
| `p.submit_and_wait(s, f, args…)` | Returns `void`. The function invokes `f` with `s` and `args...` and waits for the `wait_t<T>` it returns to complete. |

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` a selection of a type `selection_t<T>` , which satisfies [Selection](#selection_req_id), and was made by `p`.
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Policy Traits* | Description |
| -------------- | ----------- |
| `policy_traits<T>::selection_type`, `selection_t<T>` | The wrapped select type returned by `T`. Must satisfy [Selection](#selection_req_id). |
| `policy_traits<T>::wait_type`, `wait_type_t<T>` | The backend type that is returned by the user function object. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns an object of type `wait_type_t<T>`. |

We also propose to move the following function from optional to must be well-formed:

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |

This results in a greatly simplified contract for policies:

A Policy is an object with a valid dynamic selection heuristic.

The type `T` satisfies *Policy* if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| *Must* be well-formed | Description |
| --------------------- | ----------- |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |

| *Optional* | Description |
| --------------------- | ----------- |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits on the return value of the submission to complete. |

| Policy Traits* | Description |
| -------------- | ----------- |
| `policy_traits<T>::resource_type`, `resource_t<T>` | The backend-defined resource type that is passed to the user function object. |

The default implementation of these traits depends on types defined in the Policy:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using resource_type = typename std::decay_t<Policy>::resource_type;
  };
```
This would be a breaking change, but dynamic selection is an experimental API, so can modify the API in this way. However, we will want to consider this fully and perhaps investigate if there is any usage which we may break with these changes.

### TokenPolicy
The new TokenPolicy provides a way for users to control resources that require exclusive or limited access to individual resources. A capacity is set on initialization, and the policy selects the first available resource with a token slot available.

#### Implementation Approach
Calling the `submit` function for all policies so far returns quickly (before the job completes execution) and returns a wait type that matches the wait type for the backend. For TokenPolicy, this is more challenging because tokens from resources may not be available when jobs are submitted. 

We propose the following implementation Asynchronous Queueing Strategy: 

Add infrastructure to queue up jobs waiting for resources and asynchronously submit the job once resources become available. This queue will be managed by its own thread, and must synchronize with the resources to track when jobs complete and return their resources, so that the queue may then submit work.

The submission type we return must be aware and synchronized with the asynchronous queue. We must wrap this type with our own wait type, as the job's event may not be available at the time we return from the submit call. When wait is called on this wait type, it must wait for the job to be submitted and then wait on the resulting return from the submission.

The removal of the public API for selection greatly simplifies the token policy. Without this, each selection may be used for an arbitrary number of submissions or none at all. Also, when a selection handle still exists in the possession of a user, it may be used for a submission in the future. This means that we would need to track all jobs and selection handles, holding the resource's token until all are completed and/or go out of scope. Without a public selection API, we are guaranteed that each selection is mapped 1-to-1 with a submission and must only hold that resource token until the job completes.

### Benefits for Other Policies
Beyond simplifying the public interface and requirements, these changes may have some inherent benefits for existing policies in that they force users into a specific usage pattern. Selections and submissions must be paired 1-to-1, and selection should occur very close to submission time. For Dynamic Load Policy and Autotune Policy, which dynamically use statistics about resource load and performance of jobs, when selection occurs close to submission, the selection will be more accurate and up-to-date for the upcoming submission.

## Open Questions

- Should we instead require `submit_and_wait` as the required well-formed function and make the asynchronous submission optional?
    - If we did this, we could decline to implement the asynchronous `submit` call for `TokenPolicy` in favor of merely using `submit_and_wait`. This would allow us to avoid the asynchronous queue management, etc.  However, it would not allow asynchronous submission to this policy.

- Do we lose compelling use cases when removing `select` and related public API?
	- We lose the ability to submit multiple jobs to the same selection, but those jobs could be joined within a single submission instead.
	- We lose the ability to select and do something with the resource before submitting the job function.
	- We lose the ability to select and never submit to that selection.
