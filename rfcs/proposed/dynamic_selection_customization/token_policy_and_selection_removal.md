# Token Policy and Changes to Policy Contract

## Introduction
The key distinguishing feature between different dynamic selection policies is the method in which they select resources
each to submit jobs to. Currently, select is a standalone public API which can be called to obtain a resource, but can
also be invoked by directly calling `submit` or `submit_and_wait` directly. In this RFC, we propose to remove the
standalone `select` API, as well as introduce a new policy, TokenPolicy. TokenPolicy provides users a way to specify
hard concurrancy limits on resources.

We have not found that there is much utility in a standalone `select` functionality. Relying on direct submission with
an implicit selection greatly simplifies Token Policy and also makes usage of DynamicLoadPolicy and AutotunePolicy more
likely to be up to date with their instrumented metrics to make good selections.

## Proposal

### Removal of select routine
We propose to remove the following functions and traits from the public contract for a policy:

| *Must* be well-formed | Description |
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
| ------- | ----------- |
| `policy_traits<T>::selection_type`, `selection_t<T>` | The wrapped select type returned by `T`. Must satisfy [Selection](#selection_req_id). |
| `policy_traits<T>::wait_type`, `wait_type_t<T>` | The backend type that is returned by the user function object. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns on object of type `wait_type_t<T>`. |

We also propose to move the following function from optional to must be well-formed:
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
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits for the `wait_t<T>` it returns to complete. |

| Policy Traits* | Description |
| ------- | ----------- |
| `policy_traits<T>::resource_type`, `resource_t<T>` | The backend defined resource type that is passed to the user function object. Calling `unwrap` an object of type `selection_t<T>` returns an object of type `resource_t<T>`. |

The default implementation of these traits depends on types defined in the Policy:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using resource_type = typename std::decay_t<Policy>::resource_type;
  };
```
### TokenPolicy
The new TokenPolicy provides a way for users to control resources which require exclusive or limited access to
individual resources. A capacity is set on initialization, and the policy selects the first available policy with a
token slot available.

Calling the `submit` function for all policies so far returns quickly (before the job completes execution), and returns
a wait type which matches the wait type for the backend. For a TokenPolicy this is more challenging, because tokens from
resources may not be available when jobs are submitted. Either we must block until the job can acquire a resource and
submit the job to that resource or we must add infrastructure to queue up jobs waiting for resources and asynchronously
submit the job once resources become available. Additionally, we cannot simply return the same wait type that the user's
job returns. We must wrap this type with our own wait type as the job's event may not be available at the time we return
from the submit call. When wait is called on this wait type, it must wait for the the job to be submitted, and then wait
on the resulting return from the submission.

The removal of the public API for selection greatly simplifies the token policy. Without this, each selection may be
used for an arbitrary number of submissions or none at all. Also, when a selection handle still exists in the posession
of a user, it may be used for a submission in the future. This means that we would need to track all jobs and selection
handles, holding the resource's token until all are completed and/or left scope. Without a public selection API, we are
guaranteed each selection is mapped 1-to-1 with a submission and must only hold that resource token until the job
completes.

### Benefits for Other Policies
Beyond simplifying the public interface and requirements, these changes may have some inherent benefits for existing
policies in that it forces users into a specific usage pattern. Selection and submissions must be paired 1-to-1,
selection should occur very close to submission time. For Dynamic Load Policy and Autotune Policy dynamically use
statistics about resource load and performance of jobs. When selection occurs close to submission, the selection will
be more accurate and up to date for the upcoming submission.

## Open Questions
- Should we instead require `submit_and_wait` as the required well-formed function, and make the asynchronous submission
   optional?
- Do we lose compelling use cases when removing `select` and related public API?
	- We lose the ability to submit multiple jobs to the same selection, but those jobs could be joined within a single
	submission instead.
	- We lose the ability to select and do something with the resource before the submitting job function.
	- We lose the ability to select and never submit to that selection.
	