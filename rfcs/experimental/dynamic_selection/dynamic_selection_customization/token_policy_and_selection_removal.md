# Token Policy and Changes to Policy Contract

## Introduction
The key distinguishing feature between different dynamic selection policies is the method by which they select resources to submit jobs to. Currently, `select` is a standalone public API that can be called to obtain a resource, but it can also be invoked by calling `submit` or `submit_and_wait` directly. In this RFC, we propose to remove the standalone `select` API and introduce a new policy called TokenPolicy. TokenPolicy provides users with a way to specify hard concurrency limits on resources.

We have not found much utility in standalone `select` functionality. Relying on direct submission with implicit selection greatly simplifies Token Policy and also makes usage of DynamicLoadPolicy and AutotunePolicy more likely to be up-to-date with their instrumented metrics to make good selections.

## Proposal

### Removal of select routine
We propose to remove the following functions and traits from the public contract for a policy:

| *Must* be well-formed | Description |
| `p.select(args…)` | Returns `selection_t<T>` that satisfies [Selection](#selection_req_id). The selected resource must be within the set of resources returned by `p.get_resources()`. |
| `p.submit(s, f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function invokes `f` with the selected resource `s` and the arguments `args...`. |

| *Optional* | Description |
| `p.submit_and_wait(s, f, args…)` | Returns `void`. The function invokes `f` with `s` and `args...` and waits for the `wait_t<T>` it returns to complete. |

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `s` a selection of a type `selection_t<T>` , which satisfies [Selection](#selection_req_id), and was made by `p`.
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| Policy Traits* | Description |
| `policy_traits<T>::selection_type`, `selection_t<T>` | The wrapped select type returned by `T`. Must satisfy [Selection](#selection_req_id). |

We also propose that one of the following three functions must be well-formed, rather than optional:

| at least one *Must* be well-formed | Description |
| `p.try_submit(f, args…)` | Returns `std::optional<submission_t<T>>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. Returns empty optional if no resource is available for selection |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits on the return value of the submission to complete. |

This results in a greatly simplified contract for policies:

A Policy is an object with a valid dynamic selection heuristic.

The type `T` satisfies *Policy* if given,

- `p` an arbitrary identifier of type `T`
- `args` an arbitrary parameter pack of types `typename… Args`
- `f` a function object with signature `wait_t<T> fun(resource_t<T>, Args…);`

| *Must* be well-formed | Description |
| `p.get_resources()` | Returns a `std::vector<resource_t<T>>`. |

| One of the following *must* be well-formed | Description |
| `p.submit(f, args…)` | Returns `submission_t<T>` that satisfies [Submission](#submission_req_id). The function selects a resource and invokes `f` with the selected resource and `args...`. |
| `p.submit_and_wait(f, args…)` | Returns `void`. The function selects a resource, invokes `f` and waits on the return value of the submission to complete. |

| Policy Traits* | Description |
| `policy_traits<T>::resource_type`, `resource_t<T>` | The backend-defined resource type that is passed to the user function object. |
| `policy_traits<T>::has_async_submit` | Boolean that defines if a policy has an asynchronous submission function. |
| `policy_traits<T>::wait_type`, `wait_type_t<T>` | The backend type that is returned by the user function object. Calling `unwrap` on an object that satisfies [Submission](#submission_req_id) returns an object of type `wait_type_t<T>`. |

The default implementation of these traits depends on types defined in the Policy:

```cpp
  template <typename Policy>
  struct policy_traits
  {
      using resource_type = typename std::decay_t<Policy>::resource_type;
  };
```

With this contract, if `p.submit(f, args…)` is well-formed, a generic implementation of `submit_and_wait` that uses `submit` is available and waits on the result unless overridden. If `p.try_submit(f,args...)` is well-formed, then generic a generic implementation of `submit` which uses `try_submit` is available unless overridden.  Therefore, providing `try_submit` is enough to have implementations for all three submit variants automatically.

This would be a breaking change, but since dynamic selection is an experimental API, we can modify the API in this way. However, we will want to consider this fully and perhaps investigate if there is any usage that we may break with these changes.

### Removal of Wait Type
Not motivated from token policy, but rather from further simplifcation of the contract, we also propose to remove the wait_type trait. It is not necessary and can always be obtained with `decltype`, and `submit` with a valid user function. Of course, policies can provide a `wait_type` if they so choose, but it is not required. Requiring policies to provide this wait type increases complexity of customization and we have not seen a use for the public trait in use cases thus far.

### TokenPolicy
The new TokenPolicy provides a way for users to control resources that require exclusive or limited access to individual resources. A capacity is set on initialization, and the policy selects the first available resource with a token slot available.

The removal of the public API for selection greatly simplifies the token policy. Without this, each selection may be used for an arbitrary number of submissions or none at all. Also, when a selection handle still exists in the possession of a user, it may be used for a submission in the future. This means that we would need to track all jobs and selection handles, holding the resource's token until all are completed and/or go out of scope. Without a public selection API, we are guaranteed that each selection is mapped 1-to-1 with a submission and must only hold that resource token until the job completes.

In addition to the complexity of `select` within TokenPolicy, another difference between it and all previous policies is that resources are not guaranteed to be available at the time of submission. Below we offer three alternatives for addressing this.

#### Implementation Approach (1): No async submit
Our proposal for the implementation of TokenPolicy is to only provide `submit_and_wait` without an (async) `submit` function for token policy. This allows a greatly simplified implementation since resource tokens may not be available at the time of submission. With only a blocking call, we can merely wait for the token to become available and then wait for the job to complete.

#### Alternatives explored

##### Implementation Approach 2: Async submission management queue
Calling the `submit` function for all policies so far returns quickly (before the job completes execution) and returns a wait type that matches the wait type for the backend. For TokenPolicy, this is more challenging because tokens from resources may not be available when jobs are submitted.

Add infrastructure to queue up jobs waiting for resources and asynchronously submit the job once resources become available. This queue will be managed by its own thread and must synchronize with the resources to track when jobs complete and return their resources, so that the queue may then submit work.

The submission type we return must be aware of and synchronized with the asynchronous queue. We must wrap this type with our own wait type, as the job's event may not be available at the time we return from the submit call. When wait is called on this wait type, it must wait for the job to be submitted and then wait on the resulting return from the submission.

##### Implementation Approach 3: Allow submission to fail
We can provide an asynchronous `submit` call, but allow that call to fail, with some status flag in the submission object returned that would need to be queried. Currently, `submit` does not have the option of failing for any other policies, so this is extra complexity we would be adding to the API. However, it would allow us to provide both submit calls for TokenPolicy. If we do allow submit to fail, then we would not be able to generically implement `submit_and_wait`, at least without providing a similar fail status for that function as well.

### Benefits of Removing Select for Other Policies
Beyond simplifying the public interface and requirements, these changes may provide inherent benefits for existing policies by enforcing a specific usage pattern. With the removal of select interfaces, implicit selections and submissions must be paired 1-to-1, and implicit selection will occur very close to submission time. For Dynamic Load Policy and Autotune Policy, which dynamically use statistics about resource load and job performance, this means the implicit selection will be more accurate and up-to-date for the submission.

## Declined Feedback

- Bigger design alternative to TokenPolicy:
  Shift the responsibility of resource availability to the backend resource or backend, not policy. Policy becomes only about selection from available resource. Any selection mechanism (policy) can have resources with capacity caps. Use dynamic load after changes with limited resource availability in the backend or resources directly.
  - Declining this feedback as it goes against previous decision to keep universes static for the length of the program, and the cost of redesign is too high to consider at this time.


## Open Questions

- Do we lose compelling use cases when removing `select` and related public API?
	- We lose the ability to submit multiple jobs to the same selection, but those jobs could be joined within a single submission instead.
	- We lose the ability to select and do something with the resource before submitting the job function.
	- We lose the ability to select and never submit to that selection.

- Should we allow per-resource capacities rather than fixed capacities?
  - This is probably worthwhile, and should not be difficult to implement.

