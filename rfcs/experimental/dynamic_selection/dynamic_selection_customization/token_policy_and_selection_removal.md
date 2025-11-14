# Token Policy

## Introduction
The key distinguishing feature between different dynamic selection policies is the method by which they select resources to submit jobs to. This document proposes a new policy called TokenPolicy that provides users with a way to specify hard concurrency limits on resources. TokenPolicy is significantly simplified by the proposed removal of the standalone `select` API from the public contract for policies (see [Custom Policies](../customization/custom_policies.md#proposed-changes-to-policy-contract) for details on those changes).

The removal of the public API for selection greatly simplifies TokenPolicy. Without a public selection API, each selection is guaranteed to be mapped 1-to-1 with a submission and must only hold that resource token until the job completes, rather than requiring tracking of all jobs and selection handles throughout their lifetimes.

## Proposal

### TokenPolicy
The new TokenPolicy provides a way for users to control resources that require exclusive or limited access to individual resources. A capacity is set on initialization, and the policy selects the first available resource with a token slot available.

The removal of the public API for selection greatly simplifies the token policy. Without a public selection API, each selection is guaranteed to be mapped 1-to-1 with a submission and must only hold that resource token until the job completes, rather than requiring tracking of all jobs and selection handles throughout their lifetimes.

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

