# Design Documents for Experimental Features

Experimental proposals describe extensions that are implemented and
released as experimental features in the library. An experimental
feature is expected to have an implementation that is of comparable quality
to a fully supported feature. Sufficient tests are required.

An experimental feature does not yet appear as part of the oneDPL
specification. Therefore, the interface and design can change.
There is no commitment to backward compatibility for experimental features.

The documents in this directory should include a list of the exit conditions
that need to be met to move the functionality from experimental to fully supported.
These conditions might include demonstrated performance improvements, demonstrated
interest from the community, acceptance of the required specification changes, etc.

A document here needs to be updated if the corresponding feature undergoes
modifications while remaining experimental. Other changes, such as updates on the
exit conditions or on the implementation and usage experience, are also welcome.

For features that require specification changes prior to production, the document might
include wording for those changes or a link to any PRs opened against the specification.

Proposals in the `rfcs/experimental` directory do not remain there indefinitely.
They should move either to `rfcs/supported` when they become fully supported
or to `rfcs/archived` if the corresponding feature is not finally accepted but removed.
As a general rule, a proposal should not stay in the experimental folder
for longer than a year or two.
