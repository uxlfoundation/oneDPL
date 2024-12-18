# Descriptive Name for the Proposal

## Introduction

Replace the text in this section with a short description of the proposed idea.

Explain the motivation for a proposal, such as:
- Improved user experience for API changes and extensions. Code snippets to
  showcase the benefits would be nice here.
- Performance improvements.
- Improved engineering practices.

The introduction may also include any additional information that sheds light on
the proposal, such as the history of the matter, links to relevant issues and
discussions, etc.

## Proposal

Replace the text in this section with a detailed description of the proposal
with highlighted consequences. The description can be iteratively clarified
as the proposal matures.

Depending on the kind of the proposal, the description may need to cover the following:

- New use cases supported by the extension.
- The expected performance benefit for a modification, supported with the data, if available.
- The API of extensions such as class definitions and function declarations.
- Key technical design aspects, sufficient to understand how the functionality should work
  and produce the desired outcome.

A proposal should clearly outline the alternatives that were considered, 
along with their pros and cons. Each alternative should be clearly separated 
to make discussions easier to follow. Or, if you prefer, list all the alternatives
first (perhaps as subsections), and then have a dedicated section with the discussion.

Pay close attention to the following aspects of the library:
- API and ABI backward compatibility. The library follows semantic versioning
  so if any of those interfaces are to be broken, the RFC needs to state that
  explicitly.
- Performance implications, as performance is one of the main goals of the library.
- Dependencies and supported platforms. Does the proposal bring any new
  dependencies or affect the supported configurations?
- Consistency and possible interaction with the existing library functionality.

Include short explanation and links to the related proposals, if any.
Sub-proposals could be organized as separate stand-alone RFCs, but this is
not mandatory. If the related change is insignificant or does not make any sense
without the original proposal, describe it in the main RFC.

Some other common subsections could be:
- Usage examples.
- Testing aspects.
- Next steps (e.g., design review, prototype, etc.), if approved.

## Open Questions

List any questions that are not sufficiently elaborated in the proposal,
need more discussion or prototyping experience, etc. Indicate at which state
(typically, "proposed" or "experimental") a question should be addressed.
