# Introduction

This document defines the roles available in the oneDPL project and provides a list of the
[Current Code Owners](#currentcodeowners) and [Current Maintainers](#currentmaintainers) for each area of development
in oneDPL. The document also defines the process for maintainers to [Leave A Role](#leavearole) in the project.

# Roles and Responsibilities

The oneDPL project defines three primary roles, each with their own responsibilities and privileges:
* [Contributor](#contributor)
* [Code Owner](#codeowner)
* [Maintainer](#maintainer)

[permissions]: https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization#permissions-for-each-role

|                                                                                                                         _Responsibilities_  |       Contributor       |       Code Owner        |       Maintainer        |
| :------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------: | :---------------------: | :---------------------: |
|                                                                                                                                             |                         |                         |                         |
| Follow the Code of Conduct                                                                                                                  |            ✓            |            ✓           |            ✓            |
| Follow Contribution Guidelines                                                                                                              |            ✓            |            ✓           |            ✓            |
| Ensure Contribution Guidelines are followed                                                                                                 |            ✗            |            ✓           |            ✓            |
| Cleanup Stale PRs and Branches impacting a component                                                                                        |            ✗            |            ✓           |            ✓            |
| Co-own component or aspect of the library,<br>  including contributing: bug fixes, implementing features,<br> and performance optimizations |            ✗            |            ✓           |            ✓            |
| Co-own on technical direction of component or<br> aspect of the library                                                                     |            ✗            |            ✓           |            ✓            |
| Co-own the project as a whole,<br> including determining strategy and policy for the project                                                |            ✗            |            ✗           |            ✓            |
|                     |                         |                        |                         |
|                                                                                                                         **_Privileges_**    |     **Contributor**     |     **Code Owner**      |     **Maintainer**      |
|                                                                                                                         Permission granted  |   [Read][permissions]/[Write][permissions]   |   [Write][permissions]  | [Maintain][permissions] |
| Can Approve PRs     |            ✗             |             ✓           |            ✓            |
| Eligible to become                                                                                                                          |       Code Owner        |       Maintainer        |            ✗            |
| Can recommend Contributors<br> to become Code Owner                                                                                         |            ✗            |            ✓           |            ✓            |
| Can participate in promotions of<br> Code Owners and  Maintainers                                                                           |            ✗            |            ✗           |            ✓            |
| Can suggest PRs/issues for<br> Milestones during planning                                                                                   |            ✓            |            ✓           |            ✓            |
| Can choose Milestones for specific component                                                                                                |            ✗            |            ✓           |            ✓            |
| Choose project's Milestones during planning                                                                                                 |            ✗            |            ✗           |            ✓            |
| Can propose new RFC or<br> participate in review of existing RFC                                                                            |            ✓            |            ✓           |            ✓            |
| Can request rework of RFCs<br> in represented area of responsibility                                                                        |            ✗            |            ✓           |            ✓            |
| Can request rework of RFCs<br> in any part of the project                                                                                   |            ✗            |            ✗           |            ✓            |
| Can manage release process of the project                                                                                                   |            ✗            |            ✗           |            ✓            |
| Can represent the project in public as a Maintainer                                                                                         |            ✗            |            ✗           |            ✓            |

These roles are merit based. Refer to the corresponding section for specific
requirements and the nomination process.

## <a name="contributor"></a>Contributor

This role is open to anyone that would like to become actively involved in the oneDPL project. A Contributor can invest
their time and resources in several different ways to improve oneDPL
* Answering questions from community members in Discussions on Issues
* Providing feedback on RFC pull requests and Discussions
* Reviewing and/or testing pull requests
* Contribute code, including bug fixes, tests, new examples of oneDPL use, and new feature implementations
* Contribute design proposals as RFCs

Responsibilities:
* Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
* Submit issues, feature requests, and code in accordance with the [Contribution Guidelines](CONTRIBUTING.md)

Privileges:
* Code contributions recognized in the oneDPL [credits](CREDITS.txt)
* Eligible to become a Code Owner
* May request write permissions to the repository from the oneDPL Maintainers.
* Can suggest PRs and issues to be included in Milestones during planning

## <a name="codeowner"></a>Code Owner

A Code Owner is an established contributor that is recognized by active code owners and maintaininers as capable of
taking on additional activities and the responsibilities and privileges that come with them. A Code Owner is
responsible for a specific oneDPL component or functional area of the project. Code Owners are collectively responsible,
with other Code Owners, for developing and maintaining their component or functional areas, including reviewing all
changes to the corresponding areas of responsibility and indicating whether those changes are ready to be merged. Code
Owners have a track record of code contribution and reviews in the project.

Requirements:
* Track record of accepted code contributions to a specific project component.
* Track record of contributions to the code review process.
* Demonstrated in-depth knowledge of the architecture of a specific project
  component.
* Commits to being responsible for that specific area.

Responsibilities in addition to those of Contributors:
* Ensure the Contribution Guidelines are followed in contributions impacting the Code Owner's component
* Cleanup stale PRs and branches that have work impacting their component
* Participate in the discussion of RFCs that impact their component
* Co-own a component or aspect of the repository, including ensuring bugs are addressed, features are well tested,
  requested features from RFCs are implemented, and the performance of component APIs is optimized.

Privileges:
* Eligible to become a Maintainer
* Write permissions granted to the repository
* Can recommend Contributors to become Code Owners
* Can choose PRs and issues to be included in Milestones for a specific component
* Can propose new RFC or participate in review of existing RFCs

The process of becoming a Code Owner is:
1. A Contributor is nominated by opening a PR modifying the MAINTAINERS.md file including project component along with
contributor name, Github username, and affiliation.
2. At least two specific component Code Owners or Maintainers approve the PR.
3. MAINTAINERS.md file is updated to represent corresponding areas of responsibility.

## <a name="maintainer"></a>Maintainer

A Maintainer is one of the most established contributors who are responsible for the project technical direction and
participate in making decisions about strategy and priorities of the project. The responsibilities they have increase
in scope from that of Code Owners to encompass the entire project. Maintainers may make public presentations and
statements in public forums (e.g., UXL DevSummit, UXL working group meetings, etc.) about the status and direction of
the oneDPL project.

Requirements:
  * Experience as a Code Owner.
  * Track record of major project contributions to a specific project component.
  * Demonstrated deep knowledge of a specific project component.
  * Demonstrated broad knowledge of the project across multiple areas.
  * Commits to using privileges responsibly for the good of the project.
  * Is able to exercise judgment for the good of the project, independent of
    their employer, friends, or team.

Responsibilities in addition to those of Code Owners:
* Co-own the technical direction of a component or aspect of the library
* Co-own the project as a whole, including determining strategy and policy for the project

Privileges:
* Maintain permissions granted to the repository
* Can participate in promotions of Code Owners and Maintainers
* Choose project's Milestones during planning
* Can request rework of RFCs in any part of the project
* Can manage release process of the project
* Can represent the project in public as a Maintainer

The process of becoming a Maintainer is:
1. A Code Owner is nominated by a current Maintainer by opening a PR modifying the MAINTAINERS.md file to move the code owner into the Current Maintainers section.
2. A majority of the Current Maintainers must approve the PR.

## <a name="leavearole"></a>Leave A Role

Active code owners and maintainers of the oneDPL project may leave their current role by submitting a PR removing their
name from the list of current maintainers or code owners and tagging two or more active maintainers to review and
approve the PR.

## <a name="currentcodeowners"></a>Current Code Owners

| Component             | Code Owner(s)       | Github ID | Affiliation |
| --------------------- | ------------------- | --------- | ----------- |
| C++ standard policies and CPU backends | Mikhail Dvorskiy<br>Dan Hoeflinger | @MikeDvorskiy<br>@danhoeflinger | Intel<br>Intel |
| SYCL device policies and SYCL backends | Mikhail Dvorskiy<br>Adam Fidel | @MikeDvorskiy<br>@adamfidel |  Intel |
| Tested Standard C++ APIs & Utility Function Object Classes | Sergey Kopienko | @SergeyKopeinko | Intel |
| C++17 standard algorithms | Mikhail Dvorskiy<br>Adam Fidel<br>SergeyKopienko<br>Julian Miller<br>Dmitriy Sobolev | @MikeDvorskiy<br>@adamfidel<br>@SergeyKopienko<br>@julianmi<br>@dmitriy-sobolev | Intel<br>Intel<br>Intel<br>Independent<br>Intel |
| Range-Based Algorithms | Mikhail Dvorskiy<br>Ruslan Arutyunyan | @MikeDvorskiy<br>@rarutyun | Intel<br>Intel |
| Asynchronous API Algorithms | Pablo Reble | @reble | Intel |
| Additional Algorithms | Dan Hoeflinger<br>Matt Michel | @danhoeflinger<br>@mmichel11 | Intel<br>Intel |
| Iterators | Dan Hoeflinger | @danhoeflinger | Intel |
| Random Number Generators | Pavel Dyakov | @paveldyakov | Intel |
| Dynamic Selection API | Anuya Welling | @AnuyaWelling2801 | AMD |
| Kernel Templates API | Sergey Kopienko<br>Dmitriy Sobolev | @SergeyKopienko<br>@dmitriy-sobolev | Intel<br>Intel |

## <a name="currentmaintainers"></a>Current Maintainers

| Maintainer          | Github ID | Affiliation |
| ------------------- | --------- | ----------- |
| Mikhail Dvorskiy | @MikeDvorskiy | Intel |
| Dan Hoeflinger   | @danhoeflinger | Intel |
| Sergey Kopienko  | @SergeyKopeinko | Intel |
| Alexey Kukanov   | @akukanov | Intel |
| Timmie Smith     | @timmiesmith | Intel |
| Dmitriy Sobolev  | @dmitriy-sobolev | Intel |
