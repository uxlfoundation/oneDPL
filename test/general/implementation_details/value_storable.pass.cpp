// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__unseq_backend::__is_value_storable_v and __is_brace_constructible_v.

#include "support/test_config.h"

#include <oneapi/dpl/pstl/unseq_backend_simd.h>

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "support/utils.h"

namespace dpl_unseq = oneapi::dpl::__unseq_backend;

//----------------------------------------------------------------------------//
// Value types
//----------------------------------------------------------------------------//

// Satisfies every requirement: default-constructible, copy-constructible, copy-assignable.
struct Regular
{
    int val = 0;
};

struct ExplicitDefaultCtor
{
    int val;
    explicit ExplicitDefaultCtor() : val(0) {}
};

// Default-constructible, but not brace-initializable: the member is copy-initialized from an empty list, which may not
// use its explicit default constructor.
struct AggregateOfExplicitDefaultCtor
{
    ExplicitDefaultCtor member;
};

// Brace-initializable, but not default-constructible: empty braces select the initializer-list constructor.
struct BraceInitOnly
{
    int val;
    BraceInitOnly(std::initializer_list<int> init) : val(init.size() == 0 ? 0 : *init.begin()) {}
};

// A type that is not default-constructible is taken from the test utilities: TestUtils::NoDefaultCtorWrapper<int>.

struct NoCopyAssign
{
    int val = 0;
    NoCopyAssign() = default;
    NoCopyAssign(const NoCopyAssign&) = default;
    NoCopyAssign&
    operator=(const NoCopyAssign&) = delete;
};

// The copy assignment returns void instead of VoidAssign&.
struct VoidAssign
{
    int val = 0;
    void
    operator=(const VoidAssign& other)
    {
        val = other.val;
    }
};

struct MoveOnly
{
    int val = 0;
    MoveOnly() = default;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly&
    operator=(MoveOnly&&) = default;
    MoveOnly(const MoveOnly&) = delete;
    MoveOnly&
    operator=(const MoveOnly&) = delete;
};

// Copyable, but with deleted move operations.
struct CopyOnlyNoMove
{
    int val = 0;
    CopyOnlyNoMove() = default;
    CopyOnlyNoMove(const CopyOnlyNoMove&) = default;
    CopyOnlyNoMove&
    operator=(const CopyOnlyNoMove&) = default;
    CopyOnlyNoMove(CopyOnlyNoMove&&) = delete;
    CopyOnlyNoMove&
    operator=(CopyOnlyNoMove&&) = delete;
};

// Copyable and assignable from a const lvalue only.
struct ConstCopyOnly
{
    int val = 0;
    ConstCopyOnly() = default;
    ConstCopyOnly(const ConstCopyOnly&) = default;
    ConstCopyOnly&
    operator=(const ConstCopyOnly&) = default;
    ConstCopyOnly(ConstCopyOnly&) = delete;
    ConstCopyOnly&
    operator=(ConstCopyOnly&) = delete;
};

// The copy constructor is explicit, so the type is copy-constructible, but its elements cannot be copy-initialized.
struct ExplicitCopyCtor
{
    int val = 0;
    ExplicitCopyCtor() = default;
    explicit ExplicitCopyCtor(const ExplicitCopyCtor& other) : val(other.val) {}
    ExplicitCopyCtor&
    operator=(const ExplicitCopyCtor&) = default;
};

//----------------------------------------------------------------------------//
// __is_brace_constructible_v
//----------------------------------------------------------------------------//

static_assert(dpl_unseq::__is_brace_constructible_v<int>);
static_assert(dpl_unseq::__is_brace_constructible_v<int*>);
static_assert(dpl_unseq::__is_brace_constructible_v<Regular>);
static_assert(dpl_unseq::__is_brace_constructible_v<ExplicitDefaultCtor>);
static_assert(dpl_unseq::__is_brace_constructible_v<MoveOnly>);

// Brace initialization differs from default construction in both directions.
static_assert(std::is_default_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!std::is_default_constructible_v<BraceInitOnly>);
static_assert(dpl_unseq::__is_brace_constructible_v<BraceInitOnly>);

static_assert(!dpl_unseq::__is_brace_constructible_v<TestUtils::NoDefaultCtorWrapper<int>>);

//----------------------------------------------------------------------------//
// Reference types
//----------------------------------------------------------------------------//

// A reference type that does not convert to the value type.
struct OpaqueRef
{
};

template <typename _ValueType, typename _ReferenceType>
struct FakeIterator
{
    using iterator_category = std::random_access_iterator_tag;
    using value_type = _ValueType;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = _ReferenceType;

    reference
    operator*() const;
};

//----------------------------------------------------------------------------//
// __is_value_storable_v
//----------------------------------------------------------------------------//

// Accepted value types.
static_assert(dpl_unseq::__is_value_storable_v<int*>);
static_assert(dpl_unseq::__is_value_storable_v<const int*>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::const_iterator>);
static_assert(dpl_unseq::__is_value_storable_v<Regular*>);
static_assert(dpl_unseq::__is_value_storable_v<ExplicitDefaultCtor*>);
static_assert(dpl_unseq::__is_value_storable_v<BraceInitOnly*>);
// The requirements are brace initialization, copy construction and copy assignment, and nothing else.
static_assert(dpl_unseq::__is_value_storable_v<CopyOnlyNoMove*>);
static_assert(dpl_unseq::__is_value_storable_v<VoidAssign*>);
static_assert(dpl_unseq::__is_value_storable_v<const ConstCopyOnly*>);
// The reference type is not part of the requirement.
static_assert(dpl_unseq::__is_value_storable_v<std::vector<bool>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, int>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<CopyOnlyNoMove, CopyOnlyNoMove>>);
// Accepted although the bricks do not compile for them: these iterators do not meet the requirements of a forward
// iterator, which is not detected here.
static_assert(std::is_copy_constructible_v<ExplicitCopyCtor>);
static_assert(dpl_unseq::__is_value_storable_v<ExplicitCopyCtor*>);
static_assert(dpl_unseq::__is_value_storable_v<ConstCopyOnly*>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, OpaqueRef>>);

// Rejected because of the value type: the first two fail brace initialization, the third copy assignment, and the last
// one copy construction.
static_assert(!dpl_unseq::__is_value_storable_v<TestUtils::NoDefaultCtorWrapper<int>*>);
static_assert(!dpl_unseq::__is_value_storable_v<AggregateOfExplicitDefaultCtor*>);
static_assert(!dpl_unseq::__is_value_storable_v<NoCopyAssign*>);
static_assert(!dpl_unseq::__is_value_storable_v<MoveOnly*>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_v<std::back_insert_iterator<std::vector<int>>>);

int
main()
{
    return TestUtils::done();
}
