// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__unseq_backend::__is_value_storable_v and for the requirements it is built
// from.
// The only one of them that is not a standard type trait, oneapi::dpl::__unseq_backend::__is_brace_constructible_v, is
// checked on its own as well. Every requirement is checked both ways: a type that satisfies it and a type that does
// not.

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

// An explicit default constructor is enough, since _ValueType{} is a direct initialization, which may use it.
struct ExplicitDefaultCtor
{
    int val;
    explicit ExplicitDefaultCtor() : val(0) {}
};

struct ExplicitDefaultCtorMember
{
    int val;
    explicit ExplicitDefaultCtorMember() : val(0) {}
};

// Default-constructible, but not brace-initializable: an aggregate is initialized member by member, and the member is
// copy-initialized from an empty list, which may not use its explicit default constructor.
struct AggregateOfExplicitDefaultCtor
{
    ExplicitDefaultCtorMember member;
};

// Brace-initializable, but not default-constructible: with no default constructor declared, empty braces select the
// initializer-list constructor with an empty list, while _ValueType() is ill-formed. The reduction object initializes
// its members with _ValueType{}, so this is enough for it.
struct BraceInitOnly
{
    int val;
    BraceInitOnly(std::initializer_list<int> init) : val(init.size() == 0 ? 0 : *init.begin()) {}
};

struct NoDefaultCtor
{
    int val;
    explicit NoDefaultCtor(int v) : val(v) {}
};

struct NoCopyAssign
{
    int val = 0;
    NoCopyAssign() = default;
    NoCopyAssign(const NoCopyAssign&) = default;
    NoCopyAssign&
    operator=(const NoCopyAssign&) = delete;
};

// The assignment does not return VoidAssign&, which is enough here because the result is never used.
struct VoidAssign
{
    int val = 0;
    void
    operator=(const VoidAssign& other)
    {
        val = other.val;
    }
};

// The destructor is not noexcept, which is enough here because storing a value never has to be non-throwing.
struct ThrowingDtor
{
    int val = 0;
    ~ThrowingDtor() noexcept(false) {}
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

// Deleting the move operations while keeping the copy ones is enough here, because the value is never moved.
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

// Copyable and assignable from a const lvalue only, which is enough here, because the candidates are read through
// std::as_const and the element is materialized as a const _ValueType.
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

// A value type whose copy constructor is explicit, which is enough for copying the candidates, because they are copied
// by direct initialization, and so is std::is_copy_constructible_v defined. Copy-initializing an element of such a type
// is ill-formed, so an iterator over it does not meet the requirements of a forward iterator.
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

// The requirement is brace initialization, and the two directions in which it differs from default construction are
// both checked: a type which is default-constructible but not brace-initializable, and one which is the other way
// round.
static_assert(std::is_default_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!std::is_default_constructible_v<BraceInitOnly>);
static_assert(dpl_unseq::__is_brace_constructible_v<BraceInitOnly>);

static_assert(!dpl_unseq::__is_brace_constructible_v<NoDefaultCtor>);

//----------------------------------------------------------------------------//
// Reference types
//----------------------------------------------------------------------------//

// A reference type that does not convert to the value type: an iterator reporting it does not meet the requirements of
// a forward iterator, which state that *__first is convertible to the value type, so the requirement does not look at
// the reference type at all.
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

// Accepted: the value type is storable in the reduction object.
static_assert(dpl_unseq::__is_value_storable_v<int*>);
static_assert(dpl_unseq::__is_value_storable_v<const int*>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<std::vector<int>::const_iterator>);
static_assert(dpl_unseq::__is_value_storable_v<Regular*>);
static_assert(dpl_unseq::__is_value_storable_v<ExplicitDefaultCtor*>);
static_assert(dpl_unseq::__is_value_storable_v<BraceInitOnly*>);
// Accepted: the requirements are brace initialization, copy construction and copy assignment, and nothing else.
static_assert(dpl_unseq::__is_value_storable_v<CopyOnlyNoMove*>);
static_assert(dpl_unseq::__is_value_storable_v<VoidAssign*>);
static_assert(dpl_unseq::__is_value_storable_v<ThrowingDtor*>);
// Accepted: copying and storing a value only ever reads it through a const reference.
static_assert(dpl_unseq::__is_value_storable_v<const ConstCopyOnly*>);
// The reference type is not part of the requirement, so a proxy reference and an iterator returning the value type by
// value are accepted like any other.
static_assert(dpl_unseq::__is_value_storable_v<std::vector<bool>::iterator>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, int>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<CopyOnlyNoMove, CopyOnlyNoMove>>);
// Accepted although the bricks do not compile for them: an element of these iterators cannot be copy-initialized into
// the value type, so they do not meet the requirements of a forward iterator, which is not detected here. The value
// types themselves are copy-constructible, which is stated in terms of direct initialization.
static_assert(std::is_copy_constructible_v<ExplicitCopyCtor>);
static_assert(dpl_unseq::__is_value_storable_v<ExplicitCopyCtor*>);
static_assert(dpl_unseq::__is_value_storable_v<ConstCopyOnly*>);
static_assert(dpl_unseq::__is_value_storable_v<FakeIterator<int, OpaqueRef>>);

// Rejected because of the value type: the first two fail brace initialization, the third copy assignment, and the
// move-only one copy construction, and with it every other requirement that copies a value.
static_assert(!dpl_unseq::__is_value_storable_v<NoDefaultCtor*>);
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
