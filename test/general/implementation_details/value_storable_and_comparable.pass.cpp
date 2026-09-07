// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) 2025 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

// Compile-time checks for oneapi::dpl::__unseq_backend::__is_value_storable_and_comparable_v and for the requirements
// it is built from: oneapi::dpl::__unseq_backend::__is_brace_constructible_v and oneapi::dpl::__internal::
// __convertible_to_v and __predicate_v. Every requirement is checked both ways: a type that satisfies it and a type that
// does not.

#include "support/test_config.h"

#include <oneapi/dpl/pstl/unseq_backend_simd.h>
#include <oneapi/dpl/pstl/utils.h>

#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "support/utils.h"

namespace dpl_internal = oneapi::dpl::__internal;
namespace dpl_unseq = oneapi::dpl::__unseq_backend;

//----------------------------------------------------------------------------//
// Value types
//----------------------------------------------------------------------------//

// Satisfies every requirement: default-constructible, copy-constructible, copy-assignable, less-than comparable.
struct Regular
{
    int val = 0;
    bool
    operator<(const Regular& other) const
    {
        return val < other.val;
    }
};

// An explicit default constructor is enough, since _ValueType{} is a direct initialization, which may use it.
struct ExplicitDefaultCtor
{
    int val;
    explicit ExplicitDefaultCtor() : val(0) {}
    bool
    operator<(const ExplicitDefaultCtor& other) const
    {
        return val < other.val;
    }
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
    bool
    operator<(const AggregateOfExplicitDefaultCtor& other) const
    {
        return member.val < other.member.val;
    }
};

struct NoDefaultCtor
{
    int val;
    explicit NoDefaultCtor(int v) : val(v) {}
    bool
    operator<(const NoDefaultCtor& other) const
    {
        return val < other.val;
    }
};

struct NoCopyAssign
{
    int val = 0;
    NoCopyAssign() = default;
    NoCopyAssign(const NoCopyAssign&) = default;
    NoCopyAssign&
    operator=(const NoCopyAssign&) = delete;
    bool
    operator<(const NoCopyAssign& other) const
    {
        return val < other.val;
    }
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
    bool
    operator<(const VoidAssign& other) const
    {
        return val < other.val;
    }
};

// The destructor is not noexcept, which is enough here because storing a value never has to be non-throwing.
struct ThrowingDtor
{
    int val = 0;
    ~ThrowingDtor() noexcept(false) {}
    bool
    operator<(const ThrowingDtor& other) const
    {
        return val < other.val;
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
    bool
    operator<(const MoveOnly& other) const
    {
        return val < other.val;
    }
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
    bool
    operator<(const CopyOnlyNoMove& other) const
    {
        return val < other.val;
    }
};

//----------------------------------------------------------------------------//
// __is_brace_constructible_v
//----------------------------------------------------------------------------//

static_assert(dpl_unseq::__is_brace_constructible_v<int>);
static_assert(dpl_unseq::__is_brace_constructible_v<int*>);
static_assert(dpl_unseq::__is_brace_constructible_v<Regular>);
static_assert(dpl_unseq::__is_brace_constructible_v<ExplicitDefaultCtor>);
static_assert(dpl_unseq::__is_brace_constructible_v<MoveOnly>);

static_assert(std::is_default_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<AggregateOfExplicitDefaultCtor>);
static_assert(!dpl_unseq::__is_brace_constructible_v<NoDefaultCtor>);
// void{} is a valid expression, so this requirement does not reject void: that is done separately.
static_assert(dpl_unseq::__is_brace_constructible_v<void>);

//----------------------------------------------------------------------------//
// __convertible_to_v
//----------------------------------------------------------------------------//

struct ExplicitFromInt
{
    explicit ExplicitFromInt(int) {}
};

// A destination whose only constructor taking ImplicitSource is deleted and explicit: copy-initialization ignores it
// and picks the conversion operator, so std::is_convertible_v is satisfied, while static_cast selects the deleted
// constructor. This is the difference std::convertible_to catches and std::is_convertible_v does not.
struct ExplicitlyNotConvertible;

struct ImplicitSource
{
    operator ExplicitlyNotConvertible() const;
};

struct ExplicitlyNotConvertible
{
    ExplicitlyNotConvertible() = default;
    explicit ExplicitlyNotConvertible(ImplicitSource) = delete;
};

static_assert(dpl_internal::__convertible_to_v<int, int>);
static_assert(dpl_internal::__convertible_to_v<const int&, int>);
static_assert(dpl_internal::__convertible_to_v<int&, long>);
static_assert(dpl_internal::__convertible_to_v<const Regular&, Regular>);
static_assert(dpl_internal::__convertible_to_v<std::pair<int&, int&>, std::pair<int, int>>);

static_assert(!dpl_internal::__convertible_to_v<int*, int>);
static_assert(!dpl_internal::__convertible_to_v<Regular, int>);
static_assert(!dpl_internal::__convertible_to_v<int, ExplicitFromInt>);
static_assert(!dpl_internal::__convertible_to_v<const MoveOnly&, MoveOnly>);
static_assert(std::is_convertible_v<ImplicitSource, ExplicitlyNotConvertible>);
static_assert(!dpl_internal::__convertible_to_v<ImplicitSource, ExplicitlyNotConvertible>);

//----------------------------------------------------------------------------//
// __predicate_v
//----------------------------------------------------------------------------//

struct NotBool
{
};

struct IntResultLess
{
    int
    operator()(const int& lhs, const int& rhs) const
    {
        return lhs < rhs;
    }
};

struct NotBoolResultLess
{
    NotBool
    operator()(const int&, const int&) const
    {
        return NotBool{};
    }
};

// Requires modifiable arguments, so it cannot be called on const values.
struct MutableRefLess
{
    bool
    operator()(int& lhs, int& rhs) const
    {
        return lhs < rhs;
    }
};

// Callable on an rvalue only, while the requirement is stated for _Compare&.
struct RvalueOnlyLess
{
    bool
    operator()(const int&, const int&) &&
    {
        return false;
    }
};

struct UnaryLess
{
    bool
    operator()(const int&) const
    {
        return false;
    }
};

// Not copyable: the requirement is stated for _Compare&, so it must not ask for a copy.
struct MoveOnlyLess
{
    MoveOnlyLess() = default;
    MoveOnlyLess(MoveOnlyLess&&) = default;
    MoveOnlyLess(const MoveOnlyLess&) = delete;
    bool
    operator()(const int& lhs, const int& rhs) const
    {
        return lhs < rhs;
    }
};

static_assert(dpl_internal::__predicate_v<std::less<int>&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<std::less<>&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<IntResultLess&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<MoveOnlyLess&, const int&, const int&>);
static_assert(dpl_internal::__predicate_v<std::less<Regular>&, const Regular&, const Regular&>);

static_assert(!dpl_internal::__predicate_v<NotBoolResultLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<MutableRefLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<RvalueOnlyLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<UnaryLess&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<int&, const int&, const int&>);
static_assert(!dpl_internal::__predicate_v<std::less<int>&, const Regular&, const Regular&>);

//----------------------------------------------------------------------------//
// __is_value_storable_and_comparable_v
//----------------------------------------------------------------------------//

// An iterator whose reference type is not convertible to its value type.
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

// Accepted: the value type is storable and the comparator is callable on const values.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<const int*, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<int>::iterator, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<int>::const_iterator, std::less<>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<Regular*, std::less<Regular>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ExplicitDefaultCtor*, std::less<ExplicitDefaultCtor>>);
// Accepted: the requirements are default construction, copy construction and copy assignment, and nothing else.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<CopyOnlyNoMove*, std::less<CopyOnlyNoMove>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<VoidAssign*, std::less<VoidAssign>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<ThrowingDtor*, std::less<ThrowingDtor>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, MoveOnlyLess>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, IntResultLess>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, bool (*)(const int&, const int&)>);
// The comparator max_element passes down to the min_element brick.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<int*, dpl_internal::__reorder_pred<std::less<int>>>);
// A proxy reference is fine as long as it converts to the value type.
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<std::vector<bool>::iterator, std::less<bool>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<int, int>, std::less<int>>);
static_assert(dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<std::pair<int, int>, std::pair<int&, int&>>,
                                                             std::less<std::pair<int, int>>>);

// Rejected because of the value type: brace initialization, copy assignment and copy construction respectively.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<NoDefaultCtor*, std::less<NoDefaultCtor>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<AggregateOfExplicitDefaultCtor*,
                                                              std::less<AggregateOfExplicitDefaultCtor>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<NoCopyAssign*, std::less<NoCopyAssign>>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<MoveOnly*, std::less<MoveOnly>>);

// Rejected because the reference type does not convert to the value type.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<FakeIterator<int, OpaqueRef>, std::less<int>>);

// Rejected because an output iterator reports void as its value type.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<std::back_insert_iterator<std::vector<int>>,
                                                              std::less<int>>);

// Rejected because of the comparator.
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, NotBoolResultLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, MutableRefLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, RvalueOnlyLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, UnaryLess>);
static_assert(!dpl_unseq::__is_value_storable_and_comparable_v<int*, std::less<Regular>>);

int
main()
{
    return TestUtils::done();
}
