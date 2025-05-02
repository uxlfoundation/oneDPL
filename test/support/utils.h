// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _UTILS_H
#define _UTILS_H

// File contains common utilities that tests rely on

// Do not #include <algorithm>, because if we do we will not detect accidental dependencies.
#include "test_config.h"

#include _PSTL_TEST_HEADER(execution)

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <cmath>
#include <complex>
#include <type_traits>
#include <memory>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>
#include <limits>
#include <cassert>

#include "utils_const.h"
#include "iterator_utils.h"
#include "utils_sequence.h"
#include "utils_test_base.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "utils_sycl.h"
#    include "oneapi/dpl/experimental/kt/kernel_param.h"
#endif

namespace TestUtils
{

typedef double float64_t;
typedef float float32_t;

template <class T, ::std::size_t N>
constexpr size_t
const_size(const T (&)[N]) noexcept
{
    return N;
}

// Handy macros for error reporting
#define EXPECT_TRUE(condition, message) ::TestUtils::expect(true, condition, __FILE__, __LINE__, message)
#define EXPECT_FALSE(condition, message) ::TestUtils::expect(false, condition, __FILE__, __LINE__, message)

// Check that expected and actual are equal and have the same type.
#define EXPECT_EQ(expected, actual, message)                                                                           \
    ::TestUtils::expect_equal_val(expected, actual, __FILE__, __LINE__, message)

// Check that sequences started with expected and actual and have had size n are equal and have the same type.
#define EXPECT_EQ_N(expected, actual, n, message)                                                                      \
    ::TestUtils::expect_equal(expected, actual, n, __FILE__, __LINE__, message)

// Check the expected and actual ranges are equal.
#define EXPECT_EQ_RANGES(expected, actual, message)                                                                    \
    ::TestUtils::expect_equal(expected, actual, __FILE__, __LINE__, message)

// Issue error message from outstr, adding a newline.
// Real purpose of this routine is to have a place to hang a breakpoint.
inline void
issue_error_message(::std::stringstream& outstr)
{
    outstr << ::std::endl;
    ::std::cerr << outstr.str();
    ::std::exit(EXIT_FAILURE);
}

inline void
expect(bool expected, bool condition, const char* file, std::int32_t line, const char* message)
{
    if (condition != expected)
    {
        ::std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message;
        issue_error_message(outstr);
    }
}

// Do not change signature to const T&.
// Function must be able to detect const differences between expected and actual.
template <typename T1, typename T2>
bool
is_equal_val(const T1& val1, const T2& val2)
{
    using T = std::common_type_t<T1, T2>;

    if constexpr (std::is_floating_point_v<T>)
    {
        const auto eps = std::numeric_limits<T>::epsilon();
        return std::fabs(T(val1) - T(val2)) < eps;
    }

    if constexpr (std::is_same_v<T1, T2>)
    {
        return val1 == val2;
    }
    else
    {
        return T(val1) == T(val2);
    }
}

// Do not change signature to const T&.
// Function must be able to detect const differences between expected and actual.
template <typename T1, typename T2>
void
expect_equal_val(const T1& expected, const T2& actual, const char* file, std::int32_t line, const char* message)
{
    if (!is_equal_val(expected, actual))
    {
        std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected " << expected << " got "
               << actual;
        issue_error_message(outstr);
    }
}

template <typename R1, typename R2>
void
expect_equal(const R1& expected, const R2& actual, const char* file, std::int32_t line, const char* message)
{
    size_t n = expected.size();
    size_t m = actual.size();
    if (n != m)
    {
        ::std::stringstream outstr;
        outstr << "error at " << file << ":" << line << " - " << message << ", expected sequence of size " << n
               << " got sequence of size " << m;
        issue_error_message(outstr);
        return;
    }
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k)
    {
        if (!is_equal_val(expected[k], actual[k]))
        {
            ::std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k << " expected "
                   << expected[k] << " got " << actual[k];
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

template <typename T>
void
expect_equal_val(Sequence<T>& expected, Sequence<T>& actual, const char* file, std::int32_t line, const char* message)
{
    expect_equal(expected, actual, file, line, message);
}

template <typename Iterator1, typename Iterator2, typename Size>
void
expect_equal(Iterator1 expected_first, Iterator2 actual_first, Size n, const char* file, std::int32_t line,
             const char* message)
{
    size_t error_count = 0;
    for (size_t k = 0; k < n && error_count < 10; ++k, ++expected_first, ++actual_first)
    {
        if (!is_equal_val(*expected_first, *actual_first))
        {
            ::std::stringstream outstr;
            outstr << "error at " << file << ":" << line << " - " << message << ", at index " << k;
            issue_error_message(outstr);
            ++error_count;
        }
    }
}

template <typename T1, typename T2>
bool
check_data(const T1* device_iter, const T2* host_iter, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (!is_equal_val(*(host_iter + i), *(device_iter + i)))
            return false;
    }
    return true;
}

struct MemoryChecker
{
    // static counters and state tags
    static ::std::atomic<::std::size_t> alive_object_counter; // initialized outside
    // since it can truncate the value on 32-bit platforms
    // we have to explicitly cast it to desired type to avoid any warnings
    static constexpr ::std::size_t alive_state = ::std::size_t(0xAAAAAAAAAAAAAAAA);
    static constexpr ::std::size_t dead_state = 0; // only used as a set value to cancel alive_state

    ::std::int32_t _value; // object value used for algorithms
    ::std::size_t _state;  // state tag used for checks

    // ctors, dtors, assign ops
    explicit MemoryChecker(::std::int32_t value = 0) : _value(value)
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since we cannot guarantee that
        // raw memory for object being constructed does not have a bit sequence being equal to alive_state

        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker(MemoryChecker&& other) : _value(other.value())
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since
        // compiler can optimize out the move ctor call that results in false positive failure
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker(MemoryChecker&&): attempt to "
                                                  "construct an object from non-existing object");
        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker(const MemoryChecker& other) : _value(other.value())
    {
        // check for EXPECT_TRUE(state() != alive_state, ...) has not been done since
        // compiler can optimize out the copy ctor call that results in false positive failure
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker(const MemoryChecker&): attempt to "
                                                  "construct an object from non-existing object");
        // set constructed state and increment counter for living object
        inc_alive_objects();
        _state = alive_state;
    }
    MemoryChecker&
    operator=(MemoryChecker&& other)
    {
        // check if we do not assign over uninitialized memory
        EXPECT_TRUE(state() == alive_state, "wrong effect from MemoryChecker::operator=(MemoryChecker&& other): "
                                            "attempt to assign to non-existing object");
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker::operator=(MemoryChecker&& other): "
                                                  "attempt to assign from non-existing object");
        // just assign new value, counter is the same, state is the same
        _value = other.value();

        return *this;
    }
    MemoryChecker&
    operator=(const MemoryChecker& other)
    {
        // check if we do not assign over uninitialized memory
        EXPECT_TRUE(state() == alive_state, "wrong effect from MemoryChecker::operator=(const MemoryChecker& other): "
                                            "attempt to assign to non-existing object");
        EXPECT_TRUE(other.state() == alive_state, "wrong effect from MemoryChecker::operator=(const MemoryChecker& "
                                                  "other): attempt to assign from non-existing object");
        // just assign new value, counter is the same, state is the same
        _value = other.value();

        return *this;
    }
    ~MemoryChecker()
    {
        // check if we do not double destruct the object
        EXPECT_TRUE(state() == alive_state,
                    "wrong effect from ~MemoryChecker(): attempt to destroy non-existing object");
        // set destructed state and decrement counter for living object
        static_cast<volatile ::std::size_t&>(_state) = dead_state;
        dec_alive_objects();
    }

    // getters
    ::std::int32_t
    value() const
    {
        return _value;
    }
    ::std::size_t
    state() const
    {
        return _state;
    }
    static ::std::size_t
    alive_objects()
    {
        return alive_object_counter.load();
    }

  private:
    // setters
    void
    inc_alive_objects()
    {
        alive_object_counter.fetch_add(1);
    }
    void
    dec_alive_objects()
    {
        alive_object_counter.fetch_sub(1);
    }
};

inline ::std::atomic<::std::size_t> MemoryChecker::alive_object_counter{0};

inline ::std::ostream&
operator<<(::std::ostream& os, const MemoryChecker& val)
{
    return (os << val.value());
}
inline bool
operator==(const MemoryChecker& v1, const MemoryChecker& v2)
{
    return v1.value() == v2.value();
}
inline bool
operator<(const MemoryChecker& v1, const MemoryChecker& v2)
{
    return v1.value() < v2.value();
}

// Predicates for algorithms
template <typename DataType>
struct is_equal_to
{
    is_equal_to(const DataType& expected) : m_expected(expected) {}
    bool
    operator()(const DataType& actual) const
    {
        return actual == m_expected;
    }

  private:
    DataType m_expected;
};

// Low-quality hash function, returns value between 0 and (1<<bits)-1
// Warning: low-order bits are quite predictable.
inline size_t
HashBits(size_t i, size_t bits)
{
    size_t mask = bits >= 8 * sizeof(size_t) ? ~size_t(0) : (size_t(1) << bits) - 1;
    return (424157 * i ^ 0x24aFa) & mask;
}

// Stateful unary op
template <typename T, typename U>
struct Complement
{
    std::int32_t val = 1;

    U
    operator()(const T& x) const
    {
        return U(val - x);
    }
};

struct ComplementZip
{
    std::int32_t val = 1;

    template<typename T>
    auto
    operator()(const oneapi::dpl::__internal::tuple<T&>& t) const
    {
        return oneapi::dpl::__internal::tuple<T>(val - std::get<0>(t));
    }
};

template <typename In1, typename In2, typename Out>
class TheOperation
{
    Out val;

  public:
    TheOperation(Out v) : val(v) {}
    Out
    operator()(const In1& x, const In2& y) const
    {
        return Out(val + x - y);
    }
};

template <typename Out>
class TheOperationZip
{
    Out val;

  public:
    TheOperationZip(Out v) : val(v) {}

    template <typename T1, typename T2>
    auto
    operator()(const oneapi::dpl::__internal::tuple<T1&>& t1, const oneapi::dpl::__internal::tuple<T2&>& t2) const
    {
        return oneapi::dpl::__internal::tuple<Out>(val + std::get<0>(t1) - std::get<0>(t2));
    }
};

// Tag used to prevent accidental use of converting constructor, even if use is explicit.
struct OddTag
{
};

class Sum;

// Type with limited set of operations.  Not default-constructible.
// Only available operator is "==".
// Typically used as value type in tests.
class Number
{
    std::int32_t value;
    friend class Add;
    friend class Sum;
    friend class IsMultiple;
    friend class Congruent;
    friend Sum
    operator+(const Sum& x, const Sum& y);

  public:
    Number(std::int32_t val, OddTag) : value(val) {}
    friend bool
    operator==(const Number& x, const Number& y)
    {
        return x.value == y.value;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& o, const Number& d)
    {
        return o << d.value;
    }
};

// Stateful predicate for Number.  Not default-constructible.
class IsMultiple
{
    long modulus;

  public:
    // True if x is multiple of modulus
    bool
    operator()(Number x) const
    {
        return x.value % modulus == 0;
    }
    IsMultiple(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful equivalence-class predicate for Number.  Not default-constructible.
class Congruent
{
    long modulus;

  public:
    // True if x and y have same remainder for the given modulus.
    // Note: this is not quite the same as "equivalent modulo modulus" when x and y have different
    // sign, but nonetheless AreCongruent is still an equivalence relationship, which is all
    // we need for testing.
    bool
    operator()(Number x, Number y) const
    {
        return x.value % modulus == y.value % modulus;
    }
    Congruent(long modulus_, OddTag) : modulus(modulus_) {}
};

// Stateful reduction operation for Number
class Add
{
    long bias;

  public:
    explicit Add(OddTag) : bias(1) {}
    Number
    operator()(Number x, const Number& y)
    {
        return Number(x.value + y.value + (bias - 1), OddTag());
    }
};

// Class similar to Number, but has default constructor and +.
class Sum : public Number
{
  public:
    Sum() : Number(0, OddTag()) {}
    Sum(long x, OddTag) : Number(x, OddTag()) {}
    friend Sum
    operator+(const Sum& x, const Sum& y)
    {
        return Sum(x.value + y.value, OddTag());
    }
};

// Type with limited set of operations, which includes an associative but not commutative operation.
// Not default-constructible.
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
class MonoidElement
{
    size_t a, b;

  public:
    MonoidElement(size_t a_, size_t b_, OddTag) : a(a_), b(b_) {}
    friend bool
    operator==(const MonoidElement& x, const MonoidElement& y)
    {
        return x.a == y.a && x.b == y.b;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& o, const MonoidElement& x)
    {
        return o << "[" << x.a << ".." << x.b << ")";
    }
    friend class AssocOp;
};

// Stateful associative op for MonoidElement
// It's not really a monoid since the operation is not allowed for any two elements.
// But it's good enough for testing.
class AssocOp
{
    unsigned c;

  public:
    explicit AssocOp(OddTag) : c(5) {}
    MonoidElement
    operator()(const MonoidElement& x, const MonoidElement& y)
    {
        unsigned d = 5;
        EXPECT_EQ(d, c, "state lost");
        EXPECT_EQ(x.b, y.a, "commuted?");

        return MonoidElement(x.a, y.b, OddTag());
    }
};

// Multiplication of matrix is an associative but not commutative operation
// Typically used as value type in tests involving "GENERALIZED_NONCOMMUTATIVE_SUM".
template <typename T>
struct Matrix2x2
{
    T a00, a01, a10, a11;
    Matrix2x2() : a00(1), a01(0), a10(0), a11(1) {}
    Matrix2x2(T x, T y) : a00(0), a01(x), a10(x), a11(y) {}
};

template <typename T>
bool
operator==(const Matrix2x2<T>& left, const Matrix2x2<T>& right)
{
    return left.a00 == right.a00 && left.a01 == right.a01 && left.a10 == right.a10 && left.a11 == right.a11;
}

template <typename T>
struct multiply_matrix
{
    Matrix2x2<T>
    operator()(const Matrix2x2<T>& left, const Matrix2x2<T>& right) const
    {
        Matrix2x2<T> result;
        result.a00 = left.a00 * right.a00 + left.a01 * right.a10;
        result.a01 = left.a00 * right.a01 + left.a01 * right.a11;
        result.a10 = left.a10 * right.a00 + left.a11 * right.a10;
        result.a11 = left.a10 * right.a01 + left.a11 * right.a11;

        return result;
    }
};

template <typename F>
struct NonConstAdapter
{
    F my_f;
    NonConstAdapter(const F& f) : my_f(f) {}

    template <typename... Types>
    auto
    operator()(Types&&... args) -> decltype(::std::declval<F>().
                                            operator()(::std::forward<Types>(args)...))
    {
        return my_f(::std::forward<Types>(args)...);
    }
};

template <typename F>
NonConstAdapter<F>
non_const(const F& f)
{
    return NonConstAdapter<F>(f);
}

// Wrapper for types. It's need for counting of constructing and destructing objects
template <typename T>
class Wrapper
{
  public:
    Wrapper()
        : my_field(::std::make_shared<T>())
    {
        ++my_count;
    }
    Wrapper(const T& input)
        : my_field(::std::make_shared<T>(input))
    {
        ++my_count;
    }
    Wrapper(const Wrapper& input)
        : my_field(input.my_field)
    {
        ++my_count;
    }
    Wrapper(Wrapper&& input)
        : my_field(::std::move(input.my_field))
    {
        ++move_count;
    }
    Wrapper&
    operator=(const Wrapper& input)
    {
        my_field = input.my_field;
        return *this;
    }
    Wrapper&
    operator=(Wrapper&& input)
    {
        my_field = ::std::move(input.my_field);
        ++move_count;
        return *this;
    }
    bool
    operator==(const Wrapper& input) const
    {
        return my_field == input.my_field;
    }
    bool
    operator<(const Wrapper& input) const
    {
        return *my_field < *input.my_field;
    }
    bool
    operator>(const Wrapper& input) const
    {
        return *my_field > *input.my_field;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& stream, const Wrapper& input)
    {
        return stream << *(input.my_field);
    }
    ~Wrapper()
    {
        --my_count;
        if (move_count > 0)
        {
            --move_count;
        }
    }
    T*
    get_my_field() const
    {
        return my_field.get();
    };
    static size_t
    Count()
    {
        return my_count;
    }
    static size_t
    MoveCount()
    {
        return move_count;
    }
    static void
    SetCount(const size_t& n)
    {
        my_count = n;
    }
    static void
    SetMoveCount(const size_t& n)
    {
        move_count = n;
    }

  private:
    static ::std::atomic<size_t> my_count;
    static ::std::atomic<size_t> move_count;
    ::std::shared_ptr<T> my_field;
};

template <typename T>
::std::atomic<size_t> Wrapper<T>::my_count = {0};

template <typename T>
::std::atomic<size_t> Wrapper<T>::move_count = {0};

template <typename InputIterator, typename T, typename BinaryOperation, typename UnaryOperation>
T
transform_reduce_serial(InputIterator first, InputIterator last, T init, BinaryOperation binary_op,
                        UnaryOperation unary_op) noexcept
{
    for (; first != last; ++first)
    {
        init = binary_op(init, unary_op(*first));
    }
    return init;
}

inline int
done(int is_done = 1)
{
    if (is_done)
    {
#if _PSTL_TEST_SUCCESSFUL_KEYWORD
        ::std::cout << "done\n";
#else
        ::std::cout << "passed\n";
#endif
        return 0;
    }
    else
    {
        ::std::cout << "Skipped\n";
        return _SKIP_RETURN_CODE;
    }
}

// test_algo_basic_* functions are used to execute
// f on a very basic sequence of elements of type T.

// Should be used with unary predicate
template <typename T, typename F>
static void
test_algo_basic_single(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });
    invoke_on_all_host_policies()(::std::forward<F>(f), in.begin());
}

// Should be used with binary predicate
template <typename T, typename F>
static void
test_algo_basic_double(F&& f)
{
    size_t N = 10;
    Sequence<T> in(N, [](size_t v) -> T { return T(v); });
    Sequence<T> out(N, [](size_t v) -> T { return T(v); });
    invoke_on_all_host_policies()(::std::forward<F>(f), in.begin(), out.begin());
}

template <typename T, typename = bool>
struct can_use_default_less_operator : ::std::false_type
{
};

template <typename T>
struct can_use_default_less_operator<T, decltype(::std::declval<T>() < ::std::declval<T>())> : ::std::true_type
{
};

template <typename T>
inline constexpr bool can_use_default_less_operator_v = can_use_default_less_operator<T>::value;

// An arbitrary binary predicate to simulate a predicate the user providing
// a custom predicate.
template <typename _Tp>
struct UserBinaryPredicate
{
    bool
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        using KeyT = ::std::decay_t<_Tp>;
        return __y != KeyT(1);
    }
};

template <typename T>
struct MatrixPoint
{
    T m;
    T n;
    MatrixPoint() = default;
    MatrixPoint(T m, T n = {}) : m(m), n(n) {}
    bool
    operator==(const MatrixPoint& other) const
    {
        return m == other.m && n == other.n;
    }
    bool
    operator!=(const MatrixPoint& other) const
    {
        return !(*this == other);
    }
    MatrixPoint
    operator+(const MatrixPoint& other) const
    {
        return MatrixPoint(m + other.m, n + other.n);
    }
};

template <typename T>
std::ostream&
operator<<(std::ostream& os, MatrixPoint<T> matrix_point)
{
    return os << "(" << matrix_point.m << ", " << matrix_point.n << ")";
}

template <typename _Tp>
struct MaxFunctor
{
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    {
        return (__x < __y) ? __y : __x;
    }
};

template <typename _Tp>
struct MaxFunctor<MatrixPoint<_Tp>>
{
    auto
    sum(const MatrixPoint<_Tp>& __x) const
    {
        return __x.m + __x.n;
    }
    MatrixPoint<_Tp>
    operator()(const MatrixPoint<_Tp>& __x, const MatrixPoint<_Tp>& __y) const
    {
        return (sum(__x) < sum(__y)) ? __y : __x;
    }
};

template <typename _Tp>
struct MaxAbsFunctor;

// A modification of the functor we use in reduce_by_segment
template <typename _Tp>
struct MaxAbsFunctor<MatrixPoint<_Tp>>
{
    auto
    abs_sum(const MatrixPoint<_Tp>& __x) const
    {
        return std::sqrt(__x.m * __x.m + __x.n * __x.n);
    }
    MatrixPoint<_Tp>
    operator()(const MatrixPoint<_Tp>& __x, const MatrixPoint<_Tp>& __y) const
    {
        return (abs_sum(__x) < abs_sum(__y)) ? abs_sum(__y) : abs_sum(__x);
    }
};

struct TupleAddFunctor1
{
    template <typename Tup1, typename Tup2>
    auto
    operator()(const Tup1& lhs, const Tup2& rhs) const
    {
        using ::std::get;
        Tup1 tup_sum = ::std::make_tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
        return tup_sum;
    }
};

// Exercise an explicit return of std::tuple to check for issues related to ambiguous return types between
// oneapi::dpl::__internal::tuple and std::tuple
struct TupleAddFunctor2
{
    template <typename Tup1, typename Tup2>
    auto
    operator()(const Tup1& lhs, const Tup2& rhs) const
    {
        using std::get;
        using return_t =
            std::tuple<decltype(get<0>(lhs) + get<0>(rhs)), decltype(get<1>(lhs) + get<1>(rhs))>;
        return_t tup_sum{get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs)};
        return tup_sum;
    }
};

struct _Identity
{
    template< class T >
    constexpr T&& operator()(T&& t) const noexcept
    {
        return std::forward<T>(t);
    }
};

struct _ZipIteratorAdapter
{
    template< class T >
    constexpr auto operator()(T&& t) const noexcept
    {
        return dpl::make_zip_iterator(std::forward<T>(t));
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __default_alloc_vec_iter = typename std::vector<ValueType>::iterator;

template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __usm_shared_alloc_vec_iter =
    typename std::vector<ValueType, typename sycl::usm_allocator<ValueType, sycl::usm::alloc::shared>>::iterator;

template <typename Iter, typename ValueType = std::decay_t<typename std::iterator_traits<Iter>::value_type>>
using __usm_host_alloc_vec_iter =
    typename std::vector<ValueType, typename sycl::usm_allocator<ValueType, sycl::usm::alloc::host>>::iterator;

// Evaluates to true if the provided type is an iterator with a value_type and if the implementation of a
// std::vector<value_type, Alloc>::iterator can be distinguished between three different allocators, the
// default, usm_shared, and usm_host. If all are distinct, it is very unlikely any non-usm based allocator
// could be confused with a usm allocator.
template <typename Iter>
constexpr bool __vector_impl_distinguishes_usm_allocator_from_default_v =
    !std::is_same_v<__default_alloc_vec_iter<Iter>, __usm_shared_alloc_vec_iter<Iter>> &&
    !std::is_same_v<__default_alloc_vec_iter<Iter>, __usm_host_alloc_vec_iter<Iter>> &&
    !std::is_same_v<__usm_host_alloc_vec_iter<Iter>, __usm_shared_alloc_vec_iter<Iter>>;

#endif //TEST_DPCPP_BACKEND_PRESENT

////////////////////////////////////////////////////////////////////////////////
// Implementation of create_new_policy for all policies (host + hetero)
template <typename Policy>
using __is_able_to_create_new_policy =
#if TEST_DPCPP_BACKEND_PRESENT
    oneapi::dpl::__internal::__is_hetero_execution_policy<::std::decay_t<Policy>>;
#else
    ::std::false_type;
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <typename _NewKernelName, typename Policy, ::std::enable_if_t<__is_able_to_create_new_policy<Policy>::value, int> = 0>
auto
create_new_policy(Policy&& policy)
{
    return TestUtils::make_new_policy<_NewKernelName>(::std::forward<Policy>(policy));
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template <typename _NewKernelName, typename Policy, ::std::enable_if_t<!__is_able_to_create_new_policy<Policy>::value, int> = 0>
auto
create_new_policy(Policy&& policy)
{
    return ::std::forward<Policy>(policy);
}

template <int idx, typename Policy>
auto
create_new_policy_idx(Policy&& policy)
{
#if TEST_DPCPP_BACKEND_PRESENT
    return create_new_policy<TestUtils::new_kernel_name<Policy, idx>>(::std::forward<Policy>(policy));
#else
    return ::std::forward<Policy>(policy);
#endif
}

#if TEST_DPCPP_BACKEND_PRESENT
template <typename KernelName, int idx>
struct kernel_name_with_idx
{
};

template <int idx, typename KernelParam>
constexpr auto
create_new_kernel_param_idx(KernelParam)
{
#if TEST_EXPLICIT_KERNEL_NAMES
    return oneapi::dpl::experimental::kt::kernel_param<KernelParam::data_per_workitem,
                                                       KernelParam::workgroup_size,
                                                       kernel_name_with_idx<typename KernelParam::kernel_name, idx>>{};
#else
    return KernelParam{};
#endif // TEST_EXPLICIT_KERNEL_NAMES
}
#endif //TEST_DPCPP_BACKEND_PRESENT

template <typename T>
typename std::enable_if_t<std::is_arithmetic_v<T>>
generate_arithmetic_data(T* input, std::size_t size, std::uint32_t seed)
{
    std::default_random_engine gen{seed};
    // The values beyond the threshold (75%) are duplicates of the values within the threshold
    std::size_t unique_threshold = 75 * size / 100;
    if constexpr (std::is_integral_v<T>)
    {
        // no uniform_int_distribution for chars
        using GenT = std::conditional_t<sizeof(T) < sizeof(short), int, T>;
        std::uniform_int_distribution<GenT> dist(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
        std::generate(input, input + unique_threshold, [&] { return T(dist(gen)); });
    }
    else
    {
        // log2 - exp2 transformation allows generating floating point values,
        // which distribution resembles uniform distribution of their bit representation
        // This is useful for checking different cases of radix sort
        std::uniform_real_distribution<T> dist_real(std::numeric_limits<T>::min(), log2(std::numeric_limits<T>::max()));
        std::uniform_int_distribution<int> dist_binary(0, 1);
        auto randomly_signed_real = [&dist_real, &dist_binary, &gen]()
        {
            auto v = exp2(dist_real(gen));
            return dist_binary(gen) == 0 ? v : -v;
        };
        std::generate(input, input + unique_threshold, [&] { return randomly_signed_real(); });
    }
    assert(unique_threshold >= size/2 && unique_threshold < size);
    for (uint32_t i = 0, j = unique_threshold; j < size; ++i, ++j)
    {
        input[j] = input[i];
    }
}

// Utility that models __estimate_best_start_size in the SYCL backend parallel_for to ensure large enough inputs are
// used to test the large submitter path. A multiplier to the max n is added to ensure we get a few separate test inputs
// for this path. For debug testing, only test with a single large n to avoid timeouts. Returns a monotonically increasing
// sequence for use in testing.
inline std::vector<std::size_t>
get_pattern_for_test_sizes()
{
    std::size_t max_n = 0;
    // We do not enable large input size testing for FPGA devices as __parallel_for_submitter_fpga only has a single
    // implementation with the standard input sizes providing full coverage, and testing large inputs is slow with the
    // FPGA emulator.
#if TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
    sycl::queue q = TestUtils::get_test_queue();
    sycl::device d = q.get_device();
    constexpr std::size_t max_iters_per_item = 16;
    constexpr std::size_t multiplier = 4;
    constexpr std::size_t max_work_group_size = 512;
    const std::size_t large_submitter_limit =
        max_iters_per_item * max_work_group_size * d.get_info<sycl::info::device::max_compute_units>();
#endif
#if TEST_DPCPP_BACKEND_PRESENT && !PSTL_USE_DEBUG && !ONEDPL_FPGA_DEVICE
    std::size_t cap = 10000000;
    max_n = multiplier * large_submitter_limit;
    // Ensure that TestUtils::max_n <= max_n <= cap
    max_n = std::max(TestUtils::max_n, std::min(cap, max_n));
#else
    max_n = TestUtils::max_n;
#endif
    // Generate the sequence of test input sizes
    std::vector<std::size_t> sizes;
    for (std::size_t n = 0; n <= max_n; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n))
        sizes.push_back(n);
#if TEST_DPCPP_BACKEND_PRESENT && PSTL_USE_DEBUG && !ONEDPL_FPGA_DEVICE
    if (max_n < large_submitter_limit)
        sizes.push_back(large_submitter_limit);
#endif
    return sizes;
}

} /* namespace TestUtils */

#endif // _UTILS_H
