# oneDPL Test Infrastructure Guide

This document describes the C++ test utilities, conventions, and infrastructure used throughout the oneDPL test suite.

**Target Audience**: Both new contributors learning the system and maintainers needing reference material.

---

## Table of Contents

1. [Test File Structure](#test-file-structure)
2. [Core Test Utilities](#core-test-utilities)
3. [Policy Invocation Utilities](#policy-invocation-utilities)
4. [Test Data Management](#test-data-management)
5. [Iterator Testing Utilities](#iterator-testing-utilities)
6. [Device and Backend Selection](#device-and-backend-selection)
7. [Common Test Patterns](#common-test-patterns)
8. [Configuration and Feature Detection](#configuration-and-feature-detection)
9. [Best Practices](#best-practices)

---

## Test File Structure

### Standard Test Template

Every test file follows this standard structure:

```cpp
// -*- C++ -*-
//===-- test_name.pass.cpp ------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

// Include oneDPL headers via macro (supports both <oneapi/dpl/...> and direct paths)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

using namespace TestUtils;

// Test implementation...

int main() {
    // Test logic
    return TestUtils::done();
}
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `test_config.h` | Configuration, compiler workarounds, feature detection |
| `_PSTL_TEST_HEADER(header)` | Macro for flexible header inclusion |
| `utils.h` | Core test utilities (EXPECT_TRUE, Sequence, etc.) |
| `TestUtils::done()` | Test completion indicator (returns 0 for pass) |

---

## Core Test Utilities

**Location**: `test/support/utils.h`

### Assertion Macros

These macros are the primary way to check test conditions:

```cpp
// Check boolean conditions
EXPECT_TRUE(condition, message)
EXPECT_FALSE(condition, message)

// Check value equality
EXPECT_EQ(expected, actual, message)

// Check sequence equality (N elements)
EXPECT_EQ_N(expected_iter, actual_iter, n, message)

// Check range equality (containers with .size())
EXPECT_EQ_RANGES(expected_container, actual_container, message)
```

**Important**: These macros call `std::exit(EXIT_FAILURE)` on failure, printing the error message with file and line number.

#### Example Usage

```cpp
void test_transform_result() {
    Sequence<int> input(10, [](size_t i) { return i; });
    Sequence<int> output(10);
    Sequence<int> expected(10, [](size_t i) { return i * 2; });

    std::transform(input.begin(), input.end(), output.begin(),
                   [](int x) { return x * 2; });

    EXPECT_EQ_RANGES(expected, output, "transform produced incorrect result");
}
```

### Sequence Container

**Location**: `test/support/utils_sequence.h`

`Sequence<T>` is the workhorse container for oneDPL tests. It wraps a `std::vector<T>` but provides a rich set of iterator types, making it easy to test algorithms with different iterator categories without managing multiple container types.

#### Why Use Sequence?

The oneDPL parallel algorithms must work correctly with different iterator types that have varying capabilities:
- **Random access iterators**: Support `it + n`, `it - n`, `it[n]`, etc.
- **Bidirectional iterators**: Support `++it` and `--it`, but not random access
- **Forward iterators**: Support `++it` only, no decrement or random access

`Sequence<T>` provides all these iterator types from a single container, letting you thoroughly test algorithm behavior without creating separate test data for each iterator category.

#### API Reference

```cpp
template <typename T>
class Sequence {
public:
    // Constructors
    explicit Sequence(size_t size);                    // Default-initialized elements
    Sequence(size_t size, Func f);                     // Initialize with f(i) for i=0..size-1
    Sequence(const std::initializer_list<T>& data);    // Initialize from list

    // Random access iterators (underlying std::vector iterators)
    iterator begin();
    iterator end();
    const_iterator cbegin() const;
    const_iterator cend() const;

    // Forward iterators (restricted interface, only ++ supported)
    forward_iterator fbegin();
    forward_iterator fend();
    const_forward_iterator cfbegin() const;
    const_forward_iterator cfend() const;

    // Bidirectional iterators (++ and -- supported, no random access)
    bidirectional_iterator bibegin();
    bidirectional_iterator biend();
    const_bidirectional_iterator cbibegin() const;
    const_bidirectional_iterator cbiend() const;

    // Direct access
    size_t size() const;
    T* data();                      // Access underlying array
    T& operator[](size_t j);        // Element access

    // Utilities
    void fill(const T& value);      // Fill with constant value
    void fill(Func f);              // Fill with f(i) for each position i
    void print() const;             // Debug print (first 100 elements)
};
```

#### Common Usage Patterns

##### Pattern 1: Basic Initialization

```cpp
// Default-initialized (e.g., zeros for int)
Sequence<int> zeros(100);

// Initialize with lambda - creates [0, 1, 2, ..., 99]
Sequence<int> sequence(100, [](size_t i) { return i; });

// Initialize with computation - creates [0, 2, 4, ..., 198]
Sequence<int> evens(100, [](size_t i) { return i * 2; });

// From initializer list for small test cases
Sequence<int> small_test({3, 1, 4, 1, 5, 9, 2, 6});
```

##### Pattern 2: Testing with Different Iterator Types

Most algorithms work with random access iterators, but some are specified to work with weaker iterator categories. Test both:

```cpp
struct test_find {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        auto result = std::find(std::forward<Policy>(exec), first, last, 42);
        EXPECT_TRUE(result != last, "find should succeed");
    }
};

void test() {
    Sequence<int> data(1000, [](size_t i) { return i == 500 ? 42 : i; });

    // Test with random access (fastest, most capable)
    invoke_on_all_policies<0>()(test_find(), data.begin(), data.end());

    // Test with forward iterators (algorithm must not use random access)
    invoke_on_all_policies<1>()(test_find(), data.fbegin(), data.fend());

    // Test with const iterators (algorithm must not modify data)
    invoke_on_all_policies<2>()(test_find(), data.cbegin(), data.cend());
}
```

**When to use which iterator type?**
- **Random access** (`begin()`/`end()`): Default choice for most tests
- **Forward** (`fbegin()`/`fend()`): When testing algorithms that should work with forward-only iterators (e.g., `std::find`, `std::count`)
- **Bidirectional** (`bibegin()`/`biend()`): When testing algorithms that need backward iteration (e.g., `std::reverse`)
- **Const** (`cbegin()`/`cend()`): When testing read-only algorithms (e.g., `std::all_of`, `std::count_if`)

##### Pattern 3: In-Place Modification and Refilling

```cpp
Sequence<int> data(100, [](size_t i) { return i; });

// Sort the data in-place
std::sort(exec_policy, data.begin(), data.end());

// Refill for next test
data.fill(42);  // All elements = 42

// Or refill with function
data.fill([](size_t i) { return 100 - i; });  // [100, 99, 98, ..., 1]
```

##### Pattern 4: Creating Expected Results

A common pattern is to create the expected result separately, then compare:

```cpp
void test_transform_multiply() {
    Sequence<int> input(100, [](size_t i) { return i; });
    Sequence<int> output(100);  // Uninitialized
    Sequence<int> expected(100, [](size_t i) { return i * 2; });

    // Run algorithm
    std::transform(exec_policy, input.begin(), input.end(), output.begin(),
                   [](int x) { return x * 2; });

    // Verify
    EXPECT_EQ_RANGES(expected, output, "transform produced wrong result");
}
```

##### Pattern 5: Testing with Different Data Patterns

```cpp
// Sorted ascending
Sequence<int> sorted_asc(100, [](size_t i) { return i; });

// Sorted descending
Sequence<int> sorted_desc(100, [](size_t i) { return 100 - i; });

// Random-like (using HashBits for deterministic randomness)
Sequence<int> random_like(100, [](size_t i) { return HashBits(i, 20); });

// Mostly sorted with few out-of-place elements
Sequence<int> nearly_sorted(100, [](size_t i) {
    return (i == 50 || i == 75) ? 0 : i;  // Two elements out of place
});

// All same value
Sequence<int> uniform(100, [](size_t) { return 42; });

// Alternating pattern
Sequence<int> alternating(100, [](size_t i) { return i % 2; });
```

#### Direct Array Access

When you need the underlying array (e.g., for USM operations or external libraries):

```cpp
Sequence<int> seq(100, [](size_t i) { return i; });

// Get raw pointer
int* raw_ptr = seq.data();

// Pass to C-style API
some_c_function(raw_ptr, seq.size());

// Index-based access
for (size_t i = 0; i < seq.size(); ++i) {
    seq[i] *= 2;  // Modify elements directly
}
```

#### Debugging with Print

```cpp
Sequence<int> data(1000, [](size_t i) { return i; });

// Apply some transformation
std::transform(policy, data.begin(), data.end(), data.begin(),
               [](int x) { return x * x; });

// Print to inspect (prints first 100 elements)
data.print();  // Output: size = 1000: { 0 1 4 9 16 ... }
```

#### Design Note: Why Not Just Use std::vector?

You might wonder why not just use `std::vector` directly. The key advantages of `Sequence<T>`:

1. **Uniform Interface**: Provides `fbegin()`/`fend()`, `bibegin()`/`biend()` without wrapper code in every test
2. **Iterator Restriction**: The forward/bidirectional iterators are *deliberately restricted* - they compile-fail if you accidentally use random access operations
3. **Test Convenience**: Lambda-based initialization is cleaner than separate fill operations
4. **Debug Utilities**: Built-in `print()` for quick debugging

**Example of Iterator Restriction**:
```cpp
Sequence<int> seq(10);
auto fwd = seq.fbegin();

++fwd;        // ✅ OK - forward iterators support increment
fwd + 5;      // ❌ Compile error - forward iterators don't support random access
fwd[3];       // ❌ Compile error - no subscript operator
```

This compile-time checking helps ensure your algorithm implementation doesn't accidentally rely on operations not available for the iterator category you claim to support.

### Value Comparison Utilities

The `is_equal_val` function handles type-aware comparisons:

```cpp
template <typename T1, typename T2>
bool is_equal_val(const T1& val1, const T2& val2);
```

**Features**:
- Floating-point comparison using `std::numeric_limits<T>::epsilon()`
- Type promotion for mixed-type comparisons
- Exact equality for non-floating-point types

### Test Completion

```cpp
int done(int is_done = 1);
```

**Returns**:
- `0` if `is_done != 0` (test passed)
- `_SKIP_RETURN_CODE` (77) if `is_done == 0` (test skipped)

**Prints**: "passed" (or "done" if `_PSTL_TEST_SUCCESSFUL_KEYWORD` defined)

---

## Policy Invocation Utilities

**Location**: `test/support/utils_invoke.h`

These utilities allow tests to run algorithms across multiple execution policies systematically.

### Policy Invokers

```cpp
// Run on host policies: seq, unseq, par, par_unseq
invoke_on_all_host_policies()(op, args...);

// Run on device policies (SYCL)
invoke_on_all_hetero_policies<CallNumber>()(op, args...);

// Run on all policies (host + device)
invoke_on_all_policies<CallNumber>()(op, args...);
```

#### CallNumber Template Parameter

The `CallNumber` parameter (default 0) is used to generate unique kernel names for SYCL. **Each invocation in the same test must use a different CallNumber**.

#### Example Usage

```cpp
struct test_sort {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        std::sort(std::forward<Policy>(exec), first, last);
        EXPECT_TRUE(std::is_sorted(first, last), "sort failed");
    }
};

void test_sorting() {
    Sequence<int> in(1000, [](size_t v) { return 1000 - v; });

    // Test with all policies, CallNumber = 0
    invoke_on_all_policies<0>()(test_sort(), in.begin(), in.end());

    // If testing again in same function, use different CallNumber
    in.fill([](size_t v) { return v % 2; });
    invoke_on_all_policies<1>()(test_sort(), in.begin(), in.end());
}
```

### Policy Cloning Macros

These macros create policy copies with unique kernel names (essential for SYCL):

```cpp
// Clone policy preserving value category (l-value or r-value)
CLONE_TEST_POLICY(policy_src)

// Clone with index-based unique name
CLONE_TEST_POLICY_IDX(policy_src, idx)

// Clone with custom kernel name
CLONE_TEST_POLICY_NAME(policy_src, NewKernelName)
```

**Important**: Cloning creates a `test_policy_container` whose `.get()` method can only be called **once**.

#### Example Usage

```cpp
auto my_policy = TestUtils::get_dpcpp_test_policy<0, MyTest>();

// Create two clones for two algorithm calls
auto policy1 = CLONE_TEST_POLICY_IDX(my_policy, 0);
auto policy2 = CLONE_TEST_POLICY_IDX(my_policy, 1);

std::sort(policy1, data.begin(), data.end());
std::transform(policy2, data.begin(), data.end(), output.begin(), [](int x) { return x * 2; });
```

### Device Policy Creation

```cpp
// Create device policy (automatically handles kernel naming based on TEST_EXPLICIT_KERNEL_NAMES)
template <typename KernelName = DefaultKernelName>
auto make_device_policy(queue_or_policy);

// Get default test policy
template <int call_id = 0, typename PolicyName = TestPolicyName>
auto get_dpcpp_test_policy();
```

**Best Practice**: Always use `TestUtils::make_device_policy` instead of calling `oneapi::dpl::execution::make_device_policy` directly.

---

## Test Data Management

**Location**: `test/support/utils_test_base.h`, `test/support/utils_sycl.h`

The test base data infrastructure provides a structured framework for testing device algorithms with different memory types (USM, SYCL buffers). This infrastructure is **specifically designed for device/heterogeneous testing** and handles the complexities of managing data across host and device memory spaces.

### When to Use This Infrastructure

**Use the test_base infrastructure when:**
- Testing **device execution policies** (SYCL/DPC++)
- Need to test with **both USM and SYCL buffers**
- Want to systematically test across **multiple input sizes**
- Testing algorithms that require **2, 3, or 4 separate buffers** (e.g., keys, values, output)
- Need **automatic host-device data synchronization** for verification

**Don't use it when:**
- Testing **only host policies** (seq, par, unseq, par_unseq) → Use `Sequence<T>` directly
- Writing a **simple, single-case test** → Manual setup is clearer
- Testing **non-algorithm functionality** (e.g., iterators, utilities)

### Conceptual Overview

The infrastructure separates three concerns:

1. **Data Storage** (`test_base_data_*`): Where and how test data is stored (USM vs buffer vs host)
2. **Test Logic** (`test_base<T>` subclass): What algorithm to test and how to verify results
3. **Test Orchestration** (`test1buffer`, `test2buffers`, etc.): Running the test across sizes and memory types

**Data flow**:
```
                     ┌──────────────────────────┐
                     │ test1buffer/test2buffers │
                     │  (orchestration)         │
                     └───────────┬──────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
           ┌──────▼────────┐            ┌──────▼────────┐
           │  USM Memory   │            │ SYCL Buffer   │
           │ test_base_data│            │test_base_data │
           └──────┬────────┘            └───────┬───────┘
                  │                             │
                  └──────────┬──────────────────┘
                             │
                   ┌─────────▼─────────────┐
                   │  Your Test Class      │
                   │  (test_base<T>)       │
                   │  - operator()         │
                   │  - verification logic │
                   └───────────────────────┘
```

### Data Source Types

The infrastructure supports three memory types:

| Type | Memory Type | Host Access | Use Case |
|------|-------------|-------------|----------|
| `test_base_data_usm<alloc_type, T>` | USM shared/device/host | Depends on `alloc_type` | Device testing with USM pointers |
| `test_base_data_buffer<T>` | SYCL buffers | Via accessor only | Device testing with SYCL buffers |
| `test_base_data_sequence<T>` | Host (`std::vector`) | Direct | Host-only testing (rarely needed) |

**USM Allocation Types**:
- `sycl::usm::alloc::shared`: Accessible from both host and device (easiest for testing)
- `sycl::usm::alloc::device`: Device-only (requires explicit copy for verification)
- `sycl::usm::alloc::host`: Host memory pinned for device access

### Test Data Kinds

Each test can use up to 4 data buffers, identified by `UDTKind`:

```cpp
enum class UDTKind {
    eKeys,   // First input data  (e.g., elements to sort)
    eVals,   // Second input data (e.g., values in key-value sort)
    eRes,    // Output data       (e.g., sorted result)
    eRes2    // Second output     (e.g., secondary result in partition)
};
```

**Buffer count determines which helper function to use**:
- 1 buffer (in-place algorithm): `test1buffer` → uses `eKeys`
- 2 buffers (input + output): `test2buffers` → uses `eKeys`, `eVals`
- 3 buffers (2 inputs + output): `test3buffers` → uses `eKeys`, `eVals`, `eRes`
- 4 buffers (2 inputs + 2 outputs): `test4buffers` → uses `eKeys`, `eVals`, `eRes`, `eRes2`

### High-Level Test Functions

These functions orchestrate running your test across multiple sizes and memory types:

```cpp
// Single buffer test (in-place algorithms)
test1buffer<alloc_type, TestValueType, TestName>(scale_step, scale_max);

// Two buffer test (copy, transform, etc.)
test2buffers<alloc_type, TestValueType, TestName>(scale_step, scale_max);

// Three buffer test (merge, set operations, etc.)
test3buffers<alloc_type, TestValueType, TestName>(mult, scale_step, scale_max);

// Four buffer test (complex algorithms with multiple outputs)
test4buffers<alloc_type, TestValueType, TestName>(mult, scale_step, scale_max);
```

**Parameters**:
- `alloc_type`: USM allocation type (`sycl::usm::alloc::{shared, device, host}`)
- `TestValueType`: Element type (e.g., `int`, `float`, `MyStruct`)
- `TestName`: Your test class (must inherit from `test_base<TestValueType>`)
- `scale_step`: Multiplier for loop step size (default 1.0, larger = fewer iterations)
- `scale_max`: Multiplier for max size (default 1.0, e.g., 0.1 for quick tests)
- `mult`: Output buffer size multiplier (e.g., 2 if output can be 2x input size)

#### What These Functions Do

Each `testNbuffers` function:
1. **Allocates** USM memory or creates SYCL buffers of appropriate sizes
2. **Loops** over test sizes from 1 to `max_n * scale_max`, stepping by `~3.14 * scale_step`
3. **Calls** `invoke_on_all_hetero_policies` with your test operator
4. **Repeats** for both USM (if `_PSTL_SYCL_TEST_USM` defined) and SYCL buffers (if applicable)

### Creating a Test Class

To use this infrastructure, create a test class inheriting from `test_base<T>`:

```cpp
template <typename TestValueType>
struct test_my_algorithm : test_base<TestValueType> {
    // Required: specify the value type for the test framework
    using UsedValueType = TestValueType;

    // Optional: control test size scaling
    static constexpr float ScaleStep = 1.0f;  // Larger = fewer test sizes
    static constexpr float ScaleMax = 1.0f;   // Smaller = smaller max size

    // Required: inherit base constructor
    using test_base<TestValueType>::test_base;

    // Required: test operator - called for each (policy, size) combination
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last, size_t n) {
        // 1. Initialize input data
        // 2. Run algorithm
        // 3. Verify results
    }
};
```

#### Complete Example: Testing Transform

```cpp
#include "support/utils_sycl.h"

template <typename T>
struct test_transform : test_base<T> {
    using UsedValueType = T;
    static constexpr float ScaleStep = 1.0f;
    static constexpr float ScaleMax = 1.0f;

    using test_base<T>::test_base;

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last,
                    Iterator2 out_first, Iterator2 out_last, Size n) {

        // Step 1: Initialize input data on host
        {
            TestDataTransfer<UDTKind::eKeys, Size> input_data(*this, n);
            T* input_ptr = input_data.get_data();
            for (Size i = 0; i < n; ++i) {
                input_ptr[i] = static_cast<T>(i);
            }
            // Data automatically copied to device when scope ends
        }

        // Step 2: Run the algorithm
        std::transform(std::forward<Policy>(exec), in_first, in_last, out_first,
                      [](const T& x) { return x * 2; });

        // Step 3: Verify results on host
        {
            TestDataTransfer<UDTKind::eVals, Size> output_data(*this, n);
            T* output_ptr = output_data.get_data();

            for (Size i = 0; i < n; ++i) {
                T expected = static_cast<T>(i * 2);
                EXPECT_EQ(expected, output_ptr[i], "transform produced incorrect result");
            }
        }
    }
};

int main() {
    // Test with USM shared memory
    test2buffers<sycl::usm::alloc::shared, int, test_transform<int>>();

    // Can also test with specific value types
    test2buffers<sycl::usm::alloc::shared, float, test_transform<float>>();

    return TestUtils::done();
}
```

### The test_base Class

```cpp
template <typename TestValueType>
struct test_base {
    test_base_data<TestValueType>& base_data_ref;  // Reference to data storage

    test_base(test_base_data<TestValueType>& data_ref);

    // Check if data requires host-side buffering
    bool host_buffering_required() const;

    // RAII helper for host-device data transfer
    template <UDTKind kind, typename Size>
    class TestDataTransfer {
    public:
        TestDataTransfer(test_base& base, Size count);
        TestValueType* get_data();  // Get host-accessible pointer
        ~TestDataTransfer();        // Auto-syncs back to device
    };
};
```

### TestDataTransfer: The Key Utility

`TestDataTransfer` is an RAII helper that handles data movement between host and device:

```cpp
// Usage pattern:
{
    // Create transfer object (copies from device to host if needed)
    TestDataTransfer<UDTKind::eKeys, size_t> data_transfer(*this, n);

    // Get host pointer
    T* host_ptr = data_transfer.get_data();

    // Read/write data on host
    for (size_t i = 0; i < n; ++i) {
        host_ptr[i] = /* ... */;
    }

    // Destructor automatically copies data back to device if modified
}
```

**When is copying needed?**
- **USM shared**: No copy needed (accessible from both sides)
- **USM device**: Always copies to/from host buffer
- **SYCL buffer**: Creates host accessor (may involve copy depending on runtime)

### Common Patterns

#### Pattern 1: In-Place Algorithm (1 buffer)

```cpp
struct test_sort : test_base<T> {
    using UsedValueType = T;
    using test_base<T>::test_base;

    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last, size_t n) {
        // Initialize
        {
            TestDataTransfer<UDTKind::eKeys, size_t> input(*this, n);
            T* ptr = input.get_data();
            for (size_t i = 0; i < n; ++i) ptr[i] = n - i;  // Reverse order
        }

        // Sort
        std::sort(exec, first, last);

        // Verify
        {
            TestDataTransfer<UDTKind::eKeys, size_t> output(*this, n);
            T* ptr = output.get_data();
            for (size_t i = 0; i < n; ++i) {
                EXPECT_EQ(static_cast<T>(i + 1), ptr[i], "sort failed");
            }
        }
    }
};

int main() {
    test1buffer<sycl::usm::alloc::shared, int, test_sort<int>>();
    return done();
}
```

#### Pattern 2: Copy Algorithm (2 buffers)

```cpp
struct test_copy_if : test_base<T> {
    using UsedValueType = T;
    using test_base<T>::test_base;

    template <typename Policy, typename Iterator1, typename Iterator2>
    void operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last,
                    Iterator2 out_first, Iterator2 out_last, size_t n) {
        // Initialize input
        {
            TestDataTransfer<UDTKind::eKeys, size_t> input(*this, n);
            T* ptr = input.get_data();
            for (size_t i = 0; i < n; ++i) ptr[i] = i;
        }

        // Run algorithm
        auto result = std::copy_if(exec, in_first, in_last, out_first,
                                   [](T x) { return x % 2 == 0; });

        // Verify
        {
            TestDataTransfer<UDTKind::eVals, size_t> output(*this, n);
            T* ptr = output.get_data();
            size_t count = std::distance(out_first, result);
            EXPECT_EQ(n / 2, count, "wrong number of elements copied");
            for (size_t i = 0; i < count; ++i) {
                EXPECT_EQ(static_cast<T>(i * 2), ptr[i], "wrong value copied");
            }
        }
    }
};
```

#### Pattern 3: Avoiding Redundant Transfers

If you don't need to verify data (e.g., testing for crashes only), skip the transfer:

```cpp
template <typename Policy, typename Iterator>
void operator()(Policy&& exec, Iterator first, Iterator last, size_t n) {
    // Just run the algorithm, no verification
    std::sort(exec, first, last);
    // No TestDataTransfer needed
}
```

### Adjusting Test Scale

Control test execution time by adjusting scale factors:

```cpp
// Quick smoke test (10% of default sizes)
struct test_quick : test_base<int> {
    using UsedValueType = int;
    static constexpr float ScaleMax = 0.1f;  // Only test up to 10% of max_n
    using test_base<int>::test_base;
    // ...
};

// Thorough test with fewer iterations (larger steps)
struct test_thorough : test_base<int> {
    using UsedValueType = int;
    static constexpr float ScaleStep = 3.0f;  // 3x larger steps = fewer iterations
    static constexpr float ScaleMax = 2.0f;   // 2x larger max size
    using test_base<int>::test_base;
    // ...
};
```

### When NOT to Use test_base

The `test_base` infrastructure adds complexity. **Don't use it** for:

**1. Simple, single-case tests**:
```cpp
// ✅ Better: Simple and clear
int main() {
    sycl::queue q;
    sycl::buffer<int> buf(100);
    auto policy = make_device_policy<TestKernel>(q);

    std::fill(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), 42);

    // Verify...
    return done();
}
```

**2. Host-only tests** (use `Sequence<T>` + `invoke_on_all_host_policies`):
```cpp
// ✅ Better: Direct use of Sequence
int main() {
    Sequence<int> data(100);
    invoke_on_all_host_policies()(test_algorithm(), data.begin(), data.end());
    return done();
}
```

**3. Custom size patterns** (test_base uses fixed progression):
```cpp
// ✅ Better: Custom loop
for (size_t n : {0, 1, 2, 10, 100, 1000, 1000000}) {
    // Manual test setup
}
```

### Quick Decision Guide

**Should I use test_base infrastructure?**

| Question | Yes → use test_base | No → use Sequence/manual |
|----------|---------------------|--------------------------|
| Testing device policies? | ✓ | |
| Need both USM and buffer tests? | ✓ | |
| Testing across many sizes? | ✓ | |
| Simple single-size test? | | ✓ |
| Host policies only? | | ✓ |
| Custom size pattern needed? | | ✓ |
| Just checking compilation? | | ✓ |

---

## Iterator Testing Utilities

**Location**: `test/support/iterator_utils.h`

### Custom Iterator Types

```cpp
// Forward iterator adapter
template <typename Iterator, typename IteratorTag>
class ForwardIterator;

// Bidirectional iterator adapter
template <typename Iterator, typename IteratorTag>
class BidirectionalIterator;
```

**Purpose**: Restrict iterator capabilities to test algorithm requirements.

### Iterator Invokers

```cpp
// Invoke with all iterator types (random, forward, bidirectional, reverse)
invoke_on_all_iterator_types()(policy, op, args...);

// Invoke with specific iterator tag
iterator_invoker<iterator_tag, IsReverse>()(policy, op, args...);
```

### Iterator Creation

```cpp
// Create iterator with specific tag and optional reverse
template <typename InputIterator, typename IteratorTag, typename IsReverse>
struct MakeIterator;
```

#### Example

```cpp
// Create forward iterator
auto fwd_iter = MakeIterator<Iterator, std::forward_iterator_tag, std::false_type>()(it);

// Create reverse bidirectional iterator
auto rev_bidir = MakeIterator<Iterator, std::bidirectional_iterator_tag, std::true_type>()(it);
```

---

## Device and Backend Selection

### Device Type Support

**Location**: `test/support/utils_invoke.h`

```cpp
// Check if device supports type
template<typename T>
bool has_type_support(const sycl::device& device);

// Specializations for:
// - double: requires sycl::aspect::fp64
// - sycl::half: requires sycl::aspect::fp16
```

**Automatic Handling**: `invoke_on_all_hetero_policies` automatically checks type support and skips tests for unsupported types.

### SYCL Queue Access

```cpp
// Get test queue (uses default_selector or FPGA selector)
sycl::queue get_test_queue();
```

**Configuration**:
- Configured via `default_selector` (defined in `utils_sycl.h`)
- FPGA: Uses `fpga_selector` or `fpga_emulator_selector_v`
- GPU/CPU: Uses `default_selector_v`

### Runtime Device Selection

Use environment variables at test runtime:

```bash
# For SYCL devices
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
# or
export SYCL_DEVICE_FILTER=level_zero:gpu

# Run test
./my_test.pass
```

---

## Common Test Patterns

### Pattern 1: Basic Algorithm Test

```cpp
template <typename Type>
struct test_algorithm {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        // Create expected result
        Sequence<Type> expected(/* ... */);

        // Run algorithm
        my_algorithm(std::forward<Policy>(exec), first, last);

        // Verify
        EXPECT_EQ_N(expected.begin(), first, std::distance(first, last),
                    "algorithm produced incorrect result");
    }
};

void test() {
    Sequence<Type> in(100, [](size_t i) { return i; });
    invoke_on_all_policies<0>()(test_algorithm<Type>(), in.begin(), in.end());
}
```

### Pattern 2: Test with Multiple Sizes

```cpp
void test_algorithm_by_type() {
    const std::size_t max_size = 1000000;

    for (std::size_t n = 1; n <= max_size; n = n <= 16 ? n + 1 : std::size_t(3.1415 * n)) {
        Sequence<T> in(n, [](size_t v) { return T(v); });
        invoke_on_all_policies<0>()(test_operator(), in.begin(), in.end());
    }
}
```

**Pattern**: Start with 1..16 (by 1), then grow by factor of π.

### Pattern 3: Test with Different Iterator Types

```cpp
struct test_with_iterators {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        // Test implementation
    }
};

void test() {
    Sequence<int> seq(100, [](size_t i) { return i; });

    // Test with random access iterators
    invoke_on_all_policies<0>()(test_with_iterators(), seq.begin(), seq.end());

    // Test with forward iterators
    invoke_on_all_policies<1>()(test_with_iterators(), seq.fbegin(), seq.fend());

    // Test with const iterators
    invoke_on_all_policies<2>()(test_with_iterators(), seq.cbegin(), seq.cend());
}
```

### Pattern 4: Predicate Testing

```cpp
struct test_with_predicate {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        using T = typename std::iterator_traits<Iterator>::value_type;

        // Test with default comparator
        auto result1 = algorithm(std::forward<Policy>(exec), first, last);

        // Test with explicit predicate
        auto result2 = algorithm(std::forward<Policy>(exec), first, last, std::less<T>());

        EXPECT_EQ(result1, result2, "predicate version gave different result");
    }
};
```

### Pattern 5: Edge Case Testing

```cpp
void test_edge_cases() {
    // Empty sequence
    Sequence<T> empty(0);
    invoke_on_all_policies<0>()(test_op(), empty.begin(), empty.end());

    // Single element
    Sequence<T> single(1, [](size_t) { return T(42); });
    invoke_on_all_policies<1>()(test_op(), single.begin(), single.end());

    // All same value
    Sequence<T> uniform(100, [](size_t) { return T(0); });
    invoke_on_all_policies<2>()(test_op(), uniform.begin(), uniform.end());
}
```

---

## Configuration and Feature Detection

**Location**: `test/support/test_config.h`

### Backend Detection Macros

```cpp
TEST_DPCPP_BACKEND_PRESENT    // 1 if SYCL backend available
TEST_TBB_BACKEND_PRESENT      // 1 if TBB backend available
TEST_UNNAMED_LAMBDAS          // 1 if unnamed lambdas supported
TEST_EXPLICIT_KERNEL_NAMES    // 1 if explicit kernel names required
```

### Conditional Compilation

```cpp
#if TEST_DPCPP_BACKEND_PRESENT
    // Device-specific tests
    test1buffer<sycl::usm::alloc::shared, int, MyTest>();
#endif

#if !ONEDPL_FPGA_DEVICE
    // Skip on FPGA devices
    invoke_on_all_policies<>()(test, seq.cbegin(), seq.cend());
#endif
```

### Compiler Workarounds

Many compiler/platform-specific workarounds are defined in `test_config.h`:

```cpp
_PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN
_PSTL_ICPX_TEST_MINMAX_ELEMENT_PASS_BROKEN
_PSTL_TEST_COMPLEX_ACOS_BROKEN
// ... and many more
```

**Usage**:
```cpp
#if !_PSTL_SOME_KNOWN_ISSUE_BROKEN
    // Test code that triggers the issue
#endif
```

### Standard Library Feature Detection

```cpp
_ENABLE_RANGES_TESTING         // Ranges API available (C++17+ for DPCPP backend)
_ENABLE_STD_RANGES_TESTING     // C++20 std::ranges available
TEST_CPP20_SPAN_PRESENT        // std::span available
```

---

## Best Practices

### 1. Always Use Unique CallNumbers

```cpp
// ❌ BAD - Reuses CallNumber 0
invoke_on_all_policies<0>()(test1, data);
invoke_on_all_policies<0>()(test2, data);  // May cause kernel name collision!

// ✅ GOOD - Unique CallNumbers
invoke_on_all_policies<0>()(test1, data);
invoke_on_all_policies<1>()(test2, data);
```

### 2. Test Edge Cases

Always test:
- Empty sequences (size 0)
- Single element (size 1)
- Small sizes (2-16)
- Large sizes (up to max_n or beyond)
- Uniform values
- Reverse-sorted data

### 3. Use Descriptive Error Messages

```cpp
// ❌ BAD
EXPECT_TRUE(result, "failed");

// ✅ GOOD
EXPECT_TRUE(result == expected, "transform produced incorrect result at position");
```

### 4. Verify Against Serial Implementation

```cpp
// Run serial algorithm for expected result
auto expected = std::algorithm(seq.begin(), seq.end());

// Run parallel algorithm
auto result = std::algorithm(exec, seq.begin(), seq.end());

EXPECT_EQ(expected, result, "parallel result differs from serial");
```

### 5. Use TestUtils Namespace

```cpp
// ✅ GOOD - Clean, standard pattern
using namespace TestUtils;

int main() {
    Sequence<int> seq(100);
    invoke_on_all_policies<0>()(test_op(), seq.begin(), seq.end());
    return done();
}
```

### 6. Handle Type Support Gracefully

The framework automatically handles this for `invoke_on_all_hetero_policies`, but for manual tests:

```cpp
auto queue = TestUtils::get_test_queue();
if (!TestUtils::has_type_support<double>(queue.get_device())) {
    std::cout << "Device does not support double, skipping test\n";
    return TestUtils::done(0);  // Return skip code
}
```

### 7. Policy Value Categories

Test with different policy value categories when using `TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY`:

```cpp
auto policy = get_dpcpp_test_policy<0>();

// Automatic checking in invoke_on_all_hetero_policies tests:
// - policy (lvalue)
// - std::move(policy) (rvalue)
// - const policy& (const lvalue)
```

### 8. Use Appropriate Sequence Constructors

```cpp
// Initialize with function
Sequence<int> seq1(100, [](size_t i) { return i * 2; });

// Initialize with constant
Sequence<int> seq2(100);
seq2.fill(42);

// Initialize from initializer list
Sequence<int> seq3({1, 2, 3, 4, 5});
```

### 9. Understand Test Data Ownership

For `test_base_data` patterns:
- USM/device memory: Data may not be accessible from host
- Use `TestDataTransfer` or `retrieve_data`/`update_data` methods
- RAII pattern ensures data synchronization

### 10. Debug with Print Utilities

```cpp
// Print sequence (first 100 elements)
seq.print();

// Debug print (only if _ONEDPL_DEBUG_SYCL defined)
PRINT_DEBUG("Debug message");
```

---

## Helper Classes and Types

### Memory Checking

```cpp
// Tracks object construction/destruction
struct MemoryChecker {
    static std::atomic<std::size_t> alive_object_counter;
    static constexpr std::size_t alive_state;
    static constexpr std::size_t dead_state;

    static std::size_t alive_objects();  // Get current count
};
```

**Use Case**: Verify no memory leaks in algorithms.

### Test Predicates and Operations

From `utils.h`:

```cpp
// Predicates
template<typename T> struct IsEven;
template<typename T> struct IsOdd;
template<typename T> struct IsMultipleOf { T value; };
template<typename T> struct IsGreatThan { T value; };
template<typename T> struct IsLessThan { T value; };

// Binary predicates
template<typename T> struct IsGreat;
template<typename T> struct IsLess;
template<typename T> struct IsEqual;

// Operations
template<typename T1, typename T2> struct SumOp;
template<typename T> struct SumWithOp { T const_val; };
template<typename T> struct Pow2;  // x * x
```

### Special Test Types

```cpp
// Type with no default constructor, only ==
class Number;

// Type with associative but non-commutative operation
class MonoidElement;

// 2x2 matrix for testing non-commutative operations
template<typename T> struct Matrix2x2;

// Wrapper for move-only semantics
template<typename T> struct MoveOnlyWrapper;

// Wrapper without default constructor
template<typename T> struct NoDefaultCtorWrapper;
```

---

## Utilities Reference

### Size Patterns

```cpp
// Get test size pattern (handles large inputs on GPU)
std::vector<std::size_t> get_pattern_for_test_sizes();
```

**Returns**: Monotonically increasing sequence covering:
- Small sizes: 0-16 (by 1)
- Medium sizes: Grows by ~π factor
- Large sizes: Up to max_n or device-specific limits

### Data Generation

```cpp
// Generate random arithmetic data (with duplicates for testing)
template<typename T>
void generate_arithmetic_data(T* input, std::size_t size, std::uint32_t seed);
```

**Features**:
- 75% unique values
- 25% duplicates of unique values
- Special handling for floating-point (log-uniform distribution)

---

## Quick Reference

### Most Common Includes

```cpp
#include "support/test_config.h"
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)
#include "support/utils.h"
```

### Most Common Patterns

```cpp
using namespace TestUtils;

struct my_test {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last) {
        // Test implementation
    }
};

int main() {
    Sequence<int> seq(100, [](size_t i) { return i; });
    invoke_on_all_policies<0>()(my_test(), seq.begin(), seq.end());
    return done();
}
```

### Most Common Utilities

| Utility | Purpose |
|---------|---------|
| `EXPECT_TRUE(cond, msg)` | Assert condition |
| `Sequence<T>` | Test data container |
| `invoke_on_all_policies<N>()` | Run on all execution policies |
| `get_test_queue()` | Get SYCL queue |
| `done()` | Signal test completion |

---

## Additional Resources

- **Main Documentation**: `documentation/library_guide/`
- **CMake Build Guide**: `cmake/README.md`
- **Development Guide**: `CLAUDE.md` (in repository root)
- **Support Utilities Source**: `test/support/*.h`

---

*Document Version: 1.0*
*Last Updated: 2025-01-20*
*Maintained by: oneDPL Development Team*
