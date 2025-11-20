# oneDPL Development Guide for AI Assistants

This file provides high-level navigation for AI coding assistants working on the **oneAPI DPC++ Library (oneDPL)** project.

## Essential Documentation

### 📖 Core Guides
- **[Development Guide](documentation/internal/development_guide.md)** - Complete architecture, code patterns, and implementation details
- **[Test Infrastructure Guide](documentation/internal/test_infrastructure.md)** - Testing utilities, patterns, and best practices
- **[CMake Build Guide](cmake/README.md)** - Build system configuration and usage
- **[Contributing Guidelines](CONTRIBUTING.md)** - Contribution process and requirements

### 📂 Key Directories
- **`include/oneapi/dpl/`** - Public API headers (header-only library)
- **`include/oneapi/dpl/pstl/`** - Implementation layer (patterns, backends)
- **`test/`** - Comprehensive test suite with utilities in `test/support/`
- **`documentation/`** - API documentation and guides
- **`examples/`** - Example code for oneDPL usage

## Quick Architecture Overview

oneDPL is a **header-only C++ library** providing parallel algorithms for heterogeneous computing:

```
User API (oneapi::dpl::algorithm)
    ↓
Glue Layer (policy-based dispatch)
    ↓
Pattern Layer (algorithm decomposition)
    ↓
Backend Layer (TBB, OpenMP, SYCL/DPC++)
    ↓
Device Execution (CPU, GPU, FPGA)
```

### Key Concepts
1. **Execution Policies**: `seq`, `par`, `unseq`, `par_unseq` (host), `dpcpp_default` (device)
2. **Backend Abstraction**: Same algorithm works on different backends
3. **Iterator Patterns**: Custom iterators (zip, permutation, transform, counting)
4. **SYCL Integration**: Deep integration with SYCL for device execution

## Development Workflow

### Making Changes
1. **Algorithm changes**: Modify files in `include/oneapi/dpl/pstl/`
2. **Testing changes**: Add/modify tests in `test/parallel_api/`
3. **Build and test**: Use CMake (see `cmake/README.md`)
4. **Follow patterns**: Check `development_guide.md` for architecture patterns

### Testing Strategy
- **Host policies**: Use `Sequence<T>` + `invoke_on_all_host_policies`
- **Device policies**: Use `test_base` infrastructure for systematic testing across sizes and memory types
- **See**: `test_infrastructure.md` for complete testing guide

### Common Tasks

| Task | Documentation | Key Files |
|------|--------------|-----------|
| Add algorithm | `development_guide.md` § "Adding New Algorithm" | `pstl/glue_algorithm_impl.h`, `pstl/hetero/algorithm_impl_hetero.h` |
| Fix device bug | `development_guide.md` § "SYCL Buffer Management" | `pstl/hetero/dpcpp/parallel_backend_sycl*.h` |
| Write test | `test_infrastructure.md` | `test/support/utils.h`, `test/support/utils_sycl.h` |
| Build config | `cmake/README.md` | `CMakeLists.txt`, `cmake/` |

## Project Standards

### Code Style
- Follow existing conventions in the codebase
- Use SFINAE-friendly patterns for templates
- Prefix internal symbols with `__` (double underscore)
- Namespace internal code in `__internal`

### Testing Requirements
- Test all execution policies (host + device)
- Test edge cases (empty, single element, large)
- Test with different iterator types (random access, forward, const)
- Verify against serial implementation

## Debugging Tips

### Common Issues
1. **Kernel name collisions**: Use unique `CallNumber` in `invoke_on_all_policies<N>`
2. **Buffer lifetime**: Ensure `__keep` variables stay in scope until kernel completes
3. **Access mode mismatches**: Check actual data usage pattern (read/write/read_write)
4. **Type support**: Device may not support `double` or `sycl::half`

### Debugging Tools
- `Sequence::print()` - Print first 100 elements
- `PRINT_DEBUG(msg)` - Debug print (only if `_ONEDPL_DEBUG_SYCL` defined)
- `EXPECT_TRUE(cond, msg)` - Assertion with file:line output

### External
- **Homepage**: https://uxlfoundation.github.io/oneDPL
- **Spec**: https://oneapi-spec.uxlfoundation.org
- **GitHub**: https://github.com/uxlfoundation/oneDPL
- **Discussions**: https://github.com/uxlfoundation/oneDPL/discussions
- **Slack**: https://uxlfoundation.slack.com/channels/onedpl

## License

Apache 2.0 with LLVM exceptions. See `LICENSE.txt`.

---

**For detailed information on any topic, refer to the specific documentation files listed above.**
**When in doubt, check existing code patterns in the codebase - consistency is key.**
