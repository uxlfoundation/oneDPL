## Implementation Notes from POC

A proof-of-concept implementation validated the design and uncovered the
following points that should be considered during productization:

- **`device_reference` host/device bifurcation.** `device_reference::operator T()`
  and `operator=` must use different code paths depending on whether they are
  compiled for host or device. On the host, they use `sycl::queue::memcpy`
  (synchronous). On the device, they dereference the raw USM pointer directly.
  The `__SYCL_DEVICE_ONLY__` macro selects the correct path. This keeps the
  API consistent across compilation passes (same type, same member functions)
  while the implementation adapts.

- **`operator std::vector<T>()` should be non-const.** `sycl::queue::memcpy`
  is a non-const member function, so `operator std::vector<T>() const` does
  not compile without `mutable` on the queue. Making the conversion operator
  non-const is the simpler option, but const element access (`operator[] const`,
  `begin() const`, etc.) still requires `mutable sycl::queue` since they
  construct `device_reference`/`device_pointer` objects that hold a non-const
  queue pointer.

- **`sycl::queue::memcpy` requires `void*` casts.** The `sycl::queue::memcpy`
  overload set includes template overloads for `device_global` types. Passing
  typed pointers (e.g. `T*`) can be ambiguous. Explicit
  `static_cast<void*>` / `static_cast<const void*>` on the source and
  destination pointers resolves the ambiguity.

- **oneDPL `__brick_fill` / `__brick_fill_n` incompatibility with proxy
  references.** The hetero specializations of `__brick_fill` take their
  target parameter by lvalue reference (`_TargetT& __target`). When the
  iterator's `operator[]` returns a proxy reference prvalue (as
  `device_pointer` does), the prvalue cannot bind to the lvalue reference.
  Changing the parameter to a forwarding reference (`_TargetT&& __target`)
  fixes this for all proxy reference types. This is a pre-existing oneDPL
  bug, not specific to `device_vector`.

- **POC location.** The POC header is at
  `include/oneapi/dpl/experimental/device_vector.h` with tests at
  `test/parallel_api/experimental/device_vector.pass.cpp`.
