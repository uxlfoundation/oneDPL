Tested Standard C++ APIs
########################

The basic functionality for several C++ standard APIs has been tested for use in SYCL* kernels.
These APIs can be employed in device kernels similarly to how they are employed in code for a typical CPU-based platform.
The Tested Standard C++ APIs are added to the namespace ``oneapi::dpl``. The corresponding headers have been added in the
|onedpl_long| (|onedpl_short|) package. In order to use these APIs via the namespace ``oneapi::dpl``, the headers in
``<oneapi/dpl/...>`` must be included. Currently, Tested Standard C++ APIs can be used in two ways:

#. Via the namespace ``std::`` and standard headers (for example: ``<utility>...``)
#. Via the namespace ``oneapi::dpl`` and |onedpl_short| headers (for example: ``<oneapi/dpl/utility>...``)

Below is an example code that shows how to use ``oneapi::dpl::swap`` in SYCL device code:

.. code:: cpp

  #include <oneapi/dpl/utility>
  #include <sycl/sycl.hpp>
  #include <iostream>
  #include <cstdint>
  int main()
  {
      sycl::queue queue;
      constexpr std::uint32_t size = 2;
      std::uint32_t data[size] = {4, 5};
      std::cout << "Initial data: " << data[0] << ", " << data[1] << std::endl;
      sycl::buffer<std::uint32_t> buffer(data, size);
      queue.submit([&](sycl::handler& cgh) {
          auto access = buffer.get_access(cgh, sycl::read_write);
          cgh.single_task<class KernelSwap>([=]() {
              oneapi::dpl::swap(access[0], access[1]);
          });
      }).wait();
      auto host_access = buffer.get_host_access(sycl::read_only);
      std::cout << "After swap: " << host_access[0] << ", " << host_access[1] << std::endl;
      return 0;
  }

Use the following command to build and run the program (assuming it resides in the ``kernel_swap.cpp file``):

.. code:: cpp

  icpx -fsycl kernel_swap.cpp -o kernel_swap && ./kernel_swap

The printed result is:

.. code:: cpp

  Initial data: 4, 5
  After swap: 5, 4

Tested Standard C++ API Reference
=================================

===================================== ========== ========== ==========
C++ Standard API                      libstdc++  libc++     MSVC
===================================== ========== ========== ==========
``std::swap``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::lower_bound``                  Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::upper_bound``                  Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::binary_search``                Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::equal_range``                  Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::tuple``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::pair``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::reference_wrapper``            Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::ref/cref``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::divides``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::minus``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::plus``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::negate``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::modulus``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::multiplies``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::equal_to``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::greater``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::greater_equal``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::less``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::less_equal``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::not_equal_to``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::bit_and``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::bit_not``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::bit_xor``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::bit_or``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::logical_and``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::logical_or``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::logical_not``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::binary_negate``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::unary_negate``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::not1/2``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::initializer_list``             Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::forward``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::move``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::move_if_noexcept``             Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::integral_constant``            Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_same``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_base_of``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_base_of_union``             Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_convertible``               Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::extent``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::rank``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remove_all_extents``           Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remove_extent``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::add_const``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::add_cv``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::add_volatile``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remove_const``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remove_cv``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remove_volatile``              Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::decay``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::conditional``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::enable_if``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::common_type``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::declval``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::alignment_of``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_arithmetic``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_fundamental``               Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_reference``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_object``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_scalar``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_compound``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_member_pointer``            Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_const``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_assignable``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_constructible``             Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_copy_assignable``           Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_copy_constructible``        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_default_constructible``     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_destructible``              Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_empty``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_literal_type``              Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_move_assignable``           Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_move_constructible``        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_pod``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_signed``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_standard_layout``           Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_trivial``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_unsigned``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_volatile``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_trivially_assignable``      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_trivially_constructible``   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_trivially_copyable``        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::array``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::ratio``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::complex``                      Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::abs``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::arg``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::conj``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::exp``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::imag``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::norm``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::polar``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::proj``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::real``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::assert``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::sin``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::cos``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::tan``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::asin``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::acos``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::atan``                         Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::atan2``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::sinh``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::cosh``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::tanh``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::asinh``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::acosh``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::atanh``                        Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::exp``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::frexp``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::ldexp``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::log``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::log10``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::modf``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::exp2``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::expm1``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::ilogb``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::log1p``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::log2``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::logb``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::pow``                          Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::sqrt``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::cbrt``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::hypot``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::erf``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::erfc``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::tgamma``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::lgamma``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fmod``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remainder``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::remquo``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::nextafter``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::nearbyint``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::nearbyintf``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fdim``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::optional``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::reduce``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::all_of``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::any_of``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::none_of``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::count``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::count_if``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::for_each``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::find``                         Tested                Tested
------------------------------------- ---------- ---------- ----------
``std::find_if``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::find_if_not``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::for_each_n``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::ceil``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copy``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copy_backward``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copy_if``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copy_n``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copysign``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::copysignf``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fabs``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_permutation``               Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fill``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fill_n``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::floor``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fmax``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fmaxf``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fmin``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::fminf``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::move``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::move_backward``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_sorted``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_sorted_until``              Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isgreater``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isgreaterequal``               Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isinf``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isless``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::islessequal``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isnan``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::isunordered``                  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::partial_sort``                 Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::partial_sort_copy``            Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_heap``                      Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::is_heap_until``                Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::make_heap``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::max``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::min``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::nan``                          Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::nanf``                         Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::numeric_limits<T>::infinity``  Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::numeric_limits<T>::lowest``    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::numeric_limits<T>::max``       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::numeric_limits<T>::quiet_NaN`` Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::push_heap``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::pop_heap``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::generate``                     Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::generate_n``                   Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::transform``                    Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::round``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::roundf``                       Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::trunc``                        Tested     Tested     Tested
------------------------------------- ---------- ---------- ----------
``std::truncf``                       Tested     Tested     Tested
===================================== ========== ========== ==========

The testing was conducted for the following versions of the C++ standard libraries, with the help of the listed compilers:

============================================= ============================================= ==================================
C++ Standard Library                          C++ Standard Library Version                  SYCL Compiler Used
============================================= ============================================= ==================================
libstdc++ (GNU)                               Provided with GCC* 8.4.0, GCC 9.3.0,          Intel® oneAPI DPC++/C++ Compiler
                                              GCC 11.4.0, GCC 13.2.0
--------------------------------------------- --------------------------------------------- ----------------------------------
Microsoft Visual C++* (MSVC) Standard Library Provided with Microsoft Visual Studio 2019    Intel® oneAPI DPC++/C++ Compiler 
                                              and Microsoft Visual Studio 2022.
--------------------------------------------- --------------------------------------------- ----------------------------------
libc++ (LLVM)                                 Provided with Clang* 18.1, Clang 19.1,        AdaptiveCpp
                                              Clang 20.1
============================================= ============================================= ==================================
