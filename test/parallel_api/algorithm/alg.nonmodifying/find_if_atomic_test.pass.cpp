#include <sycl/sycl.hpp>

#include <chrono>

template <typename... KernelName>
struct __submitter
{
    std::size_t __n_groups;
    std::size_t __wgroup_size;

    sycl::event
    operator()(sycl::queue& __q, bool bDisplayTime)
    {
        using DataType = unsigned long;

        DataType __result = 0;

        // sycl::buffer scope
        {
            sycl::buffer<DataType, 1> __result_sycl_buf(&__result, 1);

            const auto __start_time = std::chrono::high_resolution_clock::now();

            auto __result = __q.submit([&](sycl::handler& __cgh) {

                auto __result_sycl_buf_acc = __result_sycl_buf.template get_access<sycl::access_mode::write>(__cgh);

                __cgh.parallel_for<KernelName...>(
                    sycl::nd_range</*dim=*/1>(sycl::range</*dim=*/1>(__n_groups * __wgroup_size),
                                              sycl::range</*dim=*/1>(__wgroup_size)),
                    [=](sycl::nd_item</*dim=*/1> __item_id) {

                        // Nothing doing inside
                        //static_assert(false);

                    });
            });

            const auto __stop_time = std::chrono::high_resolution_clock::now();
            const auto __elapsed = __stop_time - __start_time;
            if (bDisplayTime)
                std::cout << "sycl::queue::submit() call time = " << std::chrono::duration_cast<std::chrono::microseconds>(__elapsed).count() << " (mcs)" << std::endl;

            return __result;
        }
    }
};

int
main()
{
    sycl::queue q;
    std::cout << "Device Name = " << q.get_device().template get_info<sycl::info::device::name>() << "\n";

    __submitter<class KernelName> _s{ /* __n_groups */ 64, /* __wgroup_size */ 1024 };

    _s(q, false /* Just for JIT compile, time isn't important*/).wait();

    const auto __start_time = std::chrono::high_resolution_clock::now();

    auto __event = _s(q, true /* Already compiled, so time is important*/);

    const auto __stop_time = std::chrono::high_resolution_clock::now();
    __event.wait();
    const auto __elapsed = __stop_time - __start_time;
    std::cout << "__submitter::operator() call time = " << std::chrono::duration_cast<std::chrono::microseconds>(__elapsed).count() << " (mcs)" << std::endl;

    return 0;
}
