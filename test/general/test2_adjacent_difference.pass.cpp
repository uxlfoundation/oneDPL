// icpx -fsycl -fno-sycl-unnamed-lambda -fsycl-device-code-split=per_kernel
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <iostream>

int main()
{
    sycl::queue q;

    int n = 10;
    int *ptr = sycl::malloc_shared<int>(n, q);
    int *ptr_out = sycl::malloc_shared<int>(n, q);
    q.fill(ptr, 1, n).wait();

    oneapi::dpl::execution::device_policy<class kernel> policy{q};

    // OK
    oneapi::dpl::adjacent_difference(policy, ptr, ptr + n, ptr_out, std::plus<int>{});

    // Error case (1)
    oneapi::dpl::adjacent_difference(std::move(policy), ptr, ptr + n, ptr_out, std::plus<int>{});

    // Error case (2)
    const auto& policy_ref = policy;
    oneapi::dpl::adjacent_difference(policy_ref, ptr, ptr + n, ptr_out, std::plus<int>{});

    return 0;
}
