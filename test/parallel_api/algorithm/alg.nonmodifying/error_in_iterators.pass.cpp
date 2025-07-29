#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <vector>
#include <iostream>

//#if ONEDPL_USE_DPCPP_BACKEND
#   define POLICY oneapi::dpl::execution::dpcpp_default
//#else
//#   define POLICY oneapi::dpl::execution::par_unseq
//#endif

int main() {
    // Create a vector of bool values
    std::vector<bool> input = {true, false, true, true, false, true};

    std::cout << "Original vector of booleans:" << std::endl;
    for (const auto& val : input) {
        std::cout << (val ? "true" : "false") << " ";
    }
    std::cout << std::endl;

    // Create a vector to store the result (int type)
    std::vector<int> result(input.size());

    // Create reverse iterators
    auto input_rbegin = std::rbegin(input);
    auto input_rend = std::rend(input);
    auto result_rbegin = std::rbegin(result);

    // Use exclusive_scan with reverse iterators to convert bool to int
    // This will scan from right to left (due to reverse iterators)
    // The initial value (0) will appear at the rightmost position
    oneapi::dpl::exclusive_scan(
        POLICY,                  // Parallel execution policy
        input_rbegin,            // Start of reversed input range
        input_rend,              // End of reversed input range
        result_rbegin,           // Start of reversed output range
        int{0}                   // Initial value
    );

    std::cout << "\nOriginal vector (left to right):" << std::endl;
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << (input[i] ? 1 : 0) << " ";
    }
    std::cout << std::endl;

    std::cout << "\nResult of exclusive_scan with reverse iterators:" << std::endl;
    for (const auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}