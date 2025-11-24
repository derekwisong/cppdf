#include "df.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

int main() {
    constexpr std::size_t N{1'000'000'000};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<double> data(N);
    data.reserve(N);

    std::generate_n(data.begin(), N, [&]() { return dist(gen); });
    
    return 0;
    
    // std::cout << "df len: " << df.length() << "\n";
    // std::cout << df.column<int>("c1") << "\n";
}