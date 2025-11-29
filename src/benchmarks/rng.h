#pragma once
#include <vector>
#include <random>
#include <thread>
#include <algorithm>

namespace bench {
    template<typename T>
    std::vector<T> generate_random_series(std::size_t N) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        std::vector<T> series;
        series.reserve(N);
        std::generate_n(std::back_inserter(series), N, [&]() { return dist(gen); });

        return series;
    }

    template<typename T>
    std::vector<T> generate_random_mt(std::size_t N) {
        std::vector<T> series(N);

        // Decide how many threads to use
        const std::size_t hw = std::thread::hardware_concurrency();
        const std::size_t num_threads = hw ? hw : 4;
        const std::size_t chunk = (N + num_threads - 1) / num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        std::random_device rd;

        for (std::size_t t = 0; t < num_threads; ++t) {
            const std::size_t begin = t * chunk;
            const std::size_t end = std::min(begin + chunk, N);
            if (begin >= end) break;

            // Launch one thread per chunk
            threads.emplace_back([&, begin, end, seed = rd() + static_cast<unsigned>(t)] {
                std::mt19937 gen(seed);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (std::size_t i = begin; i < end; ++i)
                    series[i] = static_cast<T>(dist(gen));
                });
        }

        for (auto& th : threads)
            th.join();

        return series;
    }
}