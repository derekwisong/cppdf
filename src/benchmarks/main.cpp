#include "df.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <execution>
#include <benchmark/benchmark.h>
#include "rng.h"


namespace {
    using namespace df;
    constexpr auto NUM_CALCS{1'000'000'000};

	template<typename T>
    Series<T> generate_random_series(std::size_t N) {
        auto vec = bench::generate_random_series<T>(N);
        return Series<T>(std::move(vec));
	}


    void calc1_loop(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        const std::array<double, 3> c2({0.98, 1.0, 0.9});

        for (auto _ : state) {
            std::transform(
                std::execution::par_unseq,
                c1.begin(),
                c1.end(),
                c1.begin(),
                [&](double x) { return c2[2] + std::exp(c2[0] + c2[1] * x); }
            );
        }
    }
    BENCHMARK(calc1_loop);

    void calc1_series(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        const std::array<double, 3> c2({0.98, 1.0, 0.9});
        for (auto _ : state) {
            c1.mul(c2[1]).add(c2[0]).exp().add(c2[2]);
        }
    }
    BENCHMARK(calc1_series);

    void add_scalar(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        const auto val = c1[0];
        for (auto _ : state) {
            c1.add(val);
        }
    }
    BENCHMARK(add_scalar);

    void add_series(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        auto c2 = generate_random_series<double>(NUM_CALCS);
        for (auto _ : state) {
            c1.add(c2);
        }
    }
    BENCHMARK(add_series);

    void mul_scalar(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        const auto val = c1[0];
        for (auto _ : state) {
            c1.mul(val);
        }
    }
    BENCHMARK(mul_scalar);

    void mul_series(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        auto c2 = generate_random_series<double>(NUM_CALCS);
        for (auto _ : state) {
            c1.mul(c2);
        }
    }
    BENCHMARK(mul_series);

    void sqrt_series(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        for (auto _ : state) {
            c1.sqrt();
        }
    }
    BENCHMARK(sqrt_series);

    void exp_series(benchmark::State& state) {
        auto c1 = generate_random_series<double>(NUM_CALCS);
        for (auto _ : state) {
            c1.exp();
        }
    }
    BENCHMARK(exp_series);


}  // namespace

BENCHMARK_MAIN();
