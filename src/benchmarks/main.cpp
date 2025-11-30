#include "df.h"
#include "rng.h"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <execution>
#include <benchmark/benchmark.h>


namespace {
    using namespace df;
    constexpr auto NUM_CALCS{1'000'000};

	template<typename T=double>
    Series<T> generate_random_series(std::size_t N) {
        return Series<T>(bench::generate_random_mt<T>(N));
	}

    void calc1_loop(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        constexpr double a = 0.98, b = 1.0, c = 0.9;
        for (auto _ : state) {
            std::transform(
                std::execution::par_unseq,
                c1.begin(),
                c1.end(),
                c1.begin(),
                [&](double x) { return c + std::exp(a + b * x); }
            );
        }
    }
    BENCHMARK(calc1_loop);

    void calc1_series(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        constexpr double a = 0.98, b = 1.0, c = 0.9;
        for (auto _ : state) {
            c1.mul(b).add(a).exp().add(c);
        }
    }
    BENCHMARK(calc1_series);

    void add_scalar(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        const auto val = c1[0];
        for (auto _ : state) {
            c1.add(val);
        }
    }
    BENCHMARK(add_scalar);

    void add_series(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            c1.add(c1);
        }
    }
    BENCHMARK(add_series);

    void add_series_operator(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            auto c2 = c1 + c1;
            benchmark::DoNotOptimize(c2);
        }
    }
    BENCHMARK(add_series_operator);

    void mul_scalar(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        const auto val = c1[0];
        for (auto _ : state) {
            c1.mul(val);
        }
    }
    BENCHMARK(mul_scalar);

    void mul_series(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            c1.mul(c1);
        }
    }
    BENCHMARK(mul_series);

    void mul_series_operator(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            auto c2 = c1 * c1;
            benchmark::DoNotOptimize(c2);
        }
    }
    BENCHMARK(mul_series_operator);

    void sqrt_series(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            c1.sqrt();
        }
    }
    BENCHMARK(sqrt_series);

    void exp_series(benchmark::State& state) {
        auto c1 = generate_random_series(NUM_CALCS);
        for (auto _ : state) {
            c1.exp();
        }
    }
    BENCHMARK(exp_series);


}  // namespace

BENCHMARK_MAIN();
