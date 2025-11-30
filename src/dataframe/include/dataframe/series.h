#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <execution>
#include <vector>
#include <iostream>
#include <optional>


namespace df {

    enum class ExecPolicy {
        SEQ,
        PAR,
        UNSEQ,
        PAR_UNSEQ
    };

    [[noreturn]] inline void unreachable_policy() noexcept {
        assert(false && "Unknown ExecPolicy"); // in debug
        std::abort();                          // hard-stop in release
    }

    /// Helper to execute a function with the appropriate execution policy
    // policy: the execution policy to use
    // f: the function to execute, which takes an execution policy as argument
    template <class F>
    decltype(auto) with_policy(ExecPolicy policy, F&& f) {
        switch (policy) {
        case ExecPolicy::SEQ:
            return f(std::execution::seq);
        case ExecPolicy::PAR:
            return f(std::execution::par);
        case ExecPolicy::PAR_UNSEQ:
            return f(std::execution::par_unseq);
        case ExecPolicy::UNSEQ:
            return f(std::execution::unseq);
        }

        // should not reach here, but assert/abort to be safe
        unreachable_policy();
    }

    template <typename DataType_>
    class Series {
    public:
        // Construct a Series by applying a monadic functor to an input Series
        template <typename T, typename Func>
        Series(const Series<T>& other, Func&& func) : data_(other.size()) {
            other.transform_to(*this, std::forward<Func>(func));
        }
        
        // Construct a Series by applying a dyadic functor to two input Series
        template <typename T, typename J, typename Func>
        Series(const Series<T>& lhs, const Series<J>& rhs, Func&& func)  {
            if (lhs.size() != rhs.size()) {
                throw std::invalid_argument("Series sizes do not match for dyadic operation");
            }
            data_.resize(lhs.size());
            lhs.transform_to(rhs, *this, std::forward<Func>(func));
        }

        explicit Series(std::vector<DataType_> data): data_(std::move(data)) {}
        explicit Series(std::initializer_list<DataType_> data): data_(std::move(data)) {}
        Series(ExecPolicy policy, std::vector<DataType_> data): data_(std::move(data)), exec_(policy) {}
        Series() = default;

        std::size_t size() const { return data_.size(); }

        // lvalue operations (ops on named values)

        template <typename T>
        auto& add(const T& val) & {
            return transform([val](const auto& x) { return val + x; });
        }

        template <typename T>
        auto& add(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x + o; });
        }

        template <typename T>
        auto& sub(const T& val) & {
            return transform([val](const auto& x) { return x - val; });
        }

        template <typename T>
        auto& sub(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x - o; });
        }

        template <typename T>
        auto& rsub(const T& val) & {
            return transform([val](const auto& x) { return val - x; });
        }

        template <typename T>
        auto& rsub(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return o - x; });
        }

        template <typename T>
        auto& mul(const T& val) & {
            return transform([val](const auto& x) { return x * val; });
        }

        template <typename T>
        auto& mul(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x * o; });
        }

        template <typename T>
        auto& div(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x / o; });
        }

        template <typename T>
        auto& div(const T& val) & {
            return transform([val](const auto& x) { return x / val; });
        }

        template <typename T>
        auto& rdiv(const T& val) & {
            return transform([val](const auto& x) { return val / x; });
        }

        template <typename T>
        auto& rdiv(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return o / x; });
        }
        
        template <typename T>
        auto& pow(const T& val) & {
            return transform([val](const auto& x) { return std::pow(x, val); });
        }

        template <typename T>
        auto& pow(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return std::pow(x, o); });
        }

        template <typename T>
        auto& min(const T& val) & {
            return transform([val](const auto& x) { return std::min(x, val); });
        }

        template <typename T>
        auto& min(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return std::min(x, o); });
        }

        template <typename T>
        auto& max(const T& val) & {
            return transform([val](const auto& x) { return std::max(x, val); });
        }

        template <typename T>
        auto& max(const Series<T>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return std::max(x, o); });
        }

        auto& exp() & {
            return transform([](const auto& x) { return std::exp(x); });
        }

        auto& log() & {
            return transform([](const auto& x) { return std::log(x); });
        }

        auto& sqrt() & {
            return transform([](const auto& x) { return std::sqrt(x); });
        }

        auto& abs() & {
            return transform([](const auto& x) { return std::abs(x); });
        }

        auto& signum() & {
            return transform([](const auto& x) { return (x > 0) - (x < 0); });
        }

        // rvalue overloads (ops on temporary values)

        template <typename T>
        auto&& add(const T& val) && {
           add(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& add(const Series<T>& other) && {
           add(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& sub(const T& val) && {
           sub(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& sub(const Series<T>& other) && {
           sub(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& rsub(const T& val) && {
           rsub(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& rsub(const Series<T>& other) && {
           rsub(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& mul(const T& val) && {
           mul(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& mul(const Series<T>& other) && {
           mul(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& div(const T& val) && {
           div(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& rdiv(const T& val) && {
           rdiv(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& rdiv(const Series<T>& other) && {
           rdiv(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& pow(const T& val) && {
           pow(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& pow(const Series<T>& other) && {
           pow(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& min(const T& val) && {
           min(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& min(const Series<T>& other) && {
           min(other);
           return std::move(*this);
        }

        template <typename T>
        auto&& max(const T& val) && {
           max(val);
           return std::move(*this);
        }

        template <typename T>
        auto&& max(const Series<T>& other) && {
           max(other);
           return std::move(*this);
        }

        auto&& exp() && {
           exp();
           return std::move(*this);
        }

        auto&& log() && {
           log();
           return std::move(*this);
        }

        auto&& sqrt() && {
           sqrt();
           return std::move(*this);
        }

        auto&& abs() && {
           abs();
           return std::move(*this);
        }

        auto&& signum() && {
           signum();
           return std::move(*this);
        }

        // Operators

        template <typename LhsT, typename RhsT>
        friend auto operator+(const Series<LhsT>& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator+(const Series<LhsT>& lhs, const RhsT& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator+(const LhsT& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator-(const Series<LhsT>& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator-(const Series<LhsT>& lhs, const RhsT& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator-(const LhsT& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator*(const Series<LhsT>& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator*(const Series<LhsT>& lhs, const RhsT& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator*(const LhsT& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator/(const Series<LhsT>& lhs, const Series<RhsT>& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator/(const Series<LhsT>& lhs, const RhsT& rhs);

        template <typename LhsT, typename RhsT>
        friend auto operator/(const LhsT& lhs, const Series<RhsT>& rhs);

        template <typename T>
        auto& operator+=(const T& val) {
            return add(val);
        }

        template <typename T>
        auto& operator+=(const Series<T>& other) {
            return add(other);
        }

        template <typename T>
        auto& operator-=(const T& val) {
            return sub(val);
        }

        template <typename T>
        auto& operator-=(const Series<T>& other) {
            return sub(other);
        }

        template <typename T>
        auto& operator*=(const T& val) {
            return mul(val);
        }

        template <typename T>
        auto& operator*=(const Series<T>& other) {
            return mul(other);
        }

        template <typename T>
        auto& operator/=(const T& val) {
            return div(val);
        }

        template <typename T>
        auto& operator/=(const Series<T>& other) {
            return div(other);
        }

        // Aggregation functions

        DataType_ dot(const Series<DataType_>& other) const {
            return std::transform_reduce(
                exec_,
                data_.begin(), data_.end(),
                other.data_.begin(),
                DataType_{},
                std::plus<>(),
                std::multiplies<>()
            );
        }

        std::optional<DataType_> sum() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return with_policy(exec_, [&](auto& exec_){
                return std::reduce(exec_, data_.begin(), data_.end(), DataType_{}, std::plus<>{});
            });
        }

        std::optional<DataType_> mean() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return sum().value() / static_cast<DataType_>(size());
        }

        std::optional<DataType_> variance() const {
            if (size() == 0) {
                return std::nullopt;
            }
            const auto m = mean().value();
            return with_policy(exec_, [&](auto& exec_){
                return std::transform_reduce(
                    exec_,
                    data_.begin(), data_.end(),
                    DataType_{},
                    std::plus<>(),
                    [&m](const auto& x) { return (x - m) * (x - m); }
                ) / static_cast<DataType_>(size());
            });
        }

        std::optional<DataType_> stddev() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return std::sqrt(variance().value());
        }

        std::optional<std::reference_wrapper<DataType_>> min() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return *std::min_element(exec_, data_.begin(), data_.end());
        }

        std::optional<std::reference_wrapper<DataType_>> max() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return *std::max_element(exec_, data_.begin(), data_.end());
        }

        // output the series as "[ *, *, * ]" where * is the type
        friend std::ostream& operator<<(std::ostream& os, const Series<DataType_>& obj) {
            // if 10 or less, show all
            // otherwise, show first 5, ..., last 5
            constexpr auto MAX_DISPLAY{10};
            constexpr auto CHUNK{MAX_DISPLAY / 2};
            os << "[";
            if (obj.data_.size() <= MAX_DISPLAY) {
                for (auto i = 0; i < obj.data_.size(); ++i) {
                    os << obj.data_[i];
                    if (i < obj.data_.size() - 1) {
                        os << ", ";
                    }
                }
            }
            else {
                const auto f = [&os](const auto& x) { os << x << ", "; };
                std::for_each_n(obj.data_.begin(), CHUNK, f);
                os << "..., ";
                std::for_each_n(obj.data_.end() - CHUNK, CHUNK - 1, f);
                os << *(obj.data_.end() - 1);
            }
            os << "]";
            return os;

        }

        // access operator
        DataType_& operator[](std::size_t idx) {
            return data_[idx];
        }
        const DataType_& operator[](std::size_t idx) const {
            return data_[idx];
        }

        // iterator support for stl algorithms
        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }
        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }


    private:
        ExecPolicy exec_{ExecPolicy::PAR_UNSEQ};

        // The underlying data storage
        std::vector<DataType_> data_;

        // Transform this series with the result of a monadic functor applied to each element
        // functor: the monadic functor to apply: functor(this[i]) -> this[i]
        template <typename Func_>
        auto& transform(Func_&& functor) {
            transform_to(*this, functor);
            return *this;
        }

        // Transform the output with the result of a monadic functor applied to this series elementwise
        // output: the series which receives the output
        // functor: the monadic functor to apply: functor(this[i]) -> this[i]
        template <typename T, typename Func_>
        auto& transform_to(Series<T>& output, Func_&& functor) const {
            with_policy(exec_, [&](auto exec_){
                std::transform(exec_, data_.begin(), data_.end(), output.data_.begin(), functor);
            });
            return *this;
        }

        // Transform this series with a functor applied elementwise with another series
        // other: the other series which the functor receives elements from
        // functor: the dyadic functor to apply: functor(this[i], other[i]) -> this[i]
        template <typename Func_>
        auto& transform(const Series<DataType_>& other, Func_&& functor) {
            transform_to(other, *this, functor);
            return *this;
        }

        // Transform the output with the result of a functor applied to this series elementwise with another
        // other: the other series the functor receives elements from
        // output: the series which receives the output
        // functor: the dyadic functor to apply: functor(this[i], other[i]) -> output[i]
        template <typename T, typename J, typename Func_>
        auto& transform_to(const Series<T>& other, Series<J>& output, Func_&& functor) const {
            with_policy(exec_, [&](auto& exc){
                std::transform(exc, data_.begin(), data_.end(), other.data_.begin(), output.data_.begin(), functor);
            });
            return *this;
        }
    };

    // Operators that return new Series constructed from transformations

    template <typename LhsT, typename RhsT>
    auto operator+(const Series<LhsT>& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} + RhsT{})>(lhs, rhs, [](const auto& x, const auto& y) { return x + y; });
    }

    template <typename LhsT, typename RhsT>
    auto operator+(const Series<LhsT>& lhs, const RhsT& rhs) {
        return Series<decltype(LhsT{} + RhsT{})>(lhs, [rhs](const auto& x) { return x + rhs; });
    }

    template <typename LhsT, typename RhsT>
    auto operator+(const LhsT& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} + RhsT{})>(rhs, [lhs](const auto& x) { return lhs + x; });
    }

    template <typename LhsT, typename RhsT>
    auto operator-(const Series<LhsT>& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} - RhsT{})>(lhs, rhs, [](const auto& x, const auto& y) { return x - y; });
    }

    template <typename LhsT, typename RhsT>
    auto operator-(const Series<LhsT>& lhs, const RhsT& rhs) {
        return Series<decltype(LhsT{} - RhsT{})>(lhs, [rhs](const auto& x) { return x - rhs; });
    }

    template <typename LhsT, typename RhsT>
    auto operator-(const LhsT& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} - RhsT{})>(rhs, [lhs](const auto& x) { return lhs - x; });
    }

    template <typename LhsT, typename RhsT>
    auto operator*(const Series<LhsT>& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} * RhsT{})>(lhs, rhs, [](const auto& x, const auto& y) { return x * y; });
    }

    template <typename LhsT, typename RhsT>
    auto operator*(const Series<LhsT>& lhs, const RhsT& rhs) {
        return Series<decltype(LhsT{} * RhsT{})>(lhs, [rhs](const auto& x) { return x * rhs; });
    }

    template <typename LhsT, typename RhsT>
    auto operator*(const LhsT& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} * RhsT{})>(rhs, [lhs](const auto& x) { return lhs * x; });
    }

    template <typename LhsT, typename RhsT>
    auto operator/(const Series<LhsT>& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} / RhsT{})>(lhs, rhs, [](const auto& x, const auto& y) { return x / y; });
    }

    template <typename LhsT, typename RhsT>
    auto operator/(const Series<LhsT>& lhs, const RhsT& rhs) {
        return Series<decltype(LhsT{} / RhsT{})>(lhs, [rhs](const auto& x) { return x / rhs; });
    }

    template <typename LhsT, typename RhsT>
    auto operator/(const LhsT& lhs, const Series<RhsT>& rhs) {
        return Series<decltype(LhsT{} / RhsT{})>(rhs, [lhs](const auto& x) { return lhs / x; });
    }
}