#pragma once

#include <algorithm>
#include <cmath>
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

        throw std::runtime_error("Unknown execution policy");
    }

    template <typename DataType_>
    class Series {
    public:
        explicit Series(std::vector<DataType_> data): data_(std::move(data)) {}
        explicit Series(std::initializer_list<DataType_> data): data_(std::move(data)) {}
        Series(ExecPolicy policy, std::vector<DataType_> data): data_(std::move(data)), exec_(policy) {}
        Series() = default;

        std::size_t size() const { return data_.size(); }

        // lvalue operations (ops on named values)

        auto& add(const DataType_& val) & {
            return transform([val](const auto& x) { return val + x; });
        }

        auto& add(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x + o; });
        }

        auto& sub(const DataType_& val) & {
            return transform([val](const auto& x) { return x - val; });
        }

        auto& sub(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x - o; });
        }

        auto& rsub(const DataType_& val) & {
            return transform([val](const auto& x) { return val - x; });
        }

        auto& rsub(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return o - x; });
        }

        auto& mul(const DataType_& val) & {
            return transform([val](const auto& x) { return x * val; });
        }

        auto& mul(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return x * o; });
        }

        auto& div(const DataType_& val) & {
            return transform([val](const auto& x) { return x / val; });
        }

        auto& rdiv(const DataType_& val) & {
            return transform([val](const auto& x) { return val / x; });
        }

        auto& rdiv(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return o / x; });
        }
        
        auto& pow(const DataType_& val) & {
            return transform([val](const auto& x) { return std::pow(x, val); });
        }

        auto& pow(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return std::pow(x, o); });
        }

        auto& min(const DataType_& val) & {
            return transform([val](const auto& x) { return std::min(x, val); });
        }

        auto& min(const Series<DataType_>& other) & {
            return transform(other, [](const auto& x, const auto& o) { return std::min(x, o); });
        }

        auto& max(const DataType_& val) & {
            return transform([val](const auto& x) { return std::max(x, val); });
        }

        auto& max(const Series<DataType_>& other) & {
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

        auto&& add(const DataType_& val) && {
           add(val);
           return std::move(*this);
        }

        auto&& add(const Series<DataType_>& other) && {
           add(other);
           return std::move(*this);
        }

        auto&& sub(const DataType_& val) && {
           sub(val);
           return std::move(*this);
        }

        auto&& sub(const Series<DataType_>& other) && {
           sub(other);
           return std::move(*this);
        }

        auto&& rsub(const DataType_& val) && {
           rsub(val);
           return std::move(*this);
        }

        auto&& rsub(const Series<DataType_>& other) && {
           rsub(other);
           return std::move(*this);
        }

        auto&& mul(const DataType_& val) && {
           mul(val);
           return std::move(*this);
        }

        auto&& mul(const Series<DataType_>& other) && {
           mul(other);
           return std::move(*this);
        }

        auto&& div(const DataType_& val) && {
           div(val);
           return std::move(*this);
        }

        auto&& rdiv(const DataType_& val) && {
           rdiv(val);
           return std::move(*this);
        }

        auto&& rdiv(const Series<DataType_>& other) && {
           rdiv(other);
           return std::move(*this);
        }

        auto&& pow(const DataType_& val) && {
           pow(val);
           return std::move(*this);
        }

        auto&& pow(const Series<DataType_>& other) && {
           pow(other);
           return std::move(*this);
        }

        auto&& min(const DataType_& val) && {
           min(val);
           return std::move(*this);
        }

        auto&& min(const Series<DataType_>& other) && {
           min(other);
           return std::move(*this);
        }

        auto&& max(const DataType_& val) && {
           max(val);
           return std::move(*this);
        }

        auto&& max(const Series<DataType_>& other) && {
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

        auto operator+(const DataType_& val) const {
            return Series<DataType_>{*this}.add(val);            
        }

        auto operator+(const Series<DataType_>& other) const {
            return Series<DataType_>{*this}.add(other);            
        }

        auto operator-(const DataType_& val) const {
            return Series<DataType_>{*this}.sub(val);            
        }

        auto operator-(const Series<DataType_>& other) const {
            return Series<DataType_>{*this}.sub(other);            
        }

        auto operator*(const DataType_& val) const {
            return Series<DataType_>{*this}.mul(val);            
        }

        auto operator*(const Series<DataType_>& other) const {
            return Series<DataType_>{*this}.mul(other);            
        }

        auto operator/(const DataType_& val) const {
            return Series<DataType_>{*this}.div(val);            
        }

        auto operator/(const Series<DataType_>& other) const {
            return Series<DataType_>{*this}.div(other);            
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
        template <typename Func_>
        auto& transform_to(Series<DataType_>& output, Func_&& functor) {
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
        template <typename Func_>
        auto& transform_to(const Series<DataType_>& other, Series<DataType_>& output, Func_&& functor) {
            with_policy(exec_, [&](auto& exc){
                std::transform(exc, data_.begin(), data_.end(), other.data_.begin(), output.data_.begin(), functor);
            });
            return *this;
        }
    };

    // Non-member operators to support commutative operations

    // Subtraction for val - series
    template <typename DataType_>
    auto operator-(const DataType_& val, const Series<DataType_>& series) {
        return Series<DataType_>{series}.rsub(val);
    }

    // Subtraction for series - series
    template <typename DataType_>
    auto operator-(const Series<DataType_>& other, const Series<DataType_>& series) {
        return Series<DataType_>{other}.sub(series);
    }

    // Adddition for val + series
    template <typename DataType_>
    auto operator+(const DataType_& val, const Series<DataType_>& series) {
        return Series<DataType_>{series}.add(val);
    }

    // Addition for series + series
    template <typename DataType_>
    auto operator+(const Series<DataType_>& other, const Series<DataType_>& series) {
        return Series<DataType_>{other}.add(series);
    }

    // Multiplication for val * series
    template <typename DataType_>
    auto operator*(const DataType_& val, const Series<DataType_>& series) {
        return Series<DataType_>{series}.mul(val);
    }

    // Multiplication for series * series
    template <typename DataType_>
    auto operator*(const Series<DataType_>& other, const Series<DataType_>& series) {
        return Series<DataType_>{other}.mul(series);
    }

    // Division for val / series
    template <typename DataType_>
    auto operator/(const DataType_& val, const Series<DataType_>& series) {
        return Series<DataType_>{series}.rdiv(val);
    }

    // Division for series / series
    template <typename DataType_>
    auto operator/(const Series<DataType_>& other, const Series<DataType_>& series) {
        return Series<DataType_>{other}.div(series);
    }
}