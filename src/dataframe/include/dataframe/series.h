#pragma once

#include <algorithm>
#include <cmath>
#include <execution>
#include <vector>
#include <iostream>

namespace df {

    template <typename DataType_>
    class Series {
    public:
        explicit Series(std::vector<DataType_> data): data_(std::move(data)) {}
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

        // Aggregation functions

        DataType_ dot(const Series<DataType_>& other) const {
            DataType_ result{};
            std::transform_reduce(
                exec_,
                data_.begin(), data_.end(),
                other.data_.begin(),
                result,
                std::plus<>(),
                std::multiplies<>()
            );
            return result;
        }

        DataType_ sum() const {
            return std::reduce(exec_, data_.begin(), data_.end(), DataType_{}, std::plus<>{});
        }

        DataType_ mean() const {
            if (size() == 0) {
                throw std::runtime_error("Cannot compute mean of empty series");
            }
            return sum() / static_cast<DataType_>(size());
        }

        DataType_ variance() const {
            if (size() == 0) {
                throw std::runtime_error("Cannot compute variance of empty series");
            }
            const auto m = mean();
            return std::transform_reduce(
                exec_,
                data_.begin(), data_.end(),
                DataType_{},
                std::plus<>(),
                [m](const auto& x) { return (x - m) * (x - m); }
            ) / static_cast<DataType_>(size());
        }

        DataType_ stddev() const {
            return std::sqrt(variance());
        }

        DataType_& min() const {
            if (size() == 0) {
                throw std::runtime_error("Cannot compute min of empty series");
            }
            return *std::min_element(exec_, data_.begin(), data_.end());
        }

        DataType_& max() const {
            if (size() == 0) {
                throw std::runtime_error("Cannot compute max of empty series");
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

    private:
        static constexpr auto exec_ = std::execution::par_unseq;

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
            std::transform(exec_, data_.begin(), data_.end(), output.data_.begin(), functor);
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
            std::transform(exec_, data_.begin(), data_.end(), other.data_.begin(), output.data_.begin(), functor);
            return *this;
        }
    };
}