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
        using value_type = DataType_;
        using container_type = std::vector<DataType_>;
        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;

        // Construct a Series by applying a monadic functor to an input Series
        template <typename T, typename MonadicFunc>
        Series(const Series<T>& other, MonadicFunc&& func) : data_(other.size()), valid_(other.valid_) {
            other.transform_to(*this, std::forward<MonadicFunc>(func));
        }
        
        // Construct a Series by applying a dyadic functor to two input Series
        template <typename T, typename J, typename DyadicFunc>
        Series(const Series<T>& lhs, const Series<J>& rhs, DyadicFunc&& func) {
            if (lhs.size() != rhs.size()) {
                throw std::invalid_argument("Series sizes do not match for dyadic operation");
            }

            // resize data_ and valid_ to match inputs
            data_.resize(lhs.size());
            valid_.resize(lhs.size());

            // the valid_ mask is the AND of both inputs' valid_ masks
            with_policy(lhs.exec_, [&](auto& exec_) {
                std::transform(
                    exec_,
                    lhs.valid_.begin(), lhs.valid_.end(),
                    rhs.valid_.begin(), valid_.begin(),
                    [](bool a, bool b) { return a && b; }
                );
            });

            // apply the transformation
            lhs.transform_to(rhs, *this, std::forward<DyadicFunc>(func));
        }

        explicit Series(std::vector<DataType_> data): data_(std::move(data)) {}
        explicit Series(std::initializer_list<DataType_> data): data_(std::move(data)) {}
        Series(ExecPolicy policy, std::vector<DataType_> data): data_(std::move(data)), exec_(policy) {}
        Series() = default;

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

        // Dot product of this series with another
        // Will return the identity element (0) if empty
        template <typename T, typename J=DataType_>
        J dot(const Series<T>& other) const {
            if (size() != other.size()) {
                throw std::invalid_argument("Series sizes do not match for dot product");
            }

            with_policy(exec_, [&](auto& exec_){
                return std::transform_reduce(
                    exec_,
                    data_.begin(), data_.end(),
                    other.data_.begin(),
                    J{},
                    std::plus<>(),
                    std::multiplies<>()
                );
            });
        }

        // Sum of all elements in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<T> sum() const {
            if (valid_count() == 0) {
                return std::nullopt;
            }
            return with_policy(exec_, [&](auto& exec_){
                // sum the values in the series unless the corresponding item in the valid_ mask is false
                return std::transform_reduce(
                    exec_,
                    data_.begin(), data_.end(),
                    valid_.begin(),
                    T{},
                    std::plus<>(),
                    [](const auto& x, const auto& v) { return v ? x : T{}; }
                );
            });
        }

        // Mean of all elements in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<T> mean() const {
            const auto vcount = valid_count();
            if (vcount == 0) {
                return std::nullopt;
            }
            const auto summation = sum();
            if (!summation.has_value()) {
                return std::nullopt;
            }
            return summation.value() / static_cast<T>(vcount);
        }

        // Variance of all elements in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<T> variance() const {
            const auto vcount = valid_count();
            if (vcount == 0) {
                return std::nullopt;
            }
            const auto m = mean();
            if (!m.has_value()) {
                return std::nullopt;
            }
            const auto& m_val = m.value();
            return with_policy(exec_, [&](auto& exec_){
                return std::transform_reduce(
                    exec_,
                    data_.begin(), data_.end(),
                    T{},
                    std::plus<>(),
                    [&m_val](const auto& x) { return (x - m_val) * (x - m_val); }
                ) / static_cast<T>(vcount);
            });
        }

        // Standard deviation of all elements in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<T> stddev() const {
            const auto var = variance();
            if (!var.has_value()) {
                return std::nullopt;
            }
            return std::sqrt(var.value());
        }

        // Minimum element in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<std::reference_wrapper<T>> min() const {
            if (size() == 0) {
                return std::nullopt;
            }
            return *std::min_element(exec_, data_.begin(), data_.end());
        }

        // Maximum element in the series
        // Returns std::nullopt if the series is empty
        template <typename T = DataType_>
        std::optional<std::reference_wrapper<T>> max() const {
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
                    if (obj.is_null(i)) {
                        os << "null";
                    }
                    else {
                        os << obj.data_[i];
                    }
                    if (i < obj.data_.size() - 1) {
                        os << ", ";
                    }
                }
            }
            else {
                const auto f = [&os](const auto& x) {
                    if (&x == nullptr) {
                        os << "null";
                    } else {
                        os << x << ", ";
                    }
                };
                std::for_each_n(obj.data_.begin(), CHUNK, f);
                os << "..., ";
                std::for_each_n(obj.data_.end() - CHUNK, CHUNK - 1, f);
                os << *(obj.data_.end() - 1);
            }
            os << "]";
            return os;

        }

        ExecPolicy exec_policy() const { return exec_; }
        void set_exec_policy(ExecPolicy policy) { exec_ = policy; }

        iterator begin() noexcept { return data_.begin(); }
        iterator end() noexcept { return data_.end(); }
        const_iterator begin() const noexcept { return data_.begin(); }
        const_iterator end() const noexcept { return data_.end(); }

        std::size_t size() const { return data_.size(); }
        void reserve(std::size_t n) { data_.reserve(n); }
        void resize(std::size_t n) { data_.resize(n); }
    
        // Get the number of null elements in the series
        std::size_t null_count() const {
            if (valid_.size() == 0) {
                return 0;
            }
            return with_policy(exec_, [&](auto& exec_){
                return std::count_if(exec_, valid_.begin(), valid_.end(), [](bool v) { return !v; });
            });
        }

        // Get the number of valid elements in the series
        std::size_t valid_count() const {
            if (data_.size() == 0) {
                return 0;
            }

            return with_policy(exec_, [&](auto& exec_){
                return std::count_if(exec_, valid_.begin(), valid_.end(), [](bool v) { return v; });
            });
        }

        // Get element at the index without bounds checking
        DataType_& operator[](std::size_t idx) {
            return data_[idx];
        }

        // Get element at the index without bounds checking
        const DataType_& operator[](std::size_t idx) const {
            return data_[idx];
        }

        // Get element at the index with bounds checking
        DataType_& at(std::size_t idx) {
            return data_.at(idx);
        }

        // Get element at the index with bounds checking
        const DataType_& at(std::size_t idx) const {
            return data_.at(idx);
        }

        // Get an optional (could be null) element at the index with bounds checking
        std::optional<std::reference_wrapper<const DataType_>> get(std::size_t idx) const {
            if (idx >= data_.size()) {
                throw std::out_of_range("Index out of range in Series::get");
            }
            if (is_null(idx)) {
                return std::nullopt;
            }
            return data_[idx];
        }

        // Append a new element to the end of the series
        void append(const DataType_& value) {
            data_.push_back(value);
        }

        // Clear all elements from the series
        void clear() {
            data_.clear();
        }

        // Emplace a new element at the end of the series
        template <typename... Args>
        void emplace(Args&&... args) {
            data_.emplace_back(std::forward<Args>(args)...);
        }

        // set the element at idx to null
        void set_null(std::size_t idx) {
            if (data_.size() == 0) {
                // nothing to do
                return;
            }

            if (idx >= valid_.size()) {
                valid_.resize(std::max(idx + 1, data_.size()), true);
            }

            valid_[idx] = false;
        }

        // check if the element at idx is null
        bool is_null(std::size_t idx) const {
            // the valid_ vector is only used if there are nulls, so if the index is out of range, it's not null
            if (idx >= valid_.size()) {
                return true;
            }
            return !valid_[idx];
        }

    private:
        ExecPolicy exec_{ExecPolicy::PAR_UNSEQ};

        // The underlying data storage
        std::vector<DataType_> data_;
        std::vector<bool> valid_;

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