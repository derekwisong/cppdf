#pragma once

#include "series.h"

#include <memory>
#include <unordered_map>
#include <string>
#include <optional>


namespace df {
    class BaseSeries {
    public:
        virtual ~BaseSeries() = default;
        virtual std::size_t size() const noexcept = 0;
        virtual const std::type_info& type() const noexcept = 0;
    };

    template <typename T>
    class WrappedSeries final : public BaseSeries {
    public:
        explicit WrappedSeries(Series<T> series) : series_(std::move(series)) {}

        std::size_t size() const noexcept override { return series_.size(); }
        const std::type_info& type() const noexcept override { return typeid(T); }

        Series<T>& impl() noexcept { return series_; }
        const Series<T>& impl() const noexcept { return series_; }

        Series<T>* operator->() { return &series_; }
        const Series<T>* operator->() const { return &series_; }

    private:
        Series<T> series_;
    };


    using SeriesPtr = std::shared_ptr<BaseSeries>;

    class DataFrame {
    public:

        std::size_t length() const { 
            if (cols_.empty()) {
                return 0;
            }

            return cols_.begin()->second->size();
        }

        std::size_t width() const {
            return cols_.size();
        }

        std::pair<std::size_t, std::size_t> shape() const {
            return {length(), width()};
        }

        template <typename T>
        void add(std::string name, Series<T> series) {
            if (!cols_.empty() && series.size() != length()) {
                throw std::invalid_argument("Cannot add column with inconsistent length");
            }

            auto [pair, inserted] = cols_.emplace(
                std::move(name),
                std::make_shared<WrappedSeries<T>>(std::move(series))
            );

            if (inserted) {
                col_order_.emplace_back(pair->first);
            }
        }

        template <typename T>
        Series<T>& column(const std::string& name) {
            auto it = cols_.find(name);
            if (it == cols_.end()) {
                throw std::out_of_range(std::string("Column not found: ") + name);
            }

            auto* wrapped = dynamic_cast<WrappedSeries<T>*>(it->second.get());
            if (!wrapped) {
                throw std::bad_cast();
            }

            return wrapped->impl();
        }

        friend std::ostream& operator<<(std::ostream& os, const DataFrame& df);
    
    private:
        std::unordered_map<std::string, SeriesPtr> cols_;
        std::vector<std::string> col_order_;
    };

}