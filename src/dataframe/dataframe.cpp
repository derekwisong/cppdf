#include "dataframe/dataframe.h"
#include "dataframe/series.h"
#include <iostream>

namespace df {

    std::ostream& operator<<(std::ostream& os, const DataFrame& df) {
        auto [nrows, ncols] = df.shape();
        os << "DataFrame: " << nrows << " rows x " << ncols << " columns\n";
        os << "----------------------------------------\n";
        // Print column names
        for (const auto& col_name : df.col_order_) {
            os << col_name << "\t";
        }
        os << "\n";
        os << "----------------------------------------\n";

        // Print first 5 rows or all if less than 5
        std::size_t rows_to_print = std::min<std::size_t>(5, nrows);
        for (std::size_t i = 0; i < rows_to_print; ++i) {
            for (const auto& col_name : df.col_order_) {
                const auto& col = df.cols_.at(col_name);
                if (col->type() == typeid(int)) {
                    auto& series = static_cast<WrappedSeries<int>&>(*col).impl();
                    os << series[i] << "\t";
                } else if (col->type() == typeid(double)) {
                    auto& series = static_cast<WrappedSeries<double>&>(*col).impl();
                    os << series[i] << "\t";
                } else {
                    os << "N/A\t"; // Unsupported type
                }
            }
            os << "\n";
        }
        return os;
    }

}