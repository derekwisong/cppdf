#include "gtest/gtest.h"
#include "df.h"

namespace {
    using namespace df;

    TEST(SeriesTests, CreateFromInitList) {
        Series<int> s1({1, 2, 3, 4, 5});
        ASSERT_EQ(s1.size(), 5);
        ASSERT_EQ(s1[0], 1);
        ASSERT_EQ(s1[4], 5);
    }

    TEST(SeriesTests, AddScalar) {
        Series<int> s1({1, 2, 3});
        s1.add(5);
        EXPECT_EQ(s1[0], 6);
        EXPECT_EQ(s1[1], 7);
        EXPECT_EQ(s1[2], 8);
    }

    TEST(SeriesTests, AddScalarOperatorOverload) {
        const Series<int> s1({1, 2, 3});
        const auto s2 = s1 + 5;
        EXPECT_EQ(s2[0], 6);
        EXPECT_EQ(s2[1], 7);
        EXPECT_EQ(s2[2], 8);
        const auto s3 = 5 + s1;
        EXPECT_EQ(s3[0], 6);
        EXPECT_EQ(s3[1], 7);
        EXPECT_EQ(s3[2], 8);
    }

    TEST(SeriesTests, AddSeriesOperatorOverload) {
        const Series<int> s1({1, 2, 3});
        const Series<int> s2({4, 5, 6});
        const auto s3 = s1 + s2;
        EXPECT_EQ(s3[0], 5);
        EXPECT_EQ(s3[1], 7);
        EXPECT_EQ(s3[2], 9);
    }

    TEST(SeriesTests, SubtractScalar) {
        const Series<int> s1({10, 20, 30});
        const auto s2 = s1 - 5;
        EXPECT_EQ(s2[0], 5);
        EXPECT_EQ(s2[1], 15);
        EXPECT_EQ(s2[2], 25);
        const auto s3 = 5 - s1;
        EXPECT_EQ(s3[0], -5);
        EXPECT_EQ(s3[1], -15);
        EXPECT_EQ(s3[2], -25);
    }

    TEST(SeriesTests, SubtractSeries) {
        const Series<int> s1({10, 20, 30});
        const Series<int> s2({1, 2, 3});
        const auto s3 = s1 - s2;
        EXPECT_EQ(s3[0], 9);
        EXPECT_EQ(s3[1], 18);
        EXPECT_EQ(s3[2], 27);
        const auto s4 = s2 - s1;
        EXPECT_EQ(s4[0], -9);
        EXPECT_EQ(s4[1], -18);
        EXPECT_EQ(s4[2], -27);
    }

    TEST(SeriesTests, MultiplySeriesInplace) {
        Series<double> s1({1, 2, 3});
        s1.mul(7);
        EXPECT_EQ(s1[0], 7);
        EXPECT_EQ(s1[1], 14);
        EXPECT_EQ(s1[2], 21);
    }

    TEST(SeriesTests, MeanCalculation) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        const auto mean = s1.mean();
        ASSERT_TRUE(mean.has_value());
        ASSERT_DOUBLE_EQ(mean.value(), 3.0);
    }

    TEST(SeriesTests, MeanEmpyIsUndefined) {
        const Series<double> s1({});
        EXPECT_EQ(s1.mean(), std::nullopt) << "Expect mean of an empty series to be undefined";
    }

    TEST(SeriesTests, SumEmpyIsUndefined) {
        const Series<double> s1({});
        EXPECT_EQ(s1.sum(), std::nullopt) << "Expect sum of an empty series to be undefined";
    }

    TEST(SeriesTests, VarianceEmptyIsUndefined) {
        const Series<double> s1({});
        EXPECT_EQ(s1.variance(), std::nullopt) << "Expect variance of an empty series to be undefined";
    }
}