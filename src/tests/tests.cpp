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

    TEST(SeriesTests, AddScalarInplace) {
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
        const auto s3 = s1 + Series<int>({4, 5, 6});
        EXPECT_EQ(s3[0], 5);
        EXPECT_EQ(s3[1], 7);
        EXPECT_EQ(s3[2], 9);
    }

    TEST(SeriesTest, PlusEqualsScalar) {
        Series<int> s1({1, 2, 3});
        s1 += 4;
        EXPECT_EQ(s1[0], 5);
        EXPECT_EQ(s1[1], 6);
        EXPECT_EQ(s1[2], 7);
    }

    TEST(SeriesTest, PlusEqualsSeries) {
        Series<int> s1({1, 2, 3});
        s1 += Series<int>({4, 5, 6});
        EXPECT_EQ(s1[0], 5);
        EXPECT_EQ(s1[1], 7);
        EXPECT_EQ(s1[2], 9);
    }

    TEST(SeriesTests, SubtractScalarOperatorOverload) {
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

    TEST(SeriesTests, SubtractSeriesOperatorOverload) {
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

    TEST(SeriesTests, MultiplyInplaceScalar) {
        Series<int> s1({1, 2, 3});
        s1.mul(7);
        EXPECT_EQ(s1[0], 7);
        EXPECT_EQ(s1[1], 14);
        EXPECT_EQ(s1[2], 21);
    }

    TEST(SeriesTest, DivideInplaceScalar) {
        Series<double> s1({5.0, 10.0, 20.0, 30.0});
        s1.div(10);
        EXPECT_DOUBLE_EQ(s1[0], 0.5);
        EXPECT_DOUBLE_EQ(s1[1], 1.0);
        EXPECT_DOUBLE_EQ(s1[2], 2.0);
        EXPECT_DOUBLE_EQ(s1[3], 3.0);
    }

    TEST(SeriesTests, DivideInplaceSeries) {
        Series<double> s1({10.0, 20.0, 30.0});
        Series<double> s2({2.0, 4.0, 2.5});
        s1.div(s2);
        EXPECT_DOUBLE_EQ(s1[0], 5.0);
        EXPECT_DOUBLE_EQ(s1[1], 5.0);
        EXPECT_DOUBLE_EQ(s1[2], 12.0);
    }

    TEST(SeriesTests, DivideScalarOperatorOverload) {
        const Series<double> s1({10.0, 20.0, 30.0});
        const auto s2 = s1 / 10.0;
        EXPECT_DOUBLE_EQ(s2[0], 1.0);
        EXPECT_DOUBLE_EQ(s2[1], 2.0);
        EXPECT_DOUBLE_EQ(s2[2], 3.0);
        const auto s3 = 60.0 / s1;
        EXPECT_DOUBLE_EQ(s3[0], 6.0);
        EXPECT_DOUBLE_EQ(s3[1], 3.0);
        EXPECT_DOUBLE_EQ(s3[2], 2.0);
    }

    TEST(SeriesTests, MeanCalculation) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        const auto mean = s1.mean();
        ASSERT_TRUE(mean.has_value());
        ASSERT_DOUBLE_EQ(mean.value(), 3.0);
    }

    TEST(SeriesTests, AllNullMeanIsUndefined) {
        Series<double> s1({1.2, 2.3, 3.4});
        s1.set_null(0);
        s1.set_null(1);
        s1.set_null(2);
        EXPECT_EQ(s1.mean(), std::nullopt) << "Expect mean of all null series to be undefined";
    }

    TEST(SeriesTests, MeanIgnoresNulls) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        s1.set_null(1); // set second element to null
        s1.set_null(3); // set fourth element to null
        const auto mean = s1.mean();
        ASSERT_TRUE(mean.has_value());
        ASSERT_DOUBLE_EQ(mean.value(), 3.0); // (1 + 3 + 5) / 3 = 3.0
    }

    TEST(SeriesTests, SumIgnoresNulls) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        s1.set_null(0); // set first element to null
        s1.set_null(4); // set last element to null
        const auto sum = s1.sum();
        ASSERT_TRUE(sum.has_value());
        ASSERT_DOUBLE_EQ(sum.value(), 9.0); // 2 + 3 + 4 = 9.0
    }

    TEST(SeriesTests, VarianceIgnoresNulls) {
        Series<double> s1({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
        s1.set_null(2); // set one element to null
        const auto var = s1.variance();
        ASSERT_TRUE(var.has_value());
        ASSERT_DOUBLE_EQ(var.value(), 4.0); // variance should be 4.0 ignoring the null
    }

    TEST(SeriesTests, StandardDeviationIgnoresNulls) {
        Series<double> s1({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
        s1.set_null(5); // set one element to null
        const auto stddev = s1.stddev();
        ASSERT_TRUE(stddev.has_value());
        ASSERT_DOUBLE_EQ(stddev.value(), 2.0); // stddev should be 2.0 ignoring the null
    }

    TEST(SeriesTests, AllNullSumIsUndefined) {
        Series<double> s1({1.2, 2.3, 3.4});
        s1.set_null(0);
        s1.set_null(1);
        s1.set_null(2);
        EXPECT_EQ(s1.sum(), std::nullopt) << "Expect sum of all null series to be undefined";
    }

    TEST(SeriesTests, AllNullVarianceIsUndefined) {
        Series<double> s1({1.2, 2.3, 3.4});
        s1.set_null(0);
        s1.set_null(1);
        s1.set_null(2);
        EXPECT_EQ(s1.variance(), std::nullopt) << "Expect variance of all null series to be undefined";
    }

    TEST(SeriesTests, AllNullStdDevIsUndefined) {
        Series<double> s1({1.2, 2.3, 3.4});
        s1.set_null(0);
        s1.set_null(1);
        s1.set_null(2);
        EXPECT_EQ(s1.stddev(), std::nullopt) << "Expect stddev of all null series to be undefined";
    }

    TEST(SeriesTests, CountValidElements) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        s1.set_null(1); // set second element to null
        s1.set_null(3); // set fourth element to null
        const auto vcount = s1.valid_count();
        ASSERT_EQ(vcount, 3); // three valid elements remain
    }

    TEST(SeriesTests, CountNullElements) {
        Series<double> s1({1.0, 2.0, 3.0, 4.0, 5.0});
        s1.set_null(0); // set first element to null
        s1.set_null(2); // set third element to null
        s1.set_null(4); // set fifth element to null
        const auto vcount = s1.null_count();
        ASSERT_EQ(vcount, 3);
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