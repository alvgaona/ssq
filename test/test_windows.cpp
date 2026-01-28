#include <gtest/gtest.h>

#include "ssq/windows.hpp"

#include <vector>

class WindowsTest : public ::testing::Test {
protected:
    const Eigen::Index default_length = 128;
};

TEST_F(WindowsTest, KaiserLength) {
    auto w = ssq::windows::kaiser(default_length);
    EXPECT_EQ(w.size(), default_length);
}

TEST_F(WindowsTest, KaiserSymmetry) {
    auto w = ssq::windows::kaiser(default_length);

    for (Eigen::Index i = 0; i < default_length / 2; ++i) {
        EXPECT_NEAR(w(i), w(default_length - 1 - i), 1e-10)
            << "Kaiser window should be symmetric at index " << i;
    }
}

TEST_F(WindowsTest, KaiserPeakAtCenter) {
    auto w = ssq::windows::kaiser(default_length);

    Eigen::Index center = default_length / 2;
    double max_val = w.maxCoeff();

    EXPECT_NEAR(w(center), max_val, 1e-10);
    EXPECT_NEAR(w(center - 1), max_val, 1e-10);
}

TEST_F(WindowsTest, KaiserBetaEffect) {
    // Higher beta = narrower main lobe, lower sidelobes
    auto w_low = ssq::windows::kaiser(default_length, 5.0);
    auto w_high = ssq::windows::kaiser(default_length, 14.0);

    // Edge values should be smaller with higher beta
    EXPECT_LT(w_high(0), w_low(0));
    EXPECT_LT(w_high(default_length - 1), w_low(default_length - 1));
}

TEST_F(WindowsTest, HammingLength) {
    auto w = ssq::windows::hamming(default_length);
    EXPECT_EQ(w.size(), default_length);
}

TEST_F(WindowsTest, HammingSymmetry) {
    auto w = ssq::windows::hamming(default_length);

    for (Eigen::Index i = 0; i < default_length / 2; ++i) {
        EXPECT_NEAR(w(i), w(default_length - 1 - i), 1e-10)
            << "Hamming window should be symmetric at index " << i;
    }
}

TEST_F(WindowsTest, HammingEdgeValues) {
    // Hamming window has non-zero edge values (0.08)
    auto w = ssq::windows::hamming(default_length);

    EXPECT_NEAR(w(0), 0.08, 0.01);
    EXPECT_NEAR(w(default_length - 1), 0.08, 0.01);
}

TEST_F(WindowsTest, HannLength) {
    auto w = ssq::windows::hann(default_length);
    EXPECT_EQ(w.size(), default_length);
}

TEST_F(WindowsTest, HannSymmetry) {
    auto w = ssq::windows::hann(default_length);

    for (Eigen::Index i = 0; i < default_length / 2; ++i) {
        EXPECT_NEAR(w(i), w(default_length - 1 - i), 1e-10)
            << "Hann window should be symmetric at index " << i;
    }
}

TEST_F(WindowsTest, HannEdgeValues) {
    // Hann window has zero edge values
    auto w = ssq::windows::hann(default_length);

    EXPECT_NEAR(w(0), 0.0, 1e-10);
    EXPECT_NEAR(w(default_length - 1), 0.0, 1e-10);
}

TEST_F(WindowsTest, HannPeakValue) {
    // For even-length windows, the peak is slightly less than 1.0
    auto w = ssq::windows::hann(default_length);

    double max_val = w.maxCoeff();
    EXPECT_GT(max_val, 0.99);
    EXPECT_LE(max_val, 1.0);
}

TEST_F(WindowsTest, GaussianLength) {
    auto w = ssq::windows::gaussian(default_length);
    EXPECT_EQ(w.size(), default_length);
}

TEST_F(WindowsTest, GaussianSymmetry) {
    auto w = ssq::windows::gaussian(default_length);

    for (Eigen::Index i = 0; i < default_length / 2; ++i) {
        EXPECT_NEAR(w(i), w(default_length - 1 - i), 1e-10)
            << "Gaussian window should be symmetric at index " << i;
    }
}

TEST_F(WindowsTest, GaussianPeakAtCenter) {
    auto w = ssq::windows::gaussian(default_length);

    Eigen::Index center = (default_length - 1) / 2;
    double max_val = w.maxCoeff();

    EXPECT_NEAR(w(center), max_val, 1e-10);
}

TEST_F(WindowsTest, GaussianSigmaEffect) {
    // Smaller sigma = narrower window
    auto w_wide = ssq::windows::gaussian(default_length, default_length / 4.0);
    auto w_narrow = ssq::windows::gaussian(default_length, default_length / 8.0);

    // Edge values should be smaller with narrower window
    EXPECT_LT(w_narrow(0), w_wide(0));
}

TEST_F(WindowsTest, EdgeCaseLengthOne) {
    // Single-element windows should return 1.0
    EXPECT_EQ(ssq::windows::kaiser(1).size(), 1);
    EXPECT_NEAR(ssq::windows::kaiser(1)(0), 1.0, 1e-10);

    EXPECT_EQ(ssq::windows::hamming(1).size(), 1);
    EXPECT_NEAR(ssq::windows::hamming(1)(0), 1.0, 1e-10);

    EXPECT_EQ(ssq::windows::hann(1).size(), 1);
    EXPECT_NEAR(ssq::windows::hann(1)(0), 1.0, 1e-10);

    EXPECT_EQ(ssq::windows::gaussian(1).size(), 1);
    EXPECT_NEAR(ssq::windows::gaussian(1)(0), 1.0, 1e-10);
}

TEST_F(WindowsTest, EdgeCaseLengthZero) {
    // Zero-length windows should return empty vectors
    EXPECT_EQ(ssq::windows::kaiser(0).size(), 0);
    EXPECT_EQ(ssq::windows::hamming(0).size(), 0);
    EXPECT_EQ(ssq::windows::hann(0).size(), 0);
    EXPECT_EQ(ssq::windows::gaussian(0).size(), 0);
}

TEST_F(WindowsTest, AllPositiveValues) {
    // All windows should have non-negative values
    std::vector<Eigen::VectorXd> windows = {
        ssq::windows::kaiser(default_length),
        ssq::windows::hamming(default_length),
        ssq::windows::hann(default_length),
        ssq::windows::gaussian(default_length)
    };

    for (const auto& w : windows) {
        for (Eigen::Index i = 0; i < w.size(); ++i) {
            EXPECT_GE(w(i), 0.0);
        }
    }
}
