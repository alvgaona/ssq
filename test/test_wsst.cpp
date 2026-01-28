#include <gtest/gtest.h>

#include "ssq/wsst.hpp"

#include <cmath>

namespace {

const double PI = 3.14159265358979323846;

// Generate a sinusoid
Eigen::VectorXd sine_signal(Eigen::Index n, double freq, double sample_rate) {
    Eigen::VectorXd sig(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        sig(i) = std::sin(2.0 * PI * freq * static_cast<double>(i) / sample_rate);
    }
    return sig;
}

// Generate a linear chirp (frequency sweep)
Eigen::VectorXd chirp_signal(Eigen::Index n, double f0, double f1, double sample_rate) {
    Eigen::VectorXd sig(n);
    double duration = static_cast<double>(n) / sample_rate;
    double k = (f1 - f0) / duration;

    for (Eigen::Index i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double phase = 2.0 * PI * (f0 * t + 0.5 * k * t * t);
        sig(i) = std::sin(phase);
    }
    return sig;
}

}  // namespace

class WsstTest : public ::testing::Test {
protected:
    const double sample_rate = 1000.0;
};

TEST_F(WsstTest, OutputDimensions) {
    Eigen::VectorXd sig = sine_signal(500, 100.0, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);

    // Should have frequency bins and time steps
    EXPECT_GT(result.spectrum.rows(), 0);
    EXPECT_EQ(result.spectrum.cols(), sig.size());
    EXPECT_EQ(result.frequencies.size(), result.spectrum.rows());
    EXPECT_EQ(result.times.size(), result.spectrum.cols());
}

TEST_F(WsstTest, TimeAxisRange) {
    Eigen::Index sig_len = 300;
    Eigen::VectorXd sig = sine_signal(sig_len, 50.0, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);

    // Time axis should start at 0
    EXPECT_NEAR(result.times(0), 0.0, 1e-10);

    // Last time point should be (sig_len - 1) / sample_rate
    EXPECT_NEAR(result.times(result.times.size() - 1), static_cast<double>(sig_len - 1) / sample_rate, 1e-10);
}

TEST_F(WsstTest, FrequencyAxisPositive) {
    Eigen::VectorXd sig = sine_signal(500, 100.0, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);

    // All frequencies should be positive
    for (Eigen::Index i = 0; i < result.frequencies.size(); ++i) {
        EXPECT_GT(result.frequencies(i), 0.0);
    }
}

TEST_F(WsstTest, SinusoidConcentration) {
    // WSST should concentrate sinusoid energy at its frequency
    double test_freq = 100.0;
    Eigen::VectorXd sig = sine_signal(1000, test_freq, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);

    // Check a time point in the middle
    Eigen::Index t_mid = sig.size() / 2;

    // Find the frequency bin with maximum magnitude
    Eigen::Index max_bin = 0;
    double max_mag = 0.0;
    for (Eigen::Index k = 0; k < result.spectrum.rows(); ++k) {
        double mag = std::abs(result.spectrum(k, t_mid));
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = k;
        }
    }

    // Peak should be near the expected frequency
    double peak_freq = result.frequencies(max_bin);
    double tolerance = 20.0;  // Hz - wavelet frequency resolution varies
    EXPECT_NEAR(peak_freq, test_freq, tolerance) << "WSST peak should be near " << test_freq << " Hz";
}

TEST_F(WsstTest, ChirpTracking) {
    // WSST should track the instantaneous frequency of a chirp
    double f0 = 50.0;
    double f1 = 150.0;
    Eigen::Index sig_len = 2000;
    Eigen::VectorXd sig = chirp_signal(sig_len, f0, f1, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);

    // Check at middle time point
    Eigen::Index t_mid = sig_len / 2;
    double duration = static_cast<double>(sig_len) / sample_rate;
    double time = static_cast<double>(t_mid) / sample_rate;
    double expected_freq = f0 + (f1 - f0) * time / duration;

    // Find peak frequency at this time
    Eigen::Index max_bin = 0;
    double max_mag = 0.0;
    for (Eigen::Index k = 0; k < result.spectrum.rows(); ++k) {
        double mag = std::abs(result.spectrum(k, t_mid));
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = k;
        }
    }

    double peak_freq = result.frequencies(max_bin);

    // Allow reasonable tolerance for wavelet-based method
    double tolerance = 25.0;
    EXPECT_NEAR(peak_freq, expected_freq, tolerance) << "WSST should track chirp frequency at t=" << time;
}

TEST_F(WsstTest, NumVoicesParameter) {
    Eigen::VectorXd sig = sine_signal(500, 100.0, sample_rate);

    // More voices = more frequency resolution
    auto result_16 = ssq::wsst(sig, sample_rate, ssq::WaveletType::Morlet, 16);
    auto result_48 = ssq::wsst(sig, sample_rate, ssq::WaveletType::Morlet, 48);

    // More voices should give more frequency bins
    EXPECT_GT(result_48.spectrum.rows(), result_16.spectrum.rows());
}

TEST_F(WsstTest, InverseReconstruction) {
    // iwsst should reconstruct sinusoids with correct amplitude
    double test_freq = 100.0;
    Eigen::VectorXd sig = sine_signal(1000, test_freq, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);
    Eigen::VectorXd reconstructed = ssq::iwsst(result.spectrum, result.frequencies);

    // Check in the middle to avoid edge effects
    Eigen::Index mid_start = sig.size() / 4;
    Eigen::Index mid_end = 3 * sig.size() / 4;

    // Compute RMS of signal and reconstruction
    double sig_rms = 0.0, rec_rms = 0.0;
    for (Eigen::Index i = mid_start; i < mid_end; ++i) {
        sig_rms += sig(i) * sig(i);
        rec_rms += reconstructed(i) * reconstructed(i);
    }
    sig_rms = std::sqrt(sig_rms / (mid_end - mid_start));
    rec_rms = std::sqrt(rec_rms / (mid_end - mid_start));

    // Scale should be close to 1.0
    double scale = sig_rms / rec_rms;
    EXPECT_NEAR(scale, 1.0, 0.05) << "Reconstruction amplitude should match original";
}

TEST_F(WsstTest, InverseCorrelation) {
    // iwsst output should be highly correlated with input
    double test_freq = 75.0;
    Eigen::VectorXd sig = sine_signal(1000, test_freq, sample_rate);

    auto result = ssq::wsst(sig, sample_rate);
    Eigen::VectorXd reconstructed = ssq::iwsst(result.spectrum, result.frequencies);

    // Compute correlation in middle region
    Eigen::Index mid_start = sig.size() / 4;
    Eigen::Index mid_end = 3 * sig.size() / 4;
    Eigen::Index n = mid_end - mid_start;

    double mean_sig = 0.0, mean_rec = 0.0;
    for (Eigen::Index i = mid_start; i < mid_end; ++i) {
        mean_sig += sig(i);
        mean_rec += reconstructed(i);
    }
    mean_sig /= n;
    mean_rec /= n;

    double cov = 0.0, var_sig = 0.0, var_rec = 0.0;
    for (Eigen::Index i = mid_start; i < mid_end; ++i) {
        double ds = sig(i) - mean_sig;
        double dr = reconstructed(i) - mean_rec;
        cov += ds * dr;
        var_sig += ds * ds;
        var_rec += dr * dr;
    }

    double corr = cov / std::sqrt(var_sig * var_rec);
    EXPECT_GT(corr, 0.99) << "Reconstruction should be highly correlated with original";
}

TEST_F(WsstTest, InverseFrequencyIndependence) {
    // Reconstruction quality should be consistent across frequencies
    std::vector<double> test_freqs = {25.0, 50.0, 100.0, 200.0};

    for (double freq : test_freqs) {
        Eigen::VectorXd sig = sine_signal(1000, freq, sample_rate);

        auto result = ssq::wsst(sig, sample_rate);
        Eigen::VectorXd reconstructed = ssq::iwsst(result.spectrum, result.frequencies);

        // Check amplitude in middle region
        Eigen::Index mid_start = sig.size() / 4;
        Eigen::Index mid_end = 3 * sig.size() / 4;

        double sig_rms = 0.0, rec_rms = 0.0;
        for (Eigen::Index i = mid_start; i < mid_end; ++i) {
            sig_rms += sig(i) * sig(i);
            rec_rms += reconstructed(i) * reconstructed(i);
        }
        sig_rms = std::sqrt(sig_rms / (mid_end - mid_start));
        rec_rms = std::sqrt(rec_rms / (mid_end - mid_start));

        double scale = sig_rms / rec_rms;
        EXPECT_NEAR(scale, 1.0, 0.05)
            << "Reconstruction scale should be ~1.0 at " << freq << " Hz";
    }
}
