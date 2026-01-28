#include "ssq/fftw_wrapper.hpp"

namespace ssq {

FftwManager& FftwManager::instance() {
    static FftwManager instance;
    return instance;
}

FftwManager::FftwManager() = default;

FftwManager::~FftwManager() {
    cleanup();
}

FftwPlan FftwManager::get_r2c_plan(int n, double* in, fftw_complex* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    PlanKey key{n, PlanType::R2C};
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
        return it->second;
    }
    FftwPlan plan(fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE));
    plan_cache_.emplace(key, plan);
    return plan;
}

FftwPlan FftwManager::get_c2r_plan(int n, fftw_complex* in, double* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    PlanKey key{n, PlanType::C2R};
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
        return it->second;
    }
    FftwPlan plan(fftw_plan_dft_c2r_1d(n, in, out, FFTW_ESTIMATE));
    plan_cache_.emplace(key, plan);
    return plan;
}

FftwPlan FftwManager::get_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign) {
    std::lock_guard<std::mutex> lock(mutex_);
    PlanType type = (sign == FFTW_FORWARD) ? PlanType::DFT_FORWARD : PlanType::DFT_BACKWARD;
    PlanKey key{n, type};
    auto it = plan_cache_.find(key);
    if (it != plan_cache_.end()) {
        return it->second;
    }
    FftwPlan plan(fftw_plan_dft_1d(n, in, out, sign, FFTW_ESTIMATE));
    plan_cache_.emplace(key, plan);
    return plan;
}

void FftwManager::execute_r2c(int n, double* in, fftw_complex* out) {
    FftwPlan plan = get_r2c_plan(n, in, out);
    fftw_execute_dft_r2c(plan.get(), in, out);
}

void FftwManager::execute_c2r(int n, fftw_complex* in, double* out) {
    FftwPlan plan = get_c2r_plan(n, in, out);
    fftw_execute_dft_c2r(plan.get(), in, out);
}

void FftwManager::execute_dft(int n, fftw_complex* in, fftw_complex* out, int sign) {
    FftwPlan plan = get_dft_plan(n, in, out, sign);
    fftw_execute_dft(plan.get(), in, out);
}

// Legacy methods for compatibility
FftwPlan FftwManager::create_r2c_plan(int n, double* in, fftw_complex* out, unsigned flags) {
    (void)flags;
    return get_r2c_plan(n, in, out);
}

FftwPlan FftwManager::create_c2r_plan(int n, fftw_complex* in, double* out, unsigned flags) {
    (void)flags;
    return get_c2r_plan(n, in, out);
}

FftwPlan FftwManager::create_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
    (void)flags;
    return get_dft_plan(n, in, out, sign);
}

void FftwManager::execute(const FftwPlan& plan) {
    if (plan) {
        fftw_execute(plan.get());
    }
}

void FftwManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_cache_.clear();  // shared_ptr destructor handles fftw_destroy_plan
    fftw_cleanup();
}

}  // namespace ssq
