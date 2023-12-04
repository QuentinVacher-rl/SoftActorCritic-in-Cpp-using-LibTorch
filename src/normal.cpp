#include <torch/torch.h>

#include "normal.h"

torch::Tensor Normal::rsample(){
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto eps = torch::randn(1).to(device);
    return this->mean + eps * this->stddev;
}

torch::Tensor Normal::sample(){
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::NoGradGuard noGrad;
    return at::normal(mean, stddev);
}


torch::Tensor Normal::logProb(const torch::Tensor &value) {
    // log [exp(-(x-mu)^2/(2 sigma^2)) / (sqrt(2 pi) * sigma)] = 
    // = log [exp(-(x-mu)^2/(2 sigma^2))] - log [sqrt(2 pi) * sigma] = 
    // = -(x - mu)^2 / (2 sigma^2) - log(sigma) - log(sqrt(2 pi))
    return -(value - mean)*(value - mean) / (2 * var) - logStd - lz;
}