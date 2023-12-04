

#ifndef NORMAL_H
#define NORMAL_H

#include <torch/torch.h>

class Normal{
    private:
        const double lz = log(sqrt(2 * M_PI));
        const torch::Tensor mean;
        const torch::Tensor stddev;
        const torch::Tensor var;
        const torch::Tensor logStd;

    public:
        Normal(const torch::Tensor &mean, const torch::Tensor &std) 
            : mean(mean), stddev(std), var(std * std), logStd(std.log()) {}

        // Reparameterize sample of normal distribution. Used for the backward
        torch::Tensor rsample();

        // Sample of normal distribution
        torch::Tensor sample();

        // Return the log probability
        torch::Tensor logProb(const torch::Tensor &value);

};


#endif