
#ifndef NETWORKS_H
#define NETWORKS_H

#include <torch/torch.h>

namespace Networks
{

    class Critic: public torch::nn::Module
    {

        private:
            /// First hidden layer
            torch::nn::Linear hiddenLayer1;
            /// Second hidden layer
            torch::nn::Linear hiddenLayer2;
            /// Output layer
            torch::nn::Linear outputLayer;

            /// Adam optimizer
            torch::optim::Adam optimizer;

            /// Device used
            torch::Device device;

            /// Name of the ciritc
            std::string name;
            

        public:
            Critic(double lr, int stateSize, int actionSize, int sizeHL1=256, int sizeHL2=256, std::string name="Network")
            : hiddenLayer1(register_module(name+"linear"+std::to_string(1), torch::nn::Linear(stateSize+actionSize, sizeHL1))),
            hiddenLayer2(register_module(name+"linear"+std::to_string(2), torch::nn::Linear(sizeHL1, sizeHL2))),
            outputLayer(register_module(name+"linear"+std::to_string(3), torch::nn::Linear(sizeHL2, 1))),
            optimizer(this->parameters(), torch::optim::AdamOptions(lr)),
            device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
                this->name=name;
                this->to(device);
            }

            /// @brief Compute the qValue
            /// @param state state tensor
            /// @param action action tensor
            /// @return qValue
            torch::Tensor forward(torch::Tensor state, torch::Tensor action);

            /// Return a pointer to the optimizer
            torch::optim::Adam* getPtrOptimizer();

            /// Load critic model
            void loadCheckpoint();

            /// Save critic model
            void saveCheckpoint();
    };

    class Value: public torch::nn::Module
    {

        private:
            /// First hidden layer
            torch::nn::Linear hiddenLayer1;
            /// Second hidden layer
            torch::nn::Linear hiddenLayer2;
            /// Output layer
            torch::nn::Linear outputLayer;

            /// Adam optimizer
            torch::optim::Adam optimizer;

            /// Device used
            torch::Device device;

            /// Name of the value
            std::string name;
            

        public:
            Value(double lr, int stateSize, int sizeHL1=256, int sizeHL2=256, std::string name="Network")
            : hiddenLayer1(register_module(name+"linear"+std::to_string(1), torch::nn::Linear(stateSize, sizeHL1))),
            hiddenLayer2(register_module(name+"linear"+std::to_string(2), torch::nn::Linear(sizeHL1, sizeHL2))),
            outputLayer(register_module(name+"linear"+std::to_string(3), torch::nn::Linear(sizeHL2, 1))),
            optimizer(this->parameters(), torch::optim::AdamOptions(lr)),
            device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
                this->name=name;
                this->to(device);
            }
            
            /// @brief Compute the value
            /// @param state state tensor
            /// @return Value
            torch::Tensor forward(torch::Tensor state);

            /// Return a pointer to the optimizer
            torch::optim::Adam* getPtrOptimizer();

            /// Load value model
            void loadCheckpoint();

            /// Save value model
            void saveCheckpoint();
    };

    class Actor: public torch::nn::Module
    {

        private:
            /// First hidden layer
            torch::nn::Linear hiddenLayer1;
            /// Second hidden layer
            torch::nn::Linear hiddenLayer2;
            /// Mu output layer
            torch::nn::Linear muOutputLayer;
            /// Sigma output layer
            torch::nn::Linear sigmaOutputLayer;

            /// Adam optimizer
            torch::optim::Adam optimizer;

            /// Device used
            torch::Device device;

            /// ReparamNoise to avoid x/0 or log(0)
            double reparamNoise = 1e-6;

            /// Size of the state
            int stateSize;

            /// Name of the actor
            std::string name;


            

        public:
            Actor(double lr, int stateSize, int actionSize, int sizeHL1=256, int sizeHL2=256, std::string name="Network")
            : hiddenLayer1(register_module(name+"linear"+std::to_string(1), torch::nn::Linear(stateSize, sizeHL1))),
            hiddenLayer2(register_module(name+"linear"+std::to_string(2), torch::nn::Linear(sizeHL1, sizeHL2))),
            muOutputLayer(register_module(name+"linear"+std::to_string(3), torch::nn::Linear(sizeHL2, actionSize))),
            sigmaOutputLayer(register_module(name+"linear"+std::to_string(4), torch::nn::Linear(sizeHL2, actionSize))),
            optimizer(this->parameters(), torch::optim::AdamOptions(lr)),
            device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU){
                this->name=name;
                this->stateSize=stateSize;
                this->to(device);
            }
            
            /// @brief Return mu and sigma for the normal distribution
            /// @param state state Tensor
            /// @return pair of mu and sigma
            std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state);

            /// @brief Compute action and log prob with normal sample
            /// @param state state Tensor
            /// @param reparameterize true to use reparameterize sample, use for back propagation
            /// @return pair containing action and logProb
            std::pair<torch::Tensor, torch::Tensor> sampleNormal(torch::Tensor state, bool reparameterize=true);

            /// Return a pointer to the optimizer
            torch::optim::Adam* getPtrOptimizer();

            /// Load actor model
            void loadCheckpoint();

            /// Save actor model
            void saveCheckpoint();
    };

}

#endif