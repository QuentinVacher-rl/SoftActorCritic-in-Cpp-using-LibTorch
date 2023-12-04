
#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <torch/torch.h>
#include <vector>


class ReplayBuffer{
    private:
        /// Size of the buffer
        int memorySize;

        /// Current memory index
        int memoryIndex;

        /// Memory of states (double tensor with 2 dim)
        torch::Tensor stateMemory;

        /// Memory of newStates (double tensor with 2 dim)
        torch::Tensor newStateMemory;

        /// Memory of actions (double tensor with 2 dim)
        torch::Tensor actionMemory;

        /// Memory of rewards (double tensor with 1 dim)
        torch::Tensor rewardMemory;

        /// Memory of terminal (bool tensor with 1 dim)
        torch::Tensor terminalMemory;

    public:

        ReplayBuffer(int64_t maxSize, int stateSize, int actionSize)
        : memoryIndex(0){
            this->stateMemory = torch::zeros({maxSize, stateSize}, torch::kFloat);
            this->newStateMemory = torch::zeros({maxSize, stateSize}, torch::kFloat);
            this->actionMemory = torch::zeros({maxSize, actionSize}, torch::kFloat);
            this->rewardMemory = torch::zeros({maxSize}, torch::kFloat);
            this->terminalMemory = torch::zeros({maxSize}, torch::kBool);
            this->memorySize = maxSize;
        }

        /**
         * @brief Save data in the different tensor
         * 
         * @param state Last State
         * @param action Last action
         * @param reward Last reward
         * @param newState Last new State
         * @param done Last done
         */
        void storeTransition(torch::Tensor state, torch::Tensor action, double reward, torch::Tensor newState, bool done);

        /// Sample random indexes to train with
        torch::Tensor sampleBufferIndex(int batchSize);

        /// Return states corresponding to indexes
        torch::Tensor getIndexedStateMemory(torch::Tensor indexes);

        /// Return new states corresponding to indexes
        torch::Tensor getIndexedNewStateMemory(torch::Tensor indexes);
        
        /// Return actions corresponding to indexes
        torch::Tensor getIndexedActionMemory(torch::Tensor indexes);

        /// Return rewards corresponding to indexes
        torch::Tensor getIndexedRewardMemory(torch::Tensor indexes);

        /// Return terminals corresponding to indexes
        torch::Tensor getIndexedTerminalMemory(torch::Tensor indexes);

        /// Return the current memory index
        int getMemoryIndex();


};

#endif