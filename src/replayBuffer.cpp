#include <algorithm>
#include <torch/torch.h>

#include "replayBuffer.h"


void ReplayBuffer::storeTransition(torch::Tensor state, torch::Tensor action, double reward,
                                   torch::Tensor newState, bool done){
    
    // Get the current index adapted to the memory size
    auto index = memoryIndex % memorySize;

    // Store the data
    stateMemory.index_put_({index}, state);
    actionMemory.index_put_({index}, action);
    newStateMemory.index_put_({index}, newState);
    rewardMemory[index] = reward;
    terminalMemory[index] = done;

    // Incremente the index
    memoryIndex++;

}

torch::Tensor ReplayBuffer::sampleBufferIndex(int batchSize){
    // Get the max index to take
    auto index = std::min(memoryIndex, memorySize);

    // Return random indexes
    return torch::randint(index, batchSize);
}

torch::Tensor ReplayBuffer::getIndexedStateMemory(torch::Tensor indexes){
    return stateMemory.index_select(0, indexes);
}

torch::Tensor ReplayBuffer::getIndexedNewStateMemory(torch::Tensor indexes){
    return newStateMemory.index_select(0, indexes);
}

torch::Tensor ReplayBuffer::getIndexedActionMemory(torch::Tensor indexes){
    return actionMemory.index_select(0, indexes);
}

torch::Tensor ReplayBuffer::getIndexedRewardMemory(torch::Tensor indexes){
    return rewardMemory.index_select(0, indexes);
}

torch::Tensor ReplayBuffer::getIndexedTerminalMemory(torch::Tensor indexes){
    return terminalMemory.index_select(0, indexes);
}

int ReplayBuffer::getMemoryIndex(){
    return memoryIndex;
}