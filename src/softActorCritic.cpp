#include <torch/torch.h>
#include <iostream>

#include "softActorCritic.h"

torch::Tensor SoftActorCritic::chooseAction(torch::Tensor observation){

    // Sample on true normal distribution an action tensor with the given observation tensor. 
    auto outputActor = actorNet.sampleNormal(observation, false);

    // Get the action and return it
    return outputActor.first.detach();
}

void SoftActorCritic::remember(torch::Tensor state, torch::Tensor action, double reward, torch::Tensor newState, bool done){
    memory.storeTransition(state, action, reward, newState, done);
}

void SoftActorCritic::updateNetworkParameter(double currentTau){
    // If currentTau is at the default value, set it to tau
    if(currentTau==0) 
        currentTau = params.tau;
    
    // Get the target value and value parameters
    auto targetValueParams = targetValueNet.named_parameters();
    auto valueParams = valueNet.named_parameters();

    // For each layer, update the parameters of targetValueNet
    for (auto& kv : valueParams) {
        auto name = kv.key();
        targetValueParams["target" + name].data().copy_(
            params.tau * kv.value().data() + (1 - params.tau) * targetValueParams["target" + name].data()
        );
    }
}

void SoftActorCritic::learn(){
    // Do not learn if memoryIndex is under the batch size
    if (memory.getMemoryIndex() < params.batchSize)
        return;

    // Get random data
    auto indexes = memory.sampleBufferIndex(params.batchSize);
    auto stateBatch = memory.getIndexedStateMemory(indexes);
    auto newStateBatch = memory.getIndexedNewStateMemory(indexes);
    auto actionBatch = memory.getIndexedActionMemory(indexes);
    auto rewardBatch = memory.getIndexedRewardMemory(indexes);
    auto doneBatch = memory.getIndexedTerminalMemory(indexes);

    // Calcul the values with the value network
    auto value = valueNet.forward(stateBatch).view(-1);

    // Calcul the new values with the target value network. Set newValue to 0 of the newState that ended the episode
    auto newValue = targetValueNet.forward(newStateBatch).view(-1);
    newValue.masked_fill_(doneBatch, 0.0);

    // Sample action and logProbs with the state vector, reparametrize is false because the action will not be backwarded
    auto outputActorNet = actorNet.sampleNormal(stateBatch, false);
    auto actions = outputActorNet.first;
    auto logProbs = outputActorNet.second.view(-1);
    // Compute the qValue the both critic1 and critic2 networks
    auto q1NewPolicy = criticNet1.forward(stateBatch, actions);
    auto q2NewPolicy = criticNet2.forward(stateBatch, actions);
    // The criticValue is the minimum qValue
    auto criticValue = torch::min(q1NewPolicy, q2NewPolicy).view(-1);

    // Reset the gradient of the valueNet
    valueNet.getPtrOptimizer()->zero_grad();
    // Calcul the valueloss
    auto valueTarget = criticValue - logProbs;
    auto valueLoss = 0.5 * torch::mse_loss(value, valueTarget);
    // Backward and optimize
    valueLoss.backward({}, true);
    valueNet.getPtrOptimizer()->step();

    // Sample action and logProbs with the state vector, reparametrize is true because the action will be backwarded
    outputActorNet = actorNet.sampleNormal(stateBatch, true);
    actions = outputActorNet.first;
    logProbs = outputActorNet.second.view(-1);
    // Compute the qValue the both critic1 and critic2 networks
    q1NewPolicy = criticNet1.forward(stateBatch, actions);
    q2NewPolicy = criticNet2.forward(stateBatch, actions);
    // The criticValue is the minimum qValue
    criticValue = torch::min(q1NewPolicy, q2NewPolicy).view(-1);

    // Compute the actorLoss
    auto actorLoss = torch::mean(logProbs - criticValue);
    // Reset the gradient of the actorNet
    actorNet.getPtrOptimizer()->zero_grad();
    // Backward and optimize
    actorLoss.backward({}, true);
    actorNet.getPtrOptimizer()->step();

    // Reset the gradient of the criticNets
    criticNet1.getPtrOptimizer()->zero_grad();
    criticNet2.getPtrOptimizer()->zero_grad();

    // Compute Qhat. This is where the entropy is optimized
    auto q_hat = params.rewardScale * rewardBatch + params.gamma * newValue;

    // Compute qValues of old actions
    auto q1OldPolicy = criticNet1.forward(stateBatch, actionBatch).view(-1);
    auto q2OldPolicy = criticNet2.forward(stateBatch, actionBatch).view(-1);

    // Compute the criticLoss
    auto critic1Loss = 0.5 * torch::mse_loss(q1OldPolicy, q_hat);
    auto critic2Loss = 0.5 * torch::mse_loss(q2OldPolicy, q_hat);
    auto criticLoss = critic1Loss + critic2Loss;

    // Backward and optimize
    criticLoss.backward();
    criticNet1.getPtrOptimizer()->step();
    criticNet2.getPtrOptimizer()->step();

    // Update the targetValueNet
    updateNetworkParameter();

}

void SoftActorCritic::loadModels(){
    std::cout<<" ----- Loading Models ----- "<<std::endl;
    actorNet.loadCheckpoint();
    criticNet1.loadCheckpoint();
    criticNet2.loadCheckpoint();
    valueNet.loadCheckpoint();
    targetValueNet.loadCheckpoint();
}

void SoftActorCritic::saveModels(){
    std::cout<<" ----- Saving Models ----- "<<std::endl;
    actorNet.saveCheckpoint();
    criticNet1.saveCheckpoint();
    criticNet2.saveCheckpoint();
    valueNet.saveCheckpoint();
    targetValueNet.saveCheckpoint();
}

    
