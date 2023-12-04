
#include <torch/torch.h>
#include <iostream>

#include "softActorCritic.h"
#include "sacParameters.h"
#include "gridworld.h"
#include <numeric>
#include <vector>

int main() {

// Test of soft actor critic on a very simple Grid World environnement

    GridWorld env;

    int nbEpisode = 300;

    SACParameters sacParams;

    SoftActorCritic algo(sacParams, 2, 1);



    bool terminated;
    torch::Tensor state;
    torch::Tensor newState;
    torch::Tensor actionTensor;



    double reward;
    double rewardEpisode;
    std::vector<double> listRewards;

    int realAction;

    double moyenneReward;
    double moyenneLastReward;

    for(size_t i=0; i<nbEpisode; i++){

        env.reset();
        terminated = env.isTerminated();
        state = env.getState();
        rewardEpisode = 0;

        while(!terminated){
            actionTensor = algo.chooseAction(state);
            //actionTensor = torch::rand({1}).mul_(2.0).sub_(1.0);

            if (actionTensor.item<float>() < -0.5){
                realAction = 0;
            } else if (actionTensor.item<float>() < 0){
                realAction = 1;
            } else if (actionTensor.item<float>() < 0.5){
                realAction = 2;
            } else {
                realAction = 3;
            }

            reward = env.doAction(realAction);
            terminated = env.isTerminated();
            newState = env.getState();

            algo.remember(state, actionTensor, reward, newState, terminated);

            state = newState;
            rewardEpisode += reward;

            algo.learn();


        }
        listRewards.push_back(rewardEpisode);
        moyenneReward = std::accumulate(listRewards.begin(), listRewards.end(), 0.0) / listRewards.size();
        if(i>10){
            moyenneLastReward = std::accumulate(listRewards.end() - 10, listRewards.end(), 0.0) / 10;
        } else{
            moyenneLastReward = moyenneReward;
        }

        std::cout<<"episode "<<i<<" -- Reward : "<<rewardEpisode<< " -- Moyenne Rewards : "<<moyenneReward<<" -- Moyenne Rewards Last 10 ep : "<<moyenneLastReward<<std::endl;
    }

    algo.saveModels();

}
