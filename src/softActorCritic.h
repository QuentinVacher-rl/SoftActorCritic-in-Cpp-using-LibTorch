
#ifndef SOFT_ACTOR_CRITIC_H
#define SOFT_ACTOR_CRITIC_H

#include <torch/torch.h>
#include "replayBuffer.h"
#include "networks.h"
#include "sacParameters.h"

class SoftActorCritic{
    private:
        /// Parmaters
        SACParameters params;

        /// Memory for the training
        ReplayBuffer memory;

        /// Actor network
        Networks::Actor actorNet;

        /// First critic network
        Networks::Critic criticNet1;

        /// Second critic network
        Networks::Critic criticNet2;

        /// Value network
        Networks::Value valueNet;

        /// Target Value network
        Networks::Value targetValueNet;



    public:
        SoftActorCritic(
            SACParameters params, int stateSize, int actionSize)
            : memory(params.sizeBuffer, stateSize, actionSize),
            actorNet(params.lr, stateSize, actionSize, params.sizeHL1, params.sizeHL2, "actor"),
            criticNet1(params.lr, stateSize, actionSize, params.sizeHL1, params.sizeHL2, "critic1"),
            criticNet2(params.lr, stateSize, actionSize, params.sizeHL1, params.sizeHL2, "critic2"),
            valueNet(params.lr, stateSize, params.sizeHL1, params.sizeHL2, "value"),
            targetValueNet(params.lr, stateSize, params.sizeHL1, params.sizeHL2, "targetvalue") {
                this->params = params;

                // Update target value network with tau=1 is equivalent to copie the weight of value network to target value
                updateNetworkParameter(1);

                // Load the models if indicated
                if(params.loadModels) loadModels();
            }

        
        /**
         * @brief Choose an action with the actor network depending of the observation.
         * The action vector is sample on a Normal distribution
         * 
         * @param observation : Observation tensor
         * 
         * @return Action taken
         */
        torch::Tensor chooseAction(torch::Tensor observation);


        /**
         * @brief Save data in the memory buffer
         * 
         * @param state Last State
         * @param action Last action
         * @param reward Last reward
         * @param newState Last new State
         * @param done Last done
         */
        void remember(torch::Tensor state, torch::Tensor action, double reward, torch::Tensor newState, bool done);

        /**
         * @brief Update the target value parameters to the value parameters.
         * The update is scale on the tau parameter
         * 
         * @param currentTau By default=0, if currentTau=0, it is set to params.tau
        */
        void updateNetworkParameter(double currentTau=0);

        /// @brief Learn function
        void learn();

        /// @brief Load saved models, throw exception if models have not been saved before
        void loadModels();

        /// @brief save models
        void saveModels();

};

# endif