
#ifndef GRIDWORLD_H
#define GRIDWORLD_H

#include <torch/torch.h>

class GridWorld{

    private:
        int size_x=3;
        int size_y=4;
        
        torch::Tensor grid;

        int agent_x=1;
        int agent_y=0;

        int currentStep=0;

        bool terminated=false;

    public:
        GridWorld(){
            this->grid = torch::zeros({size_x, size_y}, torch::kFloat);
            
            this->grid[0][3] = 1;
            this->grid[1][2] = 3;
            this->grid[1][3] = 3;
            this->grid[2][3] = 2;
        }

        void reset();

        bool positionAvailable(int pos_x, int pos_y);

        double doAction(int action);

        torch::Tensor getState();

        bool isTerminated();

        void render();

        int getCurrentStep();
        
};

#endif