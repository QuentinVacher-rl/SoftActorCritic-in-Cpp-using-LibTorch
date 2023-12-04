#include"gridworld.h"

void GridWorld::reset(){
    agent_x = 1;
    agent_y = 0;
    currentStep = 0;
    terminated = false;
}

bool GridWorld::positionAvailable(int pos_x, int pos_y){
    if(pos_x == size_x || pos_x == -1){
        return false;
    }
    if(pos_y == size_y || pos_y == -1){
        return false;
    }
    if (grid[pos_x][pos_y].item<int>() == 3){
        return false;
    }
    return true;

}

double GridWorld::doAction(int action){
    currentStep++;

    switch (action){
        case 0:
            if (positionAvailable(agent_x - 1, agent_y)) agent_x--;
            break;
        case 1:
            if (positionAvailable(agent_x, agent_y + 1)) agent_y++;
            break;
        case 2:
            if (positionAvailable(agent_x + 1, agent_y)) agent_x++;
            break;
        case 3:
            if (positionAvailable(agent_x, agent_y - 1)) agent_y--;
            break;
    }


    double reward = -1;

    if(agent_y == 3){
        terminated=true;
        reward = (agent_x == 0) ? 100 : -100;
    }
    if(currentStep==100)
        terminated = true;
    return reward;
}

torch::Tensor GridWorld::getState(){
    return torch::tensor({{agent_x, agent_y}}, torch::kFloat);
}

bool GridWorld::isTerminated(){
    return terminated;
}

int GridWorld::getCurrentStep(){
    return currentStep;
}




