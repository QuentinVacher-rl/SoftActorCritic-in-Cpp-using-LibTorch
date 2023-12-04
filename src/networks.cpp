#include <torch/torch.h>
#include <iostream>

#include "networks.h"
#include "normal.h"



torch::Tensor Networks::Critic::forward(torch::Tensor state, torch::Tensor action){

    // Concatenate state and action
    torch::Tensor input = torch::cat({state, action}, 1);

    // Calcul and return the qvalue
    auto q_value = hiddenLayer1(input);
    q_value = torch::nn::functional::relu(q_value);
    q_value = hiddenLayer2(q_value);
    q_value = torch::nn::functional::relu(q_value);
    q_value = outputLayer(q_value);

    return q_value;
}

torch::optim::Adam* Networks::Critic::getPtrOptimizer(){
    return &optimizer;
}

void Networks::Critic::loadCheckpoint(){
    torch::serialize::InputArchive inputArchive;
    inputArchive.load_from(ROOT_DIR "/models/" + name + ".pt");
    load(inputArchive);
}

void Networks::Critic::saveCheckpoint(){
    torch::serialize::OutputArchive outputArchive;
    save(outputArchive);
    outputArchive.save_to(ROOT_DIR "/models/" + name + ".pt");
}

torch::Tensor Networks::Value::forward(torch::Tensor state){

    // Calcul and return the value
    auto value = hiddenLayer1(state);
    value = torch::nn::functional::relu(value);
    value = hiddenLayer2(value);
    value = torch::nn::functional::relu(value);
    value = outputLayer(value);

    return value;
}

torch::optim::Adam* Networks::Value::getPtrOptimizer(){
    return &optimizer;
}

void Networks::Value::loadCheckpoint(){
    torch::serialize::InputArchive inputArchive;
    inputArchive.load_from(ROOT_DIR "/models/" + name + ".pt");
    load(inputArchive);
}

void Networks::Value::saveCheckpoint(){
    torch::serialize::OutputArchive outputArchive;
    save(outputArchive);
    outputArchive.save_to(ROOT_DIR "/models/" + name + ".pt");
}



std::pair<torch::Tensor, torch::Tensor> Networks::Actor::forward(torch::Tensor state){

    // Calcul mu and sigma
    auto prob = hiddenLayer1(state);
    prob = torch::nn::functional::relu(prob);
    prob = hiddenLayer2(prob);
    prob = torch::nn::functional::relu(prob);
    auto mu = muOutputLayer(prob);
    auto sigma = sigmaOutputLayer(prob);

    // Clamp sigma with 1 and reparamNoise
    sigma = torch::clamp(sigma, reparamNoise, 1);

    return std::make_pair(mu, sigma);
}


std::pair<torch::Tensor, torch::Tensor> Networks::Actor::sampleNormal(torch::Tensor state, bool reparameterize){

    // Get the pair of action and logProb
    auto pairProb = forward(state);
    auto mu = pairProb.first;
    auto sigma = pairProb.second;

    // Create normal probabilities
    auto probabilities = Normal(mu, sigma);

    // Sample the action(s)
    auto actions = (reparameterize) ? probabilities.rsample() : probabilities.sample();

    // use tanh for the returned action
    auto action = torch::tanh(actions).to(device);

    // Calcul log prob
    auto logProbs = probabilities.logProb(actions);

    // See the paper appendix, this reduce the jacobian da/du
    logProbs -= torch::log(1 - action.pow(2) + reparamNoise);

    // Sum the log prob
    logProbs = logProbs.sum(1, true);

    return std::make_pair(action, logProbs);

}

torch::optim::Adam* Networks::Actor::getPtrOptimizer(){
    return &optimizer;
}


void Networks::Actor::loadCheckpoint(){
    torch::serialize::InputArchive inputArchive;
    inputArchive.load_from(ROOT_DIR "/models/" + name + ".pt");
    load(inputArchive);
}

void Networks::Actor::saveCheckpoint(){
    torch::serialize::OutputArchive outputArchive;
    save(outputArchive);
    outputArchive.save_to(ROOT_DIR "/models/" + name + ".pt");
}