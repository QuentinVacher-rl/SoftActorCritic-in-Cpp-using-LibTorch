
#ifndef SAC_PARAMETERS_H
#define SAC_PARAMETERS_H

#include <thread>
#include <iostream>

namespace Json {
    class Value;
}

class SACParameters {


private:
    /**
     * \brief Puts the parameters described in the derivative tree root in
     * given LearningParameters.
     *
     * Browses the JSON tree. If the node we're looking at is a leaf,
     * we call setParameterFromString. Otherwise, we browe its children to
     * follow the known parameters structure.
     *
     * \param[in] root JSON tree we will use to set parameters.
     * \param[out] params the LearningParameters being updated.
     */
    void setAllParamsFrom(const Json::Value& root);

    /**
     * \brief Reads a given json file and puts the derivative tree in root.
     *
     * Opens the file and calls the parseFromStream() method from JsonCpp
     * which handles all the parsing of the JSON file. It eventually returns
     * errors in a parameter, e.g. if the file does not respect JSON format.
     * In this case, the file is simply ignored and it is logged explicitly.
     * However, in case of JsonCpp internal errors, there can be exceptions,
     * as described in throws.
     *
     * \param[in] path path of the JSON file from which the parameters are
     *            read.
     * \param[out] root JSON tree we are going to build with the file.
     * \throws std::exception if json parser settings are not in their
     * right formats.
     */
    void readConfigFile(const char* path, Json::Value& root);

    /**
     * \brief Given a parameter name, sets its value in given
     * LearningParameters.
     *
     * To find the right parameter, the method contains a lot of if
     * statements, each of them finishing by a return. These statements
     * compare the given parameter name to known parameters names.
     * If a parameter is found, it casts value to the right type and sets
     * the given parameter to this value.
     * If no parameter was found, it simply ignores the input and logs it
     * explicitly.
     *
     * \param[in] param the name of the LearningParameters being updated.
     * \param[in] value the value we want to set the parameter to.
     */
    void setParameterFromString(const std::string& param,
                                Json::Value const& value);

public:
    /// Learning rate of the models
    double lr = 0.003;

    /// Gamma for bellman equation
    double gamma = 0.99;

    /// Reward scale 
    double rewardScale = 2.0;

    /// Size of the buffer
    int sizeBuffer = 100000;

    /// Batch size of training iterations
    int batchSize = 256;

    /// Coefficient to update the targetValue model
    double tau = 0.005;

    /// Size of the first layer of the models
    int sizeHL1 = 256;

    /// Size of the second layer of the models
    int sizeHL2 = 256;

    /// True to load the models
    bool loadModels = false;

    /**
     * \brief Loads a given json file and fills the parameters it contains
     * in given LearningParameters.
     *
     * High level method that simply calls more complicated ones as follow :
     * - readConfigFile to get the derivative tree from a JSON file path
     * - setAllParamsFrom to set the parameters given the obtained tree.
     *
     * \param[in] path path of the JSON file from which the parameters are
     *            read.
     * \param[out] params the LearningParameters being updated.
     */
    void loadParametersFromJson(const char* path);

};

#endif
