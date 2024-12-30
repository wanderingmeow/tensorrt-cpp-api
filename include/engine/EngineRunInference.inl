#pragma once
#include "engine.h"
#include "util/Util.h"

template <typename T>
bool Engine<T>::runInference(
    const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
    std::vector<std::vector<std::vector<T>>> &featureVectors) {
    // First we do some error checking
    if (inputs.empty() || inputs[0].empty()) {
        std::cerr << "Provided input vector is empty!\n";
        return false;
    }

    const auto numInputs = m_inputDims.size();
    if (inputs.size() != numInputs) {
        std::cerr << "Incorrect number of inputs provided!\n";
        return false;
    }

    // Ensure the batch size does not exceed the max
    if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
        std::cerr << "===== Error =====\n"
                  << "The batch size is larger than the model expects!\n"
                  << "Model max batch size: " << m_options.maxBatchSize << '\n'
                  << "Batch size provided to call to runInference: "
                  << inputs[0].size() << '\n';
        return false;
    }

    // Ensure that if the model has a fixed batch size that is greater than 1,
    // the input has the correct length
    if (m_inputBatchSize != -1 &&
        inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
        std::cerr
            << "===== Error =====\n"
            << "The batch size is different from what the model expects!\n"
            << "Model batch size: " << m_inputBatchSize << '\n'
            << "Batch size provided to call to runInference: "
            << inputs[0].size() << '\n';
        return false;
    }

    const auto batchSize = static_cast<int32_t>(inputs[0].size());
    // Make sure the same batch size was provided for all inputs
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].size() != static_cast<size_t>(batchSize)) {
            std::cerr << "===== Error =====\n"
                      << "The batch size is different for each input!\n";
            return false;
        }
    }

    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

    std::vector<cv::cuda::GpuMat> preprocessedInputs;

    // Preprocess all the inputs
    for (size_t i = 0; i < numInputs; ++i) {
        const auto &batchInput = inputs[i];
        const auto &dims = m_inputDims[i];

        auto &input = batchInput[0];
        if (input.channels() != dims.d[0] || input.rows != dims.d[1] ||
            input.cols != dims.d[2]) {
            std::cerr
                << "===== Error =====\n"
                << "Input does not have correct size!\n"
                << "Expected: (" << dims.d[0] << ", " << dims.d[1] << ", "
                << dims.d[2] << ")\n"
                << "Got: (" << input.channels() << ", " << input.rows << ", "
                << input.cols << ")\n"
                << "Ensure you resize your input image to the correct size\n";
            return false;
        }

        nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1],
                                     dims.d[2]};
        int32_t input_idx =
            m_engine->getBindingIndex(m_IOTensorNames[i].c_str());
        m_context->setBindingDimensions(input_idx,
                                        inputDims); // Define the batch size

        // OpenCV reads images into memory in NHWC format, while TensorRT
        // expects images in NCHW format. The following method converts NHWC to
        // NCHW. Even though TensorRT expects NCHW at IO, during optimization,
        // it can internally use NHWC to optimize cuda kernels See:
        // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-layout
        // Copy over the input data and perform the preprocessing
        auto mfloat =
            blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
        preprocessedInputs.push_back(mfloat);
        m_buffers[i] = mfloat.ptr<void>();
    }

    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        std::cerr << msg << '\n';
        throw std::runtime_error(msg);
    }

    // Set the address of the input and output buffers and run inference.
    bool status = m_context->enqueueV2(m_buffers.data(), inferenceCudaStream, nullptr);
    if (!status) {
        return false;
    }

    // Copy the outputs back to CPU
    featureVectors.clear();

    for (int batch = 0; batch < batchSize; ++batch) {
        // Batch
        std::vector<std::vector<T>> batchOutputs{};
        for (int32_t outputBinding = numInputs;
             outputBinding < m_engine->getNbBindings(); ++outputBinding) {
            // We start at index m_inputDims.size() to account for the inputs in
            // our m_buffers
            std::vector<T> output;
            auto outputLength = m_outputLengths[outputBinding - numInputs];
            output.resize(outputLength);
            // Copy the output
            Util::checkCudaErrorCode(
                cudaMemcpyAsync(output.data(),
                                static_cast<char *>(m_buffers[outputBinding]) +
                                    (batch * sizeof(T) * outputLength),
                                outputLength * sizeof(T),
                                cudaMemcpyDeviceToHost, inferenceCudaStream));
            batchOutputs.emplace_back(std::move(output));
        }
        featureVectors.emplace_back(std::move(batchOutputs));
    }

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}