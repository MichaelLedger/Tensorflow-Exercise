//
//  MUSIQHelper.cpp
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/6/6.
//

#include "MUSIQHelper.hpp"
#include <TensorFlowLiteC/TensorFlowLiteC.h>

//https://www.tensorflow.org/lite/guide/signatures

//SignatureRunner* encode_runner =
//    interpreter->GetSignatureRunner("encode");
//encode_runner->ResizeInputTensor("x", {100});
//encode_runner->AllocateTensors();
//
//TfLiteTensor* input_tensor = encode_runner->input_tensor("x");
//float* input = GetTensorData<float>(input_tensor);
//// Fill `input`.
//
//encode_runner->Invoke();
//
//const TfLiteTensor* output_tensor = encode_runner->output_tensor(
//    "encoded_result");
//float* output = GetTensorData<float>(output_tensor);
//// Access `output`.
