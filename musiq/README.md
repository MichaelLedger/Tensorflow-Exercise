# MUSIQ: Multi-scale Image Quality Transformer

This directory contains checkpoints and model inference code for the ICCV 2021
paper:
["MUSIQ: Multi-scale Image Quality Transformer"](https://arxiv.org/abs/2108.05997)
by Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang.

*Disclaimer: This is not an official Google product.*

![Model overview](images/overview.png)

## Using the models

The MUSIQ models are available on [TensorFlow Hub](https://tfhub.dev/s?q=musiq)
with documentation and a sample notebook for you to try.

But if you want to go deeper in the code, follow the instructions below.

## Pre-requisite

Install dependencies:

```
pip3 install -r requirements.txt
```

The model checkpoints can be downloaded from:
[download link](https://console.cloud.google.com/storage/browser/gresearch/musiq)

The folder contains the following checkpoints:

-   **ava_ckpt.npz**: Trained on AVA dataset.
-   **koniq_ckpt.npz**: Trained on KonIQ dataset.
-   **paq2piq_ckpt.npz**: Trained on PaQ2PiQ dataset.
-   **spaq_ckpt.npz**: Trained on SPAQ dataset.
-   **imagenet_pretrain.npz**: Pretrained checkpoint on ImageNet.

## Run Inference

```shell
python3 -m musiq.run_predict_image \
  --ckpt_path=/tmp/spaq_ckpt.npz \
  --image_path=/tmp/image.jpeg
```

## Citation

If you find this code is useful for your publication, please cite the original
paper:

```
@inproceedings{jke_musiq_iccv2021,
  title = {MUSIQ: Multi-scale Image Quality Transformer},
  author = {Junjie Ke and Qifei Wang and Yilin Wang and Peyman Milanfar and Feng Yang},
  booktitle = {ICCV},
  year = {2021}
}
```

# iOS Demo
## Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter.
```
2023-04-18 11:09:18.315345+0800 MUSIQ-Demo[2245:652100] Created TensorFlow Lite delegate for Metal.
INFO: Created TensorFlow Lite delegate for Metal.
2023-04-18 11:09:18.315941+0800 MUSIQ-Demo[2245:652100] Metal GPU Frame Capture Enabled
2023-04-18 11:09:18.316234+0800 MUSIQ-Demo[2245:652100] Metal API Validation Enabled
2023-04-18 11:09:18.389871+0800 MUSIQ-Demo[2245:652098] Initialized TensorFlow Lite runtime.
INFO: Initialized TensorFlow Lite runtime.
2023-04-18 11:09:18.412654+0800 MUSIQ-Demo[2245:652101] Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. For the Android, it can be resolved by adding "org.tensorflow:tensorflow-lite-select-tf-ops" dependency. See instructions: https://www.tensorflow.org/lite/guide/ops_select
ERROR: Select TensorFlow op(s), included in the given model, is(are) not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. For the Android, it can be resolved by adding "org.tensorflow:tensorflow-lite-select-tf-ops" dependency. See instructions: https://www.tensorflow.org/lite/guide/ops_select
2023-04-18 11:09:18.412768+0800 MUSIQ-Demo[2245:652101] Node number 0 (FlexDecodeJpeg) failed to prepare.
ERROR: Node number 0 (FlexDecodeJpeg) failed to prepare.
Failed to create the interpreter with error: Failed to allocate memory for input tensors.
Failed to initialize: internalError(Failed to allocate memory for input tensors.)
```

```
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

After running pod install, you need to provide an additional linker flag to force load the select TF ops framework into your project. In your Xcode project, go to Build Settings -> Other Linker Flags, and add:

For versions >= 2.9.0:

```
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.xcframework/ios-arm64/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

For versions < 2.9.0:

```
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

**You should then be able to run any models converted with the SELECT_TF_OPS in your iOS app. For example, you can modify the [Image Classification iOS app](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios) to test the select TF ops feature.**

https://www.tensorflow.org/lite/guide/ops_select

https://stackoverflow.com/questions/72311274/ios-tensorflow-lite-error-select-tensorflow-ops-included-in-the-given-mode

**Note: If you need to use select TF ops in an x86_64 simulator, you can build the select ops framework yourself. See Using [Bazel + Xcode](https://www.tensorflow.org/lite/guide/ops_select#using_bazel_xcode) section for more details.**

## Undefined symbols for architecture arm64:
```
  "tensorflow::MemoryDump::MemoryDump(google::protobuf::Arena*, bool)", referenced from:
      tsl::BFCAllocator::RecordMemoryMapInternal() in TensorFlowLiteSelectTfOps(bfc_allocator_0b56374a630e47938a541b3a1bc43692.o)
  "tensorflow::MemoryDump::~MemoryDump()", referenced from:
      tsl::BFCAllocator::MaybeWriteMemoryMap() in TensorFlowLiteSelectTfOps(bfc_allocator_0b56374a630e47938a541b3a1bc43692.o)
  "tensorflow::RPCOptions::MergeImpl(google::protobuf::Message&, google::protobuf::Message const&)", referenced from:
      tensorflow::ConfigProto::MergeImpl(google::protobuf::Message&, google::protobuf::Message const&) in TensorFlowLiteSelectTfOps(config.pb.o)
```

https://github.com/tensorflow/tensorflow/issues/56255

Hi, you just need to update your pod file to depend on the latest nightly build version.

```
target 'my-app' do
  use_frameworks!

  # Pods for my-app
  pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
end
```

run `pod update --verbose` and `pod install`

## [GPU] Failed to initialize: internalError(Failed to create the interpreter.)

```
2023-04-18 12:39:14.969942+0800 MUSIQ-Demo[2313:680761] Created TensorFlow Lite delegate for Metal.
INFO: Created TensorFlow Lite delegate for Metal.
2023-04-18 12:39:14.970584+0800 MUSIQ-Demo[2313:680765] Metal GPU Frame Capture Enabled
2023-04-18 12:39:14.971035+0800 MUSIQ-Demo[2313:680765] Metal API Validation Enabled
2023-04-18 12:39:15.051938+0800 MUSIQ-Demo[2313:680765] Initialized TensorFlow Lite runtime.
INFO: Initialized TensorFlow Lite runtime.
2023-04-18 12:39:15.056765+0800 MUSIQ-Demo[2313:680757] Created TensorFlow Lite delegate for select TF ops.
INFO: Created TensorFlow Lite delegate for select TF ops.
2023-04-18 12:39:15.064499+0800 MUSIQ-Demo[2313:680761] TfLiteFlexDelegate delegate: 34 nodes delegated out of 1181 nodes with 3 partitions.
INFO: TfLiteFlexDelegate delegate: 34 nodes delegated out of 1181 nodes with 3 partitions.

Failed to create the interpreter with error: Failed to create the interpreter.
[CPU] Success to initialize: MUSIQ_Demo.MUSIQTransferer
[GPU] Failed to initialize: internalError(Failed to create the interpreter.)
```

```
let createOptions: () -> Interpreter.Options? = {
    if useMetalDelegate {
        return nil
    }
    var options = Interpreter.Options()
    options.threadCount = ProcessInfo.processInfo.processorCount >= 2 ? 2 : 1
    return options
}

do {
    // Create the `Interpreter`s.
    let predictInterpreter = try Interpreter(
        modelPath: predictModelPath,
        options: createOptions(),
        delegates: createDelegates()
    )
} catch let error {
}
```

## Failed to invoke the interpreter with error: Provided data count 786432 must match the required count 0.

```
    // Copy the RGB data to the input `Tensor`.
    try self.predictInterpreter.copy(styleRGBData, toInputAt: 0)
```

```
    let byteCount = TfLiteTensorByteSize(cTensor)
    guard data.count == byteCount else {
      throw InterpreterError.invalidTensorDataCount(provided: data.count, required: byteCount)
    }
```

It seems the problem is with input buffer size.

https://github.com/tensorflow/tensorflow/issues/35864

https://stackoverflow.com/questions/63842275/tensorflow-interpreter-throwing-error-for-data-count-ios

## Thread 2: EXC_BAD_ACCESS (code=1, address=0x0)

```
// Run inference by invoking the `Interpreter`.
try self.predictInterpreter.invoke()
```

```
[CPU] Success to initialize: MUSIQ_Demo.MUSIQTransferer
[GPU] Failed to initialize: internalError(Failed to create the interpreter.)
2023-04-18 12:51:28.520108+0800 MUSIQ-Demo[2324:685191] invalid mode 'kCFRunLoopCommonModes' provided to CFRunLoopRunSpecific - break on _CFRunLoopError_RunCalledWithInvalidMode to debug. This message will only appear once per execution.
```
