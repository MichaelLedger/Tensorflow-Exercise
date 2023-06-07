//
//  MUSIQTransferer.swift
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/14.
//

import TensorFlowLite
import UIKit

class MUSIQTransferer {
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var predictInterpreter: Interpreter
    
    /// Dedicated DispatchQueue for TF Lite operations.
    private let tfLiteQueue: DispatchQueue
    
    // MARK: - Initialization
    
    /// Create a Style Transferer instance with a quantized Int8 model that runs inference on the CPU.
    static func newCPUMUSIQTransferer(
        completion: @escaping ((Result<MUSIQTransferer>) -> Void)
    ) -> () {
        return MUSIQTransferer.newInstance(predictModel: Constants.Int8.predictModel,
                                           useMetalDelegate: false,
                                           completion: completion)
    }
    
    static func newGPUMUSIQTransferer(
        completion: @escaping ((Result<MUSIQTransferer>) -> Void)
    ) -> () {
        return MUSIQTransferer.newInstance(predictModel: Constants.Float16.predictModel,
                                           useMetalDelegate: true,
                                           completion: completion)
    }
    
    /// Create a new Style Transferer instance.
    static func newInstance(predictModel: String,
                            useMetalDelegate: Bool,
                            completion: @escaping ((Result<MUSIQTransferer>) -> Void)) {
        // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
        let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.style_transfer")
        
        /*
         2023-04-20 18:16:57.353254+0800 MUSIQ-Demo[3576:1230192] invalid mode 'kCFRunLoopCommonModes' provided to CFRunLoopRunSpecific - break on _CFRunLoopError_RunCalledWithInvalidMode to debug. This message will only appear once per execution.
         Failed to invoke the interpreter with error: Failed to copy data to input tensor.
         */
//        let tfLiteQueue = DispatchQueue.main//test
        
        // Run initialization in background thread to avoid UI freeze.
        tfLiteQueue.async {
            // Construct the path to the model file.
            guard
                let predictModelPath = Bundle.main.path(
                    forResource: predictModel,
                    ofType: Constants.modelFileExtension
                )
            else {
                completion(.error(InitializationError.invalidModel(
                    "One of the following models could not be loaded: \(predictModel)"
                )))
                return
            }
            
            // Specify the delegate for the TF Lite `Interpreter`.
            let createDelegates: () -> [Delegate]? = {
                if useMetalDelegate {
                    return [MetalDelegate()]
                }
                return nil
            }
            let createOptions: () -> Interpreter.Options? = {
                if useMetalDelegate {
                    return nil
                }
                var options = Interpreter.Options()
                options.threadCount = ProcessInfo.processInfo.processorCount >= 2 ? 2 : 1
                options.isXNNPackEnabled = false
                return options
            }
            
            do {
                // Create the `Interpreter`s.
                let predictInterpreter = try Interpreter(
                    modelPath: predictModelPath,
                    options: createOptions(),
                    delegates: createDelegates()
                )
                
                // Allocate memory for the model's input `Tensor`s.
                try predictInterpreter.allocateTensors()
                
                //test
                // Get the input tensor index and shape
//                let inputIndex = predictInterpreter.input(at: 0)
                
                // Create a new tensor with the desired data type and shape
//                let inputTensor = try! Tensor.allocate(shape: Tensor.Shape([1, 224, 224, 3])), dataType: .string)
                
//                let inputTensor = try predictInterpreter.input(at: 0)
//                print("[inputTensor]:\(inputTensor)")
                
                // Create an MUSIQTransferer instance and return.
                let MUSIQTransferer = MUSIQTransferer(
                    tfLiteQueue: tfLiteQueue,
                    predictInterpreter: predictInterpreter
                )
                DispatchQueue.main.async {
                    completion(.success(MUSIQTransferer))
                }
            } catch let error {
                print("Failed to create the interpreter with error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.error(InitializationError.internalError(error)))
                }
                return
            }
        }
    }
    
    /// Initialize Style Transferer instance.
    fileprivate init(
        tfLiteQueue: DispatchQueue,
        predictInterpreter: Interpreter
    ) {
        // Store TF Lite intepreter
        self.predictInterpreter = predictInterpreter
        
        // Store the dedicated DispatchQueue for TFLite.
        self.tfLiteQueue = tfLiteQueue
    }
    
    // MARK: - Style Transfer
    
    /// Run style transfer on a given image.
    /// - Parameters
    ///   - styleImage: the image to use as a style reference.
    ///   - image: the target image.
    ///   - completion: the callback to receive the style transfer result.
    func runMUSIQTransfer(style styleImage: UIImage,
                          image: UIImage,
                          completion: @escaping ((Result<MUSIQTransferResult>) -> Void)) {
        tfLiteQueue.async {
            
//            guard let inputTensor = try? self.predictInterpreter.input(at: 0),
//                  let outputTensor = try? self.predictInterpreter.output(at: 0) else {
//                fatalError("Failed to get input/output tensors.")
//                return
//            }
            
//            let outputTensor: Tensor
            let startTime: Date = Date()
            var preprocessingTime: TimeInterval = 0
            var stylePredictTime: TimeInterval = 0
//            var styleTransferTime: TimeInterval = 0
//            var postprocessingTime: TimeInterval = 0
            
            func timeSinceStart() -> TimeInterval {
                return abs(startTime.timeIntervalSinceNow)
            }
            
            do {
                // Preprocess style image.
//                guard
//                    let styleRGBData = styleImage.scaledData(
//                        with: Constants.styleImageSize,
//                        isQuantized: false
//                    )
//                else {
//                    DispatchQueue.main.async {
//                        completion(.error(MUSIQTransfererror.invalidImage))
//                    }
//                    print("Failed to convert the style image buffer to RGB data.")
//                    return
//                }
                
                guard
                    let inputRGBData = image.scaledData(
                        with: Constants.inputImageSize,
                        isQuantized: false
                    )
                else {
                    DispatchQueue.main.async {
                        completion(.error(MUSIQTransfererror.invalidImage))
                    }
                    print("Failed to convert the input image buffer to RGB data.")
                    return
                }
                
                guard let imageBytesBase64EncodedString = image.toPngString()
                else {
                    DispatchQueue.main.async {
                        completion(.error(MUSIQTransfererror.invalidImage))
                    }
                    print("Failed to convert the input image buffer to Base64 string.")
                    return
                }
                
                let imageBytesBase64EncodedStringData = Data(imageBytesBase64EncodedString.utf8)
                
                preprocessingTime = timeSinceStart()
                
                // Copy the RGB data to the input `Tensor`.
//                try self.predictInterpreter.copy(styleRGBData, toInputAt: 0)
                //test
//                try self.predictInterpreter.copy(inputRGBData, toInputAt: 0)
//                try self.predictInterpreter.copy(imageBytesBase64EncodedStringData, toInputAt: 0)
                
                try self.predictInterpreter.resizeInput(at: 0, to: Tensor.Shape([imageBytesBase64EncodedStringData.count]))
                try self.predictInterpreter.allocateTensors()
                try self.predictInterpreter.copy(imageBytesBase64EncodedStringData, toInputAt: 0)
                let resizedInterpreterInput = try self.predictInterpreter.input(at: 0)
                print("resizedInterpreterInput:\(resizedInterpreterInput.shape)")
                
                let encode_runner: SignatureRunner = try SignatureRunner(interpreter: self.predictInterpreter, signatureKey: "serving_default")
//                let signatureInputTensor = try! Tensor(name: "keras_layer_input", dataType: Tensor.DataType.float32, shape: Tensor.Shape([1]), data: imageBytesBase64EncodedStringData)
//                let signatureInputTensor = try! Tensor.allocate(shape: Tensor.Shape([1])), dataType: .string)
                /*
                (lldb) po encode_runner.inputs
                ▿ 1 element
                  - 0 : "image_bytes_tensor"
                 */
                
                try encode_runner.allocateTensors()
                
                let signature_input_name = encode_runner.inputs[0]
                
//                let signature_input = try encode_runner.input(named: signature_input_name)
                
                let signature_input_custom = Tensor(
                  name: signature_input_name,
                  dataType: Tensor.DataType.float32,
                  shape: Tensor.Shape.init(-1),
                  data: imageBytesBase64EncodedStringData,
                  quantizationParameters: nil
                )
                
                //https://github.com/tensorflow/tensorflow/issues/22377
                /*
                 TensorFlow Lite Error: tensorflow/lite/core/subgraph.cc:1103 tensor->dims->size != dims.size() (0 != 1)
                 Failed to invoke the interpreter with error: Failed to resize input tensor with input name (image_bytes_tensor).
                 */
                
                // Only unknown dimensions can be resized with this function.
                /*
                 /// Resizes the input tensor identified as `input_name` to be the dimensions
                 /// specified by `input_dims` and `input_dims_size`. Only unknown dimensions can
                 /// be resized with this function. Unknown dimensions are indicated as `-1` in
                 /// the `dims_signature` attribute of a TfLiteTensor.
                 ///
                 /// Returns status of failure or success. Note that this doesn't actually resize
                 /// any existing buffers. A call to TfLiteSignatureRunnerAllocateTensors() is
                 /// required to change the tensor input buffer.
                 ///
                 /// NOTE: This function is similar to TfLiteInterpreterResizeInputTensorStrict()
                 /// and not TfLiteInterpreterResizeInputTensor().
                 ///
                 /// NOTE: `input_name` must match the name of an input in the signature.
                 ///
                 /// NOTE: This function makes a copy of the input dimensions, so the caller can
                 /// safely deallocate `input_dims` immediately after this function returns.
                 ///
                 /// WARNING: This is an experimental API and subject to change.
                 */
                //https://www.tensorflow.org/lite/guide/inference#run_inference_with_dynamic_shape_model
//                try encode_runner.resizeInput(named: signatureInputName, toShape: Tensor.Shape([imageBytesBase64EncodedStringData.count]))
                try encode_runner.resizeInput(named: signature_input_name, toShape: signature_input_custom.shape)
                // Note: After resizing an input tensor, the client **must** explicitly call
                // `allocateTensors()` before attempting to access the resized tensor data.
                try encode_runner.allocateTensors()
                
                try encode_runner.invoke(with: [signature_input_name: imageBytesBase64EncodedStringData])
                let signatureOutputTensor = try encode_runner.output(named: signature_input_name)
                print("[signatureOutputTensor]:\(signatureOutputTensor)")
                
                // Construct score from output tensor data
                let score = self.postprocessNIMAData(data: signatureOutputTensor.data)
                
                
//                let inputTensor = try self.predictInterpreter.input(at: 0)
//                print("[inputTensor]:\(inputTensor)")
//                let byteCount = inputTensor.data.count
                
                // input image path
                // <tf.Tensor 'image_bytes_tensor:0' shape=() dtype=string>
//                let scaledImage = UIImage(data: inputRGBData)
//                let scaledImage = image.scaledImage(with: Constants.inputImageSize)
//                if let image = scaledImage {
//                    var imageSandboxPath: URL? = nil
//                    if let data = image.jpegData(compressionQuality: 1) ?? image.pngData() {
//                        if let directory = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) as NSURL {
//                            imageSandboxPath = directory.appendingPathComponent("sample.png")
//                            print("[image_path]:\(imageSandboxPath!)")
//                            do {
//                                try data.write(to: imageSandboxPath!)
//                            } catch {
//                                print(error.localizedDescription)
//                            }
//                        }
//                    }
//                    if imageSandboxPath != nil {
//                        try self.predictInterpreter.copy(imageSandboxPath!.absoluteString, toInputAt: 0)
////                        try self.predictInterpreter.allocateTensors()
////                        if let imageSandboxPathData = imageSandboxPath!.absoluteString.data(using: .utf8) {
////                            try self.predictInterpreter.copy(imageSandboxPathData, toInputAt: 0)
////                        }
//                    }
//                }
                
                // Run inference by invoking the `Interpreter`.
//                try self.predictInterpreter.invoke()
                
                // Get the output `Tensor` to process the inference results.
                //        let predictResultTensor = try self.predictInterpreter.output(at: 0)
                
                // Grab bottleneck data from output tensor.
                //        let bottleneck = predictResultTensor.data
                
                stylePredictTime = timeSinceStart() - preprocessingTime
                
                // Copy the RGB and bottleneck data to the input `Tensor`.
                //        try self.transferInterpreter.copy(inputRGBData, toInputAt: 0)
                //        try self.transferInterpreter.copy(bottleneck, toInputAt: 1)
//                try self.predictInterpreter.copy(inputRGBData, toInputAt: 0)
                
                // Run inference by invoking the `Interpreter`.
                //        try self.transferInterpreter.invoke()
                
                // Get the result tensor
                //        outputTensor = try self.transferInterpreter.output(at: 0)
//                outputTensor = try self.predictInterpreter.output(at: 0)
                
//                styleTransferTime = timeSinceStart() - stylePredictTime - preprocessingTime
                
                // Return the result.
                DispatchQueue.main.async {
                    completion(
                        .success(
                            MUSIQTransferResult(
                                score: score,
                                preprocessingTime: preprocessingTime,
                                stylePredictTime: stylePredictTime
    //                            styleTransferTime: styleTransferTime,
    //                            postprocessingTime: postprocessingTime
                            )
                        )
                    )
                }
                
            } catch let error {
                print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.error(MUSIQTransfererror.internalError(error)))
                }
                return
            }
            
//            // Construct score from output tensor data
//            let score = self.postprocessNIMAData(data: outputTensor.data)
            
            // Construct image from output tensor data
//            guard let cgImage = self.postprocessImageData(data: outputTensor.data) else {
//                DispatchQueue.main.async {
//                    completion(.error(MUSIQTransfererror.resultVisualizationError))
//                }
//                return
//            }
            
//            let outputImage = UIImage(cgImage: cgImage)
            
//            postprocessingTime = timeSinceStart() - stylePredictTime - preprocessingTime
        }
    }
    
    // MARK: - Utils
    private func postprocessStringData(data: Data) -> String? {
        guard let score = String(data: data, encoding: .utf8) else {
            return nil
        }
        return score
    }
    
    private func postprocessNIMAData(data: Data) -> Float {
        let floatArray = data.toArray(type: Float32.self)
        // calcucate mean score
        let score = calculateMeanScore(scoreDist: floatArray)
        return score
    }
    
    private func normalizeLabels(labels: [Float]) -> [Float] {
        var normalizedLabels: [Float] = []
        labels.forEach { label in
            let sum = labels.reduce(0, +)
            normalizedLabels.append(label / sum)
        }
        return normalizedLabels
    }

    private func calculateMeanScore(scoreDist: [Float]) -> Float {
        let score_dist = normalizeLabels(labels: scoreDist)
        var score: Float = 0
        score_dist.indices.forEach { index in
            score += score_dist[index] * Float(index+1)
        }
        return score
    }
    
    /// Turns TF model's float32 array output into one supported by `CGImage`. This method
    /// assumes the provided data is the same format as the data returned from the output
    /// tensor in `runStyleTransfer`, so it should not be used for general image processing.
    /// - Parameter data: The image data to turn into a `CGImage`. This data must be a buffer of
    ///   `Float32` values between 0 and 1 in RGB format.
    /// - Parameter size: The expected size of the output image.
    private func postprocessImageData(data: Data,
                                      size: CGSize = Constants.inputImageSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        let floats = data.toArray(type: Float32.self)
        
        let bufferCapacity = width * height * 4
        let unsafePointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferCapacity)
        let unsafeBuffer = UnsafeMutableBufferPointer<UInt8>(start: unsafePointer,
                                                             count: bufferCapacity)
        defer {
            unsafePointer.deallocate()
        }
        
        for x in 0 ..< width {
            for y in 0 ..< height {
                let floatIndex = (y * width + x) * 3
                let index = (y * width + x) * 4
                let red = UInt8(floats[floatIndex] * 255)
                let green = UInt8(floats[floatIndex + 1] * 255)
                let blue = UInt8(floats[floatIndex + 2] * 255)
                
                unsafeBuffer[index] = red
                unsafeBuffer[index + 1] = green
                unsafeBuffer[index + 2] = blue
                unsafeBuffer[index + 3] = 0
            }
        }
        
        let outData = Data(buffer: unsafeBuffer)
        
        // Construct image from output tensor data
        let alphaInfo = CGImageAlphaInfo.noneSkipLast
        let bitmapInfo = CGBitmapInfo(rawValue: alphaInfo.rawValue)
            .union(.byteOrder32Big)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let imageDataProvider = CGDataProvider(data: outData as CFData),
            let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: MemoryLayout<UInt8>.size * 4 * Int(Constants.inputImageSize.width),
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: imageDataProvider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        else {
            return nil
        }
        return cgImage
    }
    
}

// MARK: - Types

/// Representation of the style transfer result.
struct MUSIQTransferResult {
    
    /// The Score of input image
    let score: Float
    
    /// The resulting image from the style transfer.
//    let resultImage: UIImage
    
    /// Time required to resize the input and style images and convert the image
    /// data to a format the model can accept.
    let preprocessingTime: TimeInterval
    
    /// The style prediction model run time.
    let stylePredictTime: TimeInterval
    
    /// The style transfer model run time.
//    let styleTransferTime: TimeInterval
    
    /// Time required to convert the model output data to a `CGImage`.
//    let postprocessingTime: TimeInterval
    
}

/// Convenient enum to return result with a callback
enum Result<T> {
    case success(T)
    case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
    // Invalid TF Lite model
    case invalidModel(String)
    
    // Invalid label list
    case invalidLabelList(String)
    
    // TF Lite Internal Error when initializing
    case internalError(Error)
}

/// Define errors that could happen when running style transfer
enum MUSIQTransfererror: Error {
    // Invalid input image
    case invalidImage
    
    // TF Lite Internal Error when initializing
    case internalError(Error)
    
    // Invalid input image
    case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
    
    // Namespace for quantized Int8 models.
    enum Int8 {
        
        static let predictModel = "spaq" //"nima"//"spaq" //"koniq"
        
    }
    
    // Namespace for Float16 models, optimized for GPU inference.
    enum Float16 {
        
        static let predictModel = "spaq" //"nima"//"spaq"//"koniq"
        
    }
    
    static let modelFileExtension = "tflite"
    
//    static let styleImageSize = CGSize(width: 256, height: 256)
    
    static let inputImageSize = CGSize(width: 224, height: 224)
    
}
