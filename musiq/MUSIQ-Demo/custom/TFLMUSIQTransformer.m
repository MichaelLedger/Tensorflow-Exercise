//
//  TFLMUSIQTransformer.m
//  MUSIQ-Demo
//
//  Created by Gavin Xiang on 2023/4/18.
//

#import "TFLMUSIQTransformer.h"
#include "musiq_transformer.h"

@interface TFLMUSIQTransformer ()
/** TfLiteMUSIQTransformer backed by C API */
@property(nonatomic) TfLiteMUSIQTransformer *imageClassifier;
@end

@implementation TFLMUSIQTransformerOptions
@synthesize baseOptions;
@synthesize classificationOptions;

- (instancetype)init {
  self = [super init];
  if (self) {
    self.baseOptions = [[TFLBaseOptions alloc] init];
    self.classificationOptions = [[TFLClassificationOptions alloc] init];
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [self init];
  if (self) {
    self.baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

@end

@implementation TFLMUSIQTransformer
- (void)dealloc {
  TfLiteImageClassifierDelete(_imageClassifier);
}

- (instancetype)initWithImageClassifier:(TfLiteImageClassifier *)imageClassifier {
  self = [super init];
  if (self) {
    _imageClassifier = imageClassifier;
  }
  return self;
}

+ (nullable instancetype)imageClassifierWithOptions:(TFLMUSIQTransformerOptions *)options
                                              error:(NSError **)error {
  if (!options) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"TFLMUSIQTransformerOptions argument cannot be nil."];
    return nil;
  }

  TfLiteImageClassifierOptions cOptions = TfLiteImageClassifierOptionsCreate();

  if (![options.classificationOptions copyToCOptions:&(cOptions.classification_options)
                                               error:error]) {
    [options.classificationOptions
        deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];
    return nil;
  }

  [options.baseOptions copyToCOptions:&(cOptions.base_options)];

  TfLiteSupportError *cCreateClassifierError = NULL;
  TfLiteImageClassifier *cImageClassifier =
      TfLiteImageClassifierFromOptions(&cOptions, &cCreateClassifierError);

  [options.classificationOptions
      deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];

  // Populate iOS error if TfliteSupportError is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:cCreateClassifierError toError:error]) {
    TfLiteSupportErrorDelete(cCreateClassifierError);
  }

  // Return nil if classifier evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cImageClassifier) {
    return nil;
  }

  return [[TFLMUSIQTransformer alloc] initWithImageClassifier:cImageClassifier];
}

- (nullable TFLClassificationResult *)classifyWithGMLImage:(GMLImage *)image
                                                     error:(NSError **)error {
  return [self classifyWithGMLImage:image
                   regionOfInterest:CGRectMake(0, 0, image.width, image.height)
                              error:error];
}

- (nullable TFLClassificationResult *)classifyWithGMLImage:(GMLImage *)image
                                          regionOfInterest:(CGRect)roi
                                                     error:(NSError **)error {
  if (!image) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"GMLImage argument cannot be nil."];
    return nil;
  }

  TfLiteFrameBuffer *cFrameBuffer = [image cFrameBufferWithError:error];

  if (!cFrameBuffer) {
    return nil;
  }

  TfLiteBoundingBox boundingBox = {.origin_x = roi.origin.x,
                                   .origin_y = roi.origin.y,
                                   .width = roi.size.width,
                                   .height = roi.size.height};

  TfLiteSupportError *classifyError = NULL;
  TfLiteClassificationResult *cClassificationResult = TfLiteImageClassifierClassifyWithRoi(
      _imageClassifier, cFrameBuffer, &boundingBox, &classifyError);

  free(cFrameBuffer->buffer);
  cFrameBuffer->buffer = NULL;

  free(cFrameBuffer);
  cFrameBuffer = NULL;

  // Populate iOS error if C Error is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:classifyError toError:error]) {
    TfLiteSupportErrorDelete(classifyError);
  }

  // Return nil if C result evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cClassificationResult) {
    return nil;
  }

  TFLClassificationResult *classificationHeadsResults =
      [TFLClassificationResult classificationResultWithCResult:cClassificationResult];
  TfLiteClassificationResultDelete(cClassificationResult);

  return classificationHeadsResults;
}
@end

