import tensorflow as tf

def inspect():
    interpreter = tf.lite.Interpreter(model_path="spaq.tflite")
    interpreter.allocate_tensors()

    # Print input shape and type
    inputs = interpreter.get_input_details()
    print('{} input(s):'.format(len(inputs)))
    for i in range(0, len(inputs)):
        print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

    # Print output shape and type
    outputs = interpreter.get_output_details()
    print('\n{} output(s):'.format(len(outputs)))
    for i in range(0, len(outputs)):
        print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))

if __name__ == '__main__':
    # .py executed as script
    # do something
    inspect()
