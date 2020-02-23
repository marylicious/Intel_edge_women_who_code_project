import argparse
import cv2
import numpy as np

from handle_models import handle_output, preprocessing
from inference import Network


def get_args():
    '''
    Gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def create_output_image(image, output):
    '''
    creates an output image showing the result of inference.
    '''
    print('image shape',image.shape[0]) 
    for box in output:
        xmin = int(box[0] * image.shape[1])
        ymin = int(box[1] * image.shape[0])
        xmax = int(box[2] * image.shape[1])
        ymax = int(box[3] * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0) , 3)

   
    return image


def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)

    ### Preprocess the input image
    preprocessed_image =  preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()
    
    #Process the output
    processed_output = handle_output(output,image.shape)

    # Create an output image based on network
    try:
        output_image = create_output_image(image, processed_output)
        print('Success')
    except:
        output_image=image
        print('Error')

    # Save down the resulting image
    cv2.imwrite("outputs/output.png", output_image)
    


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()