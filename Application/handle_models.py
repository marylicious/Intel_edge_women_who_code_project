import cv2
import numpy as np

def handle_output(output, input_shape):
    #print(output)
    boxes = []
    for val,box in enumerate(output['ActionNet/out_detection_loc'][0]):
        if output['ActionNet/out_detection_conf'][0][val][1]>0.8:
            #print('box',val,box)
            boxes.append(box)
            

   # print('output box coordinates',output['ActionNet/out_detection_loc'])
   # print('output conf', output['ActionNet/out_detection_conf'])
    return boxes



def preprocessing(input_image, height, width):

    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image