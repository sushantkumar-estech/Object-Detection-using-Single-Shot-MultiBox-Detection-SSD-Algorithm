# Object Detection

# Importing the libraries
import torch  # importing pytorch, its an efficient tool, have dynamic, efficient graph structure for fast & efficient computation
from torch.autograd import Variable  # torch.autgrad is respnsible for gradient desecent, here Variable will convert tensor into torch gradient
import cv2  # Using the draw the rectangle, but the detection will not be based on Open CV
from data import BaseTransform # BaseTransform will make the image compatible with Neural Network
from data import VOC_CLASSES as labelmap  # its a dictionary that will do the encoding of the classes, eg; planes will be encoded as 1, dog 2
from ssd import build_ssd  # build_ssd will be the construcer to built the architecture of Neural Network
import imageio  # this library is to process the images of the video and applying the detect function that we will implement on the images

# Defining a function that will do the detections
def detect(frame, net, transform):# this function will return this sync frame with the rectangle of object detection label on it
    height, width = frame.shape[:2] # generally frame.shape return 3 argument, height, width and channel (3 for color image and 1 for gray)
    frame_t = transform(frame)[0]  # just to have the first variable, that is only import for further processing.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # x here is torch tensor, the SSD NN trained for BRG color segment, so converting torch tenor from RGB to BRG
    x = Variable(x.unsqueeze(0))# adding fake dimentions because the NN can't accept the entire image, it take images only in batches, also converting into torch Variable.
    y = net(x)  # feeding the torch veriable into the neural network
    detections = y.data   # we will get the values we are interested in by adding the data attribute to the output y
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].
    for i in range(detections.size(1)): # detectio  = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1)]
        j = 0
        while detections[0, i, j, 0] >=0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy() # now we will use openCV to draw the rectange, as openCV work with numpy array, so we will convert it into numpy array as well
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)  #converting these coordinate in integer for safer side., # printing the label on the detected object
            cv2.putText(frame, labelmap[i-1],(int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # here this i-1 shows the element, as the array in python start from 0, so the index -1 is the element of that particular index
            j += 1    # trying writing j++
    return frame

# Creating the SSD neural Network, we are loading pretrained model by loading the weights in the model
net = build_ssd('test')  # making a SSD Neural Network
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # now loading the pretrained weight on the model, that torch.load function will open the library, that will open a tensor, which will contains weights

# Creating the transformation
# this transformation will make sure that the input image to the neural network will be compatible with the neural network
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # to set the color value of the image to the right scale, same as the object on which the network trained

# Doing some object detection on a video
reader = imageio.get_reader('funny_dog.mp4')  #loading the video on which the detection needs to be done
fps = reader.get_meta_data()['fps']       #taking froms from the video for application of object detection

writer = imageio.get_writer('output.mp4', fps = fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)  #just to align with the build SSD fucntuin made
    writer.append_data(frame)
    print(i)
writer.close()

