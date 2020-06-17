# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os
from PIL import Image
from numpy import *
from keras.models import load_model
from utils import *
import xlwt
from xlwt import Workbook
import time

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################
# print the arguments
#print('----- info -----')
#print('[i] The config file: ', args.model_cfg)
#print('[i] The weights of model file: ', args.model_weights)
#print('[i] Path to image file: ', args.image)
#print('[i] Path to video file: ', args.video)
#print('###########################################################\n')

# check outputs directory
if not os.path.exists(args.output_dir):
    #print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
#else:
    #print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

def _main():

    wind_name = 'face detection using YOLOv3'
    model = load_model('facenet_keras.h5')
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    org_img = Image.open('vasu.jpg')
    image = org_img.resize((160,160))
    face_array = asarray(image)
    originalembedding = get_embedding(model,face_array)

    output_file = ''

    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
    elif args.video:
        if not os.path.isfile(args.video):
            print("[!] ==> Input video file {} doesn't exist".format(args.video))
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    else:
        # Get data from the camera
        cap = cv2.VideoCapture(args.src)

    # Get the video writer initialized to save the output video
    if not args.image:
        video_writer = cv2.VideoWriter(os.path.join(args.output_dir, output_file),
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                       cap.get(cv2.CAP_PROP_FPS), (
                                           round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    count=0

    while True:

        has_frame, frame = cap.read()
        count=count+1

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            #print('[i] ==> Output file is stored at', os.path.join(args.output_dir, output_file))
            cv2.waitKey(1000)
            break
        if(count==30):  #change this to 1 for image

            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(get_outputs_names(net))

            # Remove the bounding boxes with low confidence
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            for face in faces:
                x = face[0]
                y = face[1]
                w = face[2]
                h = face[3]
                crop_img = frame[y:y+h, x:x+w]
                im = Image.fromarray(crop_img)
                image = im.resize((160,160))
                face_array = asarray(image)
                testembedding = get_embedding(model,face_array)
                dist = linalg.norm(testembedding-originalembedding)
                if(dist>7):
                    result='DIFFERENT'
                else:
                    result='SAME'
                cv2.putText(frame, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                localtime = time.asctime( time.localtime(time.time()) )
                wb = Workbook()
                sheet1 = wb.add_sheet('Reports')
                sheet1.write(1, 0, localtime)
                sheet1.write(1, 1, result)
                wb.save('Reports.xls')

            print('[i] ==> # detected faces: {}'.format(len(faces)))
            print('#' * 60)

            # initialize the set of information we'll displaying on the frame
            info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

            # Save the output video to file
            if args.image:
                cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))
            else:
                video_writer.write(frame.astype(np.uint8))

            cv2.imshow(wind_name, frame)
            count=0

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break

    cap.release()
    cv2.destroyAllWindows()

    #print('==> All done!')
    #print('***********************************************************')


if __name__ == '__main__':
    _main()
