import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy

import numpy as np
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args

image_depth1 = None
image_rgb2 = None
resize_done = False
left = 218
top = 171

top_left = (top, left)
bottom_right = (min(640, top + 300), min(480, left + 300))

def rgb_callback(ros_data):
	global image_rgb2
	global image_rgb1
	global resize_done

	#print("hi123412341234")
	#np_arr = np.fromstring(ros_data.data, np.uint8)
	#image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
	bridge = CvBridge()
	image_rgb1 = np.frombuffer(ros_data.data, dtype=np.uint8).reshape(ros_data.height, ros_data.width, -1)
	#image_rgb1 = bridge.imgmsg_to_cv2(ros_data,"bgr8")
	#print("after bgr8")
	#cv2.normalize(image_rgb1, image_rgb1, 0, 255, cv2.NORM_MINMAX)
	#print("image_rgb1", image_rgb1.shape)

	image_rgb1 = image_rgb1[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
	#print("image_rgb2", image_rgb1.shape)
	image_rgb1 = cv2.resize(image_rgb1, (300, 300), interpolation = cv2.INTER_AREA) 

	image_rgb2 = image_rgb1.astype(np.float32)/255.0
	image_rgb2 -= image_rgb2.mean()
	#print("image_rgb1", image_rgb1.shape)
	#cv2.imshow('cv_img', image_rgb1)
	##cv2.waitKey(20)
	image_rgb2 = image_rgb2.transpose((2, 0, 1))
	#image_rgb2 = image_rgb1.transpose((2, 1, 0))

	#print("image_rgb2", image_rgb2.shape)


	#dim = (300, 300)
	# resize image
	#image_rgb1 = cv2.resize(image_rgb, dim, interpolation = cv2.INTER_AREA) 
	#resize_done = True
	#print("image_rgb2", image_rgb2.shape)
	#image_rgb3 = image_rgb2.reshape(300,300,3)

	#cv2.imshow('cv_img', image_rgb3)
	#cv2.waitKey(20)

def depth_callback(ros_data):
	global image_depth1

	#print("hi123412341234")
	#np_arr = np.fromstring(ros_data.data, np.uint8)
	#image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
	bridge = CvBridge()
	image_depth = np.array(bridge.imgmsg_to_cv2(ros_data, "32FC1"), dtype=np.float32)
	#cv2.normalize(image_depth, image_depth, 0, 1, cv2.NORM_MINMAX)
	image_depth = image_depth[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
	dim = (300, 300)
	# resize image
	image_depth1 = cv2.resize(image_depth, dim, interpolation = cv2.INTER_AREA) 

	#cv2.imshow('cv_img', image_np)
	#cv2.waitKey(20)

def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))

if __name__ == '__main__':

	args = parse_args()

	rospy.init_node('image_feature', anonymous=True)

	rgb_subscriber = rospy.Subscriber("/usb_cam/image_raw/",
		Image, rgb_callback,  queue_size = 1)
	#depth_subscriber = rospy.Subscriber("/kinect2/qhd/image_depth_rect",
		#Image, depth_callback,  queue_size = 1)
	#print("subscribed to /camera/image/compressed")
	rospy.sleep(0.1)



	net = torch.load(args.network)
	#print("net",net)
	#device = torch.device("cuda:0")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# print("resize_done",resize_done)
	# print("image_depth",image_depth1.shape)
	# print("image_rgb",image_rgb1.shape)
	
	#rospy.sleep(5)
	# image_rgb2 = image_rgb1.reshape(3,300,300)
	# input_img = numpy_to_torch(np.concatenate((np.expand_dims(image_depth1, 0),image_rgb2),0))
	# input_img = input_img.reshape(1,4,300,300)
	# #print("input_img",input_img.shape)
	# #print("input_img",type(input_img))
	# input_img = input_img.float()
	# input_img = input_img.to(device)
	# pos_output, cos_output, sin_output, width_output = net(image_rgb2)
	# print("pos_output",pos_output.shape)
	# print("cos_output",cos_output.shape)
	# print("sin_output",sin_output.shape)
	# #print("width_output",width_output)
	# #print("width_output",max(width_output[:,:]))
	# #print("width_output",min(width_output))


	with torch.no_grad():
	    for idx in range(100000000000):
	    	#image_rgb3 = image_rgb2.reshape(3,300,300)
	    	#input_img = numpy_to_torch(np.concatenate((np.expand_dims(image_depth1, 0),image_rgb2),0))
	    	#print("image_rgb2",image_rgb2)
	    	input_img = numpy_to_torch(image_rgb2)
	    	input_img = input_img.reshape(1,3,300,300)
	    	input_img = input_img.float()
	    	input_img = input_img.to(device)
	    	pos_output, cos_output, sin_output, width_output = net(input_img)
	    	q_img, ang_img, width_img = post_process_output(pos_output,cos_output,sin_output,width_output)


	    	if args.vis:
	    		#print("image_rgb2",image_rgb2.shape)
	    		evaluation.plot_output(image_rgb1, q_img,ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)
	    idx = idx + 1

	rospy.spin()
