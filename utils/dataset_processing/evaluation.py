#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps


#def plot_output(rgb_img, depth_img, grasp_rectangles_gt, grasp_line_gt, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
def plot_output(rgb_img,grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    #grasp_line_gt.plot(ax,'g')
    for g in gs:
        g.plot(ax)
    
   # for g in grasp_rectangles_gt:
       # g.plot(ax,'r')
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)       # tolto per nn visualizzare
    #ax.imshow(depth_img, cmap='gray')
    #for g in gs:
       # g.plot(ax)
    #ax.set_title('Depth')
   # ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False

def calculate_line_intersection_match(grasp_q, grasp_angle, line , no_grasps=1, grasp_width=None):
    """
    param m
    param f(x0)
    param f(x1)
    param n_vertex_up
    param n_vertex_down
    
    """
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        rectangle_verteces = g.as_gr.points 
        #print("verteces",rectangle_verteces)
        n_vertex_up=0
        n_vertex_down=0
        for i in range(0, rectangle_verteces.shape[0]):
            y=rectangle_verteces[i,0]
            x=rectangle_verteces[i,1]
            if line.eval(x) > y:
                #print("line",line.eval(x))
                #print("y",y)
                n_vertex_down+=1
            else:
                n_vertex_up+=1
        #print(n_vertex_up)
        #print(n_vertex_down)
        if n_vertex_up == n_vertex_down:
            return True

    return False

                    

