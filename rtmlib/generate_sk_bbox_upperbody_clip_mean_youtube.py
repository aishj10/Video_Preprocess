import cv2
import numpy as np

from rtmlib import Wholebody, draw_skeleton, draw_bbox, Body
import os

from rtmlib.visualization.skeleton import *

import argparse

MAX_FRAME = 2000


def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))




def generate_clips(tlwh, end_sec, save_folder, raw_vid_path):
    

    x, y, w, h = tlwh
    start_sec = 0.0
    print(start_sec, end_sec, secs_to_timestr(start_sec), secs_to_timestr(end_sec))
    
    if start_sec < end_sec:
        save_path = os.path.join(save_folder, raw_vid_path.split("/")[-1])
        
        cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={w}:h={h}:x={x}:y={y} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {save_path}"

        os.system(cmd)

        

def get_bbox_upperbody(img,
                  keypoints,
                  scores,
                  kpt_ids,
                  kpt_thr=0.5):
               
        
    num_instance = keypoints.shape[0]
    
    bboxes = []
    for i in range(num_instance):
        vis_kpt = np.array([s >= kpt_thr for s in scores[i]])
        vis_kpt = vis_kpt[kpt_ids]
        
        kpt = keypoints[i]
        kpt = kpt[kpt_ids]
        kpt = kpt[vis_kpt]
        
        
        #print("after", kpt.shape)
        if len(kpt) > 0:
            x_min = np.min(kpt[:, 0])
            y_min = np.min(kpt[:, 1])
            x_max = np.max(kpt[:, 0])
            y_max = np.max(kpt[:, 1])
            if x_min != x_max and y_min!= y_max:
                #print(i,x_min, y_min, x_max, y_max )
                bboxes.append([x_min, y_min, x_max, y_max])
            
    return bboxes


def get_min_area(results):
    min_area = float('inf')
    for x0, y0, x1, y1 in results:
        w = abs(x0-x1)
        h = abs(y0-y1)
        area = w*h
        if area < min_area:
            tlwh = [x0, y0,  w, h]
            min_area = area
            
    return tlwh


def get_max_area(results):
    max_area = -float('inf')
    for x0, y0, x1, y1 in results:
        w = abs(x0-x1)
        h = abs(y0-y1)
        area = w*h
        if area > max_area:
            tlwh = [x0, y0,  w, h]
            max_area = area
            
    return tlwh


def get_mean(results):
    rx0, ry0, rx1, ry1 = 0., 0., 0., 0.
    
    rx0 = results[:,0].mean()
    ry0 = results[:,1].mean()
    rx1 = results[:,2].mean()
    ry1 = results[:,3].mean()
    
    w = abs(rx0-rx1)
    h = abs(ry0-ry1)
    
    tlwh = [rx0, ry0,  w, h]     
    
    return tlwh

def get_max(results):
    rx0, ry0, rx1, ry1 = 0., 0., 0., 0.
    
    rx0 = results[:,0].max()
    ry0 = results[:,1].max()
    rx1 = results[:,2].max()
    ry1 = results[:,3].max()
    
    w = abs(rx0-rx1)
    h = abs(ry0-ry1)
    
    tlwh = [rx0, ry0,  w, h]     
    
    return tlwh


def get_min(results):
    rx0, ry0, rx1, ry1 = 0., 0., 0., 0.
    
    rx0 = results[:,0].min()
    ry0 = results[:,1].min()
    rx1 = results[:,2].min()
    ry1 = results[:,3].min()
    
    w = abs(rx0-rx1)
    h = abs(ry0-ry1)
    
    tlwh = [rx0, ry0,  w, h]     
    
    return tlwh
    
def combine_bbox_new(ybboxes, bboxes):
    bboxes_new = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        ybbox = ybboxes[i]
        y1 = min(bbox[3],ybbox[3])
        bbox_new = [ybbox[0], ybbox[1], ybbox[2], y1]
        bboxes_new.append(bbox_new)
        
    return bboxes_new        
        

def get_sk_bbox(body, video_path, kpt_ids):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_id = 0
    tlwh = []
    
    results = []
    while True:
        if frame_id % 20 == 0:
            print('Processing frame {})'.format(frame_id))
            
        if MAX_FRAME and frame_id >= MAX_FRAME:
            break
            
        ret_val, frame = cap.read()
            
        if ret_val:
            
            frame_time = frame_id/fps            
            keypoints, scores, ybboxes = body(frame, ret_bbox=True)
            
            bboxes = get_bbox_upperbody(frame, keypoints, scores, kpt_ids, kpt_thr=0.5)
            
            #print("bboxes", len(bboxes), len(ybboxes))
            if len(bboxes) > 1 or len(bboxes)==0:
                break
                
            if len(ybboxes)==1 and len(bboxes)==len(ybboxes):
                bboxes_new = combine_bbox_new(ybboxes, bboxes)
            else:
                break
                
            
            x0, y0, x1, y1 = bboxes_new[0]
            results.append([x0, y0, x1, y1])
            
          
        else:
            break
        frame_id += 1
    
    if len(results) > 1:
        results = np.array(results)
       # tlwh = get_min_area(results)
       # tlwh = get_max_area(results)
        tlwh = get_mean(results)
       # tlwh = get_min(results)
        #tlwh = get_max(results)
    
    return tlwh, (frame_id-1)/fps
 
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='demo A')
    #parser.add_argument('--output_path', type=str, default='./image2.jpg', help='the path of image 2')
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument('--path', type=str, default='./image1.jpg', help='the path of image 1')
    args = parser.parse_args()

    device = args.device  # cpu, cuda, mps
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    
    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
    
    skeleton = 'coco133'

    to_remove = [ 'left_hip', 'right_hip','left_knee','right_knee','left_ankle','right_ankle','left_big_toe',
                 'left_small_toe','left_heel', 'right_big_toe', 'right_small_toe', 'right_heel'  ]

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    kpt_ids = []
    for i, kpt_info in keypoint_info.items():
        if kpt_info['name'] not in to_remove:
            kpt_ids.append(kpt_info['id'])
    
    for p in os.listdir(args.path):
        pn = os.path.join(args.path, p)
        video_dir = os.path.join(pn, "track_vis/bytetrack_original")
        
        save_folder = os.path.join(video_dir, "rtm_bbox_mean")
        os.makedirs(save_folder, exist_ok=True)
        
        for vpath in os.listdir(video_dir):
            if vpath.endswith('.mp4'):
                video_path = os.path.join(video_dir, vpath)
                print(video_path)
             
                tlwh, frame_time = get_sk_bbox(wholebody, video_path, kpt_ids)
                if tlwh:
                    generate_clips(tlwh, frame_time, save_folder, video_path)
