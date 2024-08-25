import argparse
import os
import os.path as osp
import time
import cv2
import torch
from collections import defaultdict

import numpy as np
import json
import copy

from loguru import logger

from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.utils.visualize import plot_tracking
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer



from rtmlib.rtmlib import Wholebody, draw_skeleton, draw_bbox, Body
from rtmlib.rtmlib.visualization.skeleton import *



import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms as Transforms

import torch.nn.functional as F

import ipdb
import pickle





def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/", help="path to images or video"
    )
    
#     parser.add_argument(
#         #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
#         "--output_path", default="./videos/palace.mp4", help="path to images or video"
#     )
    
    parser.add_argument("--MAX_FRAME", default=0, type=int, help="maximum number of frames")
    parser.add_argument("--MIN_FRAME", default=240, type=int, help="minimum number of frames")
    parser.add_argument("--FACE_FRACT", default=95, type=int, help="% of face keypoints")
    parser.add_argument("--MIN_AREA", default=100, type=int, help="maximum area of bbox")
    parser.add_argument("--MAX_DIST_AREA_RATIO", default=0.0007, type=int, help="minimum ration between area of bbox and its distance from first bbox ")
   
    parser.add_argument("--THRESHOLD", default=0.7, type=int, help="ISR similarity threshold")
    
    
    parser.add_argument(
        "--mean_bbox",
        default=True,
        action="store_true",
        help="Use this to obtain mean of bounding boxes",
    )
    
    parser.add_argument(
        "--save_clip",
        default=True,
        action="store_true",
        help="Use this if you want to save output clip",
    )
        
    
    parser.add_argument(
        "--save_stats",
        default=True,
        action="store_true",
        help="whether to save the statistics",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    
    parser.add_argument(
        "--isr_weight_path",
        default="/data/aujadhav/ISR_ICCV2023_Oral/weights/swin_base_patch4_window7_224.pth",
        type=str,
        help="Path to ISR model weights",
    )
    
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
   
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser



def convert_to_pil(img):
   
    im1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(im1_rgb)
    
    return im2 


class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = Transforms.Compose([
            Transforms.Resize((self.height, self.width)),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transformer(img).unsqueeze(0)

class SwinTransformer(nn.Module):

    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224')
        self.num_features = num_features
        self.feat = nn.Linear(1024, num_features) if num_features > 0 else None

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x

def clip_frame(clip, frame_id_out):
    dclip = []
    for c in clip:
        if c['frame_id'] <= frame_id_out:
            dclip.append(c)
            
    return dclip
        
def generate_frame_clip(clip, start, end, x, y, w, h, adjust = False, frame_dict= None):
    dclip = []
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    
    for c in clip:
        if start <= c['frame_id']<=end:
            cn= copy.deepcopy(c)
            
            if frame_dict:
                frame = frame_dict[c['frame_id']]['frame']
                cn['time'] = frame_dict[c['frame_id']]['time']
            else:
                frame = cn['frame']
            cropped_frame = frame[y:y+h, x:x+w]
            shape = cropped_frame.shape
            cn['frame'] = cropped_frame
            if adjust:
                cn['x'] = c['x']+x
                cn['y'] = c['y']+y
            
            

            cn['w'] = min(shape[1], w)
            cn['h'] = min(shape[0], h)
            dclip.append(cn)
        
    return dclip        
    
    


def combine_into_video_clips_new(first, last, d, frame_dict, args):
    clips = []
    curr = None
    prev = None
    so_far = []
    for i in range(first, last+2):
    #     print(i,d[i])
    #     print(so_far)
        if i not in d.keys():
            if so_far:
                clips.append(so_far)
                prev = None
                so_far =[]

        else:
            if prev is None:
                prev = d[i]
                so_far = prev
                continue

            else:
                curr = d[i]
                prev_obj = [p['obj_id'] for p in prev]
                curr_obj = [p['obj_id'] for p in curr]
                if set(prev_obj) == set(curr_obj):
                    so_far.extend(curr)

                else:
                    clips.append(so_far)
                    prev = curr
                    so_far = prev
                    
                    
    clip_out = []

    for clip in clips:

        start = clip[0]['frame_id']
        end = clip[-1]['frame_id']
        if end-start < args.MIN_FRAME:
            continue
        obj_ids = set([c['obj_id'] for c in clip])
        
        darea = {oid:float('inf') for oid in obj_ids}
        dxywh = {oid:[] for oid in obj_ids}
        for c in clip:
            area = c['w'] * c['h']
            if area < darea[c['obj_id']]:
                darea[c['obj_id']] = area
                dxywh[c['obj_id']] = [ c['x'],  c['y'], c['w'],  c['h']]

        for oid in obj_ids:
            dclip = {"start_frame": start, "end_frame": end, "x": dxywh[oid][0], "y": dxywh[oid][1], "w": dxywh[oid][2], "h": dxywh[oid][3]}
            
            clip_out.append(dclip)
            
    dclip = generate_bytetrack_clip(clip_out, frame_dict)
    return dclip

            
def generate_bytetrack_clip(clip_out, frame_dict):
    
    dclip = []
    k = list(frame_dict.keys())[0]
    shape = frame_dict[k]['frame'].shape
    width = shape[1]
    height = shape[0]
    for i, c in enumerate(clip_out):
        clip = []
        start = c['start_frame']
        end = c['end_frame']
        x = int(max(c['x'], 0.0))
        y = int(max(c['y'], 0.0))
        w = int(min(width, c['w']))
        h = int(min(height, c['h']))
    
        
        for fid in range(start, end+1):
            frame = frame_dict[fid]['frame']
            cropped_frame = copy.deepcopy(frame[y:y+h, x:x+w])
            cn = {'frame': cropped_frame,
                  'frame_id': fid,
                  'time' : frame_dict[fid]['time'],
                  'x' : x,
                  'y' : y,
                  'w': w,
                  'h':h}
       
            clip.append(cn)
        
        dclip.append(clip)
        
    return dclip        

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def get_bbox_upperbody(img,
                  keypoints,
                  scores,
                  kpt_ids,
                  kpt_thr=0.5):
               
        
    num_instance = keypoints.shape[0]
    filtered_keypoints = [[] for _ in range(num_instance)]
    bboxes = []
    for i in range(num_instance):
        fkpt = []
        vis_kpt = np.array([s >= kpt_thr for s in scores[i]])
        vis_kpt = vis_kpt[kpt_ids]
        
        kpt = keypoints[i]
        kpt = kpt[kpt_ids]
        kpt = kpt[vis_kpt]
        
        fkpt = np.array(kpt_ids)[vis_kpt]
        
        
        #print("after", kpt.shape)
        if len(kpt) > 0:
            x_min = np.min(kpt[:, 0])
            y_min = np.min(kpt[:, 1])
            x_max = np.max(kpt[:, 0])
            y_max = np.max(kpt[:, 1])
            if x_min != x_max and y_min!= y_max:
                #print(i,x_min, y_min, x_max, y_max )
                bboxes.append([x_min, y_min, x_max, y_max])
                filtered_keypoints[i] = fkpt
            
    return bboxes, filtered_keypoints


def single_bbox(kpt):
    x_min = np.min(kpt[:, 0])
    y_min = np.min(kpt[:, 1])
    x_max = np.max(kpt[:, 0])
    y_max = np.max(kpt[:, 1])
    return [x_min, y_min, x_max, y_max]
    
def get_bbox_all(img,
                  keypoints,
                  scores,
                  face_kpt_ids,
                  kpt_thr=0.5,
                  face_thr = 0.7):
               

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]
               
    
    num_instance = keypoints.shape[0]
    
    bboxes = []
    bboxes_faces = []
    filtered_keypoints = [[] for _ in range(num_instance)]
    for i in range(num_instance):
      
        body_box = []
        face_box = []
        vis_kpt = np.array([s >= kpt_thr for s in scores[i]])
        vis_kpt_face = np.array([s >= face_thr for s in scores[i]])
        
        kpt = keypoints[i]
        
        if len(kpt) > 0:                    
            kpt_face = np.array(kpt[face_kpt_ids])
            vis_kpt_face = vis_kpt_face[face_kpt_ids]
            fkpt = np.array(face_kpt_ids)[vis_kpt_face]
            if len(kpt_face) > 0:
                kpt_face = kpt_face[vis_kpt_face]
                
        else:
            continue

        if len(kpt) > 0:
            kpt = np.array(kpt[vis_kpt])
            #vis_kpt = vis_kpt[kpt_ids]
        
        else:
            continue
       
        
        
        if len(kpt) > 0 and len(kpt_face) > 0:
            x_min, y_min, x_max, y_max = single_bbox(kpt)
            if x_min != x_max and y_min!= y_max:
                body_box = [x_min, y_min, x_max, y_max]
                

            x_min, y_min, x_max, y_max = single_bbox(kpt_face)
            if x_min != x_max and y_min!= y_max:
                face_box = [x_min, y_min, x_max, y_max]
                
                
        if body_box and face_box:
            bboxes.append(body_box)
            bboxes_faces.append(face_box)
            filtered_keypoints[i] = fkpt
                            
            
    return bboxes, bboxes_faces, filtered_keypoints


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


    
def combine_bbox_new_max(ybboxes, bboxes):
    bboxes_new = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        ybbox = ybboxes[i]
        x0 = min(bbox[0],ybbox[0])
        y0 = min(bbox[1],ybbox[1])
        x1 = max(bbox[2],ybbox[2])
        y1 = min(bbox[3],ybbox[3])
        #bbox_new = [max(x0, 0.0), max(y0, 0.0), x1, max(y1, 0.0)]
        bbox_new = [x0, y0,x1, y1]
        bboxes_new.append(bbox_new)
        
    return bboxes_new        


def combine_bbox_new(ybboxes, bboxes):
    bboxes_new = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        ybbox = ybboxes[i]
        y1 = min(bbox[3],ybbox[3])
        bbox_new = [ybbox[0], ybbox[1], ybbox[2], y1]
        bboxes_new.append(bbox_new)
        
    return bboxes_new        

        
def is_face(keypoints, face_kpt_ids, FACE_FRACT):
    
    match = set(face_kpt_ids) & set(keypoints)
    th = len(match) *100 /len(face_kpt_ids)
    ret = (th >=  FACE_FRACT)
    return ret, th
    

def is_area(x0, y0, x1, y1, MIN_AREA):
    area = abs(x1-x0) * abs(y1-y0)
    
    return area > MIN_AREA, area
    
def is_distance(area, distance, MAX_DIST_AREA_RATIO):
    ratio = distance/float(area)    
    return ratio < MAX_DIST_AREA_RATIO, ratio    
    
def save_clip_data(dclip_out, save_path=None):
    result = []
    for clip in dclip_out:
        start_time = clip[0]['time']
        end_time = clip[-1]['time']
        x = clip[0]['x']
        y = clip[0]['y']
        w = clip[0]['w']
        h = clip[0]['h']
        res = {"start_time":start_time, "end_time": end_time, "x":x, "y": y, "w":w, "h": h}
        result.append(res)
    
    if save_path:
        with open(save_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

    return result


def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))

   

def save_single_clip(clip, clip_file, raw_vid_path):
    start_sec = clip[0]['time']
    end_sec = clip[-1]['time']
    x = clip[0]['x']
    y = clip[0]['y']
    w = clip[0]['w']
    h = clip[0]['h']
    #print(start_sec, end_sec, secs_to_timestr(start_sec), secs_to_timestr(end_sec))

    if start_sec < end_sec:
        
        cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={w}:h={h}:x={x}:y={y} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {clip_file}"

        os.system(cmd)
    

def save_clip_video(result, save_folder, raw_vid_path):
    

    for i, res in enumerate(result):
        
        start_sec = res['start_time']
        end_sec = res['end_time']
        #print(start_sec, end_sec, secs_to_timestr(start_sec), secs_to_timestr(end_sec))

        if start_sec < end_sec:
            save_path = os.path.join(save_folder, f"{i:05d}.mp4")

            cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={(res['w'])}:h={(res['h'])}:x={(res['x'])}:y={(res['y'])} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {save_path}"

            os.system(cmd)
    
def extended_face(bboxes):
    bboxes_new = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        h = abs(y1-y0)
        y0 -= h
        bboxes_new.append([x0, y0, x1, y1])
        
    return bboxes_new
        
    
def combine_bbox_face(bboxes, bboxes_face):
    bboxes_new = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        fbbox = bboxes_face[i]
        y0 = min(bbox[1],fbbox[1])
        bbox_new = [bbox[0], max(y0, 0.0), bbox[2], bbox[3]]
        bboxes_new.append(bbox_new)
        
    return bboxes_new
        
def get_sk_bbox_face(body, clip, face_kpt_ids, args):
    tlwh = []
    fx0, fy0 = None, None
    results = []
    dclip = []
    start, end = None, None
    results_stat = []
    
    for c in clip:
        if c['frame_id'] % 20 == 0:
            print('Processing frame {})'.format(c['frame_id']))
            
        frame = c['frame']
            
        frame_time = c['time']
                       
        
        keypoints, scores, ybboxes = body(frame, ret_bbox=True)

        
        bboxes, bboxes_face, filtered_keypoints = get_bbox_all(frame, keypoints, scores, face_kpt_ids, kpt_thr=0.5)
        bboxes_face_extend = extended_face(bboxes_face)

        bboxes_new = combine_bbox_face(bboxes, bboxes_face_extend)

     

        if len(ybboxes)==1 and len(bboxes_new)==len(ybboxes):
            bboxes_new = combine_bbox_new_max(ybboxes, bboxes_new)
          
            
        else:
            break

    
        
        x0, y0, x1, y1 = bboxes_new[0]
        isface, th = is_face(filtered_keypoints[0], face_kpt_ids, args.FACE_FRACT)
        isarea, area = is_area(x0, y0, x1, y1, args.MIN_AREA)

       
        
        if fx0 is None:
            fx0, fy0 = bboxes[0][0], bboxes[0][1]

        dx0, dy0 = bboxes[0][0], bboxes[0][1]    
        distance = ((fx0-dx0)**2 + (fy0-dy0)**2)**0.5
        isdist, ratio = is_distance(area, distance, args.MAX_DIST_AREA_RATIO)
        
        if args.save_stats:
            results_stat.append( f"{c['frame_id']},{ratio},{th}, {area}, {len(bboxes)} \n")                         
                        
        if isface and isarea and isdist:
            if start is None:
                start = c['frame_id']
                
            results.append([x0, y0, x1, y1])
            end = c['frame_id']
            
            
        else:
            results = []
            start, end = None, None
            break

    #ipdb.set_trace()
    if len(results) > 1:
        results = np.array(results)
        if args.mean_bbox:
            tlwh = get_mean(results)
            
        else:
            tlwh = get_min_area(results)

    else:
        tlwh = None
        
    return tlwh, start, end, results_stat

 
       
    

def get_similarity(model, clip, data_processor, args):

    frame_id = None
    frame_id_out = 0
    results = []
    results_list = []
    to_clip = False
    frame_time = 0.0
    for c in clip:
        if c['frame_id'] % 200 == 0:
            logger.info('Processing frame {})'.format(c['frame_id']))
            
        frame = c['frame']                
            
        if frame_id == None:
            image1 = convert_to_pil(frame)
            image1 = data_processor(image1).to(args.device)
            frame_id = c['frame_id'] 

        else:
            image1 = image2


        image2 = convert_to_pil(frame)
        #frame_time = frame_id/fps


        image2 = data_processor(image2).to(args.device)
        

        with torch.no_grad():
            A_feat = F.normalize(model(image1), dim=1).cpu()
            B_feat = F.normalize(model(image2), dim=1).cpu()
        simlarity = A_feat.matmul(B_feat.transpose(1, 0))
        simlarity = simlarity[0, 0]

        if simlarity < args.THRESHOLD:
            to_clip = True     
            break
            
        frame_id_out = c['frame_id'] 
        
    return frame_id_out, to_clip


def imageflow_demo(predictor, current_time, inpath,  args):
    cap = cv2.VideoCapture(inpath)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    frame_ids = []
    result_dict = defaultdict(list)
    frame_dict ={}
    first_frame = None
    last_frame = None
    while True:
        if frame_id % 200 == 0:
            logger.info('Processing frame {})'.format(frame_id))
            
        if args.MAX_FRAME and frame_id >= args.MAX_FRAME:
            break
            
        ret_val, frame = cap.read()
        
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            frame_time = frame_id/fps
            
            frame_dict[frame_id] = {"frame_id": frame_id, "time": frame_time, "frame": img_info["raw_img"]}
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                #print("online_targets",online_targets)
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        fd = {"frame_id": frame_id, "obj_id": tid, "x": tlwh[0], "y":tlwh[1], "w": tlwh[2], "h":tlwh[3]}
                             
                        
                        result_dict[frame_id].append(fd)
                       
                        
                                                
                        if first_frame is None:
                            first_frame = frame_id
                        last_frame = frame_id
                        
                timer.toc()        
            else:
                timer.toc()
                

                        
        else:
            break
        frame_id += 1
    
    #ipdb.set_trace()
    return result_dict, first_frame, last_frame, frame_dict
    #return result_out

def run_bytetrack():
    clip_out = None
    result_dict, first, last, obj_dict = imageflow_demo(predictor, current_time, inpath, args)

    if first and last:
        clip_out = combine_into_video_clips(first, last, result_dict, obj_dict) 
        
    return clip_out


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name


    logger.info("Args: {}".format(args))

    
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    
    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    
    trt_file = None
    decoder = None
        
    if args.MAX_FRAME == 0:
        args.MAX_FRAME = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
                        
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    
    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=args.device)
                        
                        
                        
    data_processor = Data_Processor(height=224, width=224)
    isr_model = SwinTransformer(num_features=512).to(args.device)
    isr_model.eval()
    
    
    
    weight = torch.load(args.isr_weight_path)
    isr_model.load_state_dict(weight['state_dict'], strict=True)                    
    
    skeleton = 'coco133'

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    
    face_kpt_ids = []
    for i, kpt_info in keypoint_info.items():
        if 'face' in kpt_info['name'] or 'eye' in kpt_info['name'] or 'nose' in kpt_info['name']:
            face_kpt_ids.append(kpt_info['id'])                    
                        
    
    start_time = time.time()
    for ipath in os.listdir(args.path):
        
        inpath = os.path.join(args.path, ipath, "original_video.mp4")
        print("inpath", inpath)  
            
        save_folder = os.path.join(args.path, ipath, "results")
        if os.path.exists(save_folder):
            continue
        os.makedirs(save_folder, exist_ok=True)
        
        logger.info("Start Tracking with ByteTrack")
        result_dict, first, last, frame_dict = imageflow_demo(predictor, current_time, inpath, args)
       
        dclip_out = [] 
        dclip_rtm = []
        results_stat_all = []
        
        if first is not None and last is not None:
            
            clip_out = combine_into_video_clips_new(first, last, result_dict, frame_dict, args) 
            logger.info("End Tracking with ByteTrack")   

            
            for i, clip in enumerate(clip_out):
              
                tlwh, start, end, results_stat = get_sk_bbox_face(wholebody, clip, face_kpt_ids, args)
                
                if len(results_stat) >0 and args.save_stats:
                    res_file = os.path.join(save_folder, f"{i:05d}.txt")
                    print(f"===>Saving stats at {res_file}")
                    with open(res_file, 'w') as f:
                        f.writelines(results_stat)
   
                if tlwh is not None:
                    
                    dclip = generate_frame_clip(clip, start, end, tlwh[0], tlwh[1], tlwh[2], tlwh[3], adjust=True)
                    
                    if len(dclip) > 0:
                        frame_id_out, to_clip = get_similarity(isr_model, dclip, data_processor, args)   
                        if to_clip :
                            dclip = clip_frame(dclip, frame_id_out) 
                            
                        if len(dclip) > args.MIN_FRAME:             
                            dclip_out.append(dclip) 
                            
                            if args.save_clip:                                 
                                clip_file = os.path.join(save_folder, f"{i:05d}.mp4")
                                print(f"===>Saving clip at {clip_file}")
                                save_single_clip(dclip, clip_file, inpath)
                        
                        
                            
            save_path = os.path.join(save_folder, "results.json")
            result = save_clip_data(dclip_out, save_path)
                

            print("===> Time", time.time()-start_time)           

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
