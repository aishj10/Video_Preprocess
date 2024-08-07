import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
import argparse
from PIL import Image
import torch.nn.functional as F
import os

from loguru import logger
import cv2
import ipdb

import numpy as np
import argparse

MAX_FRAME= 2000
THRESHOLD = 0.7


def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))


def plot_tracking(image, similarity, frame_id=0):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d similarity: %.2f' % (frame_id, similarity),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    return im

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


def convert_to_pil(img):
   
    im1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(im1_rgb)
    
    return im2


def generate_clip(save_path, end_sec, raw_vid_path, to_clip):
    
    
    start_sec = 0.0
   
    if to_clip :
        if (start_sec < end_sec):
            cmd = f"ffmpeg -i {raw_vid_path} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {save_path}"
    else:
        cmd = f"cp {raw_vid_path} {save_path}"

    os.system(cmd)

def get_similarity(model, video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    frame_id = 0
    results = []
    results_list = []
    to_clip = False
    frame_time = 0.0
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {})'.format(frame_id))
            
        if MAX_FRAME and frame_id >= MAX_FRAME:
            break
            
        ret_val, frame = cap.read()
        
            
        
        if ret_val:
            
            if frame_id == 0:
                image1 = convert_to_pil(frame)
                image1 = data_processor(image1).to(args.device)

            else:
                image1 = image2
            
            
            image2 = convert_to_pil(frame)
            frame_time = frame_id/fps
            
            
            image2 = data_processor(image2).to(args.device)
            #ipdb.set_trace()
            model = model.to(args.device)

            with torch.no_grad():
                A_feat = F.normalize(model(image1), dim=1).cpu()
                B_feat = F.normalize(model(image2), dim=1).cpu()
            simlarity = A_feat.matmul(B_feat.transpose(1, 0))
            simlarity = simlarity[0, 0]
            
            if simlarity < THRESHOLD:
                to_clip = True
                break
                

        else:
            break
        frame_id += 1
        
    return frame_time, to_clip


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo A')
#     parser.add_argument('--model-weight', type=str, default='./swin_base_patch4_window7_224.pth',
#                         help='the path of model weight')
    #parser.add_argument('--image1', type=str, default='./image1.jpg', help='the path of image 1')
    #parser.add_argument('--output_path', type=str, default='./image2.jpg', help='the path of image 2')
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument('--path', type=str, default='./image1.jpg', help='the path of image 1')
    args = parser.parse_args()

    data_processor = Data_Processor(height=224, width=224)
    model = SwinTransformer(num_features=512).cuda()
    model.eval()
    
    
    weight_path = "weights/swin_base_patch4_window7_224.pth"
    weight = torch.load(weight_path)
    model.load_state_dict(weight['state_dict'], strict=True)
    
    for p in os.listdir(args.path):
        pn = os.path.join(args.path, p)
        video_dir = os.path.join(pn, "track_vis/bytetrack_original/rtm_bbox_mean")
        
        save_folder = os.path.join(pn, "track_vis/bytetrack_original/", "isr_mean")
        os.makedirs(save_folder, exist_ok=True)
        
        for vpath in os.listdir(video_dir):
            if vpath.endswith('.mp4'):
                video_path = os.path.join(video_dir, vpath)
                save_path = os.path.join(save_folder, video_path.split("/")[-1])
                
                print(video_path)
                
                if  not os.path.exists(save_path):
                    frame_time, to_clip = get_similarity(model, video_path)

                    generate_clip(save_path, frame_time, video_path, to_clip)
                    
                
            
                
