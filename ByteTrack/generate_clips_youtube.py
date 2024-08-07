import json
from collections import defaultdict

import os
import cv2




def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))



def read_txt(txtfile):
    with open(txtfile) as f:
        lines = f.readlines()
        
    nlines = [l.strip().split(',') for l in lines]
    
    if len(nlines)==0:
        return None, None, None
    
    d  = defaultdict(list)
    
    first = int(nlines[0][0])
    last = int(nlines[-1][0])

    for l in nlines:
        frame_id = int(l[0])


        obj_id = int(l[1])

        x1 = float(l[2])
        y1 = float(l[3])
        w = float(l[4])
        h =  float(l[5])

        time = float(l[7])

        fd = {"obj_id": obj_id, "x":x1, "y":y1, "w": w, "h":h, "time": time}
        d[frame_id].append(fd)
        
    return d, first, last


def combine_into_video_clips(d, first, last):
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

        start = clip[0]['time']
        end = clip[-1]['time']
        obj_ids = set([c['obj_id'] for c in clip])
        darea = {oid:float('inf') for oid in obj_ids}
        dxywh = {oid:[] for oid in obj_ids}
        for c in clip:
            area = c['w'] * c['h']
            if area < darea[c['obj_id']]:
                darea[c['obj_id']] = area
                dxywh[c['obj_id']] = [ c['x'],  c['y'], c['w'],  c['h']]

        for oid in obj_ids:
            dclip = {"ytb_id": "M2Ohb0FAaJU",  "duration": {"start_sec": start, "end_sec": end},        
            "bbox": {"x": dxywh[oid][0], "y": dxywh[oid][1], "w": dxywh[oid][2], "h": dxywh[oid][3]}}

            clip_out.append(dclip)
            
            
    return clip_out



def generate_clips(clip_out, out_path, raw_vid_path):
    
    cap = cv2.VideoCapture(raw_vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #print(width,height)

    for i, clip in enumerate(clip_out):
        start_sec = clip['duration']['start_sec']
        end_sec = clip['duration']['end_sec']
        x = clip['bbox']['x']
        y = clip['bbox']['y']
        w = min(width, clip['bbox']['w'])
        h = min(height, clip['bbox']['h'])

        #print(start_sec, end_sec, secs_to_timestr(start_sec), secs_to_timestr(end_sec))
        out_file = os.path.join(out_path, f"{i}.mp4")
        if (start_sec < end_sec) and (not os.path.exists(out_file)):
            cmd = f"ffmpeg -i {raw_vid_path} -vf crop=w={w}:h={h}:x={x}:y={y} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_file}"

            os.system(cmd)



if __name__ == '__main__':
    
    path = "/data/aujadhav/youtube_jiarui/youtube_videos/"
    
    for ipath in os.listdir(path):
        video_path = os.path.join(path, ipath)
        data_path = os.path.join(video_path, "track_vis")
        

        txtfile = os.path.join(data_path, "original_video.txt") 
        raw_vid_path = os.path.join(video_path, "original_video.mp4")
        out_path = os.path.join(data_path, "bytetrack_original")


    #     data_path = "/data/aujadhav/youtube_jiarui/youtube_meta/speaker_3.json.0025_zCFUDrvuHgQ/speaker_clips/track_vis/"
    #     video_path = "/data/aujadhav/youtube_jiarui/youtube_meta/speaker_3.json.0025_zCFUDrvuHgQ/speaker_clips/"

    #     txtfile = os.path.join(data_path, "0.txt") 
    #     raw_vid_path = os.path.join(video_path, "0.mp4")
    #     out_path = os.path.join(data_path, "bytetrack_0")

        os.makedirs(out_path, exist_ok=True)
        
        if os.path.exists(txtfile):
            print("Processing ",raw_vid_path)

            d, first, last = read_txt(txtfile)
            
            if d:
                clip_out = combine_into_video_clips(d, first, last)
                
                if clip_out:
                    generate_clips(clip_out, out_path, raw_vid_path)
    
    
    