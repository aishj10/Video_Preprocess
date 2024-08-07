


import os
import json
import cv2


def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    min = (secs - hrs * 3600) // 60 # thanks @LeeDongYeun for finding & fixing this bug
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))

def download(video_path, ytb_id, proxy=None):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, defalut None
    """
    if proxy is not None:
        proxy_cmd = "--proxy {}".format(proxy)
    else:
        proxy_cmd = ""
    if not os.path.exists(video_path):
        down_video = " ".join([
            "yt-dlp",
            proxy_cmd,
            '-f', "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
            '--skip-unavailable-fragments',
            '--merge-output-format', 'mp4',
            "https://www.youtube.com/watch?v=" + ytb_id, "--output",
            video_path, "--external-downloader", "aria2c",
            "--external-downloader-args", '"-x 16 -k 1M"'
        ])
        print(down_video)
        status = os.system(down_video)
        if status != 0:
            print(f"video not found: {ytb_id}")




if __name__ == '__main__':
    
    # Provide path to json files
    inpath = "/data/aujadhav/yt_meta/filtered_v1/"
    
     # Provide path to output dir
    outpath =  "/data/aujadhav/youtube_videos/"
    
    jflist = os.listdir(inpath)
   
    for jf in jflist:
        
        if jf == "speaker_to_youtube_id.json":
            continue
    
        jfile = os.path.join(inpath , jf)

        jout = []
        with open(jfile, "r") as file:
            for line in file:
                jout.append(json.loads(line))

        vid_id = jout[0]['id']       
        vpath = outpath + f"{jf}_{vid_id}"
        os.makedirs(vpath, exist_ok = True)

        processed_path = os.path.join(vpath, "speaker_clips")
        os.makedirs(processed_path, exist_ok = True)



        proxy = None  

        raw_vid_path = os.path.join(vpath, "original_video.mp4")
        download(raw_vid_path, vid_id, proxy)
        

  