## Installation

### Install ByteTrack (follow ByteTrack/README.md)
### Install RTMLIB  (follow rtmlib/README.md)
### Install ISR  (follow ISR_ICCV2023_Oral/README.md)


## Generate Clips Step-By-Step
### Download Youtube Videos

```
python download_and_generate_clips.py

Change inpath for json files and outpath where videos are stored in download_and_generate_clips.py

````

### Step 1:  Run ByteTrack

```
cd ByteTrack

python tools/demo_track_youtube.py-f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result --path {outpath}

python generate_clips_youtube.py

change path to {outpath} in generate_clips_youtube.py

```

### Step 2: Run RTMLIB

```
cd rtmlib

python generate_sk_bbox_upperbody_clip_mean_youtube.py --path  {outpath}


```

### Step 3:  ISR

```
cd ISR_ICCV2023_Oral

python demo_A_youtube_clip.py.py --path  {outpath}


```

## Generate Clips in one step

```
python preprocess.py -f ByteTrack/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --path youtube_video_path --isr_weight_path weights/swin_base_patch4_window7_224.pth
```