from pathlib import Path
import os
import argparse
import cv2
import numpy as np
import glob
import unisal

def heat_map(img):

    gaze_map_path = img.split('images')[0]+'/saliency/'+ \
                    img.split('images\\')[-1]
    image = cv2.imread(img)
    gaze_map = cv2.imread(gaze_map_path, 0)
    heatmap = cv2.applyColorMap(gaze_map, cv2.COLORMAP_JET)
    gaze_heatmap =cv2.addWeighted(image,0.5,heatmap,0.5,0)
    
    return gaze_heatmap


def inference_video(video_path):

    freq_dict = {'DADA2000':30, 'BDDA':30, 'DT16':24, 'DReye':25}
    fourcc= cv2.VideoWriter_fourcc(*'XVID')
    source = video_path.split(os.sep)[-2]
    Freq = freq_dict[source]

    video_name = video_path+'/heatmap.avi'
    if os.path.exists(video_name):
        print('pass: ', video_name)
    else:
        images = sorted(glob.glob(video_path+'/images/*.png'))
        images.extend(sorted(glob.glob(video_path+'/images/*.jpg')))
        if len(images)>0:
            img_size = cv2.imread(images[0]).shape[:2]
            out = cv2.VideoWriter (video_name, \
                fourcc, Freq, (img_size[1],img_size[0]))
            for img in images:
                gaze_heatmap = heat_map(img)
                out.write(gaze_heatmap)
            
            out.release()
            print('write the video:', video_name)
        else:
            pass

def load_trainer(train_id=None):
    """Instantiate Trainer class from saved kwargs."""
    if train_id is None:
        print('pretrained weights file is not found')
    else:
        print(f"Weight file path: {train_id}")
    train_dir = Path(__file__).resolve().parent
    train_dir = train_dir / train_id
    return unisal.train.Trainer.init_from_cfg_dir(train_dir)


def predictions_from_folder(
        folder_path, is_video, source=None, train_id=None, model_domain=None):
    """Generate predictions of files in a folder with a trained model."""
    trainer = load_trainer(train_id)
    trainer.generate_predictions_from_path(
        folder_path, is_video, source=source, model_domain=model_domain, train_id=train_id)


def predict_examples(args):
    train_id = args.pretrained_weight
    path = args.inference_path
    for example_folder in (Path(__file__).resolve().parent / path).glob("*"):
        if not example_folder.is_dir():
            continue

        source = example_folder.name
        is_video = True
        print("\nGenerating predictions for %s"%str(source))

        if not example_folder.is_dir():
            continue
        for video_folder in example_folder.glob('*'):
            predictions_from_folder(
                video_folder, is_video, train_id=train_id, source=source)
            
            if args.write_video:
                inference_video(str(video_folder))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--pretrained_weight', type=str, default='./weights')
    parser.add_argument('--inference_path', type=str, default='./inference')
    parser.add_argument('--write_video', action='store_true', default=False)
    args = parser.parse_args()

    predict_examples(args)
