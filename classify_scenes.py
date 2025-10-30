import numpy as np
import torch
import json
from collections import defaultdict
import cv2
from pathlib import Path

def classify_scenes(dataset, all_detections, output_file='labels.json'):
    dets_by_clip = defaultdict(list)
    for d in all_detections:
        dets_by_clip[d['clip_id']].append(d)

    results = []
    model = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)
    model.eval()

    violent_classes = {
        6: "arm wrestling",
        150: "headbutting",
        258: "punching bag",
        259: "punching person (boxing)",
        314: "slapping",
        345: "sword fighting",
        395: "wrestling"
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    for sample in dataset:
        clip_id = Path(sample.filepath).stem
        filepath = sample.filepath
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            continue
        duration = frame_count / fps

        clip_sec = 2.0
        step_sec = 1.0
        predictions = []
        for start_s in np.arange(0, max(duration - clip_sec + step_sec, 0), step_sec):
            start_f = int(start_s * fps)
            end_f = int((start_s + clip_sec) * fps)
            tensor = get_video_tensor(cap, start_f, end_f)
            if tensor is None:
                continue
            input = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(input)
            prob = F.softmax(logit, dim=1)
            top_idx = torch.argmax(prob, dim=1).item()
            top_prob = prob[0, top_idx].item()
            predictions.append({
                'start_s': start_s,
                'end_s': start_s + clip_sec,
                'top_class': top_idx,
                'prob': top_prob
            })

        is_crime = any(p['top_class'] in violent_classes and p['prob'] > 0.5 for p in predictions)
        label = "crime" if is_crime else "normal"

        if is_crime:
            violent_segments = [p for p in predictions if p['top_class'] in violent_classes and p['prob'] > 0.5]
            just = []
            for seg in violent_segments:
                action = violent_classes[seg['top_class']]
                time_str = f"{seg['start_s']:.1f}-{seg['end_s']:.1f} seconds"
                start_frame = int(seg['start_s'] * fps) + 1
                end_frame = int(seg['end_s'] * fps)
                persons = set(d['global_id'] for d in dets_by_clip[clip_id] if start_frame <= d['frame_num'] <= end_frame)
                persons_str = ", ".join(map(str, sorted(persons))) if persons else "none"
                just.append(f"{action} detected at {time_str} involving global person IDs {persons_str}.")
            justification = " ".join(just)
        else:
            justification = "The scene appears normal with no signs of violent or criminal activity visible."

        results.append({
            'clip_id': clip_id,
            'label': label,
            'justification': justification
        })

        cap.release()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)