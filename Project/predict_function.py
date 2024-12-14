import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO
from PIL import Image
from sahi.predict import get_sliced_prediction
import glob

def predict_with_yolo_combine_model(model, image, conf_thres):
    if len(image.shape) == 2 or image.shape[2] == 1:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image, conf=conf_thres)

    bboxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy() 
    labels = results[0].boxes.cls.cpu().numpy().astype(int) 

    return bboxes, scores, labels

def combine_predictions_with_wbf(models_predictions, image_shape, weights, iou_thr=0.5, skip_box_thr=0.5):
    all_bboxes = []
    all_scores = []
    all_labels = []

    for bboxes, scores, labels in models_predictions:
        h, w = image_shape[:2]
        bboxes_norm = [[x[0] / w, x[1] / h, x[2] / w, x[3] / h] for x in bboxes]
        all_bboxes.append(bboxes_norm)
        all_scores.append(scores)
        all_labels.append(labels)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_bboxes, all_scores, all_labels, weights, iou_thr, skip_box_thr
    )

    fused_boxes = [[x[0] * w, x[1] * h, x[2] * w, x[3] * h] for x in fused_boxes]
    return fused_boxes, fused_scores, fused_labels

def draw_boxes_on_image(image, boxes, scores, labels, categories, score_threshold=0.5):
    annotated_image = image.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{categories[int(label)]}:{score:.2f}"
        color = (255, 0, 0) if label == 0 else (0, 255, 0)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_image

categories = ["no helmet", "helmet"]

def predict_with_yolo(model, image, conf_thres):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2 or image.shape[2] == 1:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: 
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model(image, conf=conf_thres)
    annotated_image = image.copy()

    bboxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy() 
    labels = results[0].boxes.cls.cpu().numpy().astype(int) 

    h, w = image.shape[:2]
    bboxes_norm = [[x[0] / w, x[1] / h, x[2] / w, x[3] / h] for x in bboxes]

    iou_thr = 0.55
    skip_box_thr = 0.5
    weights = [1.0]  
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [bboxes_norm], [scores], [labels], weights, iou_thr, skip_box_thr
    )

    fused_boxes = [[x[0] * w, x[1] * h, x[2] * w, x[3] * h] for x in fused_boxes]

    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{categories[int(label)]}:{score:.2f}"
    
        if label == 0:
            color = (255, 0, 0)  
            text_color = (0, 0, 255)  
        else:
            color = (0, 255, 0) 
            text_color = (255, 255, 0)  
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated_image, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return annotated_image

    
model_paths = glob.glob(r"**\*.pt")

def predict_with_sahi(image_np, detection_model, categories, slide_window_size= 512):
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_np

    result = get_sliced_prediction(
        image=image_rgb,
        detection_model=detection_model,
        slice_height=slide_window_size,
        slice_width=slide_window_size,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    annotated_image = image_np.copy()
    for obj in result.object_prediction_list:
        x_min, y_min, x_max, y_max = map(int, obj.bbox.to_voc_bbox())
        score = obj.score
        class_id = obj.category.id
        label_text = f"{categories[class_id]}"
        if class_id == 0:
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=2)
            cv2.putText(annotated_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
            cv2.putText(annotated_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return annotated_image