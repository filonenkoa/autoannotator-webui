import os
import subprocess
from typing import List
import streamlit as st
import time
import numpy as np
from pathlib import Path
import time
import datetime as dt
import cv2

from autoannotator.types.base import Detection
from autoannotator.utils.image_reader import ImageReader
from autoannotator.detection.human import RTDETR, IterDETR, UniHCPHuman, InternImageHuman
from autoannotator.detection.human.models.uhihcp import UniHCPHumanDetectionConfig
from autoannotator.detection.human.models.rtdetr import RTDETRDetectionConfig
from autoannotator.detection.human import HumanDetEnsemble

st.set_page_config(page_title="Human Detection", page_icon="😸", initial_sidebar_state="collapsed")

st.markdown("# Human Detection Task")
st.sidebar.header("Human Detection")

# st.session_state.data_dir = ""

st.write(
    """### Data to process"""
)
st.session_state.data_dir = st.text_input("Enter a local directory with images to process")

model_block = st.container()

with model_block:
    st.session_state.models_rtdetr_resnet101 = st.checkbox("RT-DETR ResNet-101", value=True if "models_rtdetr_resnet101" not in st.session_state.keys() else st.session_state.models_rtdetr_resnet101)
    st.session_state.models_progressive_detr = st.checkbox("Progressive DETR", value=True if "models_progressive_detr" not in st.session_state.keys() else st.session_state.models_progressive_detr)
    st.session_state.models_intern_image = st.checkbox("InternImage", value=True if "models_intern_image" not in st.session_state.keys() else st.session_state.models_intern_image)
    st.session_state.models_unihcp = st.checkbox("UniHCP", value=False if "models_unihcp" not in st.session_state.keys() else st.session_state.models_unihcp,
                                                           help="Please request the weights according to the https://github.com/OpenGVLab/UniHCP#pretrained-models")
    if st.session_state.models_unihcp:
        st.session_state.models_unihcp_path = st.text_input("Set the path to the UniHCP ONNX model file", value= "" if "models_unihcp_path" not in st.session_state.keys() else st.session_state.models_unihcp_path)
        if not Path(st.session_state.models_unihcp_path).is_file():
            st.error(f"Could not find the file {st.session_state.models_unihcp_path}")
            
st.session_state.match_iou = st.slider("IoU threshold", min_value=0.01, max_value=0.99, value=st.session_state.match_iou if "match_iou" in st.session_state.keys() else 0.65)

num_selected_models = sum([st.session_state.models_rtdetr_resnet101,
                           st.session_state.models_progressive_detr,
                           st.session_state.models_intern_image,
                           st.session_state.models_unihcp])
if "min_voters" in st.session_state:
    st.session_state.min_votes = min(num_selected_models, st.session_state.min_votes)
else:
    st.session_state.min_votes = 1
# st.write(f"{[i+1 for i in range(num_selected_models)]}")

# st.session_state.min_voters = st.select_slider("Minimun votes", options=[i+1 for i in range(num_selected_models)], value=st.session_state.min_votes)
if num_selected_models > 1:
    st.session_state.min_voters = st.slider("Minimun votes", min_value=1, max_value=num_selected_models, value=st.session_state.min_votes)
else:
    st.session_state.min_votes = 1



# @st.cache
def load_rtdetr_resnet101(container):
    container.info("Loading RT-DETR ResNet-101")
    try:
        rtdetr = RTDETR(RTDETRDetectionConfig(weights=".streamlit/models/rtdetr_r101.onnx"))
    except Exception as ex:
        container.exception(f"Failed to load RT-DETR ResNet-101: {ex}")
        raise ex
    return rtdetr

# @st.cache
def load_progressive_detr(container):
    container.info("Loading Progressive DETR")
    return IterDETR()
    
# @st.cache
def load_intern_image(container):
    container.info("Loading InternImage")
    return InternImageHuman()

# @st.cache
def loadunihcp(container):
    container.info("Loading UniHCP")
    return UniHCPHuman(UniHCPHumanDetectionConfig(weights=st.session_state.models_unihcp_path))
    
    
    
def filter_predictions(results, meta_list, min_score=0.6, min_votes=3, agreement_thr=0.9):
    out_results = []
    kept_preds = 0
    for detection, meta in zip(results, meta_list):

        if detection.score < min_score:
            continue

        if len(meta['unique_models']) < min_votes:
            continue

        out_results.append(detection)
        kept_preds += 1

    agreement_score = kept_preds / len(results)

    should_validate = (agreement_score < agreement_thr)

    return out_results, should_validate


def calc_process_time(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)

    
def auto_annotate_humans(container):
    images_list = list(Path(st.session_state.data_dir).rglob("*"))
    num_images = len(images_list)
    container.info(f"Processing {num_images} files")

    if num_images < 1:
        container.error(f"No images found in {st.session_state.data_dir}")
        return
    # st.write(f"Processing {num_images} files")
    
    # ini image file reader
    reader = ImageReader()
    
    models = []
    model_weights = []
    
    if st.session_state.models_rtdetr_resnet101:
        rt_detr = load_rtdetr_resnet101(container)
        models.append(rt_detr)
        model_weights.append(0.87)
    if st.session_state.models_progressive_detr:
        progressive_detr = load_progressive_detr(container)
        models.append(progressive_detr)
        model_weights.append(0.941)
    if st.session_state.models_intern_image:
        intern_image = load_intern_image(container)
        models.append(intern_image)
        model_weights.append(0.972)
    if st.session_state.models_unihcp:
        unihcp = loadunihcp(container)
        models.append(unihcp)
        model_weights.append(0.925)

    results_list = []
    
    
    hd_ensemble = HumanDetEnsemble(
        models=models,
        model_weights=model_weights,
        match_iou_thr=st.session_state.match_iou,
    )
    
    
    progress_bar = container.progress(0, text="Processing")
    
    t_start = time.time()
    for ind, img_path in enumerate(images_list):
        img_meta = {'img_path': img_path.as_posix()}
        # read image
        img = reader(img_meta['img_path'])

        # detect faces with ensemble
        results, meta, all_preds = hd_ensemble(img)

        # filter predictions: each prediction should have aggregated score > 0.5 and predicted by at least 2 models
        results, keep_image = filter_predictions(results, meta, min_votes=st.session_state.min_voters, min_score=0.5)

        results_list.append(results)
        
        time_elapsed, time_remaining, time_finish =  calc_process_time(t_start, ind+1, num_images)
        processed_percent = ind/num_images
        progress_bar.progress(processed_percent, text=f"Processed {ind+1}/{num_images}. Elapsed: {time_elapsed} s, Reamining: {time_remaining} s, ETA: {time_finish}")
    t_end = time.time()
    progress_bar.progress(1.0, text=f"Processed {num_images} in {t_end - t_start:.1f} seconds")
    container.balloons()
    st.session_state.results_list = results_list
    st.session_state.images_list = images_list
    return
    
container_progress = st.container()
st.button("Process", on_click=auto_annotate_humans, args=[container_progress])

def visualize(container):
    container.markdown("## Visualization of the annotations")
    for detection, image_path in zip(st.session_state.results_list, st.session_state.images_list):
        container.write(image_path)
        img = cv2.imread(image_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for det in detection:
            print(det)
            cv2.rectangle(img=img, pt1=(int(det.bbox[0]), int(det.bbox[1])), pt2=(int(det.bbox[2]), int(det.bbox[3])), color=(0,255,0), thickness = 4)
        container.image(img)


container_results = st.container()
if "results_list" in st.session_state.keys() and len(st.session_state.results_list) > 0:
    st.button("Visualize results", on_click=visualize, args=[container_results])
    

