import sys
import os
import gdown


src_path = os.path.dirname(os.path.abspath(__file__))
#print(src_path)

sys.path.append(os.path.join(src_path, "inference"))
sys.path.append(os.path.join(src_path, "src"))

#print(sys.path)

import argparse

import PIL.Image
import io
import json
import torch
import numpy as np
from inference.processing_image import Preprocess
from inference.visualizing_image import SingleImageViz
from inference.modeling_frcnn import GeneralizedRCNN
from inference.utils import Config, get_data

import wget
import pickle

import cv2

from param import parse_args
from vqa import Trainer
from tokenization import VLT5TokenizerFast


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="input.jpg")
    #parser.add_argument("--questions", type=str, default="What is the main doing, What color is the clothing the man wears, What color is the horse, What color of the suit the man wearing, What color of the horse?")
    
    args = parser.parse_args()

    return args


def preprocess_question(question: str) -> str:
    if not question.endswith("?"):
        question = "".join([question, "?"])
    if not question.startswith("vqa: "):
        question = " ".join(["vqa:", question])
    return question
    

if __name__ == "__main__":

    if not os.path.exists("snap/pretrain/VLT5"):
        os.makedirs("snap/pretrain/VLT5", exist_ok=True)

    if not os.path.exists(os.path.join("snap/pretrain/VLT5", "Epoch30.pth")):
        gdown.download('https://drive.google.com/uc?id=100qajGncE_vc4bfjVxxICwz3dwiAxbIZ', 'snap/pretrain/VLT5/Epoch30.pth', quiet=False)

    command_args = get_args()
    questions = input("Enter some questions:\n")
    args = parse_args(
        parse=False,
        backbone='t5-base',
        load='snap/pretrain/VLT5/Epoch30'
    )
    args.gpu = 0

    #print(os.getcwd())

    trainer = Trainer(args, train=False)


    URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"
    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

    objids = get_data(OBJ_URL) 
    attrids = get_data(ATTR_URL)
    gqa_answers = get_data(GQA_URL) 
    vqa_answers = get_data(VQA_URL) 
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg) 
    image_preprocess = Preprocess(frcnn_cfg) 

    #image_filename = wget.download(URL)
    image_filename = command_args.image_path

    image_dirname = image_filename
    frcnn_visualizer = SingleImageViz(image_filename, id2obj=objids, id2attr=attrids) 

    images, sizes, scales_yx = image_preprocess(image_filename) 

    output_dict = frcnn(
        images, 
        sizes, 
        scales_yx = scales_yx, 
        padding = 'max_detections', 
        max_detections = frcnn_cfg.max_detections, 
        return_tensors = 'pt' 
    )

    # add boxes and labels to the image 
    frcnn_visualizer.draw_boxes(
        output_dict.get("boxes"), 
        output_dict.get("obj_ids"),
        output_dict.get("obj_probs"),
        output_dict.get("attr_ids"), 
        output_dict.get("attr_probs"),
    )

    normalized_boxes = output_dict.get("normalized_boxes") 
    features = output_dict.get("roi_features")

    tokenizer = VLT5TokenizerFast.from_pretrained('t5-base')


    #questions = ["vqa: What is the main doing?", 
    #             "vqa: What color is the clothing the man wears?", 
    #             "vqa: What color is the horse?",
    #             "vqa: What color of the suit the man wearing?",]
    questions = questions.split(",")
    questions = map(lambda x: x.strip(), questions)
    questions = map(lambda x: preprocess_question(x), questions)

    for question in questions:
        input_ids = tokenizer(question, return_tensors='pt', padding=True).input_ids
        batch = {}
        batch['input_ids'] = input_ids
        batch['vis_feats'] = features
        batch['boxes'] = normalized_boxes

        result = trainer.model.test_step(batch)
        print(f"Q: {question}")
        print(f"A: {result['pred_ans'][0]}")

    visualized_img = frcnn_visualizer._get_buffer()
    visualized_img = np.uint8(np.clip(visualized_img, 0, 255))

    cv2.imshow("Anh", visualized_img)
    cv2.waitKey(0)
