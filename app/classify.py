import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

import sys
from detectron2.utils.visualizer import ColorMode
import cv2
import numpy as np
import functools

def classify(ddir):
    data_dir = "data/" + ddir
    setup_logger()
    for d in [ddir]: #Assign pothole metadata
        MetadataCatalog.get("data_" + d).set(thing_classes=["pothole"])
    pothole_metadata = MetadataCatalog.get("data_" + ddir)


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join("../train/output", "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)




    outFiles = []
    for filename in os.listdir(data_dir):
        im = cv2.imread(data_dir + "/" + filename)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=pothole_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(filename, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(data_dir+'|mask|'+filename, out.get_image()[:, :, ::-1])  #write to mask file
        outFiles.append(str(data_dir+'|classified|'+filename)) 

        masks = outputs['instances'].pred_masks  #output binary masks from image
        # if len(masks) > 0:
        #     combinedMask = functools.reduce(lambda a, b: a|b, masks) #logical OR to combine binary masks
        #     cv2.imwrite(data_dir+'mask'+filename, combinedMask)  #write to mask file
        #     outFiles.append(str(data_dir+'mask'+filename))    

    return outFiles



    # from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    # from detectron2.data import build_detection_test_loader
    # evaluator = COCOEvaluator("../dataset/pothole_val", output_dir="./output")
    # val_loader = build_detection_test_loader(cfg, "pothole_val")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`

# classify("2023-09-13|03")
