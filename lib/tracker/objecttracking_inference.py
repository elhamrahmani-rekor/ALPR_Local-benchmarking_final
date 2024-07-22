import sys
sys.path.append('/app')
import argparse
import os, cv2
import numpy as np
import onnxruntime
from lib.tracker.yolox.data.data_augment import preproc as preprocess
from lib.tracker.yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from lib.tracker.yolox.utils.visualize import plot_tracking
from lib.tracker.yolox.tracker.byte_tracker import BYTETracker
from lib.tracker.yolox.tracking_utils.timer import Timer
import pdb

TRACKER_CLASS_DICT = {'0':"Class 1 - Motorcycle",
    '1':'Class 2 - Passenger Car, SUV or Minivan',
    '2':'Class 3 - Pickup, Panel, Van',
    '3':'Class 4 - Commercial Bus',
    '4':'Class 5 - Single Unit 2-Axle Truck',
    '5':'Class 6 - Single Unit 3-Axle Truck',
    '6':'Class 7 - Single Unit 4 or More Axle Truck',
    '7':'Class 8 - Single Trailer 3 or 4 Axle Truck',
    '8':'Class 9 - Single Trailer 5 Axle Truck',
    '9':'Class 10 - Single Trailer 6 or More Axle Truck',
    '10':'Class 11 - Multi Trailer 5 or Less Axle Truck',
    '11':'Class 12 - Multi Trailer 6 Axle Truck',
    '12':'Class 13 - Multi Trailer 7 or More Axle Truck'
}




class Predictor(object):
    #def __init__(self, args):
    def __init__(self, model_path, args):
        args['model'] = model_path
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.args = args
        self.session = onnxruntime.InferenceSession(args['model'],providers=['CUDAExecutionProvider'])
        self.input_shape = tuple(map(int, args['input_shape'].split(',')))

    def inference(self, ori_img, timer):
        img_info = {"id": 0}
        ### wenchi ###
        if isinstance(ori_img, str):
            img_info["file_name"] = os.path.basename(ori_img)
            ori_img = cv2.imread(ori_img)
        else:
            img_info["file_name"] = None
        ##############
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        timer.tic()
        output = self.session.run(None, ort_inputs)

        predictions = demo_postprocess(output[0], self.input_shape, p6=self.args['with_p6'])[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args['nms_thr'], score_thr=self.args['score_thr'])
        return dets, img_info



class object_tracking_bytetrack():
    def __init__(self,model_path='/app/archive/tracking_model.onnx'):
        self.model_path = model_path
        pass

    def init_video(self,max_fps=10):
        #self.args = make_parser().parse_args()
        self.args = {}
        self.args['input_shape'] = "608,1088"
        self.args['nms_thr'] = 0.7
        self.args['score_thr'] = 0.1
        self.args['track_thresh'] = 0.5
        self.args['track_buffer'] = 30
        self.args['match_thresh'] = 0.8
        self.args['min_box_area'] = 10.
        self.args['with_p6'] = False
        self.args['mot20'] = False
        self.args['model'] = self.model_path

        self.predictor = Predictor(self.model_path, self.args)
        self.tracker = BYTETracker(self.args, frame_rate=max_fps)
        self.timer = Timer()
        self.frame_id = 1
        self.cls_exit = {}
        self.seen_out = {}
        self.outputs = ''
        pass

    def predict_frame(self,frame,
                      frameid,
                      show_inference=False):

        outputs, img_info = self.predictor.inference(frame, self.timer)
        results_list = []
        online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']],
                                        [img_info['height'], img_info['width']])

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            cls = t.category
            if cls not in self.cls_exit:
                self.cls_exit[cls] = []
                self.cls_exit[cls].append(tid)
                tid_curr = 1
                self.seen_out[cls] = {}
                self.seen_out[cls][tid] = tid_curr
            else:
                if tid in self.cls_exit[cls]:
                    tid_curr = self.seen_out[cls][tid]
                else:
                    tmp = self.seen_out[cls].keys()
                    tmp_max = max(tmp)
                    tid_curr = self.seen_out[cls][tmp_max] + 1
                    self.cls_exit[cls].append(tid)
                    self.seen_out[cls][tid] = tid_curr

            vertical = tlwh[2] / tlwh[
                3] > 10.0
            if tlwh[2] * tlwh[3] > self.args['min_box_area'] and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid_curr)
                online_scores.append(t.score)
                online_cls.append(t.category)
                # save results
                curr_results_dict = {}
                curr_results_dict['frame_id'] = frameid
                curr_results_dict['tid'] = tid
                curr_results_dict['top_x'] = tlwh[1]
                curr_results_dict['top_y'] = tlwh[0]
                curr_results_dict['width'] = tlwh[2]
                curr_results_dict['height'] = tlwh[3]
                curr_results_dict['class_score'] = t.score
                curr_results_dict['class'] = t.category
                results_list.append(curr_results_dict)

        self.timer.toc()

        if show_inference:
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, online_cls, frame_id=frameid + 1,
                                      fps=1. / self.timer.average_time)

            return [results_list,online_im]
        else:
            return [results_list]




