import sys
sys.path.append('/app')
from lib.tracker.track_onnx import *

class Tracker:
    def __init__(self, model_path):
        # Complete Tracks will contain:
        # 1. Timestamps and X/Y coordinates for track
        # 2. Three image crops of vehicle (beginning, ideal, end)
        self.complete_tracks = []

        self.timer = Timer()
        self.frame_id = 0
        self.results = []
        self.cls_exit = {}
        self.seen_out = {}

        class Object(object):
            pass

        self.args = Object()
        self.args.track_thresh = 0.5
        self.args.match_thresh = 0.8
        self.args.nms_thr = 0.7
        self.args.score_thr = 0.1
        self.args.min_box_area = 10
        self.args.track_buffer = 30
        self.args.input_shape = '608,1088'
        self.args.mot20 = False
        self.args.with_p6 = False
        # input_shape='608,1088', match_thresh=0.8, min_box_area=10, model='../../bytetrack_s.onnx',
        # mot20=False, nms_thr=0.7, output_dir='demo_output', score_thr=0.1, track_buffer=30, track_thresh=0.5,
        # video_path='../../videos/palace.mp4', with_p6=False

        self.predictor = Predictor(model_path, self.args)

        self.tracker = BYTETracker(self.args, frame_rate=30)

    def push(self, image_frame, debug=False):
        '''
        Push a single image frame for tracking.
        :param image_frame:
        :return: A list of all active tracks
        '''

        # if self.frame_id % 20 == 0:
        #     logger.debug('ByteTrack frame {} ({:.2f} fps)'.format(self.frame_id, 1. / max(1e-5, self.timer.average_time)))

        outputs, img_info = self.predictor.inference(image_frame, self.timer)
        online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []

        frame_results = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            cls = t.category
            if cls not in self.cls_exit:                                                # this if condition is designed for visualizing objects' counting  based on their class
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

            vertical = tlwh[2] / tlwh[3] > 10.0                                        # this threshold is for general object tracking, you could decrease this value when doing pedestrian tracking, eg. 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid_curr)
                online_scores.append(t.score)
                online_cls.append(t.category)
                # save results
                frame_results.append({
                    'frame_id': self.frame_id,
                    'tid': tid,
                    'top': tlwh[0],
                    'left': tlwh[1],
                    'width': tlwh[2],
                    'height': tlwh[3],
                    'score': t.score,
                    'category': t.category
                })

        self.timer.toc()

        if debug:

            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, online_cls, frame_id=self.frame_id + 1,
                                      fps=1. / self.timer.average_time)
            #cv2.imshow('Tracking', online_im)

        #vid_writer.write(online_im)
        self.frame_id += 1

        return frame_results, online_im

    def pop_complete_tracks(self):
        # TODO: Add integration with completed tracks
        # Race condition could exist here
        tracks = self.complete_tracks
        self.complete_tracks = []
        return tracks
