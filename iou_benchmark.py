import tensorflow as tf
import numpy as np
from numba import jit
from time import time, sleep
import pandas as pd
import json

class BenchMark:

    def __init__(self):
        conf = tf.ConfigProto()
        conf.gpu_options.per_process_gpu_memory_fraction = .95
        conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)

        np.random.seed(0)
        self.tf_bboxes1 = tf.placeholder(dtype=tf.float16, shape=[None, 4])
        self.tf_bboxes2 = tf.placeholder(dtype=tf.float16, shape=[None, 4])

    def printer(self, to_run):
        print(self.sess.run(to_run))

    def get_2_bbxoes(self, num_boxes_in_1=10000, num_boxes_in_2=100000):
        """
        :param num_boxes_in_1: int, numbers of boxes; max 10000, limitation because of the GPU memory
        :param num_boxes_in_2: int, numbers of boxes; max 100000, limitation because of the GPU memory
        :return: boxes1, boxes2
        """
        # generating random co-ordinates of [x1,y1,x2,y2]
        boxes1 = np.reshape(np.random.randint(high=1200, low=0, size=num_boxes_in_1 * 4), newshape=(num_boxes_in_1, 4))
        boxes2 = np.reshape(np.random.randint(high=1200, low=0, size=num_boxes_in_2 * 4), newshape=(num_boxes_in_2, 4))
        return boxes1, boxes2

    def np_no_vec_no_jit_iou(self, boxes1, boxes2):
        """
        Calculate IOU between a bounding box and a set of bounding boxes.
        :param box: [x1,y1,x2,y2]
        :param boxes:[[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
        :return: corresponding IOU values
        """
        def run(box, boxes):
            ww = np.maximum(np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
                            np.maximum(box[0], boxes[:, 0]),
                            0)
            hh = np.maximum(np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
                            np.maximum(box[1], boxes[:, 1]),
                            0)
            uu = box[2] * box[3] + boxes[:, 2] * boxes[:, 3]
            return ww * hh / (uu - ww * hh)

        tic = time()
        for b in boxes1:
            run(b, boxes2)
        toc = time()
        return toc - tic

    def np_no_vec_jit_iou(self, boxes1, boxes2):
        """
        Calculate IOU between a bounding box and a set of bounding boxes.
        :param box: [x1,y1,x2,y2]
        :param boxes:[[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
        :return: corresponding IOU values
        """

        @jit(nopython=True)
        def run(box, boxes):
            ww = np.maximum(np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
                            np.maximum(box[0], boxes[:, 0]),
                            0)
            hh = np.maximum(np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
                            np.maximum(box[1], boxes[:, 1]),
                            0)
            uu = box[2] * box[3] + boxes[:, 2] * boxes[:, 3]
            return ww * hh / (uu - ww * hh)

        tic = time()
        for b in boxes1:
            run(b, boxes2)
        toc = time()
        return toc - tic

    def np_vec_no_jit_iou(self, boxes1, boxes2):
        def run(bboxes1, bboxes2):
            x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
            x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = np.maximum(x11, np.transpose(x21))
            yA = np.maximum(y11, np.transpose(y21))
            xB = np.minimum(x12, np.transpose(x22))
            yB = np.minimum(y12, np.transpose(y22))

            # compute the area of intersection rectangle
            interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

            # compute the area of both the prediction and ground-truth rectangles
            boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
            boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

            iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

            return iou

        tic = time()
        run(boxes1, boxes2)
        toc = time()
        return toc - tic

    def np_vec_jit_iou(self, boxes1, boxes2):

        @jit(nopython=True)
        def run(bboxes1, bboxes2):
            x11, y11, x12, y12 = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2], bboxes1[:,3]
            x21, y21, x22, y22 = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2], bboxes2[:,3]

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = np.maximum(x11, np.transpose(x21))
            yA = np.maximum(y11, np.transpose(y21))
            xB = np.minimum(x12, np.transpose(x22))
            yB = np.minimum(y12, np.transpose(y22))

            # compute the area of intersection rectangle
            interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

            # compute the area of both the prediction and ground-truth rectangles
            boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
            boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

            iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

            return iou

        # %time np_vec_jit_iou(boxes1, boxes2)
        tic = time()
        run(boxes1, boxes2)
        toc = time()
        return toc - tic

    def tf_iou(self, boxes1, boxes2):
        def run(tb1, tb2):
            x11, y11, x12, y12 = tf.split(tb1, 4, axis=1)
            x21, y21, x22, y22 = tf.split(tb2, 4, axis=1)

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = tf.maximum(x11, tf.transpose(x21))
            yA = tf.maximum(y11, tf.transpose(y21))
            xB = tf.minimum(x12, tf.transpose(x22))
            yB = tf.minimum(y12, tf.transpose(y22))

            # compute the area of intersection rectangle
            interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB - yA + 1), 0)

            # compute the area of both the prediction and ground-truth rectangles
            boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
            boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

            iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

            return iou

        op = run(self.tf_bboxes1, self.tf_bboxes2)
        self.sess.run(op, feed_dict={self.tf_bboxes1: boxes1, self.tf_bboxes2: boxes2})
        tic = time()
        self.sess.run(op, feed_dict={self.tf_bboxes1: boxes1, self.tf_bboxes2: boxes2})
        toc = time()
        return toc - tic

    def benchmark(self, box1_max=1001, box1_step=1, box2_max=10001, box2_step=10):
        row = list()

        for num_boxes_in_1, num_boxes_in_2 in zip(range(1, box1_max, box1_step), range(10, box2_max, box2_step)):
            print("num_boxes_in_1: {}, \t num_boxes_in_2: {}".format(num_boxes_in_1, num_boxes_in_2))
            boxes1, boxes2 = self.get_2_bbxoes(num_boxes_in_1, num_boxes_in_2)

            each_row = dict()
            each_row.update({"box_A_size": num_boxes_in_1,
                             "box_B_size": num_boxes_in_2,
                             "1_np_no_vec_no_jit_iou": self.np_no_vec_no_jit_iou(boxes1, boxes2),
                             "2_np_no_vec_jit_iou": self.np_no_vec_jit_iou(boxes1, boxes2),
                             "3_np_vec_no_jit_iou": self.np_vec_no_jit_iou(boxes1, boxes2),
                             "4_np_vec_jit_iou": self.np_vec_jit_iou(boxes1, boxes2),
                             "5_tf_iou": self.tf_iou(boxes1, boxes2)
                             })
            row.append(each_row)

        return row


if __name__ == '__main__':
    analysis = BenchMark().benchmark()
    df = pd.DataFrame(data=analysis)
    df.to_csv("./analysis.csv", index=False)
