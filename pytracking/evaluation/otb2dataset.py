import numpy as np
import os
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_text import load_text
from collections import OrderedDict, defaultdict


class OTBDataset(BaseDataset):
    """ OTB-2015 dataset

    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf

    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        
        #count number of images
        totalFiles = 0
        totalDir = 0
        for base, dirs, files in os.walk("/gdrive/My Drive/Football/img"):
            for directories in dirs:
                totalDir += 1
            for Files in files:
                totalFiles += 1
        
        print("Total Files: " + str(totalFiles))


        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = 1
        end_frame = totalFiles - 1

        init_omit = 0
        

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]


            
        
        #load inital BB     
        gt = OrderedDict()
        #define numbers of tracked objects
        with open('/gdrive/My Drive/pytracking/data.TXT') as myfile:
            total_lines = sum(1 for line in myfile)
        print("detected Objects:" + str(total_lines))
        #define ground truth data for first frame
        with open('/gdrive/My Drive/pytracking/data.TXT') as myfile:
            for i in range(0,total_lines):
                line = myfile.readline()
                data = (line.strip().split(" "))
                data = [int(x) for x in data]
                gt.update({int(i):data})
        print(gt)

        objIDs = list()
        for i in range(0,total_lines):
            objIDs.append(int(i))
        
        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect=gt,
                       object_class=sequence_info['object_class'],multiobj_mode=True, object_ids=objIDs)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Objects", "path": "Football/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg","object_class": "person"}
             ]
            
            
        return sequence_info_list
