import numpy as np
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
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        #ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')
        
        ground_truth_rect2 = OrderedDict({0:[854,895,129,75],1:[3375,1172,114,83],2:[1829,0,49,62]})
        #ground_truth_rect2 = {('obj_id':0, 'gt':1, 'bboxes':[200,300,20,20]),('obj_id':1, 'gt':1, 'bboxes':[400,500,20,20])}
        #ground_truth_rect2 
        #ground_truth_rect2[0] = [1,[200,300,20,20]]
        #ground_truth_rect2[1] = [1,[400,500,30,20]]

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect=ground_truth_rect2,
                        object_class=sequence_info['object_class'],multiobj_mode=True, object_ids=[0,1,2])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            
            {"name": "Tenniswith2", "path": "Tennis/images", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg", "anno_path": "Tennis/groundtruthTennis0.txt",
             "object_class": "person"}
            
           # {"name": "uftest", "path": "uf_test/images", "startFrame": 0, "endFrame": 762, "nz": 4, "ext": "jpg", 
            #"anno_path": "uf_test/groundtruthUFtest0.txt", "object_class": "person"}
            
        
             ]
            
            
        return sequence_info_list
