import numpy as np
from pytracking.evaluation.environment import env_settings
from ltr.data.image_loader import imread_indexed
from collections import OrderedDict


class BaseDataset:
    """Base class for all datasets."""
    def __init__(self):
        self.env_settings = env_settings()

    def __len__(self):
        """Overload this function in your dataset. This should return number of sequences in the dataset."""
        raise NotImplementedError

    def get_sequence_list(self):
        """Overload this in your dataset. Should return the list of sequences in the dataset."""
        raise NotImplementedError


class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, dataset, ground_truth_rect, ground_truth_seg=None, init_data=None,
                 object_class=None, target_visible=None, object_ids=None, multiobj_mode=False):
        self.name = name
        self.frames = frames
        self.dataset = dataset
        self.ground_truth_rect = ground_truth_rect
        self.ground_truth_seg = ground_truth_seg
        self.object_class = object_class
        self.target_visible = target_visible
        self.object_ids = object_ids
        self.multiobj_mode = multiobj_mode
        self.init_data = self._construct_init_data(init_data)
        self._ensure_start_frame()

    def _ensure_start_frame(self):
        # Ensure start frame is 0
        start_frame = min(list(self.init_data.keys()))
        if start_frame > 0:
            self.frames = self.frames[start_frame:]
            if self.ground_truth_rect is not None:
                if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                    for obj_id, gt in self.ground_truth_rect.items():
                        self.ground_truth_rect[obj_id] = gt[start_frame:,:]
                else:
                    self.ground_truth_rect = self.ground_truth_rect[start_frame:,:]
            if self.ground_truth_seg is not None:
                self.ground_truth_seg = self.ground_truth_seg[start_frame:]
                assert len(self.frames) == len(self.ground_truth_seg)

            if self.target_visible is not None:
                self.target_visible = self.target_visible[start_frame:]
            self.init_data = {frame-start_frame: val for frame, val in self.init_data.items()}

    def _construct_init_data(self, init_data):
        if init_data is not None:
            if not self.multiobj_mode:
                assert self.object_ids is None or len(self.object_ids) == 1
                for frame, init_val in init_data.items():
                    if 'bbox' in init_val and isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = init_val['bbox'][self.object_ids[0]]
            # convert to list
            for frame, init_val in init_data.items():
                if 'bbox' in init_val:
                    if isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = OrderedDict({obj_id: list(init) for obj_id, init in init_val['bbox'].items()})
                    else:
                        init_val['bbox'] = list(init_val['bbox'])
        else:
            init_data = {0: dict()}     # Assume start from frame 0

            if self.object_ids is not None:
                init_data[0]['object_ids'] = self.object_ids
                
                print("Init Data new:")
                

            if self.ground_truth_rect is not None:
                if self.multiobj_mode:
                    #assert isinstance(self.ground_truth_rect, (dict, OrderedDict))
                    #init_data[0]['bbox'] = OrderedDict({obj_id: list(gt[0,:]) for obj_id, gt in self.ground_truth_rect.items()})
                    #kommentiere oberes aus, delete beloq
                    #Tennis
                    gt = OrderedDict()
                    with open('/gdrive/My Drive/data/data.txt') as myfile:
                        for i in range(0,len(self.object_ids)):
                            line = myfile.readline()
                            data = (line.strip().split(" "))
                            data = [int(x) for x in data]
                            gt.update({int(i):data})
                    init_data[0]['bbox'] = gt
                    #init_data[0]['bbox'] = OrderedDict({0:[854,895,129,75],1:[3375,1172,114,83],2:[1829,0,49,62]})
                    #UF
                    #init_data[0]['bbox'] = OrderedDict({0:[1249,1219,26,26],1:[1465,1089,28,33],2:[1666,1048,27,33],3:[1664,774,28,25],4:[1463,726,21,28],5:[1406,667,22,23],6:[1484,636,1930],7:[1557,1135,25,35],8:[1568,1027,20,35],9:[1818,1069,25,37],10:[1911,1292,23,31],11:[2042,998,29,23],12:[2054,945,20,26],13:[1939,842,21,25]})
                    #init_data[0]['bbox'] = self.ground_truth_rect
                    print(init_data)
                else:
                    assert self.object_ids is None or len(self.object_ids) == 1
                    if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                        init_data[0]['bbox'] = list(self.ground_truth_rect[self.object_ids[0]][0, :])
                    else:
                        init_data[0]['bbox'] = list(self.ground_truth_rect[0,:])

            if self.ground_truth_seg is not None:
                init_data[0]['mask'] = self.ground_truth_seg[0]

        return init_data

    def init_info(self):
        info = self.frame_info(frame_num=0)
        return info

    def frame_info(self, frame_num):
        info = self.object_init_data(frame_num=frame_num)
        return info

    def init_bbox(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_bbox')

    def init_mask(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_mask')

    def get_info(self, keys, frame_num=None):
        info = dict()
        for k in keys:
            val = self.get(k, frame_num=frame_num)
            if val is not None:
                info[k] = val
        return info

    def object_init_data(self, frame_num=None) -> dict:
        if frame_num is None:
            frame_num = 0
        if frame_num not in self.init_data:
            return dict()

        init_data = dict()
        for key, val in self.init_data[frame_num].items():
            if val is None:
                continue
            init_data['init_'+key] = val

        if 'init_mask' in init_data and init_data['init_mask'] is not None:
            anno = imread_indexed(init_data['init_mask'])
            if not self.multiobj_mode and self.object_ids is not None:
                assert len(self.object_ids) == 1
                anno = (anno == int(self.object_ids[0])).astype(np.uint8)
            init_data['init_mask'] = anno

        if self.object_ids is not None:
            init_data['object_ids'] = self.object_ids
            init_data['sequence_object_ids'] = self.object_ids

        return init_data

    def target_class(self, frame_num=None):
        return self.object_class

    def get(self, name, frame_num=None):
        return getattr(self, name)(frame_num)

    def __repr__(self):
        return "{self.__class__.__name__} {self.name}, length={len} frames".format(self=self, len=len(self.frames))



class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())