#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: infer_retrieval.py
# @Version: version 1.0

import torch
import numpy as np
import os
import random
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

from utils.config import get_args
from utils.dataloader import dataloader_msrvtt_train
from utils.dataloader import dataloader_msrvtt_test
from utils.dataloader import dataloader_msrvttfull_test


# define the dataloader
# new dataset can be added from import and inserted according to the following code
DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train,
                             "test":dataloader_msrvtt_test}
DATALOADER_DICT["msrvttfull"] = {"train":dataloader_msrvtt_train,
                                 "val":dataloader_msrvttfull_test,
                                 "test":dataloader_msrvttfull_test}



def main():

    # obtain the hyper-parameter
    args = get_args()

    # setting tokenizer
    tokenizer = ClipTokenizer()

    # init test dataloader
    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    for bid, batch in enumerate(test_dataloader):
        text_ids, text_mask, segment_ids, video, video_mask = batch

        print(text_ids.shape)
        print(text_mask.shape)
        print(segment_ids.shape)
        print(video.shape)
        print(video_mask.shape)
        exit(0)


if __name__ == "__main__":
    main()
