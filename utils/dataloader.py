#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader.py
# @Version: version 1.0
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_frame import MSRVTT_single_sentence_dataLoader
from dataloaders.dataloader_msrvtt_frame import MSRVTT_multi_sentence_dataLoader
from dataloaders.dataloader_msrvttfull_frame import MSRVTTFULL_multi_sentence_dataLoader
def dataloader_msrvtt_train(args, tokenizer):
    """return dataloader for training msrvtt-9k
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_train_set): length
        train_sampler: sampler for distributed training
    """

    msrvtt_train_set = MSRVTT_multi_sentence_dataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        frame_path=args.frame_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_train_set)

    dataloader = DataLoader(
        msrvtt_train_set,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_train_set), train_sampler

def dataloader_msrvtt_test(args, tokenizer):
    """return dataloader for testing 1k-A protocol
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_test_set): length
    """

    msrvtt_test_set = MSRVTT_single_sentence_dataLoader(
        csv_path=args.val_csv,
        frame_path=args.frame_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        msrvtt_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_test_set)

def dataloader_msrvttfull_test(args, tokenizer):
    """return dataloader for testing full protocol
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msrvtt_test_set): length
    """
    msrvtt_test_set = MSRVTTFULL_multi_sentence_dataLoader(
        subset='test',
        csv_path=args.val_csv,
        json_path=args.data_path,
        frame_path=args.frame_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    dataloader = DataLoader(
        msrvtt_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_test_set)
