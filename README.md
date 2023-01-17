# MSR-VTT_Dataloader

**Requirement**

```pip install -r requirements.txt ```

**Extract Frames**

```python sample_frame.py --input_path [raw video path] --output_path [frame path]```

**Run Sample Dataloading**

```python sample_dataloading.py --val_csv data/msrvtt_data/MSRVTT_JSFUSION_test.csv --data_path data/msrvtt_data/MSRVTT_data.json --frame_path /media/hazel/DATA1/Datasets/MSR-VTT/frames_clip/ --output_dir tmp/test_0.txt --datatype msrvttfull --max_frames 12 --batch_size_val 64 --feature_framerate 2```
