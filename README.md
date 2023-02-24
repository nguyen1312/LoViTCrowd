# LoViTCrowd
Official implementation of the accepted BMVC2022 paper "Improving Local Features with Relevant Spatial Information by Vision Transformer for Crowd Counting". 

This reporsitory contains the annotations for dataset and the code from the associated paper, for the task of crowd counting.

![image info](images/LoViTCrowd.png)

## Demo
Our proposed LoViTCrowd can estimate precisely the number of existing people in lots of scenarios with the scene complexity and scale variation.

![gif info](images/Toy_video.gif)

## Update
1. Link for pretrained weight on datasets: https://drive.google.com/drive/folders/1wssdmOJTmLCUEDHqklZHeVu5JCJw-CWn?usp=sharing
2. Link for data sample (ShangHaiB): 
3. Move data downloaded from above link to /data, /datatest.
4. Move weight downloaded from above link to /pretrained_weight.
5. Run inference: 
    ```console
    !mkdir pretrained_weight
    python test.py
    ```
