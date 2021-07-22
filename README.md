## Hands Motion Capture Processes 
### 1. hand pose estimation with mediapipe
The hand pose estimation module integrated in mediapipe [1] can distinguish between left and right hands, 
and the accuracy and stability of the 3D hand joint points are excellent.At the same time, 
Python and C++ interfaces are provided, so it is the most efficient to develop human motion capture functions on this basis. 

### 2. filtering to eliminate jitter
We Combine low-pass filter and euro filter to filter two hands points

### 3. calculate the hand size
We select the middle metacarpophalangeal to be the root joint, and the bone from
this joint to the wrist is defined as the reference bone [2].

### 4. get the hand shape
We choose MANO [3] as the hand model to get hand shape with PSO algorithm

### 5. get the joint rotations
We infer joint rotations from joint locations, known as the inverse kinematics (IK) problem. 
HybrIK [4], we adopt hybrid inverse kinematics solution, directly transforms accurate 3D joints to relative hand rotations 
for 3D hand mesh reconstruction.

### 6. get the mesh vertices
We get the mesh vertices from joints rotation with MANO hand model 
___
## Run
```
python demo_mediapipe_two_hands.py
```
![](./data/hands_capture.gif)

## Reference
[1] https://google.github.io/mediapipe/solutions/hands   
[2] https://github.com/CalciferZh/minimal-hand     
[3] https://github.com/hassony2/manopth    
[4] https://github.com/Jeff-sjtu/HybrIK













