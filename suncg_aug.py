import numpy as np


lines = [line.rstrip('\n') for line in open('../SUNCG/flistID.txt')]
gt_paths = []
normal_paths = []

for i in range(10000):
    gt_paths.append('../SUNCG/gt/' + lines[i] + '.mat')
    normal_paths.append('../SUNCG/normals_png/' + lines[i] + '.png')
    #print(gt_paths[i])
    #print(normal_paths[i])

print("DONE")