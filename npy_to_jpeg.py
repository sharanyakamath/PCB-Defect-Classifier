import numpy as np
from PIL import Image

I = np.load('s_x_train.npy')
L = np.load('s_y_train.npy')

# print(I.shape)
# print(L.shape)

# print(L[0])
# print(L[1])


cnt = 0

for i in I:
    im = Image.fromarray(i)
    if(L[cnt]==0):
        im.save("train1/defect/defect_" + str(cnt) + ".jpeg")
    else:
        im.save("train1/nodefect/nodefect_" + str(cnt) + ".jpeg")

    cnt += 1
    # if(cnt > 1):
    #     break