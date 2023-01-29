import numpy as np
import pandas as pd

def c_improve(path): #폴더 위치/폴더이름.txt

    with open(path,"r")as f:
        label = [line.rstrip('\n') for line in f]
    new_gt = []
    for l in label:
        new_gt.append(list(map(float,l.split(","))))

   
    new_gt = np.array(new_gt)
    new_gt = new_gt[new_gt[:,1].argsort(),:]

    cur_id = -1
    ret = []
    for n_idx, n in enumerate(new_gt):
        if n[1] != -1:
            if n[1] != cur_id:
                if len(ret) !=0: #fix
                    end_idx = n_idx - 1
                    real_team = int((sum(ret) / len(ret))+0.5)
                    for i in range(start_idx, end_idx+1):
                        new_gt[i][-2] = real_team
                ret = []
                cur_id = n[1]
                start_idx = n_idx #start 
                ret.append(n[-2])
            else: #second frame
                ret.append(n[-2])
        else:
            pass
    return new_gt

if __name__=="__main__":
    path = "gt_new.txt" #폴더 위치/ 폴더 이름.txt
    print(pd.DataFrame(c_improve(path)))
    pd.DataFrame(c_improve(path)).to_csv("result.txt",index=None, header=None)