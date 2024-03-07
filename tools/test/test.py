import tqdm

with tqdm.trange(0, 100, desc='epochs', dynamic_ncols=True) as tbar:
    for cur_epoch in tbar: # 到这里会显示进度条，cur_epoch会自动获取tbar中的值
        print(cur_epoch)