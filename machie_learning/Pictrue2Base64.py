# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def main():
    # 生成数据画图
    x=np.arange(0,10,0.01)
    y=np.sin(x)
    plt.figure()
    plt.plot(x,y)
    # plt.show()

    # 转base64
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()) # 将图片转为base64
    figdata_str = str(figdata_png, "utf-8") # 提取base64的字符串，不然是b'xxx'

    print(figdata_str)

    # 保存为.html
    html = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
    filename='png.html'
    with open(filename,'w') as f:
        f.write(html)

if __name__ == '__main__':
    main()

#%%
