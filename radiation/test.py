import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                                                linestyle='--', color='k')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color='r')
        return [l1, l2]


x = np.linspace(0, 3)
fig, axL = plt.subplots(figsize=(4,3))
axR = axL.twinx()

axL.plot(x, np.sin(x), color='k', linestyle='--')
axR.plot(x, 100*np.cos(x), color='r')

axL.set_ylabel('sin(x)', color='k')
axR.set_ylabel('100 cos(x)', color='r')
axR.tick_params('y', colors='r')

plt.legend([object], ['label'],
           handler_map={object: AnyObjectHandler()})

plt.show()