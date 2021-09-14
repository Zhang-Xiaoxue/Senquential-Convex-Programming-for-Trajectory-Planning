# %%
import matplotlib.pyplot as plt
import numpy as np
plt.ion() ## Note this correction
# fig1, [[ax10],[ax11],[ax12],[ax13],[ax14],[ax15],[ax16],[ax17]] = plt.subplots(nrows=8, ncols=1)
fig1,ax1 = plt.subplots(8,1)

# %%
i=0
num = 100
n = np.linspace(0, 10, 10)
y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))

while i <= num:
    temp_y0=np.random.random()
    ax1[0].scatter(i,temp_y0)
    temp_y1 = 2*np.random.random()
    ax1[1].scatter(i,temp_y1)
    ax1[2].plot([0,1],[0,1],'b')
    ax1[3].plot(n,y,'r')
    # ax1[4].set_title(r'$u_{}$'.format(str(i)))
    ax1[4].set_title(r'$u_{' + str(i) + '}$')
    i+=1
    plt.show()
    plt.pause(0.0001) #Note this correction
# %%
import numpy as np
for i in range()
A = np.random.random([6,1])

