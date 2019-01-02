
# coding: utf-8

# In[488]:

import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


# In[489]:

def peaks_detection(y, k, h):   
    """    
    Returns list of tuples of peaks in the form (the point number in series, x-coordinate, y-coordinate).
    Args:
        y - data vector;
        k (int, positive) - temporal neighbours of each point;
        h (float or int, positive) - some constant;
    
    """
        
    def S1(i, k, a):    # additional function
        """
        Returns average value of maxima differences of k neighbouring points around the i-th point (left and right).
        Args:
            i (int, positive) - position number of the point;
            k (int, positive) - temporal neighbours of i-th point;
            arr (list) - array with points' y-components.
        Description:
            If i-th point is the first or the last, we consider neighbouring points 
            only right side or left side, respectively.
            If i-th point have neighbours amount in left or right side less then k, 
            we take into account all these neighbouring points.
        """
        
        left_bound = k if i-k>=0 else i
        right_bound = k if i+k<len(a) else len(a)-i-1
        if i == 0:
            return max([float(a[i]) - float(a[i+j]) for j in range(1, int(right_bound)+1)])
        elif i == len(a)-1:
            return max([float(a[i]) - float(a[i-j]) for j in range(1, int(left_bound)+1)])
        else:
            return (max([float(a[i]) - float(a[i-j]) for j in range(1, int(left_bound)+1)]) + max([float(a[i]) - float(a[i+j]) for j in range(1, int(right_bound)+1)])) * 0.5
        
    x = np.linspace(0, len(y), len(y))
        # Compute peak function value for each of len(y) points in y
    vals = [S1(i, k, y) for i in range(len(y))]
        # Compute the mean and standard deviation of all positive values in array vals
    filtered_vals = list(filter(lambda x: x > 0, vals))
    mean = np.mean(filtered_vals)
    std = np.std(filtered_vals)
        # Remove local peaks which are “small” in global context 
    peaks = [(i, x[i], y[i]) for i in range(len(y)) if vals[i] > 0 and (vals[i] - mean) > (h * std)]
        # Retain only one peak out of any set of peaks within distance k of each other
    i = 0
    while i < len(peaks):    
        for j, peakj in enumerate(peaks):
            if peaks[i][0] != peakj[0]:
                if abs(peaks[i][0] - peakj[0]) <= k:
                    if peaks[i][2] >= peakj[2]:
                        del peaks[j]
                    else:
                        del peaks[i]
                        i -= 1
                        break
        i += 1
        # return list with tuples of such form (x_coordinate, y_coordinate)
    return peaks


# In[490]:

vtime, vel = [], []
with open('test.csv', 'r') as f:                      
    reader = csv.reader(f.read().splitlines())   
    for num, row in enumerate(reader):
        vtime.append(row[0])
        vel.append((row[1]))


# In[491]:

vtime.pop(0)
vel.pop(0)
vel1= vel
vel2=vel


# In[492]:

input_data = np.asarray(vel,np.float)


# In[493]:

x = np.linspace(0, len(vtime), len(vtime))
    # Get peaks


# In[494]:

peaks = peaks_detection(vel, 1, 1)


# In[495]:

input_data = np.asarray(vel,np.float)
type(vel)
lis = vel


# In[496]:

lis2 = []
x =np.array(lis)
for e in x:
    lis2.append(float(e))
vel =-1*np.array(lis2)


# In[497]:

peaks2 = peaks_detection(vel, 5, 1)
x = np.linspace(0, len(vtime), len(vtime))


# In[498]:

for i, peak in enumerate(peaks2):
    print(peak)


# In[499]:

import operator
def find_intermediate_min(points, points1):
    for j in range(len(points)-1):
        x1 = points[j][0]
        y1 = points[j+1][0]
        (index1, value) = min(enumerate(vel1[x1:y1]), key=operator.itemgetter(1))
        m = vel1.index(value)
        value = -1*float(value)
        points1.append([m,x[m],value])
        
find_intermediate_min(peaks, peaks2)
peaks2_final = peaks2


# In[487]:

def find_intermediate_max(point, point1):
    for i in range(len(point)-1):
        x2 = point[i][0]
        y2 = point[i+1][0]
        (index1, value) = max(enumerate(vel2[x2:y2]), key=operator.itemgetter(1))
        m = vel2.index(value)
        point1.append([m,x[m],value])
        
find_intermediate_max(peaks2, peaks)
peaks_final = peaks


# In[474]:




# In[ ]:




# In[ ]:




# In[446]:


#find_intermediates(peaks2)


# In[396]:

for i, peak in enumerate(peaks):
    print(peak)


# In[397]:

for i, peak in enumerate(peaks2):
    print(peak)


# In[500]:

pt = np.array(peaks2_final)[:,2]
pt = -1*pt


# In[501]:


fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot(111)
    # basic line plot
plt.plot(x, input_data, color='grey')
    # scatter plot with peaks
vel =-1*np.array(peaks2)
plt.scatter(np.array(peaks_final)[:,1], np.array(peaks_final)[:,2], color='green')
plt.scatter(np.array(peaks2_final)[:,1], pt, color='red')

plt.tight_layout()
plt.margins(0.01)
plt.show()


# In[332]:

# for i, group in enumerate(incl_points):
#     plt.plot(np.array(group)[:,1], np.array(group)[:,2], '--', color='black')
#     ang = 'right angle' if i%2==0 else 'left angle'
#     angle_plot = get_angle_plot(group, incl_angles[i//2][ang])
#     if i == 0: angle_text = get_angle_text(angle_plot, 5, -0.05)
#     else: angle_text = get_angle_text(angle_plot)
#     ax.add_patch(angle_plot[0]) # To display the angle arc
#     ax.text(*angle_text, fontsize=16, fontweight='bold')
#     # To display values of x-axis according to the first column of the document you should uncomment the below line
# # plt.xticks(x, vtime, rotation=70)

# plt.tight_layout()
# plt.margins(0.01)
# plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



