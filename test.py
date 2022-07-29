#%%
import torch
import numpy as np
from dataset import ShapeNetPartDataset
from model import MultiSacleUNet
import h5py


#%%
class_name = np.genfromtxt('all_object_categories.txt',dtype='U')[:,0]

dataset_path = "shapenet_prepared.h5"
f = h5py.File(dataset_path,'r')
x_test = f['x_test'][:]
y_test = f['y_test'][:]
s_test = f['s_test'][:]
p_test = f['p_test'][:]

ckpt = "C:/Users/aorhu/Masaüstü/ML3DG_Project/3d-segmentation-in-2d/runs/temp_model_1/model.ckpt"
model = MultiSacleUNet()
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

class_label_region = np.zeros((16,2),dtype=np.int64)
for i_class in range(16):
    idx_list = np.where(y_test==i_class)[0]
    gt_list  = s_test[idx_list]

    label_min = gt_list.min()
    label_max = gt_list.max()

    class_label_region[i_class,0] = label_min
    class_label_region[i_class,1] = label_max
#%%
print(class_label_region)

#%%
result_path = 'ShapeNet_testing_result.hdf5'
test_set_len = len(y_test)
f1 = h5py.File(result_path,'w')
x_set = f1.create_dataset('x_test',data = x_test) # point cloud position in 3D
y_set = f1.create_dataset('y_test',data = y_test) # point cloud shape class
s_set = f1.create_dataset('s_test',data = s_test) # point cloud segments
p_set = f1.create_dataset('p_test',data = p_test) # 2D position
pre_set = f1.create_dataset('pre_test', shape=(test_set_len,2048,1),dtype=np.int64)

#%%
test_dataset = ShapeNetPartDataset(
    path='shapenet_prepared.h5',
    split='overfit'
    )
print(test_dataset.__len__())
pre_test = np.zeros_like(s_test)
for idx_sample,pos,obj_class in zip(range(0,len(p_test)),p_test,y_test):
    print(pos)
    print(obj_class)
    print(idx_sample)
    input_tensor = torch.tensor(test_dataset.__getitem__(idx_sample)["3d_points"])
    print("got first item")
    #print(input_tensor.shape)
    input_tensor = input_tensor[None,:]
    pre_image =  model(input_tensor)[0]    
    
    pre_sample = np.zeros_like(pre_test[0])
    #print(pre_sample.shape)
    label_min = int(class_label_region[obj_class,0])
    label_max = int(class_label_region[obj_class,1]+1)
    
    #print("before argmax : ",pre_image.shape)
    #pre_image = pre_image[:,:,label_min:label_max].argmax(1)+label_min
    pre_image = torch.argmax(pre_image,dim=0) + label_min
    #print("after argmax : ",pre_image.shape)
    
    pre_sample = pre_image[pos[:,0],pos[:,1]]

    pre_test[idx_sample] = pre_sample[:,None][0]
    pre_set[idx_sample]  = pre_sample[:,None][0]
    if( idx_sample % 100 == 0):
        print('finish point segments: ',idx_sample,'/',len(s_test))
    

#%% close the result dataset
f.close()
f1.close()
#%% calculate iou for each shape
iou_shape = np.zeros(len(s_test))
for idx_sample,pre_sample,gt_sample,obj_class in zip(range(len(s_test)),pre_test,s_test,y_test):
    label_min = int(class_label_region[obj_class,0])
    label_max = int(class_label_region[obj_class,1]+1)

    iou_list = []
    # % for each segment, calculate iou
    for i_class in range(label_min,label_max):
        # break
        tp = np.sum( (pre_sample == i_class) * (gt_sample == i_class) )
        fp = np.sum( (pre_sample == i_class) * (gt_sample != i_class) )
        fn = np.sum( (pre_sample != i_class) * (gt_sample == i_class) )

        # % if current segment exists then count the iou
        iou = (tp+1e-12) / (tp+fp+fn+1e-12)

        iou_list.append(iou)

    iou_shape[idx_sample] = np.mean(iou_list)

    if( idx_sample % 100 == 0):
        print('finish iou calculation: ',idx_sample,'/',len(s_test))

# %%
print( 'iou_instance =', iou_shape.mean())
#%% calculate iou for each class
iou_class = np.zeros(16)
for obj_class in range(16):
    iou_obj_class = iou_shape[y_test[:]==obj_class]
    iou_class[obj_class] = iou_obj_class.mean()
print( 'iou_class =', iou_class.mean())

for obj_class in range(16):
    print('class',obj_class,', class name:',class_name[obj_class],",iou=",iou_class[obj_class])

# %%
#FOR OVERFIT #######################################################################
####################################################################################
####################################################################################
import torch
import numpy as np
from dataset import ShapeNetPartDataset
from model import MultiSacleUNet
import h5py

class_name = np.genfromtxt('all_object_categories.txt',dtype='U')[:,0]

dataset_path = "shapenet_prepared.h5"
f = h5py.File(dataset_path,'r')
x_test = f['x_test'][0]
y_test = f['y_test'][0]
s_test = f['s_test'][0]
p_test = f['p_test'][0]

# give the location of your pre-trained model
ckpt = "C:/Users/aorhu/Masaüstü/ML3DG_Project/3d-segmentation-in-2d/runs/temp_model_1/model.ckpt"
model = MultiSacleUNet()
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

#holds the class specific labels, will 12,13,14 in overfit case
class_label_region = np.zeros((1,2),dtype=np.int64)

label_min = s_test.min()
label_max = s_test.max()

# this double indexing is from the test script for multiple examples, not necessary to change since the rest works according to this
class_label_region[0,0] = label_min
class_label_region[0,1] = label_max

# testing for multiple examples normally reads from the h5 files in these lines but it's not necessary for a single example
x_set = x_test # point cloud position in 3D
y_set = y_test # point cloud shape class
s_set = s_test # point cloud segments
p_set = p_test # 2D position
pre_set = np.zeros((1,2048,1))
#%%
#need to have shapenet_prepared.h5 file ready under same directory with this file
test_dataset = ShapeNetPartDataset(
    path='shapenet_prepared.h5',
    split='overfit'
    )
pre_test = np.zeros_like(s_test)

#%%
pre_sample = np.zeros_like(pre_test)
#getting class specific min and max labels, it's actually same with the lines 149-150 in overfit case. 
#I'll change it once we figure out proper training
label_min = int(class_label_region[0,0])
label_max = int(class_label_region[0,1]+1)
# %%    
#reading the 0th sample from the dataset, which is the overfitted example in our case
input_tensor = torch.tensor(test_dataset.__getitem__(0)["3d_points"])
#uncomment this to view whole tensor(things get slow)
#torch.set_printoptions(profile="default")
#print(input_tensor)
print(input_tensor.shape)
input_tensor = input_tensor[None,:]
print(input_tensor.shape)
#%%
#putting the input through model
pre_image = model(input_tensor)[0]
print(pre_image.shape)
#torch.set_printoptions(profile="full")
print(pre_image)

#%%
#applying argmax to logits 
print("before argmax : ",pre_image.shape)
#pre_image = pre_image[:,:,label_min:label_max].argmax(1)+label_min
pre_image = torch.argmax(pre_image,dim=0) + label_min
print("after argmax : ",pre_image.shape)
#torch.set_printoptions(profile="full")
print(pre_image)
#%%
#this lines are also for the testing with multiple samples but no need to change for now
pre_sample = pre_image[p_test[:,0],p_test[:,1]]
print(pre_sample.shape)
pre_test[0] = pre_sample[:,None][0]
pre_set  = pre_sample[:,None][0]

# %%
#iou calculation, will only calculate one class which is chair for overfit example
iou_shape = np.zeros(len(s_test))
#for idx_sample,pos,obj_class in zip(range(0,len(p_test)),p_test,y_test):

for _pre,gt_sample in zip(pre_test,s_test):
    label_min = int(class_label_region[0,0])
    label_max = int(class_label_region[0,1]+1)

    iou_list = []
    # % for each segment, calculate iou
    for i_class in range(label_min,label_max):
        # break
        tp = np.sum( (_pre == i_class) * (gt_sample == i_class) )
        fp = np.sum( (_pre == i_class) * (gt_sample != i_class) )
        fn = np.sum( (_pre != i_class) * (gt_sample == i_class) )

        # % if current segment exists then count the iou
        iou = (tp+1e-12) / (tp+fp+fn+1e-12)

        iou_list.append(iou)

    iou_shape[0] = np.mean(iou_list)

    if( 0 % 100 == 0):
        print('finish iou calculation: ',0,'/',len(s_test))

# %%
print( 'iou_instance =', iou_shape.mean())
#%% calculate iou for each class
#iou calculation for all classes
iou_class = np.zeros(16)
for obj_class in range(16):
    iou_obj_class = iou_shape[y_test==obj_class]
    iou_class[obj_class] = iou_obj_class.mean()
print( 'iou_class =', iou_class.mean())

for obj_class in range(16):
    print('class',obj_class,', class name:',class_name[obj_class],",iou=",iou_class[obj_class])
# %%
#VISUALIZATION
#visualizatin normally will be a different script by itself but just for this overfit sample visualization
#the code is copied to here so some repetition of the code might be seen, not a problem, will be fixed
# plotly might give an error while plotting, something about "nb", only need to update some package then restart, it fixed for me
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

current_palette = sns.color_palette('bright',10)
#"all_object_categories.txt" should also be under the same directory with this script
class_name = np.genfromtxt('all_object_categories.txt',dtype='U')[:,0]


# %%
label_min = s_test.min()
label_max = s_test.max()
x_pt = x_set
s_pt = s_set-label_min
pre_pt = pre_sample-label_min
# %%
fig = go.Figure()
for i_seg in range(label_max - label_min +1):
    idxs = np.where(s_pt == i_seg)[0]
    color = current_palette.as_hex()[i_seg]
    fig.add_trace(go.Scatter3d(x=x_pt[idxs,0], y=x_pt[idxs,1], z=x_pt[idxs,2],
                                   mode='markers',
                                   marker = dict(color = color )))
    
fig.show()
# %%
fig = go.Figure()
for i_seg in range(label_max - label_min +1):
    idxs = np.where(pre_pt == i_seg)[0]
    color = current_palette.as_hex()[i_seg]
    fig.add_trace(go.Scatter3d(x=x_pt[idxs,0], y=x_pt[idxs,1], z=x_pt[idxs,2],
                                   mode='markers',
                                   marker = dict(color = color )))
    
fig.show()
# %%
