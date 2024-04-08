import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
import glob
from data_loader import ToTensorLab, RandomCrop
from data_loader import SalObjDataset
from model import thin_model
import time
# ------- 1. define loss function --------
import torch
import torch.nn as nn
import torchvision.models as models


# Loss functions
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


Percep_loss = PerceptualLoss(torch.nn.MSELoss())
L1_loss = nn.L1Loss(size_average=True)

def muti_bce_loss_fusion(clean,image_labels_v):

    loss_l1 = L1_loss(clean, image_labels_v)
    loss_PerceptualLoss = Percep_loss.get_loss(clean, image_labels_v)

    return loss_l1, loss_PerceptualLoss

def bce_loss_fusion(clean,image_labels_v):

    loss_l1 = L1_loss(clean, image_labels_v)
    return loss_l1,loss_l1


def main():
    # ------- 2. set the directory of training dataset --------

    model_name = 'thin_model'  # 'u2netp'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = "img2\\"
    tra_label_dir = "GT2\\"
    tra_mask_dir = "thick_cloud\\"
    tra_tran_dir = "trans\\"       #这种定义中间变量的就可以不用写
    image_ext = '.tif'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep) #设置保存路径

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext) #获取训练数据
    tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + image_ext) #获取训练标签
    tra_mask_name_list = glob.glob(data_dir + tra_mask_dir + '*' + image_ext) #获取测试数据
    tra_tran_name_list = glob.glob(data_dir + tra_tran_dir + '*' + image_ext) #获取测试标签
    epoch_num = 351                #设置网络迭代次数，   但是这种关键参数是需要写的
    batch_size_train = 5          #批处理大小为6
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("---")
    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset( img_name_list=tra_img_name_list,
                                    lbl_name_list=tra_lbl_name_list,
                                    mask_name_list=tra_mask_name_list,
                                    tran_name_list=tra_tran_name_list,
                                    transform=transforms.Compose([ToTensorLab(flag=0)]))
                                    #57-63行，写成注释即可，数据读取并做数据增强
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,num_workers=1) #数据分块打包

    # ------- 3. define model --------
    # define the net
    net = thin_model().cuda() # 网络模型初始化，并加载到GPU

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    first_L, second_L = 0., 0.

    ite_num4val = 0
    save_frq = 10  # save the model every 2000 iterations
    txt = open("training loss.txt", "a+")
    txt.write("epoch" + "\t\t" + "losses_all" + "\t\t" + "losses_tar_all" + "\n")
    for epoch in range(310, epoch_num):
        net.train()
        start = time.time()
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            image, image_labels, tran, tran_label = data['image'], data['image_label'], data['tran'], data['tran_label']

            image = image.type(torch.FloatTensor)
            image_labels = image_labels.type(torch.FloatTensor)
            tran = tran.type(torch.FloatTensor)
            tran_label = tran_label.type(torch.FloatTensor)


            # wrap them in Variable
            image_v, image_labels_v, tran_v, tran_label_v = Variable(image.cuda(), requires_grad=False), Variable(image_labels.cuda(), requires_grad=False),\
                                                            Variable(tran.cuda(), requires_grad=False),  Variable(tran_label.cuda(), requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            first, second = net(image_v)
            loss_image1, loss_tran1 = muti_bce_loss_fusion(first, image_labels_v)
            loss_image2, loss_tran2 = bce_loss_fusion(second, image_labels_v)
            first_loss = loss_tran1 + loss_image1
            second_loss = loss_tran2
            (first_loss+second_loss).backward()
            optimizer.step()

            # # print statistics
            first_L += first_loss.data.item()
            second_L += second_loss.data.item()

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] loss_l2: %3f, loss_l1: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, first_loss / ite_num4val,
                second_loss / ite_num4val))

            # del temporary outputs and loss
            del first_loss, second_loss, loss_image1, loss_tran1,loss_image2, loss_tran2, first, second

        log = str(epoch+1) + "\t\t" + str(np.mean(first_L / ite_num4val)) + "\t\t" + str(np.mean(second_L / ite_num4val)) + "\n"
        txt.write(log)
        txt.flush()

        if epoch>200 and epoch % save_frq == 0:

        # if epoch > 0 and epoch % 1 == 0:
            torch.save(net.state_dict(), model_dir + model_name+"_%d.pth" % (epoch))
            first_L, second_L = 0., 0.
            net.train()  # resume train
            ite_num4val = 0
        end = time.time()
        print("Time:", end - start)

if __name__ == "__main__":

    main()
