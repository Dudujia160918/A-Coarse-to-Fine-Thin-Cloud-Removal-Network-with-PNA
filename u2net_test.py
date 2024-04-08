import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import  DataLoader
from torchvision import transforms#, utils
import numpy as np
from PIL import Image
import glob, time
from model import  Net, thin_model
from data_loader import ToTensorLab
from data_loader import SalObjDataset


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze(dim=0)
    if len(predict.shape) == 3:
        predict = predict.permute(1, 2, 0)
    else:
        predict = predict.squeeze( 0)
    predict_np = predict.cpu().data.numpy()
    a = predict_np*255
    a = np.array(a).astype("uint8")
    im = Image.fromarray(a)
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='thin_model'#u2netp

    prediction_J_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_J' + os.sep)
    prediction_Clean_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_Clean' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '_320.pth')


    data_dir = os.path.join(os.getcwd(), 'test_data' + os.sep)
    tra_image_dir = "test_images\\"
    tra_label_dir = "target maps\\"
    tra_mask_dir = "thick_cloud\\"
    tra_tran_dir = "trans\\"
    image_ext = '.tif'



    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + image_ext)
    tra_mask_name_list = glob.glob(data_dir + tra_mask_dir + '*' + image_ext)
    tra_tran_name_list = glob.glob(data_dir + tra_tran_dir + '*' + image_ext)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = tra_img_name_list,
                                        lbl_name_list = tra_lbl_name_list,
                                        mask_name_list=tra_mask_name_list,
                                        tran_name_list=tra_tran_name_list,
                                        transform=transforms.Compose([ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = thin_model().cuda()
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    # total = sum([param.nelement() for param in net.parameters()])
    # print("Number of parameters: %.2fM" % (total / 1e6))
    start = time.time()
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = Variable(inputs_test).cuda()
        with torch.no_grad():
            J, clean= net(inputs_test)


        # save results to test_results folder
        if not os.path.exists(prediction_J_dir):
            os.makedirs(prediction_J_dir, exist_ok=True)
        save_output(tra_img_name_list[i_test],J,prediction_J_dir)

        if not os.path.exists(prediction_Clean_dir):
            os.makedirs(prediction_Clean_dir, exist_ok=True)
        save_output(tra_img_name_list[i_test],clean,prediction_Clean_dir)


        del clean
    print("time:", time.time() - start)

if __name__ == "__main__":
    main()
