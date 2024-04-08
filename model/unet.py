import numpy
import torch.nn.functional as F
import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
class UNet_simply(nn.Module):
    def __init__(self, input=3, output=3):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(input, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 64)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(64, 128)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(128, 128)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(128, 256)

        # right
        self.up_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_1 = DoubleConv(256, 128)

        self.up_2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.right_conv_2 = DoubleConv(256, 64)

        self.up_3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.right_conv_3 = DoubleConv(128, 64)

        self.up_4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, output, 1, 1, 0)

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output
class UNet(nn.Module):
    def __init__(self, input=3, output=3):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(input, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 128)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(128, 256)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(512, 1024)

        # right
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, output, 1, 1, 0)

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output

class AFNPBlock(nn.Module):
    def __init__(self, pool_size):
        super(AFNPBlock, self).__init__()
        self.pool_16 =  nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool_4 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool_2 = nn.AdaptiveAvgPool2d(pool_size[2])
        self.pool_1 = nn.AdaptiveAvgPool2d(pool_size[3])
        self.Conv_Value = nn.Conv2d(512, 512, 1)
        self.Conv_Key = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU()
                                      )
        self.Conv_Query = nn.Sequential(nn.Conv2d(128, 512, 1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU()
                                        )
        self.Conv_Query_temp = nn.Sequential(nn.Conv2d(128, 512, 1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU()
                                        )
        self.Conv_Query_temp2 = nn.Sequential(nn.Conv2d(256, 512, 1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU()
                                        )
        self.fc = nn.Sequential(nn.Linear(144, 72), nn.Dropout(0.2), nn.Linear(72, 144))
        self.ConvOut = nn.Conv2d(512,512,1)
        self.flatten = nn.Flatten(start_dim=2)
        # 给ConvOut初始化为0
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)

    def forward(self, gx6, binary_16, binary_8, binary_4): # color, grey

        gx6 = self.Conv_Query(gx6)
        gx6 = self.pool_16(gx6)
        Query = gx6.flatten(2).permute(0,2,1)

        binary_16 = self.Conv_Query_temp(binary_16)
        binary_8 = self.Conv_Query_temp2(binary_8)

        key_13_4 = self.pool_4(binary_16)
        Key_13_1 = self.pool_1(key_13_4)
        key_7_2 = self.pool_2(binary_8)
        key_4_1 = self.pool_1(binary_4)


        Key = torch.cat([i for i in map(self.flatten, [key_13_4,Key_13_1,binary_16,key_7_2,binary_8,key_4_1,binary_4])],dim=-1)
        Value =  Key.contiguous().permute(0,2,1)
        Concat_QK = torch.matmul(Query, Key)
        Concat_QK = (512 ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        Aggregate_QKV = torch.matmul(Concat_QK, Value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(*gx6.shape)
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        return Aggregate_QKV+gx6
class channel_atten(nn.Module):
    def __init__(self, in_channels):
        super(channel_atten, self).__init__()
        self.module = nn.Sequential( nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.Sigmoid())
    def forward(self,x):
        res = x
        atten = self.module(x)
        out = torch.mul(atten,x) + res
        return out
class AFNPBlock_256(nn.Module):
    def __init__(self):
        super(AFNPBlock_256, self).__init__()
        self.pool_16 = nn.AdaptiveAvgPool2d(16)
        self.pool_12 = nn.AdaptiveAvgPool2d(12)
        self.pool_9 = nn.AdaptiveAvgPool2d(9)
        self.pool_7 = nn.AdaptiveAvgPool2d(7)
        self.pool_6 = nn.AdaptiveAvgPool2d(6)
        self.pool_3 = nn.AdaptiveAvgPool2d(3)
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.ConvOut = nn.Conv2d(128,128,1)
        self.flatten = nn.Flatten(start_dim=2)
        # 给ConvOut初始化为0
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)
    def forward(self, x5): # color, grey
        Query = self.flatten(x5)  # 576
        Query = Query.contiguous().permute(0, 2, 1)

        Key_x5_16 = self.pool_16(x5)
        Key_x5_12 = self.pool_12(x5)
        Key_x5_9 = self.pool_9(x5)
        Key_x5_7 = self.pool_7(x5)
        Key_x5_6 = self.pool_6(x5)
        Key_x5_3 = self.pool_3(x5)
        Key_x5_1 = self.pool_1(x5)

        Key = torch.cat([i for i in map(self.flatten, [Key_x5_3, Key_x5_1, Key_x5_9, Key_x5_7, Key_x5_16])], dim=-1)
        Value = Key.contiguous().permute(0,2,1)
        Concat_QK = torch.matmul(Query, Key)
        Concat_QK = (512 ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        Aggregate_QKV = torch.matmul(Concat_QK, Value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(*x5.shape)
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        return Aggregate_QKV+x5, Key_x5_12, Key_x5_6
class UNet_simply_atten(nn.Module):
    def __init__(self, input=3, output=3):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(input, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 64)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(64, 128)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(128, 128)
        self.down_4 = nn.MaxPool2d(2, 2)

        self.left_conv_5 = DoubleConv(128, 128)
        self.down_5 = nn.MaxPool2d(2, 2)

        self.left_conv_6 = DoubleConv(128, 128)
        self.down_6 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(128, 128)
        self.center_down = nn.MaxPool2d(2, 2)
        #attention
        self.spatial_attention = AFNPBlock_256()
        self.channel_attention = channel_atten(128)


        # right
        self.up_1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.right_conv_1 = DoubleConv(256, 128)

        self.up_2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.right_conv_2 = DoubleConv(256, 64)

        self.up_3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.right_conv_3 = DoubleConv(128, 64)

        self.up_4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, output, 1, 1, 0)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        x5 = self.left_conv_5(x4_down)

        # center
        spatial_atten, x6, x7 = self.spatial_attention(x5)
        channel_atten = self.channel_attention(spatial_atten)  #24*24


        # right

        x6_up = self.up_1(channel_atten)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)
        # ############################################特征图输出######################################
        # feature = x7[0, 0:1, :, :]
        #
        # for i in range(1, x7.shape[1]):
        #     feature += x7[0,i:i+1,:,:]
        # ############################################特征图输出######################################
        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.sig(self.output(x9))

        return output
