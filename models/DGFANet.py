import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import torch.nn.utils.weight_norm as weightNorm
from pdb import set_trace as st


class FeatExtractor(nn.Module):

    def __init__(self, seq_len=599, to_binary=False):

        super(FeatExtractor, self).__init__()
        self.seq_len = seq_len
        self.to_binary = to_binary
        assert seq_len >= 10, f'seq_len ({seq_len}) must be at least 10'

        self.pad1 = nn.ReplicationPad2d((0, 0, 4, 5))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(10, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv1_bn = nn.BatchNorm2d(30)
        self.pad2 = nn.ReplicationPad2d((0, 0, 3, 4))
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(8, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2_bn = nn.BatchNorm2d(30)
        self.pad3 = nn.ReplicationPad2d((0, 0, 2, 3))
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(6, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.pad4 = nn.ReplicationPad2d((0, 0, 2, 2))
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(5, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4_bn = nn.BatchNorm2d(50)
        self.pad5 = nn.ReplicationPad2d((0, 0, 2, 2))
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(5, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5_bn = nn.BatchNorm2d(50)


    def forward(self, input):

        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1_bn(self.conv1(self.pad1(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv2_bn(self.conv2(self.pad2(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv3_bn(self.conv3(self.pad3(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv4_bn(self.conv4(self.pad4(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv5_bn(self.conv5(self.pad5(x))))
        #x = self.drop_layer(x)
        # x (batch,50,seq_len,1)
        return x

class MyFeatExtractor(nn.Module):

    def __init__(self, seq_len=599, to_binary=False):

        super(MyFeatExtractor, self).__init__()
        self.seq_len = seq_len
        self.to_binary = to_binary
        assert seq_len >= 10, f'seq_len ({seq_len}) must be at least 10'

        self.pad1 = nn.ReplicationPad2d((0, 0, 3, 3))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(7, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv1_bn = nn.BatchNorm2d(30)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,1),stride=(3,1))
        self.pad2 = nn.ReplicationPad2d((0, 0, 2, 2))
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(5, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv2_bn = nn.BatchNorm2d(40)
        self.pad3 = nn.ReplicationPad2d((0, 0, 2, 2))
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(5, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.pad4 = nn.ReplicationPad2d((0, 0, 1, 1))
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4_bn = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))


    def forward(self, input):

        x = input
        x = x.view(x.shape[0], 1, x.shape[1], 1)

        x = F.relu(self.conv1_bn(self.conv1(self.pad1(x))))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_bn(self.conv2(self.pad2(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv3_bn(self.conv3(self.pad3(x))))
        #x = self.drop_layer(x)
        x = F.relu(self.conv4_bn(self.conv4(self.pad4(x))))
        x = self.maxpool2(x)
        #print(x.size())
        # x (batch,50,99,1)
        return x

class FeatEmbedder(nn.Module):
    def __init__(self, embed_size = 1024, in_size=50*599):
        super(FeatEmbedder, self).__init__()  

        self.dense = nn.Sequential(nn.Linear(in_size, embed_size),
                                    nn.BatchNorm1d(embed_size),
                                    nn.ReLU())

        self.predict = nn.Linear(embed_size,1)
                    

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        pred = self.predict(x)
         
        return pred


class MyFeatEmbedder(nn.Module):
    def __init__(self, embed_size=1024, in_size=50 * 99):
        super(MyFeatEmbedder, self).__init__()
        # self.layer1 = nn.Linear(in_size, embed_size)
        # self.norm1 =  nn.InstanceNorm1d(1)
        # self.relu1 =  nn.ReLU()
        # self.drop1 = nn.Dropout()
        
        # self.layer2 = nn.Linear(embed_size, 512)
        # self.norm2 =  nn.InstanceNorm1d(1)
        # self.relu2 =  nn.ReLU()
        # self.drop2 = nn.Dropout()

        self.dense = nn.Sequential(nn.Linear(in_size, embed_size),
                                   nn.BatchNorm1d(embed_size, affine=True), 
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Linear(embed_size, 512),
                                   nn.BatchNorm1d(512, affine=True),
                                   nn.ReLU(),
                                   nn.Dropout())

        self.predict =  nn.Linear(512, 1)

    def forward(self, x):
        # x = x.view(x.size(0), -1)

        # x = self.layer1(x)
        # x = x.view(x.size(0), 1, -1)
        # x = self.norm1(x)
        # x = x.view(x.size(0), -1)
        # x = self.drop1(self.relu1(x))

        # x = self.layer2(x)
        # x = x.view(x.size(0), 1, -1)
        # x = self.norm2(x)
        # x = x.view(x.size(0), -1)
        # x = self.drop2(self.relu2(x))

        # pred = self.predict(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)
        pred = self.predict(x)

        return pred

class Seq2Point(nn.Module):
    def __init__(self, extractor, embedder):
        super(Seq2Point, self).__init__()  

        self.extractor = extractor

        self.embedder = embedder
    
    def forward(self, x):
        x = self.extractor(x)
        x = self.embedder(x)

        return x


class Discriminator1(nn.Module):
    def __init__(self, in_size = 50*599 ):
        super(Discriminator1,self).__init__()

        self.fc1 = nn.Linear(50*599,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,1)
        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class Discriminator2(nn.Module):
    def __init__(self, in_size=50 * 99):
        super(Discriminator2, self).__init__()

        self.fc1 = nn.Linear(in_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))



      
