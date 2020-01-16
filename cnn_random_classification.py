from torch.autograd import Variable
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import random
torch.manual_seed(0)



negative=open('rt-polarity_neg.txt',encoding='utf-8').readlines()
positive=open('rt-polarity_pos.txt',encoding='utf-8').readlines()

neg_data=[i.replace('\ufeff','').split() for i in negative]
neg_words=[j for i in neg_data for j in i if j.isalpha()==True]
neg_vocab=set(neg_words)
pos_data=[i.replace('\ufeff','').split() for i in positive]
pos_words=[j for i in pos_data for j in i if j.isalpha()==True]
pos_vocab=set(pos_words)
vocab=set(neg_words+pos_words)
print(len(neg_vocab),len(pos_vocab))

w2i={w:i for i,w in enumerate(vocab)}
i2w={i:w for i,w in enumerate(vocab)}
print(len(w2i),len(i2w))

neg_sens = []
for i in neg_data:
    neg_sen = []
    for j in i:
        if j.isalpha() == True:
            neg_sen.append(j)
    neg_sen.append(0)
    neg_sens.append(neg_sen)

pos_sens = []
for i in pos_data:
    pos_sen = []
    for j in i:
        if j.isalpha() == True:
            pos_sen.append(j)
    pos_sen.append(1)
    pos_sens.append(pos_sen)


train_sens=neg_sens[:int(len(neg_sens)*0.9)]+pos_sens[:int(len(pos_sens)*0.9)]
test_sens=neg_sens[int(len(neg_sens)*0.9):]+pos_sens[int(len(pos_sens)*0.9):]
shuffle(train_sens)
shuffle(test_sens)

print(len(test_sens))
print(int(len(neg_sens)*0.9),int(len(pos_sens)*0.9))
embeds=nn.Embedding(17185,64)



class MnistModel_tri(nn.Module):
    def __init__(self):
        super(MnistModel_tri, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv = nn.Conv1d(64, 100, 3)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        trigram = F.relu(self.conv(x))
        max_pool = nn.AdaptiveMaxPool1d(1)
        trigram = max_pool(trigram)
        trigram = trigram.squeeze(2)
        dropout = F.dropout(trigram, training=self.training)
        full_layer = self.fc(dropout)

        return F.softmax(full_layer,dim=1)


model_tri = MnistModel_tri()



class MnistModel_quad(nn.Module):
    def __init__(self):
        super(MnistModel_quad, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv1d(64, 100, 3)
        self.conv2 = nn.Conv1d(64, 100, 4)
        self.fc = nn.Linear(200, 2)

    def forward(self, x):
        trigram = F.relu(self.conv1(x))
        quadgram = F.relu(self.conv2(x))
        max_pool = nn.AdaptiveMaxPool1d(1)
        trigram = max_pool(trigram)
        trigram = trigram.squeeze(2)
        quadgram = max_pool(quadgram)
        quadgram = quadgram.squeeze(2)
        quad = torch.cat((trigram, quadgram), 1)
        dropout = F.dropout(quad, training=self.training)
        full_layer = self.fc(dropout)

        return F.softmax(full_layer,dim=1)


model_quad = MnistModel_quad()



class MnistModel_five(nn.Module):
    def __init__(self):
        super(MnistModel_five, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv1d(64, 100, 3)
        self.conv2 = nn.Conv1d(64, 100, 4)
        self.conv3 = nn.Conv1d(64, 100, 5)
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        trigram = F.relu(self.conv1(x))
        quadgram = F.relu(self.conv2(x))
        fivegram = F.relu(self.conv3(x))
        max_pool = nn.AdaptiveMaxPool1d(1)
        trigram = max_pool(trigram)
        trigram = trigram.squeeze(2)
        quadgram = max_pool(quadgram)
        quadgram = quadgram.squeeze(2)
        fivegram = max_pool(fivegram)
        fivegram = fivegram.squeeze(2)
        five = torch.cat((trigram, quadgram, fivegram), 1)
        dropout = F.dropout(five, training=self.training)
        full_layer = self.fc(dropout)

        return F.softmax(full_layer,dim=1)


model_five = MnistModel_five()

optimizer1 = optim.Adam(model_tri.parameters(), lr=0.0001)
optimizer2 = optim.Adam(model_quad.parameters(), lr=0.0001)
optimizer3 = optim.Adam(model_five.parameters(), lr=0.0001)
#answerinds=[i[-1] for i in test_sens if len(i)>3]

count=0
train_loss = []
batch_size=50
model_tri.train()
model_quad.train()
model_five.train()
for epoch in range(25):
    predictions=[]
    answerinds=[]
    total_batch=int(len(train_sens)/batch_size)
    for i in range(total_batch):
        random_sentences=random.sample(train_sens,batch_size)
        ############ another way to divide by batch #############
        #random_number=random.randrange(0,len(train_sens)-51)
        #random_number=0
        #shuffle(train_sens)
        ############################################################
        for words in random_sentences:
        #for words in train_sens[random_number:random_number+batch_size]:
        #for words in train_sens[random_number]:
            centerinds = torch.LongTensor([w2i[word] for word in words[:-1]])
            target = words[-1]
            count+=1
            #losses=nn.NLLLoss()

            if len(centerinds) == 3:
                data = Variable(embeds(centerinds).t().unsqueeze(0))
                optimizer1.zero_grad()
                #model_tri.train()
                output = model_tri(data)
                loss = 0
                loss = -torch.log(output[0][target])
                #predictions.append(output[0].argmax())
                #answerinds.append(target)
                loss.backward()  # calc gradients
                train_loss.append(loss)
                optimizer1.step()

            if len(centerinds) == 4:
                data = Variable(embeds(centerinds).t().unsqueeze(0))
                optimizer2.zero_grad()
                #model_quad.train()
                output = model_quad(data)
                loss = 0
                loss = -torch.log(output[0][target])
                #predictions.append(output[0].argmax())
                #answerinds.append(target)
                loss.backward()  # calc gradients
                train_loss.append(loss)
                optimizer2.step()  # update gradients

            if len(centerinds) >= 5:
                data = Variable(embeds(centerinds).t().unsqueeze(0))
                optimizer3.zero_grad()
                #model_five.train()
                output = model_five(data)
                loss = 0
                loss = -torch.log(output[0][target])
                #predictions.append(output[0].argmax())
                #answerinds.append(target)
                loss.backward()  # calc gradients
                train_loss.append(loss)
                optimizer3.step()  # update gradients

        if count % 1000 == 0:
            avg_loss = sum(train_loss) / len(train_loss)
            print("Loss %d: %f" % (count, avg_loss))
            train_loss = []

prediction=[]
answerind=[]
model_tri.eval()
model_quad.eval()
model_five.eval()
with torch.no_grad():
    for words in test_sens:
        centerinds = torch.LongTensor([w2i[word] for word in words[:-1]])
        target = words[-1]

        if len(centerinds) == 3:
            data = Variable(embeds(centerinds).t().unsqueeze(0))
            #model_tri.eval()
            output = model_tri(data)
            prediction.append(output[0].argmax())
            answerind.append(target)


        if len(centerinds) == 4:
            data = Variable(embeds(centerinds).t().unsqueeze(0))
            #model_quad.eval()
            output = model_quad(data)
            prediction.append(output[0].argmax())
            answerind.append(target)

        if len(centerinds) >= 5:
            data = Variable(embeds(centerinds).t().unsqueeze(0))
            #model_five.eval()
            output = model_five(data)
            prediction.append(output[0].argmax())
            answerind.append(target)

accuracy_list=[]
#print(prediction)
for x,y in zip(prediction,answerind):
    if x==y:
        accuracy_list.append(x)
print(len(prediction))
print(len(accuracy_list))
print(len(answerind))

accuracy = len(accuracy_list) / len(answerind)

print('Accuracy: {:.3f}'.format(accuracy))
