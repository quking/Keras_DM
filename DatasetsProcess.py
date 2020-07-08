import random
import numpy as np
import scipy.io as si


class DatasetsProcess(object):
    def __init__(self):
        self.pos_train, self.neg_train, self.test_Sam, self.shape = self.getData('./Datasets/Dataset1/interMatrix.mat')

    def getData(self, fileName = './Datasets/Dataset2/interMatrix.mat',ratio = 3):
        positiveSample = []
        negativeSample = []
        interation = si.loadmat(fileName)['interMatrix']
        for i in range(interation.shape[0]):
            for j in range(interation.shape[1]):
                if interation[i][j] == 1:
                    positiveSample.append((i,j,interation[i][j]))
                else:
                    negativeSample.append((i,j,interation[i][j]))
        n = 0
        num_positiveSam = len(positiveSample)
        num_negativeSam = len(negativeSample)
        test_Sam = []
        while n < num_positiveSam * 0.4:
            indx = random.randint(0,len(positiveSample)-1)
            test_Sam.append(positiveSample[indx])
            positiveSample.pop(indx)
            n = n + 1
        print('pos_test: ', len(test_Sam))
        n = 0
        while n < num_negativeSam * 0.1:
            indx = random.randint(0,len(negativeSample)-1)
            test_Sam.append(negativeSample[indx])
            negativeSample.pop(indx)
            n = n + 1
        print('neg_test', len(test_Sam)-111)
        print('训练集中正样本',len(positiveSample))
        print('训练集中fu样本', len(negativeSample))

        return positiveSample, negativeSample, test_Sam, interation.shape


    def getTrain(self):  # 测试数据集中正负样本比例 1：negnum
        return np.array(self.pos_train), np.array(self.neg_train)

    def get_train_instances(self,train):
        user_input, item_input, labels = [], [], []
        for x in train:
            user_input.append(x[0])
            item_input.append(x[1])
            labels.append(x[2])
        shuffled_idx = np.random.permutation(np.arange(len(labels)))
        user_input = np.array(user_input)[shuffled_idx]
        item_input = np.array(item_input)[shuffled_idx]
        labels = np.array(labels)[shuffled_idx]
        #print('正样本：{}, 负样本{}'.format(np.sum(labels),labels.shape[0]))
        return user_input,item_input,labels

    def get_test_instance(self):
        test = self.test_Sam
        user_input, item_input, labels = [], [], []
        for x in test:
            user_input.append(x[0])
            item_input.append(x[1])
            labels.append(x[2])
        shuffled_idx = np.random.permutation(np.arange(len(labels)))
        user_input = np.array(user_input)[shuffled_idx]
        item_input = np.array(item_input)[shuffled_idx]
        labels = np.array(labels)[shuffled_idx]
        #print('正样本：{}, 负样本{}'.format(np.sum(labels), labels.shape[0]))
        return user_input, item_input, labels