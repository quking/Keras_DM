import argparse
from sklearn.metrics import precision_score
import numpy as np
from keras.layers import Input, Dense, merge, Embedding, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
#import matplotlib.pyplot as plt
from DatasetsProcess import DatasetsProcess
from sklearn.metrics import roc_curve
import sys

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=20, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=128, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-learner', action='store', dest='learner',default = 'adam')

    args = parser.parse_args()

    classifier = DM_Model(args)

    classifier.run()


class DM_Model:
    def __init__(self, args):
        self.dataSet = DatasetsProcess()
        self.pos_train, self.neg_train = self.dataSet.pos_train, self.dataSet.neg_train
        self.test = self.dataSet.get_test_instance()
        self.shape = self.dataSet.shape
        self.learner = args.learner
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer

        self.lr = args.lr

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.earlyStop = args.earlyStop

    def get_model(self):
        # Input variables
        user_input = Input(shape=(1,), dtype='float32', name='user_input')
        item_input = Input(shape=(1,), dtype='float32', name='item_input')

        MLP_Embedding_User = Embedding(input_dim=self.shape[0], output_dim=int(10), name='user_embedding',
                                       embeddings_regularizer=l2(0), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim=self.shape[1], output_dim=int(10), name='item_embedding',
                                       embeddings_regularizer=l2(0), input_length=1)

        user_latent = Flatten()(MLP_Embedding_User(user_input))
        item_latent = Flatten()(MLP_Embedding_Item(item_input))
        for idx in range(0,len(self.userLayer)-1):
            user_latent = Dense(self.userLayer[idx],activation='relu', W_regularizer=l2(0))(user_latent)
            user_latent = Dropout(0.5)(user_latent)
        user_latent = Dense(self.userLayer[-1],activation='relu', W_regularizer=l2(0))(user_latent)

        for idx in range(0, len(self.itemLayer)-1):
            item_latent = Dense(self.itemLayer[idx],activation='relu', W_regularizer=l2(0))(item_latent)
            item_latent = Dropout(0.5)(item_latent)
        item_latent = Dense(self.itemLayer[-1],activation='relu', W_regularizer=l2(0))(item_latent)

        predict_vector = merge([user_latent, item_latent], mode='mul')

        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
        model = Model(inputs=[user_input, item_input], outputs=prediction)

        return model



    def run(self):
        num_users, num_items = self.shape
        print("Load data done #user=%d, #item=%d"
              % (num_users, num_items))

        # Build model
        model = self.get_model()
        if self.learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=self.lr), loss='binary_crossentropy',metrics=['accuracy'])
        elif self.learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=self.lr), loss='binary_crossentropy',metrics=['accuracy'])
        elif self.learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy',metrics=['accuracy'])
        else:
            model.compile(optimizer=SGD(lr=self.lr), loss='binary_crossentropy',metrics=['accuracy'])
        print(model.summary())


        # Train model
        score = 0
        best_iter = -1
        validation_ac = []
        losses = []
        precision = -1
        user_input_test, item_input_test, labels_test = self.dataSet.get_test_instance()
        sum_res = np.zeros(len(labels_test))
        train_len_p = len(self.pos_train)
        train_len_n = len(self.neg_train)
        y_pre = []
        for epoch in range(self.maxEpochs):
            # Generate training instances
            n = 0
            temp_loss = []
            for i in range(train_len_p,train_len_n,train_len_p):
                train = []
                train.extend(self.pos_train)
                train.extend(self.neg_train[i-train_len_p:i])
                user_input, item_input, labels = self.dataSet.get_train_instances(train)
            # Training

                hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=32, epochs=1, verbose=0, shuffle=True)
                temp_loss.append(hist.history['loss'])
        #print("loss: ", hist.history)
            #losses.append(hist.history['loss'])
            #validation_ac.append(hist.history['val_acc'])
        # plt.plot(hist.history['acc'])
        # plt.plot(hist.history['val_acc'])
        # plt.plot(hist.history['loss'])
        # plt.plot(hist.history['val_loss'])
        # plt.legend(['training acc', 'validation acc', 'training loss', 'val loss'], loc='upper right')
        # plt.show()


        # Evaluation
                pre_res = model.predict([user_input_test,item_input_test])
                #print('结果总和：',(pre_res.squeeze() >= 0.5).astype('int32').sum())
                n = n+1
                sum_res += (pre_res.squeeze() >= 0.5).astype('int32')
        #print("预测结果总和：",np.sum(pre_res.squeeze().astype('int32')))
            losses.append(np.mean(temp_loss))
            sum_res = (sum_res >= (n / 2)).astype('int')
            print(sum_res.sum())
            t_score = sum(sum_res.astype('int32') == labels_test.astype('int32'))/len(labels_test)
            validation_ac.append(t_score)
            t_precision = precision_score(labels_test.astype('int32'),sum_res.astype('int32'))
            print('准确率：',t_score)
            print('精准率：',t_precision)
            if t_score > score:
                score = t_score
                best_iter = epoch
                y_pre = sum_res.astype('int32')
            if precision < t_precision:
                precision = t_precision
            print('----------------{}-----------acc: {} -- mean loss: {}'.format(epoch,t_score,np.mean(temp_loss)))

        #t_score = model.evaluate([user_input_test,item_input_test],labels_test)
        # print("The accuracy of epoch {} is {}".format(epoch, t_score))
        # if score < t_score:
        #     score = t_score
        #     best_iter = epoch

        print("End. Best Iteration %d:  accuracy = %.4f  precision = %.4f " % (best_iter, score, precision))
        # plt.plot(range(0,len(losses)),losses,'r--')
        # plt.plot(range(0, len(validation_ac)), validation_ac, 'g--')
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy/loss")
        # plt.show()
        #
        # fprs, tprs, threshholds = roc_curve(labels_test.astype('int32'), y_pre)
        # plt.plot(fprs, tprs)
        # plt.show()

if __name__ == "__main__":
    main()


