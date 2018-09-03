##############################################
##    DeepVOG V0.2     BY, DaChao Cui       ##
##           Email: cuidachao@hotmail.com   ##
##############################################

import numpy as np
from collections import Counter
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, add, Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam,Adadelta
from sklearn.cross_validation import train_test_split
from keras.models import load_model, Model

label_tax = 302
model_type = 2 #1=AA_code;2=physics_Properties;3=merge(3 just for testing)

##################################################################
if (model_type == 1):
	#         C,G,P,W,Y,T,S,F,M,A,V,I,L,H,K,Q,N,E,D,R
	code_R = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
	code_D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
	code_E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
	code_N = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
	code_Q = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
	code_K = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
	code_H = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
	code_L = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
	code_I = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
	code_V = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
	code_A = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
	code_M = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
	code_F = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
	code_S = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_T = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_Y = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_W = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_P = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_G = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	code_C = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	print("only amino acid code.")
####################################################################
if (model_type == 2):
	#         o,l,a,c,h,-,p,+,s,u,t,diff
	code_R = [0,0,0,1,1,0,1,1,0,0,1,0]
	code_D = [0,0,0,1,0,1,1,0,1,0,1,0]
	code_E = [0,0,0,1,0,1,1,0,0,0,1,0]
	code_N = [0,0,0,0,0,0,1,0,1,0,1,0]
	code_Q = [0,0,0,0,0,0,1,0,0,0,1,0]
	code_K = [0,0,0,1,1,0,1,1,0,0,1,1]
	code_H = [0,0,1,1,1,0,1,1,0,0,1,0]
	code_L = [0,1,0,0,1,0,0,0,0,0,0,0]
	code_I = [0,1,0,0,1,0,0,0,0,0,0,1]
	code_V = [0,1,0,0,1,0,0,0,1,0,0,0]
	code_A = [0,0,0,0,1,0,0,0,1,1,1,0]
	code_M = [0,0,0,0,1,0,0,0,0,0,0,0]
	code_F = [0,0,1,0,1,0,0,0,0,0,0,0]
	code_S = [1,0,0,0,0,0,1,0,1,1,1,0]
	code_T = [1,0,0,0,1,0,1,0,1,0,1,0]
	code_Y = [0,0,1,0,1,0,0,0,0,0,0,1]
	code_W = [0,0,1,0,1,0,0,0,0,0,0,2]
	code_P = [0,0,0,0,0,0,0,0,1,0,0,0]
	code_G = [0,0,0,0,1,0,0,0,1,1,1,1]
	code_C = [0,0,0,0,1,0,1,0,1,0,1,0]
	print("only amino acid physics Properties.")
#######################################################################
if (model_type == 3):
	#         C,G,P,W,Y,T,S,F,M,A,V,I,L,H,K,Q,N,E,D,R,o,l,a,c,h,-,p,+,s,u,t
	code_R = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1]
	code_D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0,1]
	code_E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1]
	code_N = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1]
	code_Q = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1]
	code_K = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1]
	code_H = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1]
	code_L = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0]
	code_I = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0]
	code_V = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0]
	code_A = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1]
	code_M = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
	code_F = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0]
	code_S = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1]
	code_T = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1]
	code_Y = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0]
	code_W = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0]
	code_P = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
	code_G = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1]
	code_C = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1]
	print("chimera for amino acid code and physics Properties.")

matrix_size = len(code_R)
code_X = [1 for i in range(matrix_size)]
code_n = [0 for i in range(matrix_size)]

####################################################################


def aa_ref2npy(ref_Data, len_w):
	#####################################
	seq = []
	X = []            # seqs
	Y = []            # label for all seqs
	f = open(ref_Data,'r')
	try:
	 for line in f:
	    line = line.strip('\n')
	    if line[0] == '>':
	        ll = line.split()
	        Y.append( int(ll[-1])-1 )
	    else :
	        X.append(line)
	finally:
	 f.close( )
	
	len_input = len(Y)
	len_input_1 = len(X)
	if len_input==len_input_1: print("ok! there is the same number (" + str(len_input) + ") of labels and sequences. ")
	else: print("error! the number of labels (" + str(len_input) + ") and the number of sequences (" + str(len_input)+") are different."), exit()
	####################################
	n_size = len(X)
	XX = []
	YYY = []
	XXX = []
	
	for i in range(n_size):
	    seq = X[i]
	    seq_label = Y[i]
	    len_seq = len(X[i])                  # length of seq
	    pos_end = len_seq - 1                # seq end position
	    pos = 0                              # seq start position
	    num_s = len_seq // len_w             # number of samples in a seq
	    rem = len_seq % len_w
	    if rem >= 50 :                      # the rest number of seq be read
	        num_s = num_s + 1
	    if num_s == 0:
	        num_s = num_s + 1                # short seq fill with '-'
	    for j in range(num_s):
	        for k in range(len_w):
	            if pos <= pos_end:
	                ctr = seq[pos]
	            else:
	                ctr = '-'
	            if ctr == 'R':
	                XX.append(code_R)
	            elif ctr == 'D':
	                XX.append(code_D)
	            elif ctr == 'E':
	                XX.append(code_E)
	            elif ctr == 'N':
	                XX.append(code_N)
	            elif ctr == 'Q':
	                XX.append(code_Q)
	            elif ctr == 'K':
	                XX.append(code_K)
	            elif ctr == 'H':
	                XX.append(code_H)
	            elif ctr == 'L':
	                XX.append(code_L)
	            elif ctr == 'I':
	                XX.append(code_I)
	            elif ctr == 'V':
	                XX.append(code_V)
	            elif ctr == 'A':
	                XX.append(code_A)
	            elif ctr == 'M':
	                XX.append(code_M)
	            elif ctr == 'F':
	                XX.append(code_F)
	            elif ctr == 'S':
	                XX.append(code_S)
	            elif ctr == 'T':
	                XX.append(code_T)
	            elif ctr == 'Y':
	                XX.append(code_Y)
	            elif ctr == 'W':
	                XX.append(code_W)
	            elif ctr == 'P':
	                XX.append(code_P)
	            elif ctr == 'G':
	                XX.append(code_G)
	            elif ctr == 'C':
	                XX.append(code_C)
	            elif ctr == '-':
	                XX.append(code_n)
	            else:
	                XX.append(code_X)
	            pos = pos + 1
	        XXX.append(XX)
	        XX = []
	        YYY.append(seq_label)
	
	XXX = np.array(XXX)              # XXX.reshape(-1,1,750,20)
	YYY = np.array(YYY)
	np.save(ref_Data + ".X",XXX)
	np.save(ref_Data + ".Y",YYY)
	print ("ok! windows is " + str(len_w) + "." )
	print ("ok! raw data has been saved as a npy file " + ref_Data + ".X/Y")

####################################################################################

def aa_txt2matrix(txt, len_w):
        ####################################
        XX = []
        YYY = []
        XXX = []

        seq = str(txt)
        len_seq = len(seq)                  # length of seq
        pos_end = len_seq - 1                # seq end position
        pos = 0                              # seq start position
        num_s = len_seq // len_w             # number of samples in a seq
        rem = len_seq % len_w
        if rem >= 50 :                      # the rest number of seq be read
                num_s = num_s + 1
        if num_s == 0:
                num_s = num_s + 1                # short seq fill with '-'
        for j in range(num_s):
                for k in range(len_w):
                    if pos <= pos_end:
                        ctr = seq[pos]
                    else:
                        ctr = '-'
                    if ctr == 'R':
                        XX.append(code_R)
                    elif ctr == 'D':
                        XX.append(code_D)
                    elif ctr == 'E':
                        XX.append(code_E)
                    elif ctr == 'N':
                        XX.append(code_N)
                    elif ctr == 'Q':
                        XX.append(code_Q)
                    elif ctr == 'K':
                        XX.append(code_K)
                    elif ctr == 'H':
                        XX.append(code_H)
                    elif ctr == 'L':
                        XX.append(code_L)
                    elif ctr == 'I':
                        XX.append(code_I)
                    elif ctr == 'V':
                        XX.append(code_V)
                    elif ctr == 'A':
                        XX.append(code_A)
                    elif ctr == 'M':
                        XX.append(code_M)
                    elif ctr == 'F':
                        XX.append(code_F)
                    elif ctr == 'S':
                        XX.append(code_S)
                    elif ctr == 'T':
                        XX.append(code_T)
                    elif ctr == 'Y':
                        XX.append(code_Y)
                    elif ctr == 'W':
                        XX.append(code_W)
                    elif ctr == 'P':
                        XX.append(code_P)
                    elif ctr == 'G':
                        XX.append(code_G)
                    elif ctr == 'C':
                        XX.append(code_C)
                    elif ctr == '-':
                        XX.append(code_n)
                    else:
                        XX.append(code_X)
                    pos = pos + 1
                XXX.append(XX)
                XX = []

        XXX = np.array(XXX)          
        return XXX
###############################################################################################
def seven_add(len_w,matrix_size,n_classes,nb_filters):
	dropout1=0.3
	inpt = Input(shape=(1,len_w,matrix_size))
	x = Conv2D(filters=nb_filters,kernel_size=(2,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(inpt)
	x = add([inpt, x])
	C1 = Conv2D(filters=nb_filters,kernel_size=(3,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(x)
	x = add([C1, x])
	C2 = Conv2D(filters=nb_filters,kernel_size=(5,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(x)
	x = add([C2, x])
	C3 = Conv2D(filters=nb_filters,kernel_size=(7,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(x)
	x = add([C3, x])
	C4 = Conv2D(filters=nb_filters,kernel_size=(11,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(x)
	x = add([C4, x])
	C5 = Conv2D(filters=nb_filters,kernel_size=(17,1),padding='same',input_shape=(1,len_w,matrix_size),data_format='channels_first')(x)
	x = Dropout(dropout1)(C5)
	x = Dense(40, activation='relu')(x)
	x = MaxPooling2D(pool_size=(1,5))(x)
	x = Dense(20, activation='relu')(x)
	x = MaxPooling2D(pool_size=(1,2))(x)
	x = Dense(40, activation='relu')(x)
	x = Dropout(dropout1)(x)
	#model.add(Dropout(dropout1))
	x = Flatten()(x)
	#model.add(Dropout(dropout2))
	x = Dense(n_classes, activation='softmax')(x)
	model = Model(inputs=inpt, outputs=x)
	return model
#####################################################################

def DL_TrainTest(ref_Data, len_w):
	test_split_rate=0.1
	nb_filters = matrix_size
	kernel_s = 3
	n_batch = 10
	n_echos = 30
	dropout1 = 0.10
	dropout2 = 0.10	
	print ("now runing is DL_TrainTest.")
	print ("test_split_rate: ", test_split_rate)
	print ("nb_filters: ", nb_filters)
	print ("kernel_s: ", kernel_s)
	print ("n_batch: ", n_batch)
	print ("n_echos: ", n_echos)
	print ("dropout1: ", dropout1)
	print ("dropout2: ", dropout2)
	X = np.load(ref_Data + ".X.npy")
	Y = np.load(ref_Data + ".Y.npy")
	print ("ok! the npy file " + ref_Data + ".X/Y.npy are loaded!" )
	n_classes = len(np.unique(Y))
	print ("ok! all labels are in " + str(n_classes) + " kinds." )
	YY_t = []
	for i in Y:
	    ll = np.zeros(n_classes)
	    ll[i] = 1
	    YY_t.append(ll)
	YY_t = np.array(YY_t)
	print (YY_t)
	for i in range(5):
		print("now test "+ str(i) + ".")
		X_train,X_test,Y_train,Y_test = train_test_split(X,YY_t,test_size=test_split_rate)
		X_train = X_train.reshape(-1,1,len_w,matrix_size)
		X_test = X_test.reshape(-1,1,len_w,matrix_size)
		model = seven_add(len_w,matrix_size,n_classes,nb_filters)
		model.summary()
		model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])   #
		model.fit(x=X_train,y=Y_train,batch_size=n_batch,epochs=n_echos,verbose=1)
		model.save(ref_Data + '.'+ str(i) +'.all.h5')
		score = model.evaluate(X_test,Y_test,verbose=0)
		print("Ended: ",score[0],score[1])

###############################################################################################
def DL_Train(ref_Data, len_w):
	nb_filters = matrix_size
	kernel_s = 3
	n_batch = 10
	n_echos = 20
	dropout1 = 0.30
	dropout2 = 0.30	
	print ("now runing is DL_Train.")
	print ("nb_filters: ", nb_filters)
	print ("kernel_s: ", kernel_s)
	print ("n_batch: ", n_batch)
	print ("n_echos: ", n_echos)
	print ("dropout1: ", dropout1)
	print ("dropout2: ", dropout2)
	print ("matrix_size: ", matrix_size)
	X = np.load(ref_Data + ".X.npy")
	Y = np.load(ref_Data + ".Y.npy")
	print ("ok! the npy file " + ref_Data + ".X/Y.npy are loaded!" )
	n_classes = len(np.unique(Y))
	print ("ok! all labels are in " + str(n_classes) + " kinds." )
	YY_t = []
	for i in Y:
	    ll = np.zeros(n_classes)
	    ll[i] = 1
	    YY_t.append(ll)
	YY_t = np.array(YY_t)
	print (YY_t)
	print("now training for all, Be noted here no test part !!!")
	X_train = X.reshape(-1,1,len_w,matrix_size)
	Y_train = YY_t
	# model = Sequential()
	model = seven_add(len_w,matrix_size,n_classes,nb_filters)
	model.summary()
	model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])   #
	model.fit(x=X_train,y=Y_train,batch_size=n_batch,epochs=n_echos,verbose=1)
	model.save(ref_Data + '.all.phy.h5')
	print("the model (" + ref_Data + ".all.phy.h5) has saved!")
	

##########################################################################################

def predict_and_loss(model, data_to_predict, len_w):
	##########################################################
	seq = []
	X = []            # seqs
	Y = []            # label for all seqs
	f = open(data_to_predict,'r')
	try:
	 for line in f:
	    line = line.strip('\n')
	    if line[0] == '>':
	        ll = line.split()
	        Y.append( int(ll[-1])-1 )
	    else :
	        X.append(line)
	finally:
	 f.close( )
	
	len_input = len(Y)
	len_input_1 = len(X)
	if len_input==len_input_1: print("ok! there is the same number (" + str(len_input) + ") of labels and sequences. ")
	else: print("error! the number of labels (" + str(len_input) + ") and the number of sequences (" + str(len_input_1)+") are different."), exit()
	##########################################################################
	mod_x = load_model(model)
	print("model has loaded.")
	print("predicting!")
	for mm in range(len_input_1):
		Xm = X[mm]
		Xm = aa_txt2matrix(txt=Xm,len_w=len_w)
		Xm = Xm.reshape(-1,1,len_w,matrix_size)
		Yp = mod_x.predict(Xm,verbose=0)
		Ypl = len(Yp)
		loss_bin = []
		for i in range(Ypl):
			Yp_t = []
			Ypli = np.argmax(Yp[i])
			ll = np.zeros(label_tax)
			ll[ Ypli ] = 1
			Yp_t.append(ll)
			Yp_t = np.array(Yp_t)
			Xmi = Xm[i].reshape(-1,1,len_w,matrix_size)
			# print(Xmi," ",Yp_t)
			scores = mod_x.evaluate(Xmi,Yp_t)
			loss_bin.append((Ypli, scores[0]))
		if len(loss_bin)>1:
			loss_bin.pop()
		loss_bin.sort(reverse = True, key=lambda x:x[1])
		loss_dic = dict(loss_bin)
		min_loss_label = min(loss_dic.items(), key=lambda x: x[1])[0]
		Yp = np.ones(Ypl, dtype=int)
		Yp = Yp.dot(min_loss_label)
		yll = []
		for i in range(Ypl):
		    ll = np.zeros(label_tax)
		    ll[Yp[i]] = 1
		    yll.append(ll)
		yll = np.array(yll)
		scores = mod_x.evaluate(Xm,yll)
		print("id:",mm," orgin label:",str(Y[mm])," predict label:",str(Yp[i])," loss:",scores[0])
	print("all is ended.")

# 5x cross validation
unknown_data = 'pvog_5parts/m.a0.txt'
ref_Data = 'pvog_5parts/m.a_1234.txt'
len_w = 500
if 1:
	aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
if 0:  #JUST FOR SELF-TESTING
	DL_TrainTest(ref_Data=ref_Data,len_w=len_w)
if 1:
	DL_Train(ref_Data=ref_Data,len_w=len_w)
if 1:
	predict_and_loss(model=str(ref_Data)+".all.phy.h5",data_to_predict=unknown_data, len_w=len_w)	

# unknown_data = 'pvog_5parts/m.a1.txt'
# ref_Data = 'pvog_5parts/m.a0_234.txt'
# len_w = 500
# if 0:
	# aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
# if 0:
	# DL_TrainTest(ref_Data=ref_Data,len_w=len_w)
# if 0:
	# DL_Train(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# predict_and_loss(model=str(ref_Data)+".all.phy.h5",data_to_predict=unknown_data, len_w=len_w)	
	
# unknown_data = 'pvog_5parts/m.a2.txt'
# ref_Data = 'pvog_5parts/m.a01_34.txt'
# len_w = 500
# if 1:
	# aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
# if 0:
	# DL_TrainTest(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# DL_Train(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# predict_and_loss(model=str(ref_Data)+".all.phy.h5",data_to_predict=unknown_data, len_w=len_w)		
	
# unknown_data = 'pvog_5parts/m.a3.txt'
# ref_Data = 'pvog_5parts/m.a012_4.txt'
# len_w = 500
# if 1:
	# aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
# if 0:
	# DL_TrainTest(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# DL_Train(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# predict_and_loss(model=str(ref_Data)+".all.phy.h5",data_to_predict=unknown_data, len_w=len_w)		
	
# unknown_data = 'pvog_5parts/m.a4.txt'
# ref_Data = 'pvog_5parts/m.a0123_.txt'
# len_w = 500
# if 1:
	# aa_ref2npy(ref_Data=ref_Data,len_w=len_w)
# if 0:
	# DL_TrainTest(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# DL_Train(ref_Data=ref_Data,len_w=len_w)
# if 1:
	# predict_and_loss(model=str(ref_Data)+".all.phy.h5",data_to_predict=unknown_data, len_w=len_w)		