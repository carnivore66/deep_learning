from keras.models import Sequential
#from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import models
from keras import layers
from keras import losses
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
import xlrd
import numpy as np 
from keras import models
from keras import layers
from keras.regularizers import l2,l1
from keras import backend as K

###readind data

wb = xlrd.open_workbook('C:/Users/Gholamrezaee/Desktop/HCV1.xlsx')
sheet = wb.sheet_by_index(0) 


data_set = np.zeros((1386,29))

for i in range(1,1386):
    for j in range(29):
        data_set[i, j] = sheet.cell_value(i, j)
#    
print(data_set)
print('datase type=',type(data_set))

data_set = data_set[1:,:]
print('data_set.shape = \n',data_set.shape)  

###seprating data and targets

data = np.zeros((1385,28))
label = np.zeros((1385,1))

for i in range(1385):
    data[i,0:28]= data_set[i,0:28]
    label[i]= data_set[i,28]
    label[i]=label[i]-1
           
print('data\n: ', data[0])
print('\nlabel:\n ', label)

######################seprating data and targets		

train_data = data[:1000]
test_data = data[1000:]


train_targets = label[0:1000]
test_targets = label[1000:]

#####################convert data type to float 32

train_data = np.asarray(train_data).astype('float32')
test_data = np.asarray(test_data).astype('float32')

#####################normalization
mean = train_data.mean(axis=0)
train_data = train_data - mean

std = train_data.std(axis=0)
train_data = train_data / std

test_data -= mean
test_data /= std
#print('test data areeeee:\n',test_data)

######################lable's one_hot coding

y_train = to_categorical(train_targets)
y_test = to_categorical(test_targets)
print(len(train_data))

#############################precision/recall

#from keras import backend as K
def mcor(y_true, y_pred):
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())
 
def precision(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



########################build_model func
def build_model_dense():
    model=models.Sequential()
    model.add(layers.Dense(64, activation='tanh', input_shape= (28,)))
    #model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(16, activation='tanh'))
    # model.add(layers.Dense(16, activation='tanh'))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer = optimizers.RMSprop(lr=0.001) , loss = 'categorical_crossentropy' , metrics = [f1])
    return model

#######################model dropout
def build_model_dropout():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='tanh'
                           ,kernel_initializer='lecun_uniform',bias_initializer='lecun_uniform', input_shape= (28,)))
    model.add(Dropout(0.2))
   
    model.add(layers.Dense(14, activation='tanh'
                           ,kernel_initializer='lecun_uniform',bias_initializer='lecun_uniform'))
     
    model.add(Dropout(0.2))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer =optimizers.RMSprop(0.001)  ,
                  loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    return model

#####################regularizer model
def build_model_l2():
    model=models.Sequential()
    model.add(layers.Dense(128, activation='relu',input_shape=(28,)))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(0.2), bias_regularizer=l2(0.001)))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=l2(0.2), bias_regularizer=l2(0.001)))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = [f1])
    return model

#######################kfold
k=4
num_val_sample=len(train_data)//k
num_epochs=100
all_acc_histories=[]
all_acc_val_histories=[]
all_f1_histories = []
all_f1_val_histories=[]
all_loss_histories=[]
all_loss_val_histories=[]

for i in range(k):
    print('processing fold #',i)
    val_data = train_data [i*num_val_sample: (i+1)*num_val_sample]
    val_targets=y_train[i*num_val_sample:(i+1)*num_val_sample]
    
    partial_train_data=np.concatenate([
        train_data[:i*num_val_sample],
        train_data[(i+1)*num_val_sample:]], 
        axis=0)
   # print('partial_train_data shape:\n',partial_train_data.shap      
    partial_train_targets=np.concatenate([ 
       y_train[:i*num_val_sample],
       y_train[(i+1)*num_val_sample:]],
        axis=0)
    
 #######creat model  
    modell=build_model_l2()
    #modell=build_model_dense()
    # modell=build_model_dense()
   
    
    history = modell.fit(partial_train_data, partial_train_targets
                        ,validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16)  
  

    loss_history = history.history['loss']
    loss_val_history = history.history['val_loss']
    
    
    all_loss_histories.append(loss_history)
    all_loss_val_histories.append(loss_val_history)
    

    f1_history = history.history['f1']
    f1_val_history = history.history['val_f1']

    all_f1_histories.append(f1_history)
    all_f1_val_histories.append(f1_val_history)
    
    ####
   # result = modell.evaluate(test_data,y_test)
    #print(result)
    
###########################accuracy and loss diagram

plt.clf()
average_f1_history = [np.mean([x[i] for x in all_f1_histories]) for i in range(num_epochs)]
average_f1_val_history = [np.mean([x[i] for x in all_f1_val_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_f1_history) + 1),average_f1_history, 'bo', label = 'Training f1' )
plt.plot(range(1, len(average_f1_val_history) + 1),average_f1_val_history, 'b', label = 'Validation f1' )
plt.title('Training and Validation f1')
plt.xlabel('Epochs')
plt.ylabel('f1')
plt.legend()
plt.show()




plt.clf()
average_loss_history = [np.mean([x[i] for x in  all_loss_histories]) for i in range(num_epochs)]
average_loss_val_history = [np.mean([x[i] for x in all_loss_val_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_loss_history) + 1),average_loss_history, 'bo', label = 'Training loss')
plt.plot(range(1, len(average_loss_val_history) + 1),average_loss_val_history,'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

results = modell.evaluate(test_data , y_test)
print(results)
