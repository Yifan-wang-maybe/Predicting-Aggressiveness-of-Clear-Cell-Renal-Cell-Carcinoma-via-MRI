import sys
import numpy as np
import matplotlib.pyplot as plt



def find_sublist_index(main_list, sublist):
    sublist_len = len(sublist)
    for i in range(len(main_list) - sublist_len + 1):
        if main_list[i:i + sublist_len] == sublist:
            return i
        


Train_loss = []


Val_loss = []
test_loss = []

Val_AUC_0 = []
Val_AUC_1 = []
Val_AUC_2 = []
Val_AUC_3 = []
Val_AUC_4 = []

test_AUC_0 = []
test_AUC_1 = []
test_AUC_2 = []
test_AUC_3 = []
test_AUC_4 = []

test_class_0 = []
test_class_1 = []
test_class_2 = []
test_class_3 = []
test_class_4 = []



# MAE_tiny_augment
with open('output_dir_tiny_finetune_5_with_new_Fundation.output','r') as f:
    for line in f:  
        if line[18:22] == 'Aver':
            Train_loss.append(float(line[54:60]))

        if line[18:22] == 'Eval':            
            if line[24:28] == 'loss':
                Val_loss.append(float(line[29:35])) 

        
        if line[18:26] == 'Eval auc':     
            
            index = find_sublist_index(line,'AUC_0')
            Val_AUC_0.append(float(line[index+6:index+12]))
            index = find_sublist_index(line,'AUC_1')
            Val_AUC_1.append(float(line[index+6:index+12]))  

            index = find_sublist_index(line,'AUC_2')
            Val_AUC_2.append(float(line[index+6:index+12]))         
            index = find_sublist_index(line,'AUC_3')
            Val_AUC_3.append(float(line[index+6:index+12])) 
            index = find_sublist_index(line,'AUC_4')
            Val_AUC_4.append(float(line[index+6:index+12]))    

        if line[18:22] == 'Test':
            if line[24:28] == 'loss':
                test_loss.append(float(line[29:35]))   
       
        if line[18:26] == 'Test auc':   
            index = find_sublist_index(line,'AUC_0')
            test_class_0.append(float(line[index+6:index+12]))
            index = find_sublist_index(line,'AUC_1')
            test_class_1.append(float(line[index+6:index+12]))            
            index = find_sublist_index(line,'AUC_2')
            test_class_2.append(float(line[index+6:index+12]))         
            index = find_sublist_index(line,'AUC_3')
            test_class_3.append(float(line[index+6:index+12])) 
            index = find_sublist_index(line,'AUC_4')
            test_class_4.append(float(line[index+6:index+12]))             
        

plt.subplot(1, 3, 1)
plt.plot(Train_loss,label="Train_loss")
plt.plot(Val_loss,label="Val_loss")
plt.plot(test_loss,label="test_loss")
plt.ylim([0.001,5])
plt.legend(loc = "best")

plt.subplot(1, 3, 2)
plt.plot(Val_AUC_0,label="GGG>=1")
plt.plot(Val_AUC_1,label="GGG>=2")
plt.plot(Val_AUC_2,label="GGG>=3")
plt.plot(Val_AUC_3,label="PIRADS>=3")
plt.plot(Val_AUC_4,label="PIRADS>=4")
plt.ylim([0.5,1])
plt.legend(loc = "best")

'''
plt.subplot(1, 4, 3)
plt.plot(test_AUC_0,label="GGG>=1")
plt.plot(test_AUC_1,label="GGG>=2")
plt.plot(test_AUC_2,label="GGG>=3")
plt.plot(test_AUC_3,label="PIRADS>=3")
plt.plot(test_AUC_4,label="PIRADS>=4")
plt.ylim([0,0.5])
plt.legend(loc = "best")
'''

plt.subplot(1, 3, 3)
plt.plot(test_class_0,label="GGG>=1")
plt.plot(test_class_1,label="GGG>=2")
plt.plot(test_class_2,label="GGG>=3")
plt.plot(test_class_3,label="PIRADS>=3")
plt.plot(test_class_4,label="PIRADS>=4")
plt.ylim([0.5,1])
plt.legend(loc = "best")


plt.show()



min_1 = np.max(test_class_0)
Index_1 = np.where(test_class_0 == min_1)

#min_1 = np.max(test_AUC_3)
#Index_1 = np.where(test_AUC_3 == min_1)

Index_1 = Index_1[0]
Index_1 = Index_1[0]

print(Index_1)
'''
print(test_AUC_0[Index_1])
print(test_AUC_1[Index_1])
print(test_AUC_2[Index_1])
print(test_AUC_3[Index_1])
print(test_AUC_4[Index_1])
'''
print(test_class_0[Index_1])
print(test_class_1[Index_1])
print(test_class_2[Index_1])
print(test_class_3[Index_1])
print(test_class_4[Index_1])