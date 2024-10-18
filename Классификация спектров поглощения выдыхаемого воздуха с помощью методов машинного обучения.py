import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
import math

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from numpy.polynomial import Polynomial

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


cmap1 = ListedColormap(['blue','orange'])
cmap2 = ListedColormap(['green','red'])

def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x - x0))) + b
    return y
  


cm = 1/2.54




"""
поиск спектров молекул в папке path и копирование от туда данных

"""
path = 'Gas_comp'
pathCompon = []

nameCompon = os.listdir(path)

for i in nameCompon:
    i = (path) + "/" + (i)
    pathCompon.append(i)
    
df = pd.read_csv(pathCompon[0], sep="\t", usecols=[0], names=["wavenumber"])

for a, b in zip(pathCompon, nameCompon):
    a = pd.read_csv(a, sep="\t", usecols=[0, 1], names=["wavenumber", b])
    df = df.merge(a)
    
wavenumber = df['wavenumber'].values
del df['wavenumber'] # удаление обьекта столбца у длины волны
    
df_coef = np.zeros((1, len(nameCompon)))
df_coef = pd.DataFrame(df_coef, columns = nameCompon)

"""
конец

"""


# шляпа не нужная хз зачем ее сделал
"""
тут мы будем задавать коэфиценты
"""

def Coef_create(name_com = []):
        name = name_com[::3]
        num = name_com[1::3]
        df_coef_copy = df_coef.copy()
        
        if len(name_com) != 0:
            for a, b in zip(name, num):
                df_coef_copy[a][0] = b
            return df_coef_copy.values
        else:
            print("не указаны значения")

"""
конец

"""




"""
тут мы создаем смесь
"""

def Mix(data = list, name_com = [] ,name_num = []):

    if len(data) != 0:
        data0 = data[0]
        data1 = data[1]
    else:
        data0 = []
        
    name = name_num[0::3]
    var1 = name_num[1::3]
    var2 = name_num[2::3]
    
    if len(name_com) != 0:
        arr_coef = Coef_create(name_com)
    else:
        print("нету заданных данных")
    
    # тутц
    arr_coefall = np.array([])
    for a, b in zip(var1, var2):
        arr_coef1 = np.random.uniform(a, b, size=(1, 100))
        
        if len(arr_coefall) == 0:
            arr_coefall = arr_coef1
        else:
            arr_coefall = np.vstack((arr_coefall, arr_coef1))
    
    # тутц        
    arr_coefall = arr_coefall.T
    
    
    
    # тут
    df_coef_copy = df_coef.copy()
    df_coef_copy = pd.concat([df_coef]*100, ignore_index=True)
    
    
    
    # тут
    
    for a, b in zip(name, range(0, len(name))):
        
        df_coef_copy[a] = arr_coefall[:,b]
    
    # матричное произведение коэфицентов компонент на интенсивности компонент
    df_pre = np.dot(df.values, df_coef_copy.values.T)
    
    # именование концентраций
    name_mix = ''
    for a, b in zip(name, var1):
        name_mix = name_mix + f'{a}:{b} '        
    name_mix = name_mix[:-1]
    
    # добовление имени с концентрациями в список data0
    data0.append(f'спектр с концентрациями {name_mix}')
    
    # добавлени данных смеси в список data1
    if len(data) != 0:
        data1 = np.hstack((data1, df_pre))
    else:
        data1 = df_pre
        
    # добавление списков data1 и data0 в общий список data
    data = [data0, data1]

    return data


"""
конец

"""


"""
конец



"""
def calcul(pre_data, numcompon = 0, begin_wave = int ,end_wave = int, z = float):
    name_id = pre_data[0]
    df = pre_data[1]
         
    if numcompon != 0:
        pca = PCA(numcompon)
        df_Pca = pca.fit_transform(df[begin_wave:end_wave,:])
    
    else:
        pca = PCA()
        df_Pca = pca.fit_transform(df[begin_wave:end_wave,:])
            
    df_score = pca.components_
    df_score = df_score.T
    
    
    # _round_to = lambda z, digits: 0 if z == 0 else round(z, max(int(-math.floor(math.log10(abs(z - int(z))))) + digits - 1, digits))
    h = str(z).replace('.',',')
    # тут мы создаем дефолтные графики счетов
    for i in range(1,3):
        
        plt.scatter(df_score[100*(i-1):100*i,0],df_score[100*(i-1):100*i,2])
        
    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.savefig(fname=f'C:/Users/TON/Desktop/project_dip/graph_{begin_wave}-{end_wave}_wave/PCA/shym_{h}.png')
    plt.show()
    
    return df_score
"""
конец

"""





"""
шум и SVM
"""

def shym_SVM(data, begin_wave, end_wave):
    
    sd_Sensitivity = [] # чувстыитеьность 
    sd_Specificity = [] # спецификация
    
    shym = np.linspace(0.00000001, 0.004, 20)
    std_Sensitivity = []
    std_Specificity = []
    
    for z in shym:
          
        newdata = [e.copy() for e in data]
        newdata[1] = newdata[1] + (z *np.random.normal(0, 0.5, size=(newdata[1].shape)))
        a = calcul(newdata, 0 , begin_wave, end_wave, z) # закидываем копию данных с шумом

        # метод опрорных векторов
        Sensitivity = [] # чувстыитеьность 
        Specificity = [] # спецификация
        
        y = np.zeros(200)
        y[:100] = y[:100] - 1
        y[100:] = y[100:] + 1
        
        
        for i in range (0, 1000):
            
            X_train, X_test, y_train, y_test = train_test_split(a[:,:3:2], y, test_size = 0.3)
            
            clf = svm.SVC(kernel = 'sigmoid')
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            
            """
            tn - количество истинных негативов 
            fp - ложные срабатывания
            fn - ложноотрицательные результаты
            tp - положительные срабатывания
            """
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, normalize='true').ravel()
            Sensitivity.append((tp/(tp + fn)))
            Specificity.append((tn/(tn + fp)))
            
        
        sd_Sensitivity.append(np.mean(Sensitivity))
        sd_Specificity.append(np.mean(Specificity))
       
        std_Sensitivity.append(np.std(Sensitivity))
        std_Specificity.append(np.std(Sensitivity))
        
        
        mask1 = (y_test==1)
        mask2 = (y_test==-1)
        ax = plt.gca()
        ax.scatter(X_test[mask1,0], X_test[mask1,1], label='здоровые',edgecolors='black',c=['green'])
        ax.scatter(X_test[mask2,0], X_test[mask2,1], label='болеющие раком легких',edgecolors='black', c=['red'])
        ax.legend()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # plt.imshow(Z, interpolation='nearest',
        #     extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
        #     origin='lower', cmap=plt.cm.PuOr_r)
        
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=3, linestyles='dashed')
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.4) 
        
        
        plt.scatter(X_test[mask1,0], X_test[mask1,1], label='здоровые',edgecolors='black',c=['green'])
        plt.scatter(X_test[mask2,0], X_test[mask2,1], label='болеющие раком легких',edgecolors='black',c=['red'])
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        
        plt.title(f'на спектральном диапазоне от {wavenumber[begin_wave]} см^-1 до {wavenumber[end_wave]} см^-1')

        
        plt.savefig(fname=f'C:/Users/TON/Desktop/project_dip/graph_{begin_wave}-{end_wave}_wave/SVM/shym_{z}.png')
        # plt.grid()
        plt.show()
        
    sd_Sensitivity = np.array(sd_Sensitivity) 
    sd_Specificity = np.array(sd_Specificity) 
    std_Sensitivity = np.array(std_Sensitivity) 
    std_Specificity = np.array(std_Specificity)
    
    print('***********')
    print(f'шум = {shym[6]}, на частотном диапазоне - от {wavenumber[begin_wave]} см^-1 до {wavenumber[end_wave]} см^-1')
    print(f'чувствительность среднее - {sd_Sensitivity[6]} ,дисперсия - {std_Sensitivity[6]}')
    print(f'спецефичность среднее - {sd_Specificity[6]} ,дисперсия - {std_Specificity[6]}')
    print('***********')
    # отрисовка чувстыитеьности и спецефичности
    
    fig, axs = plt.subplots(figsize=(20*cm, 10*cm), nrows=1, ncols=2, layout="tight")
    
    popt, _ = spo.curve_fit(sigmoid, shym, sd_Sensitivity,  maxfev=30000)
    L_fit, x0_fit, k_fit, b_fit = popt
    
    # axs[0].plot(shym, sd_Sensitivity)
    axs[0].plot(shym, sigmoid(shym, L_fit, x0_fit, k_fit, b_fit)*100)
    axs[0].errorbar(shym, sigmoid(shym, L_fit, x0_fit, k_fit, b_fit)*100, std_Sensitivity*100, ls=' ', marker='x', capsize=2.5, elinewidth=1, color='black')
    # axs[0].set_title('Sensitivity')
    # axs[0].legend(['средние значения Sensitivity', 'среднеквадратичное откланение Sensitivity'], fontsize=10)
    
    popt, _ = spo.curve_fit(sigmoid, shym, sd_Specificity,  maxfev=30000)
    L_fit, x0_fit, k_fit, b_fit = popt
    # axs[1].plot(shym, sd_Specificity)
    axs[1].plot(shym, sigmoid(shym, L_fit, x0_fit, k_fit, b_fit)*100)
    axs[1].errorbar(shym, sigmoid(shym, L_fit, x0_fit, k_fit, b_fit)*100, std_Specificity*100, ls=' ', marker='x', capsize=2.5, elinewidth=1, color='black')
    # axs[1].set_title('Specificity')
    # axs[1].legend(['средние значения Specificity', 'среднеквадратичное откланение Specificity'])
    
    
    axs[0].set_xlabel('шум, см^-1')
    axs[0].set_ylabel('Sensitivity, %')
    axs[1].set_xlabel('шум, см^-1')
    axs[1].set_ylabel('Specificity, %')
    
    fig.legend(['ср. значение', 'ср. кв. откланение'], fontsize=8,loc='center', bbox_to_anchor=(0.9, 0.82))
    fig.suptitle(f'Sensitivity и Specificity на спектральном диапазоне от {wavenumber[begin_wave]} см^-1 до {wavenumber[end_wave]} см^-1')
    plt.savefig(fname=f'C:/Users/TON/Desktop/project_dip/graph_{begin_wave}-{end_wave}_wave/sens&spec.png')
    plt.show()
    del fig, axs
    
    
    
    
    
    
    




data = []
# df_coef['H2O'] = 0.03
# df_coef['CO2'] = 0.04

# задаем смеси, которые мы измерили у 100 человек типа больных и 100 здоровых
data = Mix(data ,[],[
    'H2O', 0.05, 0.06,
    'CO2', 0.03, 0.05,
    'CO', 1*10**-6, 10**-5,
    'NO', 1000*10**-9, 3500*10**-9,
    'NH3', 1*10**-9, 1000*10**-9,
    'CH3OH', 0.02*10**-6, 1.09*10**-6,
    'CH3CN', 0.2*10**-6, 0.37*10**-6,
    'CH4',30*10**-6,100*10**-6,
    'C2H4',1*10**-9,100*10**-9
    ])

data = Mix(data ,[],[
    'H2O', 0.05, 0.06,
    'CO2', 0.03, 0.05,
    'CO', 0.1*10**-6, 6*10**-6,
    'NO', 3*10**-9, 1000*10**-9,
    'NH3', 1*10**-9, 500*10**-9,
    'CH3OH', 1*10**-9, 10*10**-9,
    'CH3CN', 0.2*10**-6, 0.21*10**-6,
    'CH4',1.5*10**-6,3*10**-6,
    'C2H4',1*10**-9,30*10**-9
    ])



x=np.array([0,1,2,3])

for i in 2**x:
    
    wave_list = []
    wave_step = (wavenumber.shape[0]-1)/i
    v = 0
    while v != (wavenumber.shape[0]-1):
        wave_list.append(v)
        wave_list.append(v)
        v = v + wave_step
        
    wave_list.append((wavenumber.shape[0]-1))
    wave_list = [int(item) for item in wave_list]
    wave_list.pop(0)    
    
    
    for j, k in zip(wave_list[0::2],wave_list[1::2]):
        
        # os.mkdir(f'C:/Users/TON/Desktop/project_dip/graph_{j}-{k}_wave/SVM')
        
        shym_SVM(data, j, k)





plt.plot(wavenumber,data[1][:,0:100])
plt.xticks([wavenumber[0], wavenumber[1262], wavenumber[2525], wavenumber[3787], wavenumber[5050], wavenumber[6312], wavenumber[7575], wavenumber[8837], wavenumber[10100]])

