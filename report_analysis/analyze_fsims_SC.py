
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import json 
import struct
import random
import argparse


def float_to_hex(f):
    h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return h[2:len(h)]

def hex_to_float(h):
    return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

def int_to_float(h):
    return float(struct.unpack(">f",struct.pack(">I",h))[0])



def load_fsim_report(file_name):
    pwd=os.getcwd()
    results_path=os.path.join(pwd,"FSIM_logs/Analysis")
    df_report= pd.read_csv(file_name,index_col=[0]) 
    num_kernels=max(df_report['kernel'])+1
    print(num_kernels)
    num_channels=max(df_report['channel'])+1
    
    df_report['Abs_error']=df_report['Abs_error']
    df_report=df_report.sort_values(by=['kernel','channel','Abs_error'])
    df_report['MRAD']=(100*(df_report['fault_ACC@1']-df_report['gold_ACC@1'])/df_report['gold_ACC@1'])
    df_report['MRADk']=(100*(df_report['fault_ACC@k']-df_report['gold_ACC@k'])/df_report['gold_ACC@k'])

    layer=int(df_report['Layer'].min())

    #let's perform the analisys per bit
    df_bit_wise_results_MRAD=pd.DataFrame()
    for bit in range(19,32):
        df_new=df_report.loc[((df_report['BitMask']==2**bit))]
        df_new_col=pd.DataFrame()
        df_new_col[f"bit{bit}"]=df_new['MRAD']
        df_bit_wise_results_MRAD=pd.concat([df_bit_wise_results_MRAD,df_new_col],axis=1)
    #df_bit_wise_results_MRAD.set_axis(['bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31'],axis="columns",inplace=True)
    df_bit_wise_results_MRAD.describe().to_csv(f"{results_path}/MRAD1_layer{layer}.csv")
    myFig = plt.figure()
    bp=df_bit_wise_results_MRAD.boxplot(rot=45)
    myFig.savefig(f"{results_path}/MRAD1_layer{layer}.jpg", format="jpg")

    df_bit_wise_results_MRAD=pd.DataFrame()
    for bit in range(19,32):
        df_new=df_report.loc[((df_report['BitMask']==2**bit))]
        df_new_col=pd.DataFrame()
        df_new_col[f"bit{bit}"]=df_new['MRADk']
        df_bit_wise_results_MRAD=pd.concat([df_bit_wise_results_MRAD,df_new_col],axis=1)
    #df_bit_wise_results_MRAD.set_axis(['bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31'],axis="columns",inplace=True)
    df_bit_wise_results_MRAD.describe().to_csv(f"{results_path}/MRADk_layer{layer}.csv")
    myFig = plt.figure()
    bp=df_bit_wise_results_MRAD.boxplot(rot=45)
    myFig.savefig(f"{results_path}/MRADk_layer{layer}.jpg", format="jpg")
    


    df_bit_wise_results_Abs_error=pd.DataFrame()
    for bit in range(19,32):
        df_new=df_report.loc[((df_report['BitMask']==2**bit)&(df_report['MRAD']<0))]
        df_new_col=pd.DataFrame()
        df_new_col[f"bit{bit}"]=df_new['Abs_error']
        df_bit_wise_results_Abs_error=pd.concat([df_bit_wise_results_Abs_error,df_new_col], axis=1)
    #df_bit_wise_results_Abs_error.set_axis(['bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31'],axis="columns",inplace=True)
    df_bit_wise_results_Abs_error.describe().to_csv(f"{results_path}/Error_figures_layer{layer}.csv")
    myFig = plt.figure()
    bp=df_bit_wise_results_Abs_error.boxplot(column=['bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit31'])
    #plt.yscale('log')
    myFig.savefig(f"{results_path}/Error_figures_layer{layer}.jpg", format="jpg")


    df_bit_wise_results_Abs_error=pd.DataFrame()
    
    for ch in range(0,int(num_kernels)):
        df_new=df_report.loc[((df_report['kernel']==ch))]
        df_new_col=pd.DataFrame()
        df_new_col[f"{ch}"]=df_new['MRAD']
        df_bit_wise_results_Abs_error=pd.concat([df_bit_wise_results_Abs_error,df_new_col], ignore_index=True, axis=1)
    #df_bit_wise_results_Abs_error.set_axis(['bit19','bit20','bit21','bit22','bit23','bit24','bit25','bit26','bit27','bit28','bit29','bit30','bit31'],axis="columns",inplace=True)
    
    df_bit_wise_results_Abs_error.describe().to_csv(f"{results_path}/Kernels_layer{layer}.csv")


    fig, axes = plt.subplots(1)
    df_bit_wise_results_Abs_error.iloc[:,0:20].boxplot()
    #plt.yscale('log')
    fig.savefig(f"{results_path}/Kernels_layer{layer}.jpg", format="jpg")

    print(df_report['MRAD'].describe())
    # dft=df_report.loc[(df_report['BitMask']!=2**30)]
    dft=df_report
    dftx=df_report.loc[(df_report['BitMask']!=2**30)&(df_report['MRAD']<-0)&(df_report['MRAD']<-0)]

    df= dft[['kernel','Abs_error','Ffree_Weight','Faulty_weight','MRAD','BitMask']]
    df['kernel']=(df['kernel'].astype(int)).astype('category')
    #df['Abs_error']=df['Faulty_weight'].astype(int).apply(int_to_float)
    df['Ffree_Weight']=df['Ffree_Weight'].astype(int).apply(int_to_float)
    df['Faulty_weight']=df['Faulty_weight'].astype(int).apply(int_to_float)

    dftx = dftx[['kernel','Abs_error','Ffree_Weight','Faulty_weight','MRAD','BitMask']]
    dftx['kernel']=(dftx['kernel'].astype(int)).astype('category')
    #dftx['Abs_error']=dftx['Faulty_weight'].astype(int).apply(int_to_float)
    dftx['Ffree_Weight']=dftx['Ffree_Weight'].astype(int).apply(int_to_float)
    dftx['Faulty_weight']=dftx['Faulty_weight'].astype(int).apply(int_to_float)
    

    # df.to_csv(f"{results_path}/masked_report{layer}.csv")

    print(df)
    

    fig, ax = plt.subplots()                
    df.hist(column='Ffree_Weight', bins=25,ax=ax,layout=(1,1)) 
    dftx.hist(column='Ffree_Weight',bins=25,ax=ax,layout=(1,1)) 
    # df.hist(column='Ffree_Weight', bins=25,ax=ax,layout=(1,1)) 
    #dftx.hist(column='Abs_error',bins=25,ax=ax,layout=(1,1)) 
    #for axs in ax:
    ax.set_yscale('log')
    fig.savefig(f"{results_path}/hist.jpg")

    print(df.describe())


    # degradation_data=df.loc[(df['MRAD']<0)]
    # improvement_data=df.loc[(df['MRAD']>0)]
    # nochange_data=df.loc[(df['MRAD']==0)]

    # degradation_data=df_report.loc[(df_report['MRAD']<0)]
    # improvement_data=df_report.loc[(df_report['MRAD']>0)]
    # nochange_data=df_report.loc[(df_report['MRAD']==0)]


    degradation_data=df_report.loc[(df_report['MRAD']<0)]
    improvement_data=df_report.loc[(df_report['MRAD']>0)]
    nochange_data=df_report.loc[(df_report['MRAD']==0)]



    print(degradation_data.describe())
    print(improvement_data.describe())
    print(nochange_data.describe())


    flip_0_1=df_report.loc[(df_report['Ffree_Weight']<df_report['Faulty_weight'])&(df_report['BitMask']!=2**30)][['kernel','Abs_error','Ffree_Weight','Faulty_weight','MRAD','BitMask']]
    flip_1_0=df_report.loc[(df_report['Ffree_Weight']>df_report['Faulty_weight'])&(df_report['BitMask']!=2**30)][['kernel','Abs_error','Ffree_Weight','Faulty_weight','MRAD','BitMask']]
    
    flip_0_1['Exp']=flip_0_1['Ffree_Weight'].astype(int).apply(lambda x: (x&2139095040)>>23)
    flip_1_0['Exp']=flip_1_0['Ffree_Weight'].astype(int).apply(lambda x: (x&2139095040)>>23)
    
    flip_0_1['Ffree_Weight']=flip_0_1['Ffree_Weight'].astype(int).apply(int_to_float)
    flip_0_1['Faulty_weight']=flip_0_1['Faulty_weight'].astype(int).apply(int_to_float)
    flip_1_0['Ffree_Weight']=flip_1_0['Ffree_Weight'].astype(int).apply(int_to_float)
    flip_1_0['Faulty_weight']=flip_1_0['Faulty_weight'].astype(int).apply(int_to_float)

    
    flip_0_1['BitMask']=flip_0_1['BitMask'].astype(int).apply(np.log2)
    flip_1_0['BitMask']=flip_1_0['BitMask'].astype(int).apply(np.log2)

    flip_0_1.to_csv(f"{results_path}/flip_0_1{layer}.csv")
    flip_1_0.to_csv(f"{results_path}/flip_1_0{layer}.csv")

    print(flip_0_1.describe())
    print(flip_1_0.describe())



def fp32_error_plot2(bits=[31],data=0.75):
    hex_data=float_to_hex(data)
    data_int=int(hex_data,16)
    #print(bits)
    for bit in bits:
        data_int=(data_int)^(2**bit)
    return(data-int_to_float(data_int))



def fp32_error_plot(bit,data):
    hex_data=float_to_hex(data)
    sign=(int(hex_data,16) & int('80000000',16))>>31
    exponent=(int(hex_data,16) & int('7f800000',16))>>23
    mantiza_bin=(int(hex_data,16) & int('007fffff',16))
    mantiza_real=0
    i=23
    while(i>0):
        bitm=((mantiza_bin>>(23-i))&1)
        mantiza_real+=bitm*2**(-i)
        i-=1
        
    #print(hex_data,sign,(exponent),mantiza_bin,mantiza_real)

    #sign=0
    #exponent=126
    #mantiza_bin=4194304
    #mantiza_real=0.5


    en_bit_flip_sign=0
    en_bit_flip_exp=0
    en_bit_flip_mantiza=0

    sign_bit_exp=1
    sign_bit_man=1
    exp_val=0
    bit_man_val=0

    if bit==31:
        en_bit_flip_sign=1

    elif(bit<=30 and bit>=23):
        en_bit_flip_exp=1
        exp_val=2**(bit-23)
        
        if (exp_val & exponent != 0):
            sign_bit_exp=(-1)
        else:
            sign_bit_exp=(1)

    else:
        en_bit_flip_mantiza=1
        
        bit_man_val=2**(bit)

        if (bit_man_val & mantiza_bin != 0):
            sign_bit_man=(-1)
        else:
            sign_bit_man=(1)

    
    p1=2**(exponent-127)
    p2=(-1)**sign
    p3=(-1)**(en_bit_flip_sign*(sign ^ 1))
    p4=2**(en_bit_flip_exp*(sign_bit_exp*exp_val))
    p5=(1+en_bit_flip_mantiza*(sign_bit_man)*(bit_man_val)*(2**(-23)))
    error=p1*(p2+mantiza_real*(p2-p3*p4)-p3*p4*p5)
        
    #print(p1,p2,p3,p4,p5)
    return(error)

def main():
    bit=sys.argv[1]
    pwd=os.getcwd()
    results_path=os.path.join(pwd,f"FSIM_logs/Analysis/{bit}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data=[]
    for _ in range(2000000):
        data.append(random.random()*2)
    data=sorted(data)
    #print(data, errors)
    
    lbits=[]
    nbit=8
    while(nbit!=0):
        rbit=random.randint(0,31)
        if(rbit not in lbits) and (rbit != 30 and rbit != 31):
            lbits.append(rbit)  
            nbit-=1  
    lbits=sorted(lbits)
    print(lbits)

    for trace_bit in range(1,32):
        data=[]
        for _ in range(2000000):
            data.append(random.random()*2)
        data=sorted(data)
        fig=plt.figure()

        for i in range(trace_bit):
            errors1=[abs(fp32_error_plot2([j if "up" in bit else i-j for j in range(i+1)],x)) for x in data]
            plt.plot(data,errors1,'r')
        
        errors2=[abs(fp32_error_plot2([trace_bit],x)) for x in data]
        plt.plot(data,errors2,'b')
        #plt.plot(data,[abs((errors1[idx]-errors2[idx])/errors1[idx]) for idx in range(len(data))],'g')
        plt.yscale('log')
        fig.savefig(f"{results_path}/error_bit_{trace_bit}.jpg")
        plt.close()



def main_rnd():
    bit=sys.argv[1]
    pwd=os.getcwd()
    results_path=os.path.join(pwd,f"FSIM_logs/Analysis/{bit}")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    data=[]
    for _ in range(2000000):
        data.append(random.random()*2)
    data=sorted(data)
    #print(data, errors)
    
    lbits=[]
    nbit=8
    while(nbit!=0):
        rbit=random.randint(0,31)
        if(rbit not in lbits) and (rbit != 30 and rbit != 31):
            lbits.append(rbit)  
            nbit-=1  
    lbits=sorted(lbits)
    print(lbits)

    for trace_bit in range(10,32):       
        fig=plt.figure()        
        for i in range(0,trace_bit-2):
            data=[]
            for _ in range(2000000):
                data.append(random.random()*2)
            data=sorted(data)  
            nnbbits=[trace_bit-2]        
            #nnbbits=[random.randint(0,trace_bit-1) for _ in range(i+1)]
            nbit=i+1
            while(nbit!=0):
                rbit=random.randint(0,trace_bit-3)
                if(rbit not in nnbbits):
                    nnbbits.append(rbit)  
                    nbit-=1  
            nnbbits=sorted(nnbbits)
            print(nnbbits)
            errors1=[abs(fp32_error_plot2(nnbbits,x)) for x in data]
            #print(errors1)
            plt.plot(data,errors1,'r')
        
        data=[]
        for _ in range(2000000):
            data.append(random.random()*2)
        data=sorted(data)
        #fig=plt.figure()
        errors2=[abs(fp32_error_plot2([trace_bit-1,trace_bit],x)) for x in data]
        plt.plot(data,errors2,'b')
        #plt.plot(data,[abs((errors1[idx]-errors2[idx])/errors1[idx]) for idx in range(len(data))],'g')
        plt.yscale('log')
        plt.xscale('log')
        fig.savefig(f"{results_path}/error_bit_{trace_bit}.jpg")
        plt.close()

def main2():
    file_name=sys.argv[1]
    pwd=os.getcwd()
    results_path=os.path.join(pwd,f"FSIM_logs/Analysis")
    if not os.path.exists(results_path):
        os.makedirs(results_path)    
    load_fsim_report(file_name)




if __name__=="__main__":  
    if len(sys.argv)>2: 
        if sys.argv[2]:
            main_rnd()
        else:
            main()
    else:
        main2()