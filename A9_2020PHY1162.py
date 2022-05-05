import numpy as np
import matplotlib.pyplot as plt
from MyIVP import *
import pandas as pd
from scipy import stats

def slope(x,S):
    
    return [*S[1:],-S[0] + np.sin(3*x)]

#Dirichlet or neumann bc

def linearshooting(a,b,z_a,z_b,M,f,s0,s1,tol,key = 'd',cf = None):
    dict1 = {'d':0,'n':1,'r1':0,'r2':1}
    def phi(s):
        if key == 'd':
           Ic = [z_a,s]
        elif key == 'n':
            Ic = [s,z_a]
        else :   
            Ic = [s, (cf*s + z_a)]
            
        res = RK4_vec(Ic,a,b,M,2,f)
        y_c = res[1][:,dict1[key]]
        
        return abs(z_b - y_c[-1]),res
    S = np.zeros(1000)
    
    S[0] = s0 ; S[1] = s1
    res_list = [phi(S[0])[1],phi(S[1])[1]]
    for i in range(2,len(S)):
        S[i] = S[i-1] - (((S[i-1]-S[i-2])*(phi(S[i-1])[0]))/(phi(S[i-1])[0] - phi(S[i-2])[0]))
        res_list.append(phi(S[i])[1])
        
        if phi(S[i])[0] <= tol:
           return phi(S[i])[1],S[:i+1],res_list
    return phi(S[-1])[1]  


def Y_anay(x):
    return (3/8)*np.sin(x) - np.cos(x) - (1/8)*np.sin(3*x)

Y_anay = np.vectorize(Y_anay)


def plotting(f,arr,N,title,k_p,sp1_title,sp2_title):
    sig_l = ['o','1','v','*','2','x','o','2','x','*','v','d']
    fig,(ax1,ax2) = plt.subplots(1,2)
    plt.suptitle(title)
    
    for i in range(len(N)):
        ax1.plot(arr[i][0],arr[i][1][:,0],f'--{sig_l[i]}',label = f'for {k_p} = {N[i]}')
        ax2.plot(arr[i][0],arr[i][1][:,1],f'--{sig_l[i]}',label = f'for {k_p} = {N[i]}')
    ax1.plot(arr[2][0],f(arr[2][0]),'b')
    ax1.legend()
    ax2.legend() 
    ax1.set_title(sp1_title)
    ax2.set_title(sp2_title)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel("Y'")



def plot_s(a,b,z_a,z_b,key,tol,condition,cf = None):
    S_arr = linearshooting(a,b,z_a,z_b,32,slope,0,1,tol,key,cf)[1:]
    itr = S_arr[0]
    plt_ar = S_arr[1]
    
    plotting(Y_anay,plt_ar,itr,condition,'s',f'Y vs X for N = 32 and tol = {tol}', f"Y' vs X for N = 32 and tol = {tol}")
    

N = np.logspace(1,6,base=2,num = 6)

def solve_cnt(a,b,z_a,z_b,key,tol,condition,cf = None):

    df = []
    for i in range(len(N)):
        df1 = linearshooting(a,b,z_a,z_b,N[i],slope,0,1,tol,key,cf)[0]
        df.append(df1)
    
    plotting(Y_anay,df,N,condition,'N','Y vs X',"Y' vs X")       
    



#Robin 2
solve_cnt(0,np.pi/2,-1,1,'r2',10**(-12),'Robin-2 BC',-1)
plot_s(0,np.pi/2,-1,1,'r2',10**(-12),'Robin-2 BC',-1)



def Error(a,b,f,N,tol):
    mat = linearshooting(a,b,-1,1,N,slope,0,1,tol,'r2',-1)[0]
    x = mat[0]
    h = x[1] - x[0]
    ynum = mat[1][:,0]
    E_l = abs(ynum - f(x)) 
    E_n = max(E_l)
    E_r = np.sqrt(np.sum(np.power(E_l,2))/len(E_l))
    data = {'x':x,'y_num':ynum,'y_analytic':f(x),'Error':E_l}
    df  = pd.DataFrame(data)
    return np.log(N),np.log(h),np.log(E_n),np.log(E_r),df


Error = np.vectorize(Error,otypes = [float,float,float,float,pd.DataFrame])

rev_t = Error(0,np.pi/2,Y_anay,N,10**(-8))

data1 = {'N':N,'h':np.exp(rev_t[1]),'E_max':np.exp(rev_t[2]),'E_rms':np.exp(rev_t[3])}
df2 = pd.DataFrame(data1)
df2.to_csv('er_n.csv')
print(df2)

cnv_max = []
cnv_rms = []
for i in range(len(rev_t[2])-1):
    cnv_max.append(rev_t[2][i+1]/rev_t[2][i])
    cnv_rms.append(rev_t[3][i+1]/rev_t[3][i])
    
data5 = {'N':N[1:],'Ratio of Max err':cnv_max,'Ratio of rms error':cnv_rms}
df5 = pd.DataFrame(data5)
df5.to_csv('er_ratio.csv')
print(df5)


rev_t[4][1].to_csv('y_er_4.csv')
rev_t[4][2].to_csv('y_er_8.csv')

def err_line(mat,key=0):
    dict2 = {0:'N',1:'h'}
    
    slope_max, intercept_max,r_value, p_value, std_err = stats.linregress(mat[key],mat[2])
    slope_rms, intercept_rms,r_value, p_value, std_err = stats.linregress(mat[key],mat[3])
    
    fig,ax = plt.subplots(1,2)
    plt.suptitle(f'Log Error plots vs {dict2[key]}')
    ax[0].plot(mat[key],mat[2],'o',label = 'error points')
    ax[0].plot(mat[key],mat[key]*slope_max + intercept_max,label = f'Slope = {slope_max}')
    ax[1].plot(mat[key],mat[3],'x',label = 'error points') 
    ax[1].plot(mat[key],mat[key]*slope_rms + intercept_rms,label = f'Slope = {slope_rms}')
    ax[0].set_title(f'E_max vs {dict2[key]}')
    ax[1].set_title(f'E_rms vs {dict2[key]}')
    ax[0].set_xlabel(f'log{dict2[key]}')
    ax[1].set_xlabel(f'log{dict2[key]}')
    ax[0].set_ylabel('log(E_max)')
    ax[1].set_ylabel('log(E_rms)')
    ax[0].legend()
    ax[1].legend()
    plt.show()  

    return slope_max,slope_rms


tu = err_line(rev_t,0)

print(tu)

