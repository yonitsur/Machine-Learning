#################################
# Yoni Tsur 204617963
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

def run_adaboost(X_train, y_train, T, j_theta_sign = None):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    h= [None]*T
    w = np.zeros(T)
    D_t = np.array([1/len(X_train)]*len(X_train))

    for t in range(T):
        h_t = WL(D_t, X_train, y_train, j_theta_sign)
        e_t = sum(D_t[i] for i in range(len(X_train)) if h_t(X_train[i]) != y_train[i])
        w_t = 0.5 * np.log((1-e_t)/e_t )
        exp = [-y_train[i]*w_t*h_t(X_train[i]) for i in range(len(X_train))]  
        D_t = np.multiply(D_t, np.exp(exp)) / np.sum(np.multiply(D_t, np.exp(exp)))
        h[t] = h_t
        w[t] = w_t
                     
    return h, w

def WLbySign(D, X_train, y_train, sign):
    """
    """
    min_err = np.Infinity      
    best_theta = 0
    best_j = 0
    
    for j in range(len(X_train[0])):     
        J = np.array([[X_train[i][j], y_train[i], D[i]] for i in range(len(X_train))])
        J = J[np.argsort(J[:, 0])]
        J = np.vstack((J, np.array([J[len(X_train)-1][0]+1, 0,0])))   # add one more element to the end of J
        err = 0
        for i in range(len(X_train)): 
            if(J[i][1] == sign):
                err += J[i][2]
        if(err < min_err):
            min_err = err
            best_theta = J[0][0]-1
            best_j = j
        for i in range(len(X_train)): 
            err = err - sign*J[i][1]*J[i][2]
            if(err < min_err and J[i][0]!=J[i+1][0]):
                min_err = err
                best_theta = (J[i][0]+J[i+1][0])/2
                best_j = j
                
    return min_err, best_j, best_theta 

def WL(D, X_train, y_train, j_theta_sign):
    """ 
    return best weak learner h: X -> {-1,1}
    """
    min_err1, j1, t1 = WLbySign(D, X_train, y_train, 1)
    min_err2, j2, t2 = WLbySign(D, X_train, y_train, -1)
    if min_err1 < min_err2:
        if(j_theta_sign != None):
            j_theta_sign.append((j1, t1, 1))
        return lambda x: -1+2*(x[j1]<=t1)
    else:
        if(j_theta_sign != None):
            j_theta_sign.append((j2, t2, -1))
        return lambda x: 1-2*(x[j2]<=t2)
        
def empirical_error(X, y, h, alpha):
    """
    """
    errors = np.zeros(len(h))
    p = np.zeros(len(X))
    for t in range(len(h)):
        err = 0
        for i in range(len(X)):
            p[i] += alpha[t] * h[t](X[i])
            err += (p[i]*y[i]<0)
        errors[t] = err/len(X)

    return errors
    

def loss(X, y, h, alpha):
    """
    """
    loss = np.zeros(len(h))
    exp = np.zeros(len(X))    

    for t in range(len(h)):
        for i in range(len(X)):
            exp[i] += -y[i] * alpha[t] * h[t](X[i])
        loss[t] = sum(np.exp(exp))/len(X)

    return loss

##################################################################

def main():

    data = parse_data()
    if not data:
        return

    (X_train, y_train, X_test, y_test, vocab) = data
   
    ###################### a ######################

    def a():
        T = 80
        hypotheses, alpha_vals= run_adaboost(X_train, y_train, T)
        training_error = empirical_error(X_train, y_train, hypotheses, alpha_vals)
        test_error = empirical_error(X_test, y_test, hypotheses, alpha_vals)  
        plt.plot(np.arange(1,T+1), training_error , label = 'training error')
        plt.plot(np.arange(1,T+1), test_error, label = 'test error')
        plt.xlabel('iteration(t)')
        plt.legend()
        plt.savefig('a')

    ###################### b ######################

    def b():
        T = 10
        j_theta_sign = []
        hypotheses, alpha_vals = run_adaboost(X_train, y_train, T, j_theta_sign)

        for i in range(T):
           print("h{} ; word: \"{}\" ; theta:{} ; sign:{} ; weight:{}".
           format(i+1, vocab[j_theta_sign[i][0]], j_theta_sign[i][1], j_theta_sign[i][2], alpha_vals[i]))
    
    ###################### c ######################

    def c():
        T = 80
        hypotheses, alpha_vals= run_adaboost(X_train, y_train, T)
        training_loss = loss(X_train, y_train, hypotheses, alpha_vals) 
        test_loss = loss(X_test, y_test, hypotheses, alpha_vals)  
        plt.plot(np.arange(1,T+1), training_loss ,label = 'training loss')
        plt.plot(np.arange(1,T+1), test_loss ,label = 'test loss')
        plt.xlabel('iteration(t)')
        plt.legend()
        plt.savefig('c')


    # a()
    # b()
    # c()

if __name__ == '__main__':
    main()



